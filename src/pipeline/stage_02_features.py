import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils.config_loader import load_config, get_symbols
from src.utils.state_manager import is_stage_complete, update_project_state
from src.utils.logger import get_logger
from src.utils.io_utils import read_checkpoint, checkpoint_exists, write_features
from src.features.feature_pipeline import build_features_for_symbol, save_feature_manifest
import pandas as pd

logger = get_logger("stage_02_features")


def _feature_config_hash(cfg) -> str:
    from omegaconf import OmegaConf
    payload = {
        "features": OmegaConf.to_container(cfg.features, resolve=True),
        "train_end": str(cfg.data.train_end),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _read_saved_hash(features_dir: Path) -> str:
    p = features_dir / "feature_config_hash.txt"
    return p.read_text().strip() if p.exists() else ""


def _write_saved_hash(features_dir: Path, h: str) -> None:
    features_dir.mkdir(parents=True, exist_ok=True)
    (features_dir / "feature_config_hash.txt").write_text(h)


def _process_symbol(args: tuple) -> tuple:
    symbol, cfg_dict, train_end, force = args
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(cfg_dict)

    checkpoints_dir = Path(cfg.data.checkpoints_dir)
    features_dir = Path(cfg.data.features_dir)
    processed_dir = Path(cfg.data.processed_dir)

    # Check 15m checkpoint exists (primary timeframe)
    if not checkpoint_exists("ingest", symbol, "15m", checkpoints_dir):
        return symbol, None, f"No ingest checkpoint for {symbol} 15m"

    # Skip if feature file already exists (allows resuming after cross-sectional OOM).
    # With --force, delete the existing file so features are recomputed from scratch.
    feature_path = features_dir / f"{symbol}_15m_features.parquet"
    if feature_path.exists():
        if force:
            feature_path.unlink()
        else:
            import pyarrow.parquet as _pq
            existing_cols = _pq.read_schema(str(feature_path)).names
            return symbol, existing_cols, None

    try:
        df_15m = read_checkpoint("ingest", symbol, "15m", checkpoints_dir)

        # Load higher timeframes if available
        df_1h = None
        if checkpoint_exists("ingest", symbol, "1h", checkpoints_dir):
            df_1h = read_checkpoint("ingest", symbol, "1h", checkpoints_dir)

        df_4h = None
        if checkpoint_exists("ingest", symbol, "4h", checkpoints_dir):
            df_4h = read_checkpoint("ingest", symbol, "4h", checkpoints_dir)

        df_1d = None
        if checkpoint_exists("ingest", symbol, "1d", checkpoints_dir):
            df_1d = read_checkpoint("ingest", symbol, "1d", checkpoints_dir)

        # Load macro and onchain panels (15m resolution)
        macro_panel_path = processed_dir / "macro_panel_15m.parquet"
        onchain_panel_path = processed_dir / "onchain_panel_15m.parquet"
        macro_panel = pd.read_parquet(macro_panel_path) if macro_panel_path.exists() else None
        onchain_panel = pd.read_parquet(onchain_panel_path) if onchain_panel_path.exists() else None

        # BTC reference for funding divergence (skip for BTC itself)
        btc_df = None
        if symbol != "BTCUSDT" and checkpoint_exists("ingest", "BTCUSDT", "15m", checkpoints_dir):
            btc_df = read_checkpoint("ingest", "BTCUSDT", "15m", checkpoints_dir)

        features_df = build_features_for_symbol(
            symbol, df_15m, df_1h, df_4h, df_1d,
            macro_panel, onchain_panel, btc_df, cfg, train_end
        )

        write_features(features_df, symbol, "15m", features_dir)
        return symbol, features_df.columns.tolist(), None

    except Exception as e:
        import traceback
        return symbol, None, f"{e}\n{traceback.format_exc()}"


def run(cfg, force: bool = False, symbol_filter: str = None) -> None:
    from omegaconf import OmegaConf

    features_dir = Path(cfg.data.features_dir)
    current_hash = _feature_config_hash(cfg)
    saved_hash = _read_saved_hash(features_dir)
    config_changed = current_hash != saved_hash

    if not force and is_stage_complete("features"):
        if not config_changed:
            logger.info("Stage 2 already complete and feature config unchanged — skipping.")
            return
        logger.info(f"Stage 2: feature config changed (hash {saved_hash[:8]} → {current_hash[:8]}) — recomputing all features.")

    if force:
        logger.info("Stage 2: --force, recomputing all features.")
    elif config_changed:
        pass  # already logged above

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    all_symbols = get_symbols(cfg)
    if symbol_filter:
        all_symbols = [s for s in all_symbols if s.get("name", s.get("symbol")) == symbol_filter]
    symbol_names = [s.get("name", s.get("symbol")) for s in all_symbols]

    train_end = str(cfg.data.train_end)
    issues = []
    all_feature_cols = {}

    # Phase 1: per-symbol features — parallel with ProcessPoolExecutor
    # Use max_workers=4 to avoid OOM on large feature sets
    args_list = [(sym, cfg_dict, train_end, force) for sym in symbol_names]

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_process_symbol, args): args[0] for args in args_list}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                sym, cols, err = future.result()
                if err:
                    logger.error(f"{sym}: {err}")
                    issues.append(f"{sym}: {str(err)[:200]}")
                else:
                    all_feature_cols[sym] = cols
                    logger.info(f"{sym}: {len(cols)} features computed")
            except Exception as e:
                logger.error(f"{symbol}: executor error: {e}")
                issues.append(f"{symbol}: {str(e)[:200]}")

    # Phase 2: cross-sectional features (serial — requires all symbols loaded)
    # Applied after all per-symbol features are computed
    if len(all_feature_cols) > 1:
        logger.info("Phase 2: cross-sectional features")
        _apply_cross_sectional_features(symbol_names, cfg, train_end, issues)

    # Save feature manifest
    features_dir = Path(cfg.data.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    save_feature_manifest(all_feature_cols, features_dir / "feature_manifest.json")

    update_project_state("features", "done", issues, output_dir=str(features_dir))
    logger.info(f"Stage 2 complete. {len(all_feature_cols)}/{len(symbol_names)} symbols processed.")


def _apply_cross_sectional_features(symbol_names: list, cfg, train_end: str, issues: list) -> None:
    import pyarrow.parquet as pq
    from src.utils.io_utils import read_features, write_features
    from src.features.cross_sectional import fit_cross_sectional_stats_from_files, apply_cross_sectional_ranks

    features_dir = Path(cfg.data.features_dir)
    checkpoints_dir = Path(cfg.data.checkpoints_dir)
    cs_stats_path = checkpoints_dir / "cross_sectional_stats.pkl"
    train_end_ts = pd.Timestamp(train_end, tz="UTC")

    # Pass 1: scan column names via parquet metadata only (no data loaded)
    logger.info("Cross-sectional pass 1: scanning feature columns")
    col_sets = []
    available_symbols = []
    parquet_paths = []
    for sym in symbol_names:
        path = features_dir / f"{sym}_15m_features.parquet"
        if not path.exists():
            logger.warning(f"Feature file missing for {sym}, skipping cross-sectional")
            continue
        try:
            schema = pq.read_schema(str(path))
            cols = set(schema.names)
            col_sets.append(cols)
            available_symbols.append(sym)
            parquet_paths.append(path)
        except Exception as e:
            logger.warning(f"Could not read schema for {sym}: {e}")

    if len(available_symbols) < 2:
        return

    common_cols = list(set.intersection(*col_sets))
    feature_cols = [c for c in common_cols if c not in ("is_warmup", "__index_level_0__")]
    logger.info(f"Cross-sectional: {len(feature_cols)} common features across {len(available_symbols)} symbols")

    # Pass 2: column-by-column streaming fit — peak RAM ~5MB per column, never loads a full DataFrame
    logger.info(f"Cross-sectional pass 2: fitting stats (column-streaming, {len(available_symbols)} symbols)")
    try:
        fit_cross_sectional_stats_from_files(parquet_paths, feature_cols, train_end_ts, cs_stats_path)
    except Exception as e:
        logger.error(f"Cross-sectional stat fitting failed: {e}")
        issues.append(f"cross_sectional_fit: {e}")
        return

    # Pass 3: apply ranks — one full DataFrame at a time, write and discard
    logger.info("Cross-sectional pass 3: applying ranks per symbol")
    for sym in available_symbols:
        try:
            df = read_features(sym, "15m", features_dir)
            df_ranked = apply_cross_sectional_ranks(df, cs_stats_path, feature_cols)
            write_features(df_ranked, sym, "15m", features_dir)
            del df, df_ranked
        except Exception as e:
            logger.warning(f"Cross-sectional apply failed for {sym}: {e}")
            issues.append(f"cross_sectional_apply_{sym}: {e}")
