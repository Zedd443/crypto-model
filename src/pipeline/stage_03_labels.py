import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
from src.utils.config_loader import get_symbols
from src.utils.state_manager import is_stage_complete, update_project_state
from src.utils.logger import get_logger
from src.utils.io_utils import checkpoint_exists, read_checkpoint, write_pipeline_diagnostics
from src.labels.triple_barrier import label_all_bars
from src.labels.sample_weights import compute_return_weights, compute_label_uniqueness, combine_weights

logger = get_logger("stage_03_labels")


def _save_labels(labels_df: pd.DataFrame, weights: pd.Series, symbol: str, labels_dir: Path) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels_path = labels_dir / f"{symbol}_15m_labels.parquet"
    weights_path = labels_dir / f"{symbol}_15m_weights.parquet"
    labels_df.to_parquet(labels_path)
    weights.to_frame("sample_weight").to_parquet(weights_path)
    logger.debug(f"Labels saved: {labels_path}")


def _label_symbol(symbol: str, cfg, checkpoints_dir: Path, labels_dir: Path) -> tuple:
    try:
        if not checkpoint_exists("ingest", symbol, "15m", checkpoints_dir):
            return symbol, f"No ingest checkpoint for {symbol}"

        df = read_checkpoint("ingest", symbol, "15m", checkpoints_dir)

        # Apply triple barrier labeling
        labels_df = label_all_bars(df["close"], df["high"], df["low"], cfg)

        # Sample weights
        return_weights = compute_return_weights(df["close"], labels_df, cfg)
        uniqueness = compute_label_uniqueness(labels_df)
        weights = combine_weights(return_weights, uniqueness)

        _save_labels(labels_df, weights, symbol, labels_dir)

        n_long = int((labels_df["label"] == 1).sum())
        n_short = int((labels_df["label"] == -1).sum())
        n_neutral = int((labels_df["label"] == 0).sum())
        n_total = len(labels_df)
        pct_neutral = round(n_neutral / max(n_total, 1), 4)

        # Count fee-reclassified bars: neutral bars where tp_level < fee threshold.
        # These are bars that were TP hits but reclassified to 0 by label_all_bars fee-adjust.
        # Original label is not stored separately — but only TP hits near the clip floor
        # have tp_level < threshold after reclassification, so this approximation is tight.
        fee_reclassified = 0
        if "tp_level" in labels_df.columns and getattr(cfg.labels, "fee_adjust_labels", False):
            cost = float(getattr(cfg.labels, "round_trip_cost_pct", 0.003))
            multiple = float(getattr(cfg.labels, "dead_zone_cost_multiple", 1.0))
            threshold = cost * multiple
            fee_reclassified = int((
                (labels_df["label"] == 0) & (labels_df["tp_level"] < threshold)
            ).sum())

        logger.info(f"{symbol}: {n_total} labels — long={n_long}, short={n_short}, neutral={n_neutral}")
        return symbol, {
            "n_long": n_long, "n_short": n_short, "n_neutral": n_neutral,
            "pct_neutral": pct_neutral, "fee_reclassified": fee_reclassified,
        }, None

    except Exception as e:
        import traceback
        return symbol, None, f"{e}\n{traceback.format_exc()}"


def _label_symbol_worker(symbol: str, cfg_dict: dict, checkpoints_dir_str: str, labels_dir_str: str) -> tuple:
    # Worker entry point for ProcessPoolExecutor — all args must be picklable.
    cfg = OmegaConf.create(cfg_dict)
    return _label_symbol(symbol, cfg, Path(checkpoints_dir_str), Path(labels_dir_str))


def run(cfg, force: bool = False, symbol_filter: str = None) -> None:
    if not force and is_stage_complete("labels"):
        logger.info("Stage 3 already complete, skipping.")
        return

    all_symbols = get_symbols(cfg)
    if symbol_filter:
        _sf = set(symbol_filter) if isinstance(symbol_filter, list) else {symbol_filter}
        all_symbols = [s for s in all_symbols if s.get("name", s.get("symbol")) in _sf]
    symbol_names = [s.get("name", s.get("symbol")) for s in all_symbols]

    checkpoints_dir = Path(cfg.data.checkpoints_dir)
    labels_dir = Path(cfg.data.labels_dir)

    max_workers = int(os.environ.get("LABEL_WORKERS", 4))
    # Serialize OmegaConf and Paths so they survive multiprocessing pickling
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    checkpoints_dir_str = str(checkpoints_dir)
    labels_dir_str = str(labels_dir)

    issues = []
    success_count = 0
    diag_rows = []

    logger.info(f"Stage 3: labeling {len(symbol_names)} symbols with max_workers={max_workers}")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_label_symbol_worker, sym, cfg_dict, checkpoints_dir_str, labels_dir_str): sym
            for sym in symbol_names
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="stage_03", unit="sym"):
            sym, stats, err = future.result()
            if err:
                logger.error(f"{sym}: labeling failed — {err}")
                issues.append(f"{sym}: {str(err)[:200]}")
            else:
                success_count += 1
                if stats:
                    diag_rows.append({"symbol": sym, "stage": "labels", **stats})

    # Write label stats to unified diagnostics CSV
    if diag_rows:
        results_dir = Path(cfg.data.results_dir) if hasattr(cfg.data, "results_dir") else Path("results")
        write_pipeline_diagnostics(diag_rows, results_dir)

    # Validate label distribution across all symbols
    _validate_label_distributions(symbol_names, labels_dir, issues)

    update_project_state("labels", "done", issues, output_dir=str(labels_dir))
    logger.info(f"Stage 3 complete. {success_count}/{len(symbol_names)} symbols labeled.")


def _validate_label_distributions(symbol_names: list, labels_dir: Path, issues: list) -> None:
    # Flag symbols with extreme class imbalance (>80% one class)
    for symbol in symbol_names:
        labels_path = labels_dir / f"{symbol}_15m_labels.parquet"
        if not labels_path.exists():
            continue
        try:
            df = pd.read_parquet(labels_path)
            if "label" not in df.columns:
                continue
            value_counts = df["label"].value_counts(normalize=True)
            for cls, pct in value_counts.items():
                if pct > 0.80:
                    msg = f"{symbol}: label class {cls} dominates at {pct:.1%}"
                    logger.warning(msg)
                    issues.append(msg)
        except Exception as e:
            logger.warning(f"Label validation failed for {symbol}: {e}")
