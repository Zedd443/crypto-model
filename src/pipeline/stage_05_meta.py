import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
from src.utils.config_loader import get_symbols
from src.utils.state_manager import is_stage_complete, update_project_state, update_completed_symbol
from src.utils.logger import get_logger
from src.utils.io_utils import read_features
from src.models.meta_labeler import (
    create_meta_labels, build_meta_features, train_meta_labeler,
    save_meta_model,
)
from src.models.model_versioning import get_latest_model, register_model, generate_version_string

logger = get_logger("stage_05_meta")

_TF = "15m"


def _train_meta_symbol(
    symbol: str,
    cfg,
    checkpoints_dir: Path,
    labels_dir: Path,
    features_dir: Path,
    models_dir: Path,
) -> tuple:
    logger.info(f"Meta-labeling {symbol}")
    train_end = pd.Timestamp(cfg.data.train_end, tz="UTC")

    # Load OOF predictions from stage 04 (.npy for speed + separate index file)
    oof_path = checkpoints_dir / "oof" / f"{symbol}_{_TF}_oof_proba.npy"
    oof_idx_path = checkpoints_dir / "oof" / f"{symbol}_{_TF}_oof_index.npy"
    if not oof_path.exists():
        return symbol, None, f"OOF predictions not found: {oof_path}"

    oof_arr = np.load(str(oof_path))  # shape (n_directional, 2)
    if oof_idx_path.exists():
        # New format: index saved as int64 ns-since-epoch
        idx_int = np.load(str(oof_idx_path))
        oof_index = pd.DatetimeIndex(idx_int.astype("datetime64[ns]"), tz="UTC")
    else:
        # Old format (pre-index-fix): no index file — fall back to positional alignment with warning
        logger.warning(f"{symbol}: OOF index file missing — using positional alignment (less accurate). Retrain stage_04 to fix.")
        oof_index = None

    if oof_index is not None:
        oof_df = pd.DataFrame(oof_arr, index=oof_index, columns=["prob_short", "prob_long"])
    else:
        oof_df = None  # handled below with positional fallback

    # Load labels
    labels_path = labels_dir / f"{symbol}_{_TF}_labels.parquet"
    weights_path = labels_dir / f"{symbol}_{_TF}_weights.parquet"
    if not labels_path.exists():
        return symbol, None, f"Labels not found: {labels_path}"

    labels_df = pd.read_parquet(labels_path)
    weights_df = pd.read_parquet(weights_path) if weights_path.exists() else None
    sample_weights = weights_df["sample_weight"] if weights_df is not None else pd.Series(1.0, index=labels_df.index)

    # Load features for regime and micro features
    try:
        features_df = read_features(symbol, _TF, features_dir)
    except Exception as e:
        return symbol, None, f"Features not found: {e}"

    # Train data only
    train_labels_all = labels_df[labels_df.index <= train_end]

    if oof_df is not None:
        # New format: align by DatetimeIndex — OOF only has directional bars
        train_labels_aligned = train_labels_all.loc[
            train_labels_all.index.intersection(oof_df.index)
        ]
        oof_df = oof_df.loc[train_labels_aligned.index]
        oof_proba = oof_df[["prob_short", "prob_long"]].values
    else:
        # Old format fallback: drop neutrals positionally (less accurate alignment)
        train_dir = train_labels_all[train_labels_all["label"] != 0]
        min_len = min(len(train_dir), len(oof_arr))
        train_labels_aligned = train_dir.iloc[:min_len]
        oof_proba = oof_arr[:min_len]

    train_weights = sample_weights.reindex(train_labels_aligned.index).fillna(1.0)
    y_train_series = (train_labels_aligned["label"] == 1).astype(int)
    y_train = y_train_series.values

    if len(y_train) != len(oof_proba):
        return symbol, None, (
            f"OOF/label mismatch after alignment — y_train={len(y_train)}, "
            f"oof_proba={len(oof_proba)}. Rerun stage_04."
        )

    if len(y_train) < 100:
        return symbol, None, f"Insufficient train samples for meta-labeling: {len(y_train)}"

    # Create meta labels: 1 if primary model was correct, 0 otherwise
    meta_y = create_meta_labels(y_train, oof_proba)

    # Build meta features from OOF predictions + regime + microstructure
    # Use .loc slice instead of boolean mask to avoid OOM with 1.2GB DataFrames
    train_features = features_df.loc[:train_end]

    # Extract regime prob columns — reindex to directional bars (aligned_index)
    aligned_index = train_labels_aligned.index
    regime_cols = [c for c in train_features.columns if c.startswith("regime_prob_")]
    regime_probs = train_features[regime_cols].reindex(aligned_index) if regime_cols else None

    # Realized vol proxy (volume_zscore or rv_daily)
    realized_vol = (
        train_features["rv_daily"].reindex(aligned_index)
        if "rv_daily" in train_features.columns
        else pd.Series(0.0, index=aligned_index)
    )

    volume_zscore = (
        train_features["volume_surprise_20"].reindex(aligned_index)
        if "volume_surprise_20" in train_features.columns
        else pd.Series(0.0, index=aligned_index)
    )

    ofi = (
        train_features["ofi_20"].reindex(aligned_index)
        if "ofi_20" in train_features.columns
        else pd.Series(0.0, index=aligned_index)
    )

    # regime_probs, realized_vol, volume_zscore, ofi are already on aligned_index
    realized_vol_arr = realized_vol
    volume_zscore_arr = volume_zscore
    ofi_arr = ofi

    meta_X = build_meta_features(
        oof_proba, regime_probs, realized_vol_arr, volume_zscore_arr, ofi_arr
    )
    meta_X = meta_X.fillna(0.0)

    # Align weights (already on aligned_index, same length as meta_y)
    w_meta = train_weights

    # Train meta-labeler
    try:
        meta_model = train_meta_labeler(meta_X, meta_y, w_meta, cfg)
    except Exception as e:
        return symbol, None, f"Meta training failed: {e}"

    # Meta model accuracy on train (informational only)
    meta_pred = (meta_model.predict_proba(meta_X.values)[:, 1] > 0.5).astype(int)
    meta_acc = float(np.mean(meta_pred == meta_y))
    logger.info(f"  {symbol}: meta accuracy on OOF = {meta_acc:.3f}")

    # Version and register meta model
    primary_model_entry = get_latest_model(symbol, _TF, model_type="primary")
    primary_version = primary_model_entry["version"] if primary_model_entry else "unknown"

    version = generate_version_string(
        symbol, _TF,
        feature_names=meta_X.columns.tolist(),
        hyperparams={"primary_version": primary_version},
        train_start=str(train_labels_aligned.index.min().date()),
        train_end=str(train_labels_aligned.index.max().date()),
    )

    save_meta_model(meta_model, symbol, _TF, version, models_dir)
    register_model(
        symbol=symbol,
        tf=_TF,
        version=version,
        metrics={"meta_accuracy_oof": meta_acc},
        feature_names=meta_X.columns.tolist(),
        hyperparams={"primary_version": primary_version},
        train_period=(str(train_labels_aligned.index.min().date()), str(train_labels_aligned.index.max().date())),
        model_path=str(models_dir / f"{version}_meta.pkl"),
        model_type="meta",
    )

    return symbol, {"version": version, "meta_accuracy": meta_acc}, None


def _train_meta_symbol_worker(
    symbol: str,
    cfg_dict: dict,
    checkpoints_dir_str: str,
    labels_dir_str: str,
    features_dir_str: str,
    models_dir_str: str,
) -> tuple:
    # Worker entry point for ProcessPoolExecutor — all args must be picklable.
    # Reconstruct OmegaConf and Path objects inside the subprocess.
    cfg = OmegaConf.create(cfg_dict)
    return _train_meta_symbol(
        symbol, cfg,
        Path(checkpoints_dir_str),
        Path(labels_dir_str),
        Path(features_dir_str),
        Path(models_dir_str),
    )


def run(cfg, force: bool = False, symbol_filter: str = None) -> None:
    if not force and is_stage_complete("meta_labeling"):
        logger.info("Stage 5 already complete, skipping.")
        return

    all_symbols = get_symbols(cfg)
    if symbol_filter:
        all_symbols = [s for s in all_symbols if s.get("name", s.get("symbol")) == symbol_filter]
    symbol_names = [s.get("name", s.get("symbol")) for s in all_symbols]

    # Skip symbols already meta-labeled in a prior run (pipeline is resumable)
    # --force bypasses resume: re-run meta-labeling for all symbols in symbol_names
    if not force:
        from src.utils.state_manager import load_state
        _state = load_state()
        _completed = set(_state.get("stages", {}).get("meta_labeling", {}).get("completed_symbols", []))
        if _completed:
            _pending = [s for s in symbol_names if s not in _completed]
            logger.info(f"Resuming meta-labeling — {len(_completed)} already done, {len(_pending)} remaining")
            symbol_names = _pending

    checkpoints_dir = Path(cfg.data.checkpoints_dir)
    labels_dir = Path(cfg.data.labels_dir)
    features_dir = Path(cfg.data.features_dir)
    models_dir = Path(cfg.data.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # GPU cannot be shared across processes — enforce serial when XGB_DEVICE=cuda
    device = os.environ.get("XGB_DEVICE", "cpu")
    max_workers = 1 if device == "cuda" else int(os.environ.get("META_WORKERS", 4))

    # Serialize OmegaConf and Paths so they survive multiprocessing pickling
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    checkpoints_dir_str = str(checkpoints_dir)
    labels_dir_str = str(labels_dir)
    features_dir_str = str(features_dir)
    models_dir_str = str(models_dir)

    issues = []
    success_count = 0

    logger.info(f"Stage 5: meta-labeling {len(symbol_names)} symbols with max_workers={max_workers} (device={device})")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _train_meta_symbol_worker,
                sym, cfg_dict, checkpoints_dir_str, labels_dir_str, features_dir_str, models_dir_str,
            ): sym
            for sym in symbol_names
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="stage_05", unit="sym"):
            sym, result, err = future.result()
            if err:
                logger.error(f"{sym}: meta-labeling failed — {err}")
                issues.append(f"{sym}: {str(err)[:200]}")
            else:
                update_completed_symbol("meta_labeling", sym)
                success_count += 1
                logger.info(f"{sym}: meta-labeler registered — v{result['version'][:20]}...")

    update_project_state("meta_labeling", "done", issues, output_dir=str(models_dir))
    logger.info(f"Stage 5 complete. {success_count}/{len(symbol_names)} meta-labelers trained.")
