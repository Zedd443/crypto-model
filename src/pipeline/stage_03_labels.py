from pathlib import Path
import pandas as pd
from src.utils.config_loader import get_symbols
from src.utils.state_manager import is_stage_complete, update_project_state
from src.utils.logger import get_logger
from src.utils.io_utils import checkpoint_exists, read_checkpoint
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
        uniqueness = compute_label_uniqueness(labels_df, len(df))
        weights = combine_weights(return_weights, uniqueness)

        _save_labels(labels_df, weights, symbol, labels_dir)

        n_long = (labels_df["label"] == 1).sum()
        n_short = (labels_df["label"] == -1).sum()
        n_neutral = (labels_df["label"] == 0).sum()
        logger.info(f"{symbol}: {len(labels_df)} labels — long={n_long}, short={n_short}, neutral={n_neutral}")
        return symbol, None

    except Exception as e:
        import traceback
        return symbol, f"{e}\n{traceback.format_exc()}"


def run(cfg, force: bool = False, symbol_filter: str = None) -> None:
    if not force and is_stage_complete("labels"):
        logger.info("Stage 3 already complete, skipping.")
        return

    all_symbols = get_symbols(cfg)
    if symbol_filter:
        all_symbols = [s for s in all_symbols if s.get("name", s.get("symbol")) == symbol_filter]
    symbol_names = [s.get("name", s.get("symbol")) for s in all_symbols]

    checkpoints_dir = Path(cfg.data.checkpoints_dir)
    labels_dir = Path(cfg.data.labels_dir)

    issues = []
    success_count = 0

    for symbol in symbol_names:
        sym, err = _label_symbol(symbol, cfg, checkpoints_dir, labels_dir)
        if err:
            logger.error(f"{sym}: labeling failed — {err}")
            issues.append(f"{sym}: {str(err)[:200]}")
        else:
            success_count += 1

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
