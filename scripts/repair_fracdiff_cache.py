#!/usr/bin/env python3
"""
One-off repair script: generate missing fracdiff d-values cache.

Reads existing feature parquets, slices to train_end, and calls fit_and_save_d_values
to populate data/checkpoints/fracdiff/ with d-value JSON files for all symbols.

CRITICAL: Only processes train-period data (index <= train_end). Safe to run anytime.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.features.fracdiff import fit_and_save_d_values
from src.utils.io_utils import read_features
from src.utils.logger import get_logger

# Fracdiff columns (same as in live_features.py)
_FRACDIFF_COLS = [
    "close_5_mean", "close_10_mean", "close_20_mean",
    "close_50_mean", "close_100_mean", "close_200_mean",
    "obv", "vwap_20",
]

logger = get_logger("repair_fracdiff")


def repair_fracdiff_cache():
    """Regenerate fracdiff d-value cache from existing feature parquets."""
    cfg = load_config()

    features_dir = Path(cfg.data.features_dir)
    checkpoints_dir = Path(cfg.data.checkpoints_dir)
    fracdiff_cache = checkpoints_dir / "fracdiff"
    train_end = pd.Timestamp(cfg.data.train_end, tz="UTC")

    logger.info(f"Repairing fracdiff cache — train_end={cfg.data.train_end}")
    logger.info(f"Reading from: {features_dir}")
    logger.info(f"Writing to: {fracdiff_cache}")

    # List all symbols with feature parquets
    feature_files = sorted(features_dir.glob("*_15m_features.parquet"))
    symbol_names = [f.stem.replace("_15m_features", "") for f in feature_files]

    if not symbol_names:
        logger.error(f"No feature parquets found in {features_dir}")
        return

    logger.info(f"Found {len(symbol_names)} symbols")

    success_count = 0
    skip_count = 0

    for symbol in symbol_names:
        try:
            # Read full feature DataFrame
            df = read_features(symbol, "15m", features_dir)

            # Slice to train period
            df_train = df[df.index <= train_end]

            if len(df_train) == 0:
                logger.warning(f"{symbol}: no train data found (index all > train_end)")
                skip_count += 1
                continue

            # Extract fracdiff columns
            fracdiff_cols = [c for c in _FRACDIFF_COLS if c in df_train.columns]

            if not fracdiff_cols:
                logger.warning(f"{symbol}: no fracdiff columns found in feature set")
                skip_count += 1
                continue

            # Fit and save d-values
            fit_and_save_d_values(df_train, fracdiff_cols, symbol, "15m", fracdiff_cache)
            success_count += 1

        except Exception as e:
            logger.error(f"{symbol}: failed — {e}")
            skip_count += 1
            continue

    logger.info(f"Fracdiff repair complete: {success_count} succeeded, {skip_count} skipped")


if __name__ == "__main__":
    repair_fracdiff_cache()
