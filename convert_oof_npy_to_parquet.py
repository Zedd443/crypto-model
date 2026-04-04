"""
One-time script: convert existing OOF .npy files to .parquet with DatetimeIndex.

The .npy shape is (n_directional_bars, 2) matching X_train_final after neutral filtering.
We reconstruct the index by re-applying the same filtering logic used in stage_04.

Run once from project root:
    .venv/Scripts/python convert_oof_npy_to_parquet.py
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.config_loader import load_config
from src.utils.io_utils import read_features
from src.utils.logger import get_logger

logger = get_logger("convert_oof")

cfg = load_config()

oof_dir = Path(cfg.data.checkpoints_dir) / "oof"
labels_dir = Path(cfg.data.labels_dir)
features_dir = Path(cfg.data.features_dir)

train_end = pd.Timestamp(cfg.data.train_end, tz="UTC")
_TF = str(cfg.data.primary_timeframe)

converted = []
failed = []

for npy_path in sorted(oof_dir.glob("*_oof_proba.npy")):
    symbol = npy_path.stem.replace(f"_{_TF}_oof_proba", "")
    parquet_path = oof_dir / f"{symbol}_{_TF}_oof_proba.parquet"

    if parquet_path.exists():
        logger.info(f"{symbol}: parquet already exists, skipping")
        continue

    try:
        oof_arr = np.load(str(npy_path))  # shape (n, 2)

        # Reconstruct directional train index — same logic as stage_04
        labels_path = labels_dir / f"{symbol}_{_TF}_labels.parquet"
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels not found: {labels_path}")
        labels_df = pd.read_parquet(labels_path)

        features_df = read_features(symbol, _TF, features_dir)
        common_idx = features_df.index.intersection(labels_df.index)
        labels_aligned = labels_df.loc[common_idx]
        features_aligned = features_df.loc[common_idx]

        train_mask = features_aligned.index <= train_end
        train_label_series = labels_aligned.loc[train_mask, "label"]

        # Remove warmup rows if flag exists
        if "is_warmup" in features_aligned.columns:
            warmup_mask = features_aligned.loc[train_mask, "is_warmup"] == 1
            train_label_series = train_label_series[~warmup_mask]

        # Drop neutral — same as stage_04
        dir_mask = train_label_series != 0
        directional_index = train_label_series[dir_mask].index

        if len(directional_index) != len(oof_arr):
            raise ValueError(
                f"Index length {len(directional_index)} != OOF length {len(oof_arr)}. "
                "OOF may be from a different training run — retrain stage_04 instead."
            )

        oof_df = pd.DataFrame(
            oof_arr,
            index=directional_index,
            columns=["prob_short", "prob_long"],
        )
        oof_df.to_parquet(str(parquet_path))
        logger.info(f"{symbol}: converted {len(oof_df)} bars → {parquet_path.name}")
        converted.append(symbol)

    except Exception as e:
        logger.error(f"{symbol}: FAILED — {e}")
        failed.append((symbol, str(e)))

print(f"\nDone. Converted: {len(converted)}, Failed: {len(failed)}")
if failed:
    print("Failed symbols (need stage_04 retrain):")
    for sym, err in failed:
        print(f"  {sym}: {err}")
else:
    print("All OOF files converted. You can now run stage_05.")
