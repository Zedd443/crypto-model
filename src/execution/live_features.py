import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.technical import build_technical_features
from src.models.model_versioning import get_latest_model
from src.utils.logger import get_logger

logger = get_logger("live_features")


def get_lookback_bars_needed(cfg) -> int:
    # Extra 200 bars ensures rolling indicators (longest window=200) have enough history
    return int(cfg.features.warmup_bars) + 200


def compute_live_features(symbol: str, cfg, lookback_df: pd.DataFrame) -> pd.Series:
    # lookback_df must have UTC DatetimeIndex and cols: open, high, low, close, volume
    if lookback_df.index.tz is None:
        raise ValueError(f"{symbol}: lookback_df index must be UTC-aware")

    # Compute the full technical feature matrix on the lookback window
    feat_df = build_technical_features(lookback_df, cfg)

    # Take only the last row — this is the just-closed bar
    last_row = feat_df.iloc[[-1]].copy()

    # Load imputer — fit was done on train split only (no leakage)
    imputer_path = Path(cfg.data.checkpoints_dir) / "imputers" / f"{symbol}_15m_imputer.pkl"
    if not imputer_path.exists():
        raise FileNotFoundError(f"Imputer not found: {imputer_path}")
    with open(imputer_path, "rb") as f:
        imputer = pickle.load(f)

    # Load scaler — fit was done on train split only (no leakage)
    scaler_path = Path(cfg.data.checkpoints_dir) / "scalers" / f"{symbol}_15m_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Look up which features the registered model was trained on
    registry_entry = get_latest_model(symbol, "15m", model_type="primary")
    if registry_entry is None:
        raise RuntimeError(f"No registered primary model for {symbol} 15m")

    selected_features = registry_entry["feature_names"]

    # Subset to model features; fill missing cols with NaN so imputer handles them
    missing_cols = [c for c in selected_features if c not in last_row.columns]
    if missing_cols:
        logger.warning(f"{symbol}: {len(missing_cols)} feature cols missing from live compute — filling NaN")
        for c in missing_cols:
            last_row[c] = np.nan

    last_row = last_row[selected_features]

    # Transform with imputer then scaler (transform-only, never fit)
    transformed = imputer.transform(last_row.values)
    transformed = scaler.transform(transformed)

    result = pd.Series(transformed[0], index=selected_features, name=last_row.index[0])
    logger.debug(f"{symbol}: live features computed — {len(result)} features")
    return result
