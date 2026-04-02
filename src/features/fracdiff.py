import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import json
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("fracdiff")


def _get_fracdiff_weights(d: float, size: int, threshold: float = 1e-5) -> np.ndarray:
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    return np.array(w[::-1])


def fracdiff_series(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    weights = _get_fracdiff_weights(d, len(series), threshold)
    width = len(weights)
    result = np.full(len(series), np.nan)
    arr = series.values
    for i in range(width - 1, len(arr)):
        result[i] = np.dot(weights, arr[i - width + 1: i + 1])
    return pd.Series(result, index=series.index, name=series.name)


def estimate_min_d(series: pd.Series, max_d: float = 1.0, d_step: float = 0.1, adf_threshold: float = 0.05) -> float:
    # CRITICAL: call only on TRAIN data
    for d in np.arange(0.0, max_d + d_step, d_step):
        if d == 0.0:
            test_series = series.dropna()
        else:
            test_series = fracdiff_series(series, d).dropna()
        if len(test_series) < 20:
            continue
        try:
            p_value = adfuller(test_series, maxlag=1, autolag=None)[1]
            if p_value < adf_threshold:
                logger.debug(f"Stationary at d={d:.1f}, p={p_value:.4f} for {series.name}")
                return round(d, 2)
        except Exception:
            continue
    return 1.0  # fallback: full difference


def fit_and_save_d_values(train_df: pd.DataFrame, feature_cols: list, symbol: str, tf: str, cache_path: Path) -> dict:
    # CRITICAL: only called on training data
    d_values = {}
    for col in feature_cols:
        if col not in train_df.columns:
            continue
        series = train_df[col].dropna()
        if len(series) < 100:
            d_values[col] = 1.0
            continue
        d_values[col] = estimate_min_d(series)
        logger.info(f"{symbol} {tf} {col}: d={d_values[col]}")

    cache_path.mkdir(parents=True, exist_ok=True)
    out_path = cache_path / f"fracdiff_d_{symbol}_{tf}.json"
    with open(out_path, "w") as f:
        json.dump(d_values, f)
    return d_values


def load_d_values(symbol: str, tf: str, cache_path: Path) -> dict:
    path = cache_path / f"fracdiff_d_{symbol}_{tf}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def apply_fracdiff_transform(df: pd.DataFrame, d_values: dict) -> pd.DataFrame:
    result = df.copy()
    for col, d in d_values.items():
        if col not in df.columns or d == 0.0:
            continue
        result[col] = fracdiff_series(df[col], d)
    return result
