import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("sample_weights")


def compute_return_weights(close: pd.Series, labels_df: pd.DataFrame, cfg) -> pd.Series:
    # weight_i = |log(close[t1_i] / close[t0_i])|, normalized by mean, clipped.
    # Vectorized: map t0→p0 and t1→p1 via searchsorted instead of iterrows O(n²).
    clip_min = float(cfg.labels.weight_clip_min)
    clip_max = float(cfg.labels.weight_clip_max)

    close_idx_arr = close.index.values
    close_val_arr = close.values

    t0_arr = labels_df.index.values
    t1_arr = pd.to_datetime(labels_df["t1"].values)

    # p0: price at t0 — direct lookup via index map
    idx_map = {ts: i for i, ts in enumerate(close.index)}
    p0_idx = np.array([idx_map.get(t, -1) for t in t0_arr])
    valid_p0 = p0_idx >= 0
    p0 = np.where(valid_p0, close_val_arr[np.clip(p0_idx, 0, len(close_val_arr) - 1)], np.nan)

    # p1: price at closest bar <= t1 — searchsorted on sorted index
    # close.index must be monotonic (it is for OHLCV timeseries)
    t1_pos = np.searchsorted(close_idx_arr, t1_arr, side="right") - 1
    t1_pos = np.clip(t1_pos, 0, len(close_val_arr) - 1)
    p1 = close_val_arr[t1_pos]

    # |log(p1/p0)|, NaN where p0 invalid or zero
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.where(
            valid_p0 & (p0 > 0) & ~np.isnan(p0) & ~np.isnan(p1),
            np.abs(np.log(p1 / np.where(p0 > 0, p0, 1.0))),
            np.nan,
        )

    weights = pd.Series(log_ret, index=labels_df.index, name="return_weight")

    mean_w = weights.mean()
    if mean_w > 0:
        weights = weights / mean_w
    else:
        weights = weights.fillna(1.0)

    weights = weights.clip(clip_min, clip_max).fillna(1.0)
    return weights


def compute_label_uniqueness(labels_df: pd.DataFrame) -> pd.Series:
    # uniqueness_i = 1 / number of labels overlapping with label i
    # Vectorized via numpy broadcasting — replaces O(n²) Python loop.
    n = len(labels_df)
    if n == 0:
        return pd.Series(dtype=float)

    t0_arr = np.asarray(labels_df.index.values.astype("int64"))
    t1_arr = np.asarray(pd.to_datetime(labels_df["t1"].values).astype("int64"))

    # Cap n for very large datasets to avoid OOM (100k × 100k = 10GB)
    # For n > 10k, subsample overlap counting via chunked approach
    if n <= 10_000:
        # Full vectorized: (n, n) boolean matrix
        overlap = (t0_arr[:, None] <= t1_arr[None, :]) & (t1_arr[:, None] >= t0_arr[None, :])
        overlap_count = overlap.sum(axis=1).astype(float)
    else:
        # Chunked to keep memory bounded at ~400MB per chunk
        chunk = 2_000
        overlap_count = np.zeros(n, dtype=float)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            ov = (
                (t0_arr[start:end, None] <= t1_arr[None, :]) &
                (t1_arr[start:end, None] >= t0_arr[None, :])
            )
            overlap_count[start:end] = ov.sum(axis=1)

    uniqueness = np.where(overlap_count > 0, 1.0 / overlap_count, 1.0)
    return pd.Series(uniqueness, index=labels_df.index, name="uniqueness")


def combine_weights(return_weights: pd.Series, uniqueness: pd.Series) -> pd.Series:
    common_idx = return_weights.index.intersection(uniqueness.index)
    rw = return_weights.loc[common_idx]
    u = uniqueness.loc[common_idx]

    combined = rw * u
    mean_c = combined.mean()
    if mean_c > 0:
        combined = combined / mean_c
    return combined.rename("sample_weight")
