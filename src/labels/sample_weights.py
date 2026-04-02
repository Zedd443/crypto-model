import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("sample_weights")


def compute_return_weights(close: pd.Series, labels_df: pd.DataFrame, cfg) -> pd.Series:
    # weight_i = |log(close[t1_i] / close[t0_i])|
    # Normalized by mean weight, clipped to configured bounds
    clip_min = float(cfg.labels.weight_clip_min)
    clip_max = float(cfg.labels.weight_clip_max)

    weights = pd.Series(np.nan, index=labels_df.index)

    for t0, row in labels_df.iterrows():
        t1 = row["t1"]
        if t0 not in close.index:
            continue
        p0 = close.loc[t0]
        # Find closest bar at or before t1
        valid_t1_candidates = close.index[close.index <= t1]
        if len(valid_t1_candidates) == 0:
            continue
        t1_actual = valid_t1_candidates[-1]
        p1 = close.loc[t1_actual]
        if p0 <= 0 or np.isnan(p0) or np.isnan(p1):
            continue
        weights.loc[t0] = abs(np.log(p1 / p0))

    # Normalize by mean
    mean_w = weights.mean()
    if mean_w > 0:
        weights = weights / mean_w
    else:
        weights = weights.fillna(1.0)

    # Clip
    weights = weights.clip(clip_min, clip_max).fillna(1.0)
    return weights.rename("return_weight")


def compute_label_uniqueness(labels_df: pd.DataFrame, total_bars: int) -> pd.Series:
    # uniqueness_i = 1 / average number of labels that overlap with label i
    # Based on t0 and t1 overlap across all labels
    n = len(labels_df)
    if n == 0:
        return pd.Series(dtype=float)

    t0_arr = labels_df.index.values
    t1_arr = labels_df["t1"].values
    uniqueness = np.zeros(n)

    for i in range(n):
        # Count how many other labels [j] overlap with label i
        # Overlap: t0_j <= t1_i and t1_j >= t0_i
        overlap_count = np.sum(
            (t0_arr <= t1_arr[i]) & (t1_arr >= t0_arr[i])
        )
        # Average concurrent labels over the holding period of label i
        if overlap_count > 0:
            uniqueness[i] = 1.0 / overlap_count
        else:
            uniqueness[i] = 1.0

    result = pd.Series(uniqueness, index=labels_df.index, name="uniqueness")
    return result


def combine_weights(return_weights: pd.Series, uniqueness: pd.Series) -> pd.Series:
    # Align indices
    common_idx = return_weights.index.intersection(uniqueness.index)
    rw = return_weights.loc[common_idx]
    u = uniqueness.loc[common_idx]

    combined = rw * u
    # Renormalize to mean=1
    mean_c = combined.mean()
    if mean_c > 0:
        combined = combined / mean_c
    return combined.rename("sample_weight")
