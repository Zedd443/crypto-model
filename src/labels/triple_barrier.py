import numpy as np
import pandas as pd
from src.utils.logger import get_logger
from src.features.technical import compute_atr

logger = get_logger("triple_barrier")


def compute_daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
    log_ret = close.pct_change()
    vol = log_ret.ewm(span=span, adjust=False).std()
    return vol.rename("daily_vol")


def get_vertical_barriers(close_index: pd.Index, max_hold_bars: int) -> pd.Series:
    # For each bar t, t1 = t + max_hold_bars (or last bar of index)
    timestamps = pd.Series(close_index, index=close_index)
    t1 = timestamps.shift(-max_hold_bars)
    # Fill any NaTs at the end with last valid timestamp
    last_ts = close_index[-1]
    t1 = t1.fillna(last_ts)
    return t1


def apply_triple_barrier(
    close: pd.Series,
    events_index: pd.Index,
    tp_mult: float,
    sl_mult: float,
    max_hold_bars: int,
    vol: pd.Series,
) -> pd.DataFrame:
    # Triple barrier labeling per Lopez de Prado AFML Ch.3
    # Returns DataFrame with columns [label, t1] indexed by events_index
    # label: +1 TP hit first, -1 SL hit first, 0 time barrier
    # IMPORTANT: forward-looking on prices — correct for label construction

    labels = []
    t1_out = []

    close_arr = close.values
    close_idx = close.index
    idx_map = {ts: i for i, ts in enumerate(close_idx)}

    for t0 in events_index:
        if t0 not in idx_map:
            labels.append(0)
            t1_out.append(t0)
            continue

        i0 = idx_map[t0]
        i_end = min(i0 + max_hold_bars, len(close_arr) - 1)
        t_end = close_idx[i_end]

        p0 = close_arr[i0]
        if np.isnan(p0) or p0 <= 0:
            labels.append(0)
            t1_out.append(t_end)
            continue

        v = vol.loc[t0] if t0 in vol.index else 0.0
        if np.isnan(v):
            v = 0.01  # fallback vol

        tp_level = p0 * (1 + tp_mult * v)
        sl_level = p0 * (1 - sl_mult * v)

        hit_label = 0
        hit_time = t_end

        for j in range(i0 + 1, i_end + 1):
            p_j = close_arr[j]
            if np.isnan(p_j):
                continue
            ts_j = close_idx[j]
            if p_j >= tp_level:
                hit_label = 1
                hit_time = ts_j
                break
            elif p_j <= sl_level:
                hit_label = -1
                hit_time = ts_j
                break

        labels.append(hit_label)
        t1_out.append(hit_time)

    result = pd.DataFrame({
        "label": labels,
        "t1": t1_out,
    }, index=events_index)
    return result


def compute_atr_barriers(close: pd.Series, high: pd.Series, low: pd.Series, cfg) -> pd.DataFrame:
    # ATR-based TP/SL levels
    atr = compute_atr(high, low, close, period=14)
    sl_atr_mult = float(cfg.labels.sl_atr_mult)
    tp_atr_mult = float(cfg.labels.tp_atr_mult)
    sl_min = float(cfg.labels.sl_min_pct)
    sl_max = float(cfg.labels.sl_max_pct)
    tp_min = float(cfg.labels.tp_min_pct)
    tp_max = float(cfg.labels.tp_max_pct)

    natr = atr / close  # normalize

    sl_level = (natr * sl_atr_mult).clip(sl_min, sl_max)
    tp_level = (natr * tp_atr_mult).clip(tp_min, tp_max)

    return pd.DataFrame({
        "tp_level": tp_level,
        "sl_level": sl_level,
        "atr": atr,
        "natr": natr,
    }, index=close.index)


def label_all_bars(close: pd.Series, high: pd.Series, low: pd.Series, cfg) -> pd.DataFrame:
    max_hold_bars = int(cfg.labels.max_hold_bars)
    vol_lookback = int(cfg.labels.vol_lookback)

    # Use ATR-normalized vol as the barrier width scaler
    barriers = compute_atr_barriers(close, high, low, cfg)
    # vol = natr (percentage-based)
    vol = barriers["natr"]

    tp_mult = float(cfg.labels.tp_atr_mult)
    sl_mult = float(cfg.labels.sl_atr_mult)

    events_index = close.dropna().index
    labels_df = apply_triple_barrier(
        close, events_index, tp_mult, sl_mult, max_hold_bars, vol
    )

    # Add barrier info
    labels_df = labels_df.join(barriers[["tp_level", "sl_level", "atr", "natr"]], how="left")
    labels_df.index.name = "t0"

    n_long = (labels_df["label"] == 1).sum()
    n_short = (labels_df["label"] == -1).sum()
    n_neutral = (labels_df["label"] == 0).sum()
    logger.info(f"Labels: long={n_long}, short={n_short}, neutral={n_neutral}")

    return labels_df
