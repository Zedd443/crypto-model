import numpy as np
import pandas as pd
from src.utils.logger import get_logger
from src.features.technical import compute_atr

logger = get_logger("triple_barrier")



def apply_triple_barrier_clipped(
    close: pd.Series,
    events_index: pd.Index,
    tp_level: pd.Series,
    sl_level: pd.Series,
    max_hold_bars: int,
) -> pd.DataFrame:
    # Triple barrier labeling per Lopez de Prado AFML Ch.3
    # tp_level and sl_level are fractional distances (already clipped by compute_atr_barriers).
    # Barrier prices: tp_price = p0 * (1 + tp_level[t0]), sl_price = p0 * (1 - sl_level[t0])
    # label: +1 TP hit first, -1 SL hit first, 0 time barrier
    # bars_to_exit: actual bars held before exit (for Sharpe annualization in stage_04)

    close_arr = close.values
    close_idx = close.index
    idx_map = {ts: i for i, ts in enumerate(close_idx)}

    tp_arr = tp_level.reindex(close_idx).values
    sl_arr = sl_level.reindex(close_idx).values

    n_close = len(close_arr)

    # --- Partition events into invalid (fast path) vs valid (vectorized path) ---
    # "invalid" = not in idx_map, NaN p0, p0<=0, NaN tp/sl fracs
    valid_mask = np.zeros(len(events_index), dtype=bool)
    i0_arr = np.empty(len(events_index), dtype=np.intp)

    for ev_idx, t0 in enumerate(events_index):
        if t0 not in idx_map:
            continue
        i0 = idx_map[t0]
        p0 = close_arr[i0]
        if np.isnan(p0) or p0 <= 0:
            continue
        if np.isnan(tp_arr[i0]) or np.isnan(sl_arr[i0]):
            continue
        valid_mask[ev_idx] = True
        i0_arr[ev_idx] = i0

    valid_ev_indices = np.where(valid_mask)[0]
    n_valid = len(valid_ev_indices)

    # Pre-allocate output arrays (length = n_events)
    out_labels = np.zeros(len(events_index), dtype=np.int8)
    out_bars   = np.zeros(len(events_index), dtype=np.int64)
    # t1 defaults: for invalid events use t0 (if not in map) or t_end; fill after

    if n_valid > 0:
        # --- Build (n_valid, max_hold_bars) forward-price matrix ---
        # Row r corresponds to event valid_ev_indices[r]; columns 0..max_hold_bars-1
        # represent bars i0+1 .. i0+max_hold_bars.  Out-of-bounds positions → NaN.
        i0_valid = i0_arr[valid_ev_indices]  # shape (n_valid,)

        # Column offsets: 1-based (bars after entry)
        col_offsets = np.arange(1, max_hold_bars + 1, dtype=np.intp)  # shape (max_hold_bars,)

        # Absolute bar indices for every (event, offset) pair; shape (n_valid, max_hold_bars)
        abs_idx = i0_valid[:, None] + col_offsets[None, :]  # broadcasting

        # Clip to valid range; mark out-of-bounds with sentinel -1
        oob = abs_idx >= n_close
        abs_idx_clipped = np.where(oob, 0, abs_idx)  # safe index (will be masked)

        # Gather prices; shape (n_valid, max_hold_bars)
        price_matrix = close_arr[abs_idx_clipped].astype(np.float64)
        price_matrix[oob] = np.nan  # out-of-bounds → NaN (no-hit, search continues past)

        # TP and SL prices for each event; shape (n_valid,)
        p0_valid  = close_arr[i0_valid].astype(np.float64)
        tp_prices = p0_valid * (1.0 + tp_arr[i0_valid])  # shape (n_valid,)
        sl_prices = p0_valid * (1.0 - sl_arr[i0_valid])  # shape (n_valid,)

        # Hit masks; NaN comparisons produce False — NaN bars are naturally skipped
        tp_hit = price_matrix >= tp_prices[:, None]  # (n_valid, max_hold_bars)
        sl_hit = price_matrix <= sl_prices[:, None]  # (n_valid, max_hold_bars)

        # Combined hit: True where either barrier is breached
        any_hit = tp_hit | sl_hit  # (n_valid, max_hold_bars)

        # np.argmax on boolean returns index of first True; if all False returns 0.
        # We disambiguate "no hit" via any_hit.any(axis=1).
        first_hit_col = np.argmax(any_hit, axis=1)  # (n_valid,) — column index (0-based offset)
        has_hit = any_hit.any(axis=1)                # (n_valid,) bool

        # Effective bars to exit: first_hit_col+1 if hit, else max_hold_bars (time barrier)
        # But clamp to actual available bars (n_close - 1 - i0_valid)
        max_avail = np.minimum(max_hold_bars, n_close - 1 - i0_valid)  # (n_valid,)
        bars_hit  = np.where(has_hit, first_hit_col + 1, max_avail)    # (n_valid,)

        # Labels: check which barrier was hit first
        # Use advanced indexing: tp_hit[r, bars_hit[r]-1]
        row_idx = np.arange(n_valid)
        col_idx = bars_hit - 1  # 0-based column of the exit bar

        hit_tp = tp_hit[row_idx, col_idx] & has_hit
        hit_sl = sl_hit[row_idx, col_idx] & has_hit & ~hit_tp  # SL only if not TP first

        # Resolve label: +1 TP, -1 SL, 0 timeout (or simultaneous → TP wins, matching original)
        labels_valid = np.where(hit_tp, 1, np.where(hit_sl, -1, 0)).astype(np.int8)

        # Write back into output arrays
        out_labels[valid_ev_indices] = labels_valid
        out_bars[valid_ev_indices]   = bars_hit

    # --- Resolve t1 timestamps ---
    # For each event: t1 = close_idx[i0 + bars_to_exit] if in idx_map, else t0
    events_list = list(events_index)
    t1_out = []
    for ev_idx, t0 in enumerate(events_list):
        if t0 not in idx_map:
            # not in map — use t0 as fallback (matches original)
            t1_out.append(t0)
        else:
            i0 = idx_map[t0]
            bars = int(out_bars[ev_idx])
            i_exit = min(i0 + bars, n_close - 1)
            t1_out.append(close_idx[i_exit])

    return pd.DataFrame({
        "label": out_labels.tolist(),
        "t1": t1_out,
        "bars_to_exit": out_bars.tolist(),
    }, index=events_index)


def compute_atr_barriers(close: pd.Series, high: pd.Series, low: pd.Series, cfg, atr_period: int = 14) -> pd.DataFrame:
    atr = compute_atr(high, low, close, period=atr_period)
    sl_atr_mult = float(cfg.labels.sl_atr_mult)
    tp_atr_mult = float(cfg.labels.tp_atr_mult)
    sl_min = float(cfg.labels.sl_min_pct)
    sl_max = float(cfg.labels.sl_max_pct)
    tp_min = float(cfg.labels.tp_min_pct)
    tp_max = float(cfg.labels.tp_max_pct)

    natr = atr / close

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
    # vol_lookback drives ATR period — config key is now respected here
    vol_lookback = int(cfg.labels.vol_lookback)

    # compute_atr_barriers returns clipped tp_level/sl_level as fractional distances.
    # Used both as actual barriers (consistent geometry) and stored in label parquet
    # for price_returns reconstruction in stage_04.
    barriers = compute_atr_barriers(close, high, low, cfg, atr_period=vol_lookback)

    events_index = close.dropna().index
    labels_df = apply_triple_barrier_clipped(
        close, events_index, barriers["tp_level"], barriers["sl_level"], max_hold_bars
    )

    labels_df = labels_df.join(barriers[["tp_level", "sl_level", "atr", "natr"]], how="left")
    labels_df.index.name = "t0"

    # Fee-adjusted reclassification: TP hits where net gain < round-trip cost → neutral.
    # tp_level is the barrier size (fractional). A trade is marginal when the barrier
    # barely covers cost — threshold = cost * dead_zone_cost_multiple from config.
    if getattr(getattr(cfg, 'labels', cfg), 'fee_adjust_labels', False):
        cost = float(getattr(cfg.labels, 'round_trip_cost_pct', 0.003))
        multiple = float(getattr(cfg.labels, 'dead_zone_cost_multiple', 1.0))
        threshold = cost * multiple
        if "tp_level" in labels_df.columns:
            tp_mask = labels_df["label"] == 1
            marginal_tp = tp_mask & (labels_df["tp_level"] < threshold)
            n_reclassified = marginal_tp.sum()
            if n_reclassified > 0:
                labels_df.loc[marginal_tp, "label"] = 0
                logger.info(
                    f"Fee-adjust: reclassified {n_reclassified} marginal TP hits as neutral "
                    f"(tp_level < {threshold:.4f} = cost {cost:.4f} × {multiple})"
                )

    n_long = (labels_df["label"] == 1).sum()
    n_short = (labels_df["label"] == -1).sum()
    n_neutral = (labels_df["label"] == 0).sum()
    logger.info(f"Labels: long={n_long}, short={n_short}, neutral={n_neutral}")

    return labels_df
