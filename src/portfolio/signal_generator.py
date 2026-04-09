import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("signal_generator")


def compute_adaptive_threshold(
    primary_proba_series: pd.Series,
    top_pct: float = 0.40,
    signal_floor: float = 0.55,
    window_bars: int = 480,
) -> pd.Series:
    # Rolling window-based threshold: percentile of recent signal strengths
    # threshold = max(percentile(1-top_pct), signal_floor)
    # window_bars = 5 days × 96 bars/day for 15m data
    rolling_pct = primary_proba_series.rolling(window_bars, min_periods=50).quantile(1.0 - top_pct)
    threshold = rolling_pct.clip(lower=signal_floor)
    return threshold.fillna(signal_floor).rename("adaptive_threshold")


def generate_signals(
    primary_proba_df: pd.DataFrame,
    meta_proba_series: pd.Series,
    regime_df: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    # primary_proba_df: columns [symbol_prob_long, symbol_prob_short] or indexed by symbol
    # meta_proba_series: probability that primary prediction is correct
    # regime_df: regime probability columns per timestamp
    signal_floor = float(cfg.portfolio.signal_floor_prob)
    window_bars = int(5 * 96)  # 5 days of 15m bars

    # Compute direction from primary proba
    if "prob_long" in primary_proba_df.columns:
        prob_long = primary_proba_df["prob_long"]
        prob_short = primary_proba_df["prob_short"]
    else:
        # Assume single column is prob_long
        prob_col = primary_proba_df.columns[0]
        prob_long = primary_proba_df[prob_col]
        prob_short = 1.0 - prob_long

    # Dead zone thresholds from config
    dead_zone_direction_base = float(cfg.portfolio.dead_zone_direction)
    dead_zone_signal = float(cfg.portfolio.dead_zone_signal)

    # Adaptive dead zone: scale up when conformal width signals model uncertainty
    # conf_width is currently a static 0.20 placeholder — activates with real per-bar widths
    conf_width_full = float(getattr(cfg.model, 'conformal_width_full', 0.20))
    conf_width_60pct = float(getattr(cfg.model, 'conformal_width_60pct', 0.40))
    conf_width_series = pd.Series(0.20, index=primary_proba_df.index)  # placeholder
    scale = np.where(conf_width_series < conf_width_full, 1.0,
                     np.where(conf_width_series < conf_width_60pct, 1.25, 1.5))
    dead_zone_direction = dead_zone_direction_base * scale

    direction = np.where(prob_long > 0.5, 1, -1)
    dead_zone = np.abs(prob_long - 0.5) < dead_zone_direction
    direction = np.where(dead_zone, 0, direction)
    primary_conf = np.where(direction == 1, prob_long, np.where(direction == -1, prob_short, 0.5))

    # Meta probability aligned to index
    meta_prob_aligned = meta_proba_series.reindex(primary_proba_df.index).fillna(0.5)

    signal_strength = primary_conf * meta_prob_aligned.values

    # Adaptive threshold per bar
    adaptive_thresh = compute_adaptive_threshold(
        pd.Series(signal_strength, index=primary_proba_df.index),
        signal_floor=signal_floor,
        window_bars=window_bars,
    )

    is_signal = (signal_strength >= adaptive_thresh.values).astype(int)
    # No signal if direction is ambiguous (prob_long near 0.5)
    is_signal = np.where(np.abs(prob_long - 0.5) < dead_zone_signal, 0, is_signal)

    # Dominant regime state
    if regime_df is not None and len(regime_df) > 0:
        regime_aligned = regime_df.reindex(primary_proba_df.index, method="ffill")
        # idxmax(axis=1) raises ValueError when a row is entirely NaN — handle by
        # pre-computing a "unknown" default for all-NaN rows and only calling
        # idxmax on rows that have at least one valid probability
        all_nan_mask = regime_aligned.isna().all(axis=1)
        if all_nan_mask.any():
            logger.warning(
                f"regime_aligned has {all_nan_mask.sum()} all-NaN rows — "
                "defaulting those rows to 'unknown'"
            )
        regime_state = pd.Series("unknown", index=primary_proba_df.index, dtype=object)
        valid_mask = ~all_nan_mask
        if valid_mask.any():
            regime_state.loc[valid_mask] = (
                regime_aligned.loc[valid_mask].idxmax(axis=1).fillna("unknown")
            )
    else:
        regime_state = pd.Series("unknown", index=primary_proba_df.index)

    # Conformal width placeholder — updated in pipeline with actual conformal q
    conformal_width = pd.Series(0.20, index=primary_proba_df.index, name="conformal_width")

    signals = pd.DataFrame({
        "direction": direction,
        "primary_prob": prob_long.values,
        "meta_prob": meta_prob_aligned.values,
        "signal_strength": signal_strength,
        "regime_state": regime_state.values,
        "is_signal": is_signal,
        "conformal_width": conformal_width.values,
    }, index=primary_proba_df.index)

    n_signals = int(is_signal.sum())
    n_long_sig = int((direction == 1).sum())
    n_short_sig = int((direction == -1).sum())
    n_dead = int((direction == 0).sum())
    logger.info(f"Signals generated: {n_signals}/{len(signals)} active — long={n_long_sig}, short={n_short_sig}, dead_zone={n_dead}")
    return signals


def apply_h4_filter(
    signals_15m: pd.DataFrame,
    h4_proba_df: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    # h4_conflict: sign(h4_pred) != sign(h1_pred) AND h4_conf > 0.60 → reduce_size_50pct
    # h4_exit: sign(h4_pred) != sign(h1_pred) AND h4_conf > 0.80 → exit flag
    signals = signals_15m.copy()
    signals["h4_reduce_size"] = 0
    signals["h4_exit_flag"] = 0

    if h4_proba_df is None or len(h4_proba_df) == 0:
        return signals

    # Align 4h signal to 15m timestamps by forward filling
    if "prob_long" in h4_proba_df.columns:
        h4_prob = h4_proba_df["prob_long"]
    else:
        h4_prob = h4_proba_df.iloc[:, 0]

    h4_aligned = h4_prob.reindex(signals.index, method="ffill")
    h4_dir = np.where(h4_aligned > 0.5, 1, -1)
    h4_conf = np.where(h4_aligned > 0.5, h4_aligned, 1.0 - h4_aligned)

    h15_dir = signals["direction"].values
    conflict = h4_dir != h15_dir

    # Reduce size flag
    reduce_mask = conflict & (h4_conf > 0.60)
    signals.loc[reduce_mask, "h4_reduce_size"] = 1

    # Exit flag (stronger conflict)
    exit_mask = conflict & (h4_conf > 0.80)
    signals.loc[exit_mask, "h4_exit_flag"] = 1

    # Cancel signal if h4 strongly disagrees
    signals.loc[exit_mask, "is_signal"] = 0

    logger.info(f"H4 filter: {reduce_mask.sum()} reduce, {exit_mask.sum()} exit flags applied")
    return signals


def apply_conformal_size_scaling(position_size: float, conformal_width: float, cfg) -> float:
    # Scale position based on prediction uncertainty
    w_full = float(cfg.model.conformal_width_full)       # < 0.20 → 1.0×
    w_partial = float(cfg.model.conformal_width_60pct)   # 0.20-0.40 → 0.6×; >0.40 → 0.3×

    if conformal_width < w_full:
        scale = 1.0
    elif conformal_width < w_partial:
        scale = 0.6
    else:
        scale = 0.3

    return position_size * scale
