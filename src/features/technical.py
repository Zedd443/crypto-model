import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("technical")


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.rename(f"rsi_{period}")


def compute_bb(close: pd.Series, period: int, std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(period, min_periods=period).mean()
    sigma = close.rolling(period, min_periods=period).std()
    upper = mid + std * sigma
    lower = mid - std * sigma
    bb_width = (upper - lower) / (mid + 1e-9)
    # percent B: (close - lower) / (upper - lower)
    bb_pct = (close - lower) / (upper - lower + 1e-9)
    return pd.DataFrame({
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_pct": bb_pct,
        "bb_width": bb_width,
    }, index=close.index)


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, min_periods=signal_period).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": hist,
    }, index=close.index)


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    # True Range
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional movement
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)

    # Smoothed (Wilder's EMA using alpha=1/period)
    atr_smooth = tr.ewm(alpha=1.0 / period, min_periods=period).mean()
    plus_smooth = plus_dm.ewm(alpha=1.0 / period, min_periods=period).mean()
    minus_smooth = minus_dm.ewm(alpha=1.0 / period, min_periods=period).mean()

    di_plus = 100.0 * plus_smooth / (atr_smooth + 1e-9)
    di_minus = 100.0 * minus_smooth / (atr_smooth + 1e-9)
    dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
    adx = dx.ewm(alpha=1.0 / period, min_periods=period).mean()

    return pd.DataFrame({
        "adx": adx,
        "di_plus": di_plus,
        "di_minus": di_minus,
    }, index=close.index)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period).mean()
    return atr.rename(f"atr_{period}")


def compute_natr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    atr = compute_atr(high, low, close, period)
    return (atr / close).rename(f"natr_{period}")


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    obv = (direction * volume).cumsum()
    return obv.rename("obv")


def compute_vwap_rolling(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    typical = (high + low + close) / 3.0
    pv = typical * volume
    vwap = pv.rolling(window, min_periods=window).sum() / (volume.rolling(window, min_periods=window).sum() + 1e-9)
    return vwap.rename(f"vwap_{window}")


def compute_vwap_deviation(close: pd.Series, vwap: pd.Series) -> pd.Series:
    return ((close - vwap) / (vwap + 1e-9)).rename("vwap_deviation")


def compute_rolling_stats(series: pd.Series, name: str, windows: list) -> pd.DataFrame:
    frames = {}
    for w in windows:
        roll = series.rolling(w, min_periods=w)
        frames[f"{name}_{w}_mean"] = roll.mean()
        frames[f"{name}_{w}_std"] = roll.std()
        # Skew and kurt are numerically unstable for n<20 (Jondeau-Rockinger 2003; Bai-Ng):
        # small-sample higher moments produce ±Inf / huge values that dominate tree splits.
        # Skip these columns entirely for w<20; downstream iterators must not assume every
        # window produces skew/kurt columns — use df.filter() or explicit column checks.
        if w >= 20:
            frames[f"{name}_{w}_skew"] = series.rolling(w, min_periods=w).skew()
            frames[f"{name}_{w}_kurt"] = series.rolling(w, min_periods=w).kurt()
    return pd.DataFrame(frames, index=series.index)


def compute_lag_features(series: pd.Series, name: str, lags: list) -> pd.DataFrame:
    return pd.DataFrame(
        {f"{name}_lag_{l}": series.shift(l) for l in lags},
        index=series.index,
    )


def compute_har_rv(log_return: pd.Series, daily_bars: int = 96, weekly_days: int = 5, monthly_days: int = 22) -> pd.DataFrame:
    r2 = log_return ** 2
    rv_daily = r2.rolling(daily_bars, min_periods=daily_bars).sum()
    rv_weekly = rv_daily.rolling(weekly_days, min_periods=weekly_days).mean()
    rv_monthly = rv_daily.rolling(monthly_days, min_periods=monthly_days).mean()
    return pd.DataFrame({
        "rv_daily": rv_daily,
        "rv_weekly": rv_weekly,
        "rv_monthly": rv_monthly,
    }, index=log_return.index)


def compute_jump_decomposition(log_return: pd.Series, daily_bars: int = 96) -> pd.DataFrame:
    r2 = log_return ** 2
    rv_daily = r2.rolling(daily_bars, min_periods=daily_bars).sum()
    abs_r = log_return.abs()
    # BV = (pi/2) * sum(|r_t| * |r_{t-1}|) — bipower variation
    bv_term = abs_r * abs_r.shift(1)
    bv = (np.pi / 2.0) * bv_term.rolling(daily_bars, min_periods=daily_bars).sum()
    jump_component = (rv_daily - bv).clip(lower=0.0)
    continuous_vol = np.sqrt(bv.clip(lower=0.0))
    return pd.DataFrame({
        "bv": bv,
        "jump_component": jump_component,
        "continuous_vol": continuous_vol,
    }, index=log_return.index)


def compute_realized_skewness(log_return: pd.Series, window: int = 96) -> pd.Series:
    r2 = log_return ** 2
    r3 = log_return ** 3
    rv = r2.rolling(window, min_periods=window).sum()
    sum_r3 = r3.rolling(window, min_periods=window).sum()
    rs = np.sqrt(window) * sum_r3 / (rv ** 1.5 + 1e-20)
    return rs.rename("realized_skewness")


def compute_realized_kurtosis(log_return: pd.Series, window: int = 96) -> pd.Series:
    r2 = log_return ** 2
    r4 = log_return ** 4
    rv = r2.rolling(window, min_periods=window).sum()
    sum_r4 = r4.rolling(window, min_periods=window).sum()
    rk = window * sum_r4 / (rv ** 2 + 1e-20)
    return rk.rename("realized_kurtosis")


def build_technical_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]

    windows = list(cfg.features.lookback_windows)
    lags = list(cfg.features.lag_periods)
    daily_bars = int(cfg.features.rv_daily_bars)

    # shift(1) uses previous bar close — backward-looking; the global shift(1) in
    # feature_pipeline.py pushes this further so feature at t = log(close_{t-1}/close_{t-2})
    log_ret = np.log(close / close.shift(1)).rename("log_return")

    parts = [log_ret.to_frame()]

    for w in windows:
        parts.append(compute_rsi(close, w).to_frame())

    bb_period = int(getattr(cfg.features, 'bb_period', 20))
    bb_std_mult = float(getattr(cfg.features, 'bb_std_mult', 2.0))
    parts.append(compute_bb(close, bb_period, std=bb_std_mult))

    macd_fast = int(getattr(cfg.features, 'macd_fast', 12))
    macd_slow = int(getattr(cfg.features, 'macd_slow', 26))
    macd_signal = int(getattr(cfg.features, 'macd_signal', 9))
    parts.append(compute_macd(close, fast=macd_fast, slow=macd_slow, signal_period=macd_signal))

    adx_period = int(getattr(cfg.features, 'adx_period', 14))
    parts.append(compute_adx(high, low, close, adx_period))

    atr_periods = list(getattr(cfg.features, 'atr_periods', [14, 50]))
    for w in atr_periods:
        parts.append(compute_atr(high, low, close, w).to_frame())
        parts.append(compute_natr(high, low, close, w).to_frame())

    parts.append(compute_obv(close, volume).to_frame())

    vwap = compute_vwap_rolling(high, low, close, volume, int(cfg.features.vwap_window))
    parts.append(vwap.to_frame())
    parts.append(compute_vwap_deviation(close, vwap).to_frame())

    parts.append(compute_rolling_stats(log_ret, "log_ret", windows))
    parts.append(compute_rolling_stats(volume, "volume", windows))

    parts.append(compute_lag_features(log_ret, "log_ret", lags))
    parts.append(compute_lag_features(volume, "volume", lags))

    weekly_days = int(getattr(cfg.features, 'rv_weekly_days', 5))
    monthly_days = int(getattr(cfg.features, 'rv_monthly_days', 22))
    parts.append(compute_har_rv(log_ret, daily_bars, weekly_days, monthly_days))
    parts.append(compute_jump_decomposition(log_ret, daily_bars))
    parts.append(compute_realized_skewness(log_ret, daily_bars).to_frame())
    parts.append(compute_realized_kurtosis(log_ret, daily_bars).to_frame())

    # Time-of-day cyclical features — no leakage (uses bar timestamp only)
    parts.append(compute_time_of_day_cyclical(close.index))

    # Rolling ACF lag-1 and lag-5 — momentum vs mean-reversion signal
    acf_window = int(getattr(cfg.features, 'acf_window', 96))
    parts.append(compute_rolling_acf(log_ret, lag=1, window=acf_window).to_frame())
    parts.append(compute_rolling_acf(log_ret, lag=5, window=acf_window).to_frame())

    # EMA deviation and crossover features
    ema_spans = list(getattr(cfg.features, 'ema_spans', [9, 21, 50]))
    if ema_spans:
        parts.append(compute_ema_features(close, ema_spans))

    # Supertrend — ATR-based trend line with direction and normalized distance
    st_period = int(getattr(cfg.features, 'supertrend_period', 10))
    st_mult = float(getattr(cfg.features, 'supertrend_mult', 3.0))
    parts.append(compute_supertrend(high, low, close, period=st_period, mult=st_mult))

    # StochRSI — RSI of RSI; more sensitive overbought/oversold detector
    sr_period = int(getattr(cfg.features, 'stochrsi_period', 14))
    sr_smooth = int(getattr(cfg.features, 'stochrsi_smooth_k', 3))
    parts.append(compute_stochrsi(close, rsi_period=sr_period, stoch_period=sr_period, smooth_k=sr_smooth))

    # Williams %R — momentum oscillator complementary to RSI
    wr_period = int(getattr(cfg.features, 'williams_r_period', 14))
    parts.append(compute_williams_r(high, low, close, period=wr_period).to_frame())

    # Keltner Channel — EMA-based envelope; used standalone and for Squeeze
    kc_period = int(getattr(cfg.features, 'keltner_period', 20))
    kc_mult = float(getattr(cfg.features, 'keltner_mult', 1.5))
    parts.append(compute_keltner_channel(high, low, close, period=kc_period, atr_mult=kc_mult))

    # Squeeze Momentum (TTM Squeeze) — detects BB inside KC compression before breakout
    sq_mom_period = int(getattr(cfg.features, 'squeeze_momentum_period', 12))
    parts.append(compute_squeeze_momentum(high, low, close,
                                           bb_period=bb_period, bb_std=bb_std_mult,
                                           kc_period=kc_period, kc_mult=kc_mult,
                                           mom_period=sq_mom_period))

    # Session-anchored VWAP — reset at 0h/8h/16h UTC (same as funding settlement)
    session_hours = list(getattr(cfg.features, 'vwap_session_hours', [0, 8, 16]))
    parts.append(compute_vwap_session(high, low, close, volume, close.index, session_hours))

    return pd.concat(parts, axis=1)


def compute_time_of_day_cyclical(index: pd.DatetimeIndex) -> pd.DataFrame:
    # Cyclical encoding of hour-of-day using sin/cos — preserves circular structure
    # All UTC — no leakage (uses bar open timestamp)
    hour = index.hour + index.minute / 60.0
    tod_sin = np.sin(2 * np.pi * hour / 24.0)
    tod_cos = np.cos(2 * np.pi * hour / 24.0)
    return pd.DataFrame({"tod_sin": tod_sin, "tod_cos": tod_cos}, index=index)


def compute_rolling_acf(series: pd.Series, lag: int, window: int) -> pd.Series:
    # Rolling autocorrelation at given lag — positive = momentum, negative = mean-reversion
    return series.rolling(window, min_periods=window // 2).corr(
        series.shift(lag)
    ).rename(f"acf_lag{lag}_w{window}")


def compute_supertrend(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    atr = compute_atr(high, low, close, period)
    hl_mid = (high + low) / 2.0

    # Work on numpy arrays for speed (state-dependent loop can't be fully vectorized)
    raw_upper = (hl_mid + mult * atr).values
    raw_lower = (hl_mid - mult * atr).values
    close_v = close.values
    n = len(close_v)

    upper = raw_upper.copy()
    lower = raw_lower.copy()
    direction = np.ones(n)
    supertrend = np.full(n, np.nan)

    for i in range(1, n):
        # Lock bands: upper can only decrease, lower can only increase
        upper[i] = raw_upper[i] if close_v[i - 1] > upper[i - 1] else min(raw_upper[i], upper[i - 1])
        lower[i] = raw_lower[i] if close_v[i - 1] < lower[i - 1] else max(raw_lower[i], lower[i - 1])

        if direction[i - 1] == -1 and close_v[i] > upper[i]:
            direction[i] = 1.0
        elif direction[i - 1] == 1 and close_v[i] < lower[i]:
            direction[i] = -1.0
        else:
            direction[i] = direction[i - 1]

        supertrend[i] = lower[i] if direction[i] == 1 else upper[i]

    atr_v = atr.values
    distance = (close_v - supertrend) / (atr_v + 1e-9)
    flip = np.abs(np.diff(direction, prepend=direction[0])).clip(0, 1)

    return pd.DataFrame({
        "supertrend_direction": direction,
        "supertrend_distance": distance,
        "supertrend_flip": flip,
    }, index=close.index)


def compute_stochrsi(close: pd.Series, rsi_period: int = 14, stoch_period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
    rsi = compute_rsi(close, rsi_period)
    rsi_min = rsi.rolling(stoch_period, min_periods=stoch_period).min()
    rsi_max = rsi.rolling(stoch_period, min_periods=stoch_period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-9)
    k = stoch_rsi.rolling(smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(smooth_d, min_periods=smooth_d).mean()
    return pd.DataFrame({
        "stochrsi_k": k,
        "stochrsi_d": d,
        "stochrsi_kd_diff": k - d,
    }, index=close.index)


def compute_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest_high = high.rolling(period, min_periods=period).max()
    lowest_low = low.rolling(period, min_periods=period).min()
    wr = -100.0 * (highest_high - close) / (highest_high - lowest_low + 1e-9)
    return wr.rename(f"williams_r_{period}")


def compute_keltner_channel(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, atr_mult: float = 1.5) -> pd.DataFrame:
    ema_mid = close.ewm(span=period, adjust=False).mean()
    atr = compute_atr(high, low, close, period)
    upper = ema_mid + atr_mult * atr
    lower = ema_mid - atr_mult * atr
    kc_pct = (close - lower) / (upper - lower + 1e-9)  # 0=at lower, 1=at upper
    kc_width = (upper - lower) / (ema_mid + 1e-9)
    return pd.DataFrame({
        "kc_upper": upper,
        "kc_lower": lower,
        "kc_pct": kc_pct,
        "kc_width": kc_width,
    }, index=close.index)


def compute_squeeze_momentum(high: pd.Series, low: pd.Series, close: pd.Series,
                              bb_period: int = 20, bb_std: float = 2.0,
                              kc_period: int = 20, kc_mult: float = 1.5,
                              mom_period: int = 12) -> pd.DataFrame:
    # Squeeze = Bollinger Band inside Keltner Channel → low volatility compression
    bb = compute_bb(close, bb_period, std=bb_std)
    kc = compute_keltner_channel(high, low, close, kc_period, kc_mult)

    squeeze_on = ((bb["bb_upper"] < kc["kc_upper"]) & (bb["bb_lower"] > kc["kc_lower"])).astype(float)

    # Momentum oscillator: close vs midpoint of recent range (TTM Squeeze style)
    highest_high = high.rolling(mom_period, min_periods=mom_period).max()
    lowest_low = low.rolling(mom_period, min_periods=mom_period).min()
    delta = close - (highest_high + lowest_low) / 2.0
    momentum = delta.ewm(span=mom_period, adjust=False).mean()

    return pd.DataFrame({
        "squeeze_on": squeeze_on,
        "squeeze_momentum": momentum,
        "squeeze_momentum_sign": np.sign(momentum),
    }, index=close.index)


def compute_vwap_session(high: pd.Series, low: pd.Series, close: pd.Series,
                          volume: pd.Series, index: pd.DatetimeIndex,
                          session_hours: list = None) -> pd.DataFrame:
    if session_hours is None:
        session_hours = [0, 8, 16]
    typical = (high + low + close) / 3.0
    pv = typical * volume

    # Assign each bar to its session boundary (the most recent session_hours mark)
    hour = index.hour
    session_id = pd.Series(np.zeros(len(index), dtype=int), index=index)
    for h in sorted(session_hours, reverse=True):
        session_id[hour >= h] = h
    # Build cumulative session groups: group changes when session_id changes
    session_group = (session_id != session_id.shift(1)).cumsum()

    cum_pv = pv.groupby(session_group).cumsum()
    cum_vol = volume.groupby(session_group).cumsum()
    session_vwap = cum_pv / (cum_vol + 1e-9)

    anchor_dist = (close - session_vwap) / (session_vwap + 1e-9)
    # Rolling z-score of distance within session
    anchor_std = anchor_dist.groupby(session_group).transform(lambda x: x.expanding().std())
    anchor_zscore = anchor_dist / (anchor_std + 1e-9)

    return pd.DataFrame({
        "session_vwap": session_vwap,
        "session_vwap_dist": anchor_dist,
        "session_vwap_zscore": anchor_zscore.clip(-5, 5),
    }, index=index)


def compute_cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    # Money flow multiplier: position of close in the bar's range
    mf_mult = ((close - low) - (high - close)) / (high - low + 1e-9)
    mf_vol = mf_mult * volume
    cmf = mf_vol.rolling(window, min_periods=window).sum() / (volume.rolling(window, min_periods=window).sum() + 1e-9)
    return cmf.rename(f"cmf_{window}")


def compute_ema_features(close: pd.Series, spans: list[int]) -> pd.DataFrame:
    # EMA values, EMA crossover signal, and price-vs-EMA deviation
    parts = {}
    emas = {}
    for span in spans:
        ema = close.ewm(span=span, adjust=False).mean()
        emas[span] = ema
        parts[f"ema_{span}"] = ema
        parts[f"ema_{span}_dev"] = (close - ema) / (ema + 1e-9)  # price deviation from EMA

    # Crossover signals: EMA fast vs EMA slow pairs (9/21, 9/50, 21/50)
    span_pairs = [(s, l) for i, s in enumerate(spans) for l in spans[i+1:]]
    for fast, slow in span_pairs:
        cross_col = f"ema_{fast}_{slow}_cross"
        # +1 = fast above slow (bullish), -1 = fast below slow (bearish)
        parts[cross_col] = np.sign(emas[fast] - emas[slow])
        # Distance as fraction — magnitude of trend
        parts[f"ema_{fast}_{slow}_gap"] = (emas[fast] - emas[slow]) / (emas[slow] + 1e-9)

    return pd.DataFrame(parts, index=close.index)
