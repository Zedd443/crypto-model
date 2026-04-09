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
        frames[f"{name}_{w}_skew"] = roll.skew()
        frames[f"{name}_{w}_kurt"] = roll.kurt()
    return pd.DataFrame(frames, index=series.index)


def compute_lag_features(series: pd.Series, name: str, lags: list) -> pd.DataFrame:
    return pd.DataFrame(
        {f"{name}_lag_{l}": series.shift(l) for l in lags},
        index=series.index,
    )


def compute_har_rv(log_return: pd.Series, daily_bars: int = 96) -> pd.DataFrame:
    r2 = log_return ** 2
    rv_daily = r2.rolling(daily_bars, min_periods=daily_bars).sum()
    rv_weekly = rv_daily.rolling(5, min_periods=5).mean()
    rv_monthly = rv_daily.rolling(22, min_periods=22).mean()
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

    bb_period = 20
    parts.append(compute_bb(close, bb_period))
    parts.append(compute_macd(close))
    parts.append(compute_adx(high, low, close, 14))

    for w in [14, 50]:
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

    parts.append(compute_har_rv(log_ret, daily_bars))
    parts.append(compute_jump_decomposition(log_ret, daily_bars))
    parts.append(compute_realized_skewness(log_ret, daily_bars).to_frame())
    parts.append(compute_realized_kurtosis(log_ret, daily_bars).to_frame())

    # Time-of-day cyclical features — no leakage (uses bar timestamp only)
    parts.append(compute_time_of_day_cyclical(close.index))

    # Rolling ACF lag-1 and lag-5 — momentum vs mean-reversion signal
    acf_window = int(getattr(cfg.features, 'acf_window', 96))
    parts.append(compute_rolling_acf(log_ret, lag=1, window=acf_window).to_frame())
    parts.append(compute_rolling_acf(log_ret, lag=5, window=acf_window).to_frame())

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
