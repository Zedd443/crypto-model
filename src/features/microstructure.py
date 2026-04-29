import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("microstructure")


def compute_log_return(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1)).rename("log_return")


def compute_ofi(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    # Signed volume proxy based on bar position
    buy_vol_proxy = ((close - open_) / (high - low + 1e-9)).clip(-1.0, 1.0) * volume
    sell_vol_proxy = volume - buy_vol_proxy
    net_flow = buy_vol_proxy - sell_vol_proxy
    ofi = net_flow.rolling(window, min_periods=window).sum()
    return ofi.rename(f"ofi_{window}")


def compute_amihud(log_return: pd.Series, volume_usd: pd.Series, window: int = 20) -> pd.Series:
    # Amihud illiquidity: |return| / volume_usd, rolling mean
    ratio = log_return.abs() / (volume_usd + 1e-9)
    amihud = ratio.rolling(window, min_periods=window).mean()
    return amihud.rename(f"amihud_{window}")


def compute_kyle_lambda(log_return: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    # |return| / sqrt(volume), rolling mean
    ratio = log_return.abs() / (np.sqrt(volume + 1e-9))
    kyle = ratio.rolling(window, min_periods=window).mean()
    return kyle.rename(f"kyle_lambda_{window}")


def compute_roll_measure(close: pd.Series, window: int = 20) -> pd.Series:
    # Effective bid-ask spread proxy: 2 * sqrt(max(-cov(r_t, r_{t-1}), 0))
    log_ret = np.log(close / close.shift(1))
    lag1 = log_ret.shift(1)
    # Rolling covariance: cov(r_t, r_{t-1})
    roll_cov = log_ret.rolling(window, min_periods=window).cov(lag1)
    roll_measure = 2.0 * np.sqrt((-roll_cov).clip(lower=0.0))
    return roll_measure.rename(f"roll_measure_{window}")


def compute_parkinson_vol(high: pd.Series, low: pd.Series, window: int = 14) -> pd.Series:
    # Parkinson (1980): uses high-low range as vol estimator
    term = (np.log(high) - np.log(low)) ** 2 / (4.0 * np.log(2.0))
    pv = term.rolling(window, min_periods=window).mean()
    return pv.rename(f"parkinson_vol_{window}")


def compute_volume_surprise(volume: pd.Series, window: int = 20) -> pd.Series:
    # Deviation from EWM baseline, normalized by EWM std
    ewm_mean = volume.ewm(span=window, adjust=False).mean()
    ewm_std = volume.ewm(span=window, adjust=False).std()
    surprise = (volume - ewm_mean) / (ewm_std + 1e-9)
    return surprise.rename(f"volume_surprise_{window}")


def compute_volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    # Current volume relative to rolling mean
    ratio = volume / (volume.rolling(window, min_periods=window).mean() + 1e-9)
    return ratio.rename(f"volume_ratio_{window}")


def compute_spread_proxy(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    # High-low spread relative to close, rolling mean
    hl_spread = (high - low) / (close + 1e-9)
    spread = hl_spread.rolling(window, min_periods=window).mean()
    return spread.rename(f"spread_proxy_{window}")


def compute_cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    mf_mult = ((close - low) - (high - close)) / (high - low + 1e-9)
    mf_vol = mf_mult * volume
    cmf = mf_vol.rolling(window, min_periods=window).sum() / (volume.rolling(window, min_periods=window).sum() + 1e-9)
    return cmf.rename(f"cmf_{window}")


def build_microstructure_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]

    ofi_window = int(cfg.features.ofi_window)
    amihud_window = int(cfg.features.amihud_window)
    kyle_window = int(cfg.features.kyle_window)
    spread_window = int(cfg.features.spread_window)

    log_ret = compute_log_return(close)
    # Volume in USD (approximate)
    volume_usd = volume * close

    cmf_window = int(getattr(cfg.features, 'cmf_window', 20))
    parkinson_windows = list(getattr(cfg.features, 'parkinson_vol_windows', [14, 50]))

    parts = [
        compute_ofi(open_, high, low, close, volume, ofi_window).to_frame(),
        compute_amihud(log_ret, volume_usd, amihud_window).to_frame(),
        compute_kyle_lambda(log_ret, volume, kyle_window).to_frame(),
        compute_roll_measure(close, amihud_window).to_frame(),
        compute_cmf(high, low, close, volume, cmf_window).to_frame(),
        compute_volume_surprise(volume, ofi_window).to_frame(),
        compute_volume_ratio(volume, ofi_window).to_frame(),
        compute_spread_proxy(high, low, close, spread_window).to_frame(),
    ]
    for pw in parkinson_windows:
        parts.append(compute_parkinson_vol(high, low, pw).to_frame())

    result = pd.concat(parts, axis=1)
    result.index = df.index
    return result
