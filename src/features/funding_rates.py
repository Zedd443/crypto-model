import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("funding_rates")

# Funding is settled at these UTC hours
_FUNDING_HOURS = [0, 8, 16]


def compute_funding_proxy(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    # Basis proxy: directional signal mimicking funding rate pressure
    proxy = (close - open_) / (open_ + 1e-9)
    return proxy.rename("funding_proxy")


def compute_funding_velocity(funding_proxy: pd.Series, periods: int = 1) -> pd.Series:
    vel = funding_proxy.diff(periods)
    return vel.rename(f"funding_velocity_{periods}")


def compute_funding_zscore(funding_proxy: pd.Series, window: int) -> pd.Series:
    roll = funding_proxy.rolling(window, min_periods=window)
    zscore = (funding_proxy - roll.mean()) / (roll.std() + 1e-9)
    return zscore.rename(f"funding_zscore_{window}")


def compute_funding_percentile(funding_proxy: pd.Series, window: int) -> pd.Series:
    # Vectorized rolling rank: expand + rank, then take diagonal-equivalent
    # Equivalent to rolling apply rank but ~20x faster via cumulative approach
    rolled = funding_proxy.rolling(window, min_periods=window)
    rank_min = rolled.rank(method="min", pct=True)
    rank_max = rolled.rank(method="max", pct=True)
    pct = (rank_min + rank_max) / 2.0  # midrank percentile
    return pct.rename(f"funding_percentile_{window}")


def compute_hours_to_funding(index: pd.DatetimeIndex) -> pd.Series:
    # Compute minimum hours until next settlement at 0h, 8h, 16h UTC
    hours = pd.Series(index.hour + index.minute / 60.0, index=index)
    def _min_hours_to_next(h_float):
        candidates = [fh - h_float for fh in _FUNDING_HOURS if fh - h_float > 0]
        # Wrap around midnight: next 0h next day
        candidates.append(24.0 - h_float)
        return min(candidates)
    hours_to_funding = hours.apply(_min_hours_to_next)
    return hours_to_funding.rename("hours_to_funding")


def compute_pre_funding_window(hours_to_funding: pd.Series, threshold: float = 1.0) -> pd.Series:
    flag = (hours_to_funding <= threshold).astype(int)
    return flag.rename(f"pre_funding_window_{threshold}")


def compute_cross_coin_funding_divergence(funding_proxy: pd.Series, btc_funding_proxy: pd.Series) -> pd.Series:
    # Align by index before subtracting
    aligned_btc = btc_funding_proxy.reindex(funding_proxy.index)
    divergence = funding_proxy - aligned_btc
    return divergence.rename("funding_btc_divergence")


def build_funding_features(df: pd.DataFrame, btc_df, cfg) -> pd.DataFrame:
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]

    zscore_window = int(cfg.features.funding_zscore_window)
    pct_window = int(cfg.features.funding_percentile_window)

    proxy = compute_funding_proxy(open_, high, low, close)

    parts = [
        proxy.to_frame(),
        compute_funding_velocity(proxy, 1).to_frame(),
        compute_funding_velocity(proxy, 3).to_frame(),
        compute_funding_zscore(proxy, zscore_window).to_frame(),
        compute_funding_percentile(proxy, pct_window).to_frame(),
        compute_hours_to_funding(df.index).to_frame(),
        compute_pre_funding_window(compute_hours_to_funding(df.index), 1.0).to_frame(),
    ]

    if btc_df is not None:
        btc_proxy = compute_funding_proxy(btc_df["open"], btc_df["high"], btc_df["low"], btc_df["close"])
        parts.append(compute_cross_coin_funding_divergence(proxy, btc_proxy).to_frame())

    result = pd.concat(parts, axis=1)
    result.index = df.index
    return result
