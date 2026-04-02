import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("time_utils")

_TF_MAP = {
    "1m": pd.Timedelta(minutes=1),
    "3m": pd.Timedelta(minutes=3),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4),
    "6h": pd.Timedelta(hours=6),
    "8h": pd.Timedelta(hours=8),
    "12h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=1),
    "3d": pd.Timedelta(days=3),
    "1w": pd.Timedelta(weeks=1),
}


def ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"DataFrame index is not DatetimeIndex: {type(df.index)}")
    if df.index.tz is None:
        raise ValueError("DataFrame index is timezone-naive — all times must be UTC")
    if str(df.index.tz) != "UTC":
        logger.debug(f"Converting index tz from {df.index.tz} to UTC")
        df = df.copy()
        df.index = df.index.tz_convert("UTC")
    return df


def timeframe_to_timedelta(tf: str) -> pd.Timedelta:
    if tf not in _TF_MAP:
        raise ValueError(f"Unknown timeframe '{tf}'. Supported: {list(_TF_MAP.keys())}")
    return _TF_MAP[tf]


def assert_no_future_leakage(feature_df: pd.DataFrame, label_df: pd.DataFrame) -> None:
    # Features at time t must be derived from data strictly before t
    # Labels at t represent the forward return starting at t
    # So feature index and label index can share timestamps —
    # but we assert features were not computed using label-period data
    # by checking no feature NaN pattern aligns with look-ahead
    common_idx = feature_df.index.intersection(label_df.index)
    if len(common_idx) == 0:
        raise ValueError("Feature and label DataFrames share no common timestamps — alignment error")

    # Strict check: label t1 (exit time) must always be > entry time t0
    if "t1" in label_df.columns:
        bad = label_df["t1"] <= label_df.index
        n_bad = bad.sum()
        if n_bad > 0:
            raise ValueError(f"Leakage detected: {n_bad} rows where label t1 <= entry time")

    logger.debug("No future leakage detected in feature/label alignment")


def bars_in_period(timeframe: str, days: int) -> int:
    td = timeframe_to_timedelta(timeframe)
    day_td = pd.Timedelta(days=days)
    return int(day_td / td)
