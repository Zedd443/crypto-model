from pathlib import Path
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("io_utils")

REQUIRED_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def read_raw(symbol: str, timeframe: str, raw_dir: str = "data/raw") -> pd.DataFrame | None:
    path = Path(raw_dir) / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        logger.warning(f"Raw file not found: {path}")
        return None

    df = pd.read_parquet(path)
    df = _enforce_utc_index(df)

    missing = [c for c in REQUIRED_OHLCV_COLS if c not in df.columns]
    if missing:
        logger.error(f"{symbol}_{timeframe}: missing columns {missing}")
        return None

    return df


def write_checkpoint(df: pd.DataFrame, stage: str, symbol: str, timeframe: str, base_dir: str = "data/checkpoints") -> None:
    out_dir = Path(base_dir) / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_{timeframe}.parquet"
    df.to_parquet(path)
    logger.debug(f"Checkpoint saved: {path}")


def checkpoint_exists(stage: str, symbol: str, timeframe: str, base_dir: str = "data/checkpoints") -> bool:
    path = Path(base_dir) / stage / f"{symbol}_{timeframe}.parquet"
    return path.exists()


def read_checkpoint(stage: str, symbol: str, timeframe: str, base_dir: str = "data/checkpoints") -> pd.DataFrame:
    path = Path(base_dir) / stage / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    df = pd.read_parquet(path)
    return _enforce_utc_index(df)


def write_features(df: pd.DataFrame, symbol: str, timeframe: str, features_dir: str = "data/features") -> None:
    out_dir = Path(features_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_{timeframe}_features.parquet"
    df.to_parquet(path)
    logger.debug(f"Features saved: {path}")


def read_features(symbol: str, timeframe: str, features_dir: str = "data/features") -> pd.DataFrame:
    path = Path(features_dir) / f"{symbol}_{timeframe}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    df = pd.read_parquet(path)
    return _enforce_utc_index(df)


def _enforce_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to convert first column as timestamp if index is not datetime
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    elif str(df.index.tz) != "UTC":
        df.index = df.index.tz_convert("UTC")
    return df
