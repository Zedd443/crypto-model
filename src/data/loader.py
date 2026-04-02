from pathlib import Path
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("loader")

REQUIRED_OHLCV = ["open", "high", "low", "close", "volume"]


def _enforce_utc(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    elif str(df.index.tz) != "UTC":
        df.index = df.index.tz_convert("UTC")
    return df


def load_ohlcv(symbol: str, timeframe: str, raw_dir: str = "data/raw") -> pd.DataFrame | None:
    path = Path(raw_dir) / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        logger.warning(f"OHLCV file not found: {path}")
        return None

    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]

    missing = [c for c in REQUIRED_OHLCV if c not in df.columns]
    if missing:
        logger.error(f"{symbol}_{timeframe}: missing columns {missing}")
        return None

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    df = _enforce_utc(df)

    # Validate OHLCV integrity
    bad_high = (df["high"] < df["open"]) | (df["high"] < df["close"])
    bad_low = (df["low"] > df["open"]) | (df["low"] > df["close"])
    bad_vol = df["volume"] < 0

    n_bad = bad_high.sum() + bad_low.sum() + bad_vol.sum()
    if n_bad > 0:
        logger.warning(f"{symbol}_{timeframe}: {n_bad} rows with OHLCV integrity violations — dropping")
        valid = ~(bad_high | bad_low | bad_vol)
        df = df[valid]

    if not df.index.is_monotonic_increasing:
        logger.warning(f"{symbol}_{timeframe}: index not monotonic, sorting")
        df = df.sort_index()

    # Drop exact duplicates
    df = df[~df.index.duplicated(keep="last")]

    return df


def load_all_symbols(timeframe: str, raw_dir: str, symbols_list: list[str]) -> dict[str, pd.DataFrame]:
    result = {}
    for symbol in symbols_list:
        df = load_ohlcv(symbol, timeframe, raw_dir)
        if df is not None:
            result[symbol] = df
        else:
            logger.warning(f"Skipping {symbol} for timeframe {timeframe} — file missing or invalid")
    logger.info(f"Loaded {len(result)}/{len(symbols_list)} symbols for {timeframe}")
    return result


def load_macro(raw_dir: str = "data/raw") -> dict[str, pd.DataFrame]:
    macro_files = [
        "macro_bond_10y", "macro_euro", "macro_industrial_prod",
        "macro_inflation_cpi", "macro_sp500", "macro_unemployment",
        "macro_usd_index", "macro_vix",
    ]
    result = {}
    for name in macro_files:
        path = Path(raw_dir) / f"{name}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            df = _enforce_utc(df)
            result[name] = df
        else:
            logger.warning(f"Macro file not found: {path}")
    logger.info(f"Loaded {len(result)} macro series")
    return result


def load_market(raw_dir: str = "data/raw") -> dict[str, pd.DataFrame]:
    market_files = [
        "market_commodity", "market_dji", "market_ftse100", "market_gold",
        "market_hushen300", "market_kospi", "market_msci_world", "market_nikkei",
        "market_silver",
    ]
    result = {}
    for name in market_files:
        path = Path(raw_dir) / f"{name}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            df = _enforce_utc(df)
            result[name] = df
        else:
            logger.warning(f"Market file not found: {path}")
    logger.info(f"Loaded {len(result)} market series")
    return result


def load_fear_greed(raw_dir: str = "data/raw") -> pd.DataFrame | None:
    path = Path(raw_dir) / "fear_greed_cache.parquet"
    if not path.exists():
        logger.warning(f"fear_greed_cache.parquet not found at {path}")
        return None

    df = pd.read_parquet(path)

    # File stores date as a column (not index) — promote it to DatetimeIndex
    if "date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    df = _enforce_utc(df)

    # Normalise value column name
    if "value" not in df.columns:
        candidates = [c for c in df.columns if "value" in c.lower() or "index" in c.lower() or "score" in c.lower()]
        if candidates:
            df = df.rename(columns={candidates[0]: "value"})
            logger.info(f"fear_greed: renamed column '{candidates[0]}' to 'value'")
        else:
            logger.error(f"fear_greed_cache.parquet has no 'value' column — columns: {list(df.columns)}")
            return None

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    logger.info(f"fear_greed loaded: {len(df)} rows, {df.index.min().date()} – {df.index.max().date()}, value range {df['value'].min()}–{df['value'].max()}")
    return df


def load_onchain(symbol: str, raw_dir: str = "data/raw") -> pd.DataFrame:
    path = Path(raw_dir) / f"{symbol}_onchain_coinmetrics.parquet"
    if not path.exists():
        logger.info(f"Onchain file not found for {symbol}: {path} — returning empty DataFrame")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    df = _enforce_utc(df)
    logger.info(f"Loaded onchain data for {symbol}: {len(df)} rows, columns: {list(df.columns)}")
    return df
