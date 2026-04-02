import pandas as pd
from omegaconf import DictConfig, OmegaConf
from src.utils.logger import get_logger

logger = get_logger("macro_merger")


def apply_release_lag(series: pd.Series, source_name: str, cfg: DictConfig) -> pd.Series:
    release_lags = OmegaConf.to_container(cfg.macro.release_lags, resolve=True)

    if source_name not in release_lags:
        logger.debug(f"No release lag configured for {source_name}, using 0")
        return series

    lag_cfg = release_lags[source_name]
    lag_days = lag_cfg.get("lag_days", 0)
    safety_buffer = lag_cfg.get("safety_buffer_days", 0)
    total_lag = lag_days + safety_buffer

    if total_lag > 0:
        series = series.shift(total_lag, freq="D")
        logger.debug(f"{source_name}: applied lag of {total_lag} calendar days")

    return series


def merge_macro_to_index(
    master_index: pd.DatetimeIndex,
    macro_dict: dict[str, pd.DataFrame],
    market_dict: dict[str, pd.DataFrame],
    cfg: DictConfig,
) -> pd.DataFrame:
    master_index.name = "timestamp"
    master_df = pd.DataFrame(index=master_index)

    all_sources = {**macro_dict, **market_dict}

    macro_data_quality = pd.Series(0, index=master_index, dtype=float)
    total_sources = len(all_sources)

    for source_name, df in all_sources.items():
        if df is None or df.empty:
            logger.warning(f"Skipping empty macro source: {source_name}")
            continue

        # Use first numeric column if 'value' not present
        value_col = "value" if "value" in df.columns else None
        if value_col is None:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                logger.warning(f"{source_name}: no numeric columns found, skipping")
                continue
            value_col = numeric_cols[0]

        series = df[value_col].copy()

        # Apply release lag (OECD data already shifted +1 period by caller convention)
        series = apply_release_lag(series, source_name, cfg)

        # merge_asof requires sorted index
        series_df = series.reset_index()
        series_df.columns = ["timestamp", source_name]
        series_df = series_df.sort_values("timestamp")

        master_reset = master_df.reset_index()

        merged = pd.merge_asof(
            master_reset,
            series_df,
            on="timestamp",
            direction="backward",
        )
        merged = merged.set_index("timestamp")

        # Only ffill — NEVER bfill
        master_df[source_name] = merged[source_name].ffill()

        # Track how many sources have data at each timestamp
        has_data = master_df[source_name].notna().astype(float)
        macro_data_quality += has_data

        # Days since last release per source
        days_col = f"{source_name}_days_since"
        last_valid = master_df[source_name].notna()
        # Cumulative count of bars since last valid value
        cumcount = (~last_valid).groupby(last_valid.cumsum()).cumcount()
        master_df[days_col] = cumcount

    if total_sources > 0:
        master_df["macro_data_quality"] = macro_data_quality / total_sources
    else:
        master_df["macro_data_quality"] = 0.0

    logger.info(f"Macro merge complete: {len(master_df.columns)} columns, {len(master_df)} rows")
    return master_df
