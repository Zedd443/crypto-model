import pandas as pd
from omegaconf import DictConfig
from src.utils.logger import get_logger

logger = get_logger("onchain_merger")


def merge_onchain_to_index(
    master_index: pd.DatetimeIndex,
    fear_greed_df: pd.DataFrame | None,
    cfg: DictConfig,
) -> pd.DataFrame:
    # Only fear & greed is available — coinmetrics data (NVT/SOPR/exchange_flow) never existed
    result = pd.DataFrame(index=master_index)

    # Fear & greed: 1-day lag then ffill to intraday
    if fear_greed_df is not None and not fear_greed_df.empty and "value" in fear_greed_df.columns:
        fg = fear_greed_df[["value"]].copy()
        if fg.index.tz is None:
            fg.index = fg.index.tz_localize("UTC")
        elif str(fg.index.tz) != "UTC":
            fg.index = fg.index.tz_convert("UTC")

        # Shift 1 day to prevent look-ahead (index announcement date → next day availability)
        fg = fg.shift(1, freq="D")

        fg_reset = fg.reset_index()
        # Index name may be "date", "index", or the original column name — normalise to "timestamp"
        idx_col = fg_reset.columns[0]
        fg_reset = fg_reset.rename(columns={idx_col: "timestamp", "value": "fear_greed_value"})
        fg_reset = fg_reset.sort_values("timestamp")

        master_reset = pd.DataFrame({"timestamp": master_index})
        merged = pd.merge_asof(master_reset, fg_reset, on="timestamp", direction="backward")
        merged = merged.set_index("timestamp")

        # Only ffill — no bfill
        result["fear_greed_value"] = merged["fear_greed_value"].ffill()
        result["fear_greed_zscore"] = (
            (result["fear_greed_value"] - result["fear_greed_value"].rolling(30).mean())
            / (result["fear_greed_value"].rolling(30).std() + 1e-9)
        )
        logger.info("Fear & greed merged successfully")
    else:
        logger.warning("Fear & greed not available — columns will be NaN")
        result["fear_greed_value"] = float("nan")
        result["fear_greed_zscore"] = float("nan")

    return result
