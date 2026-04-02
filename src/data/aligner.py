import pandas as pd
from omegaconf import DictConfig
from src.utils.logger import get_logger

logger = get_logger("aligner")


def align_symbols_to_master_index(
    symbols_dict: dict[str, pd.DataFrame],
    timeframe: str,
    cfg: DictConfig,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    if not symbols_dict:
        logger.error("Empty symbols_dict passed to aligner")
        return {}, []

    # Build master index as union of all symbol timestamps
    all_indices = [df.index for df in symbols_dict.values()]
    master_index = all_indices[0]
    for idx in all_indices[1:]:
        master_index = master_index.union(idx)
    master_index = master_index.sort_values()
    logger.info(f"Master index: {len(master_index)} bars for {timeframe}")

    # Forward-fill limit from config — how many consecutive NaN bars to ffill
    htf_limits = dict(cfg.features.htf_ffill_limits) if hasattr(cfg.features, "htf_ffill_limits") else {}
    ffill_limit = htf_limits.get(timeframe, None)

    aligned = {}
    excluded = []
    min_bars = cfg.data.min_align_bars

    for symbol, df in symbols_dict.items():
        # Reindex to master index; data that didn't exist → NaN
        aligned_df = df.reindex(master_index)

        # Forward fill with configured limit (no backward fill ever)
        aligned_df = aligned_df.ffill(limit=ffill_limit)

        # Count non-NaN close bars as valid bar count
        valid_bars = aligned_df["close"].notna().sum()
        if valid_bars < min_bars:
            logger.warning(f"Excluding {symbol}: only {valid_bars} valid bars after alignment (min={min_bars})")
            excluded.append(symbol)
            continue

        aligned[symbol] = aligned_df

    logger.info(f"Alignment complete: {len(aligned)} symbols kept, {len(excluded)} excluded")
    return aligned, excluded
