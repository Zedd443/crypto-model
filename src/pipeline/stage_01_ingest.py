from pathlib import Path
from src.utils.config_loader import load_config, get_symbols
from src.utils.state_manager import is_stage_complete, update_project_state
from src.utils.logger import get_logger
from src.data.loader import load_all_symbols, load_macro, load_market, load_fear_greed
from src.data.aligner import align_symbols_to_master_index
from src.data.macro_merger import merge_macro_to_index
from src.data.onchain_merger import merge_onchain_to_index
from src.data.market_data_fetcher import fetch_symbol_market_data
from src.utils.io_utils import write_checkpoint
import pandas as pd

logger = get_logger("stage_01_ingest")


def run(cfg, force: bool = False, symbol_filter: str = None) -> None:
    if not force and is_stage_complete("ingest"):
        logger.info("Stage 1 already complete, skipping. Use --force to re-run.")
        return

    issues = []
    processed_dir = Path(cfg.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_symbols = get_symbols(cfg)
    if symbol_filter:
        _sf = set(symbol_filter) if isinstance(symbol_filter, list) else {symbol_filter}
        all_symbols = [s for s in all_symbols if s.get("name", s.get("symbol")) in _sf]
    symbol_names = [s.get("name", s.get("symbol")) for s in all_symbols]

    logger.info(f"Ingesting {len(symbol_names)} symbols across {len(cfg.data.timeframes)} timeframes")

    for tf in cfg.data.timeframes:
        logger.info(f"Processing timeframe {tf}")
        raw_dir = Path(cfg.data.raw_dir)

        symbols_dict = load_all_symbols(tf, raw_dir, symbol_names)
        logger.info(f"  Loaded {len(symbols_dict)} symbols for {tf}")

        if not symbols_dict:
            issues.append(f"No data loaded for {tf}")
            continue

        aligned, excluded = align_symbols_to_master_index(symbols_dict, tf, cfg)
        if excluded:
            logger.warning(f"  Excluded {len(excluded)} symbols with insufficient bars: {excluded}")
            issues.append(f"Excluded for {tf}: {excluded}")

        if not aligned:
            logger.error(f"No aligned data for {tf}")
            issues.append(f"No aligned data for {tf}")
            continue

        # Build master index from union of all aligned symbol indices
        all_indices = [df.index for df in aligned.values()]
        master_index = all_indices[0]
        for idx in all_indices[1:]:
            master_index = master_index.union(idx)

        # Load macro and market data
        macro_dict = load_macro(raw_dir)
        market_dict = load_market(raw_dir)
        macro_panel = merge_macro_to_index(master_index, macro_dict, market_dict, cfg)

        # Load onchain data (fear & greed only — coinmetrics not available)
        fear_greed_df = load_fear_greed(raw_dir)
        onchain_panel = merge_onchain_to_index(master_index, fear_greed_df, cfg)

        # Save per-symbol checkpoints
        checkpoints_dir = Path(cfg.data.checkpoints_dir)
        for symbol, df in aligned.items():
            write_checkpoint(df, "ingest", symbol, tf, checkpoints_dir)

        # Fetch market positioning data (OI, LS ratios, taker ratio) for 15m only
        # These endpoints only support 15m granularity on Binance FAPI
        if tf == "15m":
            logger.info("Fetching market positioning data (OI, LS ratio, taker ratio) per symbol")
            for symbol in list(aligned.keys()):
                try:
                    fetch_symbol_market_data(symbol, raw_dir)
                except Exception as e:
                    logger.warning(f"{symbol}: market data fetch failed — {e}")
                    issues.append(f"{symbol}_market_data: {e}")

        # Save panel data
        macro_path = processed_dir / f"macro_panel_{tf}.parquet"
        onchain_path = processed_dir / f"onchain_panel_{tf}.parquet"
        macro_panel.to_parquet(macro_path)
        onchain_panel.to_parquet(onchain_path)
        logger.info(
            f"  Saved macro panel {macro_panel.shape} -> {macro_path.name} | "
            f"onchain panel {onchain_panel.shape} -> {onchain_path.name}"
        )

    update_project_state("ingest", "done", issues, output_dir=str(processed_dir))
    logger.info(f"Stage 1 complete. Issues: {len(issues)}")
