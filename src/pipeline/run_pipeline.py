# Entry point for the crypto ML pipeline.
# Run as module: python -m src.pipeline.run_pipeline [options]
#
# Examples:
#   python -m src.pipeline.run_pipeline                        # run all stages 1-7
#   python -m src.pipeline.run_pipeline --stage 2              # run stage 2 only (features)
#   python -m src.pipeline.run_pipeline --stage 4 --force      # re-run training even if done
#   python -m src.pipeline.run_pipeline --stage 4 --symbol BTCUSDT  # single symbol debug
#   python -m src.pipeline.run_pipeline --stage 8              # run live execution loop
#
# Stage map:
#   1 ingest       — download OHLCV + on-chain + macro data from APIs
#   2 features     — compute technical/regime/cross-sectional features per symbol
#   3 labels       — triple-barrier labeling + sample weights
#   4 training     — XGBoost primary model with Optuna tuning + feature selection
#   5 meta_labeling— meta-labeler trained on OOF predictions (signal confidence)
#   6 portfolio    — generate signals, position sizing, correlation filter
#   7 backtest     — walk-forward backtest with realistic costs
#   8 live         — real-time execution loop (Binance Demo/Mainnet FAPI)

import argparse
import sys
from pathlib import Path
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.pipeline import (
    stage_01_ingest,
    stage_02_features,
    stage_03_labels,
    stage_04_train,
    stage_05_meta,
    stage_06_portfolio,
    stage_07_backtest,
    stage_08_live,
)

logger = get_logger("run_pipeline")

STAGES = {
    1: stage_01_ingest.run,
    2: stage_02_features.run,
    3: stage_03_labels.run,
    4: stage_04_train.run,
    5: stage_05_meta.run,
    6: stage_06_portfolio.run,
    7: stage_07_backtest.run,
    8: stage_08_live.run,
}

STAGE_NAMES = {
    1: "ingest",
    2: "features",
    3: "labels",
    4: "training",
    5: "meta_labeling",
    6: "portfolio",
    7: "backtest",
    8: "live",
}


def main():
    parser = argparse.ArgumentParser(description="Crypto ML Trading Pipeline")
    parser.add_argument(
        "--stage", type=int, choices=range(1, 9),
        help="Run specific stage only (1-8). Omit to run all stages sequentially."
    )
    parser.add_argument(
        "--from-stage", type=int, choices=range(1, 9), dest="from_stage",
        help="Run stages from this stage onwards (e.g. --from-stage 4 runs stages 4-7)."
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Restrict to a single symbol (e.g. BTCUSDT). Useful for debugging."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-run even if stage is already marked complete."
    )
    parser.add_argument(
        "--config", type=str, default="config/base.yaml",
        help="Path to config YAML file."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.stage:
        stages_to_run = [args.stage]
    elif args.from_stage:
        stages_to_run = list(range(args.from_stage, 8))  # 8 (live) not in batch run
    else:
        stages_to_run = list(range(1, 8))

    for stage_num in stages_to_run:
        stage_name = STAGE_NAMES[stage_num]
        logger.info(f"=== Stage {stage_num}: {stage_name} ===")
        try:
            STAGES[stage_num](cfg, force=args.force, symbol_filter=args.symbol)
            logger.info(f"=== Stage {stage_num} ({stage_name}) finished ===")
        except Exception as e:
            logger.error(f"Stage {stage_num} ({stage_name}) failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)

    logger.info("Pipeline run complete.")


if __name__ == "__main__":
    main()
