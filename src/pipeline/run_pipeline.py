# Entry point for the crypto ML pipeline.
# Run as module: python -m src.pipeline.run_pipeline [options]
#
# Examples:
#   python -m src.pipeline.run_pipeline                        # run all stages 1-7
#   python -m src.pipeline.run_pipeline --stage 2              # run stage 2 only (features)
#   python -m src.pipeline.run_pipeline --stage 4 --force      # re-run training even if done
#   python -m src.pipeline.run_pipeline --stage 4b             # train 4h/1d HTF models
#   python -m src.pipeline.run_pipeline --stage 4 --symbol BTCUSDT SOLUSDT  # one or more symbols
#   python -m src.pipeline.run_pipeline --stage 8              # run live execution loop
#
# Stage map:
#   1  ingest       — download OHLCV + on-chain + macro data from APIs
#   2  features     — compute technical/regime/cross-sectional features per symbol
#   3  labels       — triple-barrier labeling + sample weights
#   4  training     — XGBoost primary model with Optuna tuning + feature selection
#   4b htf_train    — train lightweight 4h/1d XGBoost models (predictions injected into features)
#   5  meta_labeling— meta-labeler trained on OOF predictions (signal confidence)
#   6  portfolio    — generate signals, position sizing, correlation filter
#   7  backtest     — walk-forward backtest with realistic costs
#   8  live         — real-time execution loop (Binance Demo/Mainnet FAPI)
#
# Full retrain with HTF models:
#   1. --stage 1                       (ingest)
#   2. --stage 4b                      (train 4h/1d models)
#   3. --stage 2 --force               (inject htf_pred_4h/1d, recompute all features)
#   4. --stage 3 --force               (re-label)
#   5. --stage 4 --force               (retrain 15m primary models)
#   6. --stage 5 --force  through 7    (meta, portfolio, backtest)

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
    stage_04b_htf_train,
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
    "4b": stage_04b_htf_train.run,
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
    "4b": "htf_train",
    5: "meta_labeling",
    6: "portfolio",
    7: "backtest",
    8: "live",
}


def main():
    parser = argparse.ArgumentParser(description="Crypto ML Trading Pipeline")
    parser.add_argument(
        "--stage", type=str,
        help="Run specific stage only (1-8, or '4b' for HTF model training). Omit to run all stages sequentially."
    )
    parser.add_argument(
        "--from-stage", type=int, choices=range(1, 9), dest="from_stage",
        help="Run stages from this stage onwards (e.g. --from-stage 4 runs stages 4-7)."
    )
    parser.add_argument(
        "--symbol", type=str, nargs="+", default=None,
        help="Restrict to one or more symbols (e.g. --symbol SOLUSDT AVAXUSDT). Useful for debugging."
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
        # Allow "4b" as a string stage key; convert numeric strings to int
        stage_key = args.stage if args.stage == "4b" else int(args.stage)
        stages_to_run = [stage_key]
    elif args.from_stage:
        stages_to_run = list(range(args.from_stage, 8))  # 8 (live) not in batch run
    else:
        stages_to_run = list(range(1, 8))

    # symbol_filter: None = all symbols, list with one item = single symbol (backwards compat), list = multi-symbol
    symbol_filter = args.symbol  # list[str] | None

    for stage_num in stages_to_run:
        stage_name = STAGE_NAMES[stage_num]
        logger.info(f"=== Stage {stage_num}: {stage_name} ===")
        try:
            STAGES[stage_num](cfg, force=args.force, symbol_filter=symbol_filter)
            logger.info(f"=== Stage {stage_num} ({stage_name}) finished ===")
        except Exception as e:
            logger.error(f"Stage {stage_num} ({stage_name}) failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)

    logger.info("Pipeline run complete.")


if __name__ == "__main__":
    main()
