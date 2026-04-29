# Entry point for the crypto ML pipeline.
# Run as module: python -m src.pipeline.run_pipeline [options]
#
# Examples:
#   python -m src.pipeline.run_pipeline                        # run all stages (1→2→3→4→4b→5→6→7)
#   python -m src.pipeline.run_pipeline --stage 2              # run stage 2 only (features)
#   python -m src.pipeline.run_pipeline --stage 4 --force      # re-run training even if done
#   python -m src.pipeline.run_pipeline --stage 4b             # train 1h/4h/1d HTF models only
#   python -m src.pipeline.run_pipeline --stage 4 --symbol BTCUSDT SOLUSDT  # one or more symbols
#   python -m src.pipeline.run_pipeline --stage 8              # run live execution loop
#
# Stage map:
#   1  ingest       — download OHLCV + on-chain + macro data from APIs
#   2  features     — compute technical/regime/cross-sectional features (no HTF model dependency)
#   3  labels       — triple-barrier labeling + sample weights
#   4  training     — XGBoost primary model with Optuna tuning + feature selection
#   4b htf_train    — train independent 1h/4h/1d XGBoost approval-filter models (parallel to stage 4)
#   5  meta_labeling— meta-labeler trained on OOF predictions (signal confidence, 15m only)
#   6  portfolio    — signals + HTF approval filter + position sizing + correlation filter
#   7  backtest     — walk-forward backtest with realistic costs
#   8  live         — real-time execution loop (Binance Demo/Mainnet FAPI)

import argparse
import sys
import warnings
from pathlib import Path

# Silence noisy sklearn parallel warning emitted when third-party libs (xgboost,
# optuna, imblearn) invoke sklearn.utils.parallel.delayed without the paired
# Parallel — not fixable from our code and not actionable.
warnings.filterwarnings(
    "ignore",
    message=r".*sklearn\.utils\.parallel\.delayed.*should be used with.*",
    category=UserWarning,
)

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.cli_progress import PipelineProgress, console
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

    # Full sequential order: 1 → 2 → 3 → 4 → 4b → 5 → 6 → 7
    # 4b runs after stage 4 — both are training stages with no cross-dependency.
    # HTF approval logic lives in stage 6 (loads models at signal time), not in features.
    _FULL_SEQUENCE = [1, 2, 3, 4, "4b", 5, 6, 7]

    if args.stage:
        # Allow "4b" as a string stage key; convert numeric strings to int
        stage_key = args.stage if args.stage == "4b" else int(args.stage)
        stages_to_run = [stage_key]
    elif args.from_stage:
        # Slice _FULL_SEQUENCE from the requested stage onwards
        from_key = int(args.from_stage)
        try:
            start_pos = _FULL_SEQUENCE.index(from_key)
        except ValueError:
            start_pos = 0
        stages_to_run = _FULL_SEQUENCE[start_pos:]
    else:
        stages_to_run = _FULL_SEQUENCE

    # symbol_filter: None = all symbols, list with one item = single symbol (backwards compat), list = multi-symbol
    symbol_filter = args.symbol  # list[str] | None

    console.print()
    for stage_num in stages_to_run:
        with PipelineProgress(stage_num):
            try:
                STAGES[stage_num](cfg, force=args.force, symbol_filter=symbol_filter)
            except Exception as e:
                import traceback
                logger.error(traceback.format_exc())
                sys.exit(1)

    console.print("[bold green]All stages complete.[/bold green]")


if __name__ == "__main__":
    main()
