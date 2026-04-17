"""
Stage 4b — Train 4h and 1d HTF models per symbol.

These lightweight XGBoost classifiers are trained directly on raw 4h/1d OHLCV bars.
Their calibrated predictions are injected into the 15m feature frame at stage_02
as `htf_pred_4h` and `htf_pred_1d` columns, giving the primary 15m model access to
higher-timeframe trend context that HTF technical features alone cannot fully capture.

Run order: after stage_01 (ingest) but before or alongside stage_02 (features).
Stage_02 --force is needed after this stage to inject HTF predictions into features.

Usage:
  python -m src.pipeline.run_pipeline --stage 4b
  python -m src.pipeline.run_pipeline --stage 4b --force --symbol BTCUSDT SOLUSDT
"""
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from src.utils.config_loader import get_symbols
from src.utils.state_manager import update_project_state
from src.utils.io_utils import read_checkpoint, checkpoint_exists
from src.utils.logger import get_logger
from src.models.htf_model import train_htf_model, save_htf_model

logger = get_logger("stage_04b_htf_train")

_HTF_TIMEFRAMES = ["4h", "1d"]


def _htf_models_exist(symbol: str, models_dir: Path) -> bool:
    for tf in _HTF_TIMEFRAMES:
        if not (models_dir / f"{symbol}_{tf}_htf_model.json").exists():
            return False
    return True


def _train_symbol_htf(symbol: str, cfg, checkpoints_dir: Path, models_dir: Path) -> tuple:
    train_end = pd.Timestamp(cfg.data.train_end, tz="UTC")
    val_end = pd.Timestamp(cfg.data.val_end, tz="UTC")

    results = {}
    for tf in _HTF_TIMEFRAMES:
        if not checkpoint_exists("ingest", symbol, tf, checkpoints_dir):
            logger.warning(f"{symbol} {tf}: no ingest checkpoint, skipping")
            continue
        try:
            df_htf = read_checkpoint("ingest", symbol, tf, checkpoints_dir)
            model, calibrator, feature_names = train_htf_model(
                symbol, tf, df_htf, train_end, val_end, cfg
            )
            save_htf_model(model, calibrator, feature_names, symbol, tf, models_dir)
            results[tf] = "ok"
        except Exception as e:
            logger.error(f"{symbol} {tf}: HTF training failed — {e}")
            results[tf] = str(e)

    return symbol, results


def run(cfg, force: bool = False, symbol_filter=None) -> None:
    logger.info("=== Stage 4b: HTF Model Training (4h + 1d) ===")

    all_symbols = get_symbols(cfg)
    if symbol_filter:
        _sf = set(symbol_filter) if isinstance(symbol_filter, list) else {symbol_filter}
        all_symbols = [s for s in all_symbols if s.get("name", s.get("symbol")) in _sf]
    symbol_names = [s.get("name", s.get("symbol")) for s in all_symbols]

    checkpoints_dir = Path(cfg.data.checkpoints_dir)
    models_dir = Path(cfg.data.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    issues = []
    completed = []

    for symbol in tqdm(symbol_names, desc="stage_04b", unit="sym"):
        if not force and _htf_models_exist(symbol, models_dir):
            logger.debug(f"{symbol}: HTF models already exist, skipping (use --force to retrain)")
            completed.append(symbol)
            continue

        sym, results = _train_symbol_htf(symbol, cfg, checkpoints_dir, models_dir)
        errors = [f"{tf}:{v}" for tf, v in results.items() if v != "ok"]
        if errors:
            issues.append(f"{sym}: {'; '.join(errors)}")
        else:
            completed.append(sym)
            logger.info(f"{sym}: HTF models trained (4h + 1d)")

    logger.info(
        f"Stage 4b complete. {len(completed)}/{len(symbol_names)} symbols OK. "
        f"Reminder: run --stage 2 --force to inject HTF predictions into 15m feature frames."
    )
    # We don't gate stage_04b in project_state — it's a sub-stage.
    # Issues are logged but don't block pipeline progression.
