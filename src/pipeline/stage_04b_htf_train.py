# Stage 4b — Train 4h and 1d HTF approval-filter models per symbol.
#
# Each model is a lightweight XGBoost classifier trained directly on raw HTF OHLCV bars.
# They act as independent directional confirmation filters in stage_06 (signal generation):
# a 15m signal is zeroed out when HTF models disagree with it beyond the approval threshold.
# These models are NOT features fed into the 15m primary model.
#
# Label: ATR-based triple-barrier on NATIVE HTF bars (not resampled from 15m).
# Macro features: merged from macro_panel_{tf}.parquet at native HTF resolution.
#
# Run order: after stage_04 (both are training stages, no cross-dependency).
# Stage_06 loads HTF models at signal time — no stage_02 re-run required.
#
# Usage:
#   python -m src.pipeline.run_pipeline --stage 4b
#   python -m src.pipeline.run_pipeline --stage 4b --force --symbol BTCUSDT SOLUSDT
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


def _load_macro_panels(cfg) -> dict:
    """Load macro panels for each HTF — keyed by timeframe string."""
    processed_dir = Path(cfg.data.processed_dir)
    panels = {}
    for tf in _HTF_TIMEFRAMES:
        p = processed_dir / f"macro_panel_{tf}.parquet"
        if p.exists():
            try:
                panels[tf] = pd.read_parquet(p)
                logger.debug(f"Loaded macro_panel_{tf}: {panels[tf].shape}")
            except Exception as exc:
                logger.warning(f"Could not load macro_panel_{tf}: {exc}")
        else:
            logger.warning(f"macro_panel_{tf}.parquet not found at {p} — macro features skipped for {tf}")
    return panels


def _train_symbol_htf(
    symbol: str,
    cfg,
    checkpoints_dir: Path,
    models_dir: Path,
    macro_panels: dict,
) -> tuple:
    train_end = pd.Timestamp(cfg.data.train_end, tz="UTC")
    val_end = pd.Timestamp(cfg.data.val_end, tz="UTC")

    results = {}
    for tf in _HTF_TIMEFRAMES:
        if not checkpoint_exists("ingest", symbol, tf, checkpoints_dir):
            logger.warning(f"{symbol} {tf}: no ingest checkpoint, skipping")
            continue
        try:
            df_htf = read_checkpoint("ingest", symbol, tf, checkpoints_dir)
            macro_panel = macro_panels.get(tf, None)
            model, calibrator, feature_names = train_htf_model(
                symbol, tf, df_htf, train_end, val_end, cfg,
                macro_panel=macro_panel,
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

    macro_panels = _load_macro_panels(cfg)
    if not macro_panels:
        logger.warning("No macro panels loaded — HTF models will train without macro features")

    issues = []
    completed = []

    for symbol in tqdm(symbol_names, desc="stage_04b", unit="sym"):
        if not force and _htf_models_exist(symbol, models_dir):
            logger.debug(f"{symbol}: HTF models already exist, skipping (use --force to retrain)")
            completed.append(symbol)
            continue

        sym, results = _train_symbol_htf(symbol, cfg, checkpoints_dir, models_dir, macro_panels)
        errors = [f"{tf}:{v}" for tf, v in results.items() if v != "ok"]
        if errors:
            issues.append(f"{sym}: {'; '.join(errors)}")
        else:
            completed.append(sym)
            logger.info(f"{sym}: HTF models trained (4h + 1d)")

    logger.info(f"Stage 4b complete. {len(completed)}/{len(symbol_names)} symbols OK.")
    if issues:
        logger.warning(f"Issues: {issues}")
