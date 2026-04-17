from pathlib import Path
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from src.utils.config_loader import get_symbols
from src.utils.state_manager import is_stage_complete, update_project_state
from src.utils.logger import get_logger
from src.utils.io_utils import read_checkpoint, checkpoint_exists
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import compute_all_metrics, write_backtest_summary
from src.backtest.survivorship import load_delisted_coins, compute_survivorship_note
from src.models.model_versioning import get_active_models
from omegaconf import OmegaConf

logger = get_logger("stage_07_backtest")

_TF = "15m"


def _load_signals(symbol: str, signals_dir: Path) -> pd.DataFrame | None:
    signals_path = signals_dir / f"{symbol}_{_TF}_signals.parquet"
    if not signals_path.exists():
        return None
    return pd.read_parquet(signals_path)



def run(cfg, force: bool = False, symbol_filter: str = None) -> None:
    if not force and is_stage_complete("backtest"):
        logger.info("Stage 7 already complete, skipping.")
        return

    all_symbols = get_symbols(cfg)
    if symbol_filter:
        _sf = set(symbol_filter) if isinstance(symbol_filter, list) else {symbol_filter}
        all_symbols = [s for s in all_symbols if s.get("name", s.get("symbol")) in _sf]
    symbol_names = [s.get("name", s.get("symbol")) for s in all_symbols]

    checkpoints_dir = Path(cfg.data.checkpoints_dir)
    signals_dir = checkpoints_dir / "signals"
    results_dir = Path(cfg.data.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load delisted coins for survivorship bias handling
    delisted_path = Path("delisted_coins.json")
    delisted_dict = load_delisted_coins(delisted_path)
    logger.info(f"Loaded {len(delisted_dict)} delisted coins")

    # Load price data for all symbols
    prices_dict = {}
    for sym in symbol_names:
        if checkpoint_exists("ingest", sym, _TF, checkpoints_dir):
            try:
                df = read_checkpoint("ingest", sym, _TF, checkpoints_dir)
                prices_dict[sym] = df
            except Exception as e:
                logger.warning(f"Cannot load prices for {sym}: {e}")

    if not prices_dict:
        logger.error("No price data available for backtest")
        update_project_state("backtest", "failed", ["No price data"])
        return

    issues = []

    # Run backtest per symbol (single-symbol engine, aggregate results)
    all_nav_series = []
    all_trade_logs = []
    per_symbol_metrics = []

    for sym in tqdm(symbol_names, desc="stage_07", unit="sym"):
        sig_df = _load_signals(sym, signals_dir)
        if sig_df is None:
            logger.warning(f"No signals for {sym} — skipping from backtest")
            issues.append(f"{sym}: no signals checkpoint")
            continue

        test_start_ts = pd.Timestamp(cfg.data.test_start, tz="UTC")
        sig_df = sig_df[sig_df.index >= test_start_ts]
        if len(sig_df) == 0:
            logger.warning(f"{sym}: no signals after test_start filter ({cfg.data.test_start})")
            continue
        logger.info(f"{sym}: {len(sig_df)} signal bars after test_start filter")

        if sym in prices_dict:
            prices_dict[sym] = prices_dict[sym][prices_dict[sym].index >= test_start_ts]

        sym_prices = {sym: prices_dict[sym]} if sym in prices_dict else {}
        if not sym_prices:
            logger.warning(f"No price data for {sym} — skipping")
            continue

        logger.info(f"Running backtest for {sym}: {len(sig_df)} signal bars, {len(sym_prices[sym])} price bars")

        engine = BacktestEngine(
            signals_df=sig_df,
            prices_dict=sym_prices,
            costs_cfg=cfg.backtest,
            delisted_dict=delisted_dict,
            cfg=cfg,
        )

        try:
            result = engine.run()
        except Exception as e:
            logger.error(f"{sym}: backtest engine failed — {e}")
            issues.append(f"{sym}: engine error: {str(e)[:200]}")
            continue

        nav = result["nav_series"]
        trades = result["trade_log"]
        n_trades = len(trades) if len(trades) > 0 else 0
        logger.info(f"{sym}: backtest complete — {n_trades} trades, final equity={result['final_equity']:.2f}")

        # Save per-symbol results
        nav.to_frame("nav").to_parquet(results_dir / f"{sym}_{_TF}_nav.parquet")
        if n_trades > 0:
            trades.to_csv(results_dir / f"{sym}_{_TF}_trades.csv", index=False)

        # Compute per-symbol metrics (reuse same function as portfolio — no extra compute cost)
        sym_metrics = compute_all_metrics(nav, trades, cfg)
        sym_metrics["symbol"] = sym
        per_symbol_metrics.append(sym_metrics)

        all_nav_series.append(nav)
        if n_trades > 0:
            all_trade_logs.append(trades)

    if not all_nav_series:
        logger.error("No backtest results generated")
        update_project_state("backtest", "failed", issues + ["No results"])
        return

    # Aggregate NAV series (sum across symbols for portfolio NAV)
    combined_nav = _aggregate_nav(all_nav_series, cfg)
    combined_trades = pd.concat(all_trade_logs, ignore_index=True) if all_trade_logs else pd.DataFrame()

    # Compute metrics
    metrics = compute_all_metrics(combined_nav, combined_trades, cfg)
    logger.info(
        f"Portfolio metrics — Sharpe={metrics.get('sharpe', 0):.3f} "
        f"MaxDD={metrics.get('max_drawdown', 0):.3f} "
        f"Trades={metrics.get('n_trades', 0)}"
    )

    # Save combined results
    combined_nav.to_frame("nav").to_parquet(results_dir / "portfolio_nav.parquet")
    if len(combined_trades) > 0:
        combined_trades.to_csv(results_dir / "trade_log.csv", index=False)

    # Save per-symbol metrics CSV
    if per_symbol_metrics:
        sym_df = pd.DataFrame(per_symbol_metrics)
        # Put symbol column first
        cols = ["symbol"] + [c for c in sym_df.columns if c != "symbol"]
        sym_df[cols].to_csv(results_dir / "per_symbol_metrics.csv", index=False)
        logger.info(f"Per-symbol metrics saved: {results_dir / 'per_symbol_metrics.csv'}")

    # Generate per-symbol summary chart from training_summary.csv
    try:
        from src.visualization.training_diagnostics import plot_per_symbol_summary
        training_summary_csv = Path(cfg.data.models_dir) / "training_summary.csv"
        plot_per_symbol_summary(training_summary_csv, results_dir / "diagnostics")
    except Exception as e:
        logger.debug(f"Per-symbol summary chart skipped: {e}")

    # Survivorship note
    survivorship_note = compute_survivorship_note(delisted_dict, symbol_names)

    # Config snapshot for reproducibility
    config_snapshot = OmegaConf.to_container(cfg, resolve=True)

    # Active model versions
    model_versions = get_active_models()

    write_backtest_summary(
        metrics=metrics,
        survivorship_note=survivorship_note,
        config_snapshot={
            "train_end": config_snapshot["data"]["train_end"],
            "val_start": config_snapshot["data"]["val_start"],
            "test_start": config_snapshot["data"]["test_start"],
            "slippage_pct": config_snapshot["backtest"]["slippage_pct"],
            "commission_pct": config_snapshot["backtest"]["commission_pct"],
        },
        model_versions={k: v.get("version", "unknown") for k, v in model_versions.items()},
        output_path=results_dir / "backtest_summary.json",
    )

    update_project_state("backtest", "done", issues, output_dir=str(results_dir))
    logger.info(f"Stage 7 complete. Results saved to {results_dir}")


def _aggregate_nav(nav_series_list: list, cfg) -> pd.Series:
    if len(nav_series_list) == 1:
        return nav_series_list[0]

    # Build portfolio NAV as weighted sum of individual NAVs
    # Normalize each NAV to start at 1, then average
    normalized = []
    for nav in nav_series_list:
        if len(nav) > 0 and nav.iloc[0] != 0:
            normalized.append(nav / nav.iloc[0])

    if not normalized:
        return nav_series_list[0]

    # Align on common index
    combined = pd.concat(normalized, axis=1).ffill()
    portfolio_nav = combined.mean(axis=1)

    # Scale to initial equity
    initial_equity = float(getattr(getattr(cfg, "account", None), "current_equity", None) or 120.0)
    portfolio_nav = portfolio_nav * initial_equity

    return portfolio_nav.rename("portfolio_nav")
