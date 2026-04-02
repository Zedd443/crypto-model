from pathlib import Path
import numpy as np
import pandas as pd
import json
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


def _build_combined_signals_df(symbol_names: list, signals_dir: Path) -> pd.DataFrame:
    # For single-symbol mode: return the symbol's signals directly
    # For multi-symbol: combine into a per-timestamp lookup
    all_sig_dfs = {}
    for sym in symbol_names:
        sig_df = _load_signals(sym, signals_dir)
        if sig_df is not None:
            all_sig_dfs[sym] = sig_df

    if not all_sig_dfs:
        return pd.DataFrame()

    # For single symbol pipelines, return directly
    if len(all_sig_dfs) == 1:
        sym = list(all_sig_dfs.keys())[0]
        return all_sig_dfs[sym]

    # Multi-symbol: build wide DataFrame where each column-group is a symbol
    # BacktestEngine expects signals_df with rows=timestamps, accessible by timestamp
    # We store per-symbol signals separately and pass each to the engine
    # Return the first symbol's signals for now (engine handles multi via prices_dict)
    return pd.concat(list(all_sig_dfs.values()), keys=list(all_sig_dfs.keys()), axis=1)


def run(cfg, force: bool = False, symbol_filter: str = None) -> None:
    if not force and is_stage_complete("backtest"):
        logger.info("Stage 7 already complete, skipping.")
        return

    all_symbols = get_symbols(cfg)
    if symbol_filter:
        all_symbols = [s for s in all_symbols if s.get("name", s.get("symbol")) == symbol_filter]
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

    for sym in symbol_names:
        sig_df = _load_signals(sym, signals_dir)
        if sig_df is None:
            logger.warning(f"No signals for {sym} — skipping from backtest")
            issues.append(f"{sym}: no signals checkpoint")
            continue

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
        nav.to_parquet(results_dir / f"{sym}_{_TF}_nav.parquet")
        if n_trades > 0:
            trades.to_csv(results_dir / f"{sym}_{_TF}_trades.csv", index=False)

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
    combined_nav.to_parquet(results_dir / "portfolio_nav.parquet")
    if len(combined_trades) > 0:
        combined_trades.to_csv(results_dir / "trade_log.csv", index=False)

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
    initial_equity = float(cfg.account.get("current_equity", 120.0))
    portfolio_nav = portfolio_nav * initial_equity

    return portfolio_nav.rename("portfolio_nav")
