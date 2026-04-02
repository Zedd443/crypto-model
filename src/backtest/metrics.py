import numpy as np
import pandas as pd
import json
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("metrics")

# For 15m bars: 252 trading days × 96 bars per day
_PERIODS_PER_YEAR_15M = 252 * 96


def compute_sharpe(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = _PERIODS_PER_YEAR_15M) -> float:
    excess = returns - risk_free / periods_per_year
    std = excess.std()
    if std < 1e-12 or len(returns) < 2:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def compute_sortino(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = _PERIODS_PER_YEAR_15M) -> float:
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("inf") if excess.mean() > 0 else 0.0
    downside_std = downside.std()
    if downside_std < 1e-12:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def compute_calmar(returns: pd.Series, max_dd: float) -> float:
    if abs(max_dd) < 1e-12:
        return 0.0
    ann_ret = float(returns.mean() * _PERIODS_PER_YEAR_15M)
    return float(ann_ret / abs(max_dd))


def compute_max_drawdown(nav_series: pd.Series) -> float:
    if len(nav_series) < 2:
        return 0.0
    rolling_max = nav_series.cummax()
    drawdowns = (nav_series - rolling_max) / (rolling_max + 1e-9)
    return float(drawdowns.min())


def compute_portfolio_cvar_from_nav(nav_series: pd.Series, confidence: float = 0.95) -> float:
    returns = nav_series.pct_change().dropna()
    if len(returns) < 10:
        return float("nan")
    cutoff = int((1.0 - confidence) * len(returns))
    sorted_ret = returns.sort_values()
    if cutoff <= 0:
        return float(sorted_ret.iloc[0])
    return float(sorted_ret.iloc[:cutoff].mean())


def compute_hit_rate(trade_log_df: pd.DataFrame) -> float:
    if len(trade_log_df) == 0:
        return float("nan")
    profitable = (trade_log_df["pnl_usd"] > 0).sum()
    return float(profitable / len(trade_log_df))


def compute_profit_factor(trade_log_df: pd.DataFrame) -> float:
    if len(trade_log_df) == 0:
        return float("nan")
    gross_profit = trade_log_df.loc[trade_log_df["pnl_usd"] > 0, "pnl_usd"].sum()
    gross_loss = abs(trade_log_df.loc[trade_log_df["pnl_usd"] < 0, "pnl_usd"].sum())
    if gross_loss < 1e-9:
        return float("inf") if gross_profit > 0 else float("nan")
    return float(gross_profit / gross_loss)


def compute_all_metrics(nav_series: pd.Series, trade_log_df: pd.DataFrame, cfg) -> dict:
    if len(nav_series) < 2:
        logger.warning("NAV series too short to compute metrics")
        return {}

    returns = nav_series.pct_change().dropna()
    max_dd = compute_max_drawdown(nav_series)

    # Annualized return
    n_bars = len(nav_series)
    total_return = float((nav_series.iloc[-1] / nav_series.iloc[0]) - 1.0)
    n_years = n_bars / _PERIODS_PER_YEAR_15M
    ann_return = float((1.0 + total_return) ** (1.0 / max(n_years, 1e-6)) - 1.0) if total_return > -1 else float("-inf")

    metrics = {
        "total_return": total_return,
        "annualized_return": ann_return,
        "sharpe": compute_sharpe(returns),
        "sortino": compute_sortino(returns),
        "max_drawdown": max_dd,
        "calmar": compute_calmar(returns, max_dd),
        "cvar_95": compute_portfolio_cvar_from_nav(nav_series, 0.95),
        "cvar_99": compute_portfolio_cvar_from_nav(nav_series, 0.99),
        "n_bars": n_bars,
        "final_nav": float(nav_series.iloc[-1]),
        "initial_nav": float(nav_series.iloc[0]),
    }

    if len(trade_log_df) > 0:
        metrics.update({
            "n_trades": len(trade_log_df),
            "hit_rate": compute_hit_rate(trade_log_df),
            "profit_factor": compute_profit_factor(trade_log_df),
            "avg_pnl_pct": float(trade_log_df["pnl_pct"].mean()),
            "avg_hold_bars": float(trade_log_df.get("hold_bars", pd.Series([0])).mean()),
            "exit_tp_pct": float((trade_log_df["exit_reason"] == "TP").sum() / len(trade_log_df)),
            "exit_sl_pct": float((trade_log_df["exit_reason"] == "SL").sum() / len(trade_log_df)),
        })
    else:
        metrics.update({"n_trades": 0, "hit_rate": float("nan"), "profit_factor": float("nan")})

    return metrics


def write_backtest_summary(
    metrics: dict,
    survivorship_note: str,
    config_snapshot: dict,
    model_versions: dict,
    output_path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "metrics": metrics,
        "survivorship_note": survivorship_note,
        "model_versions": model_versions,
        "config_snapshot": config_snapshot,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Backtest summary written to {output_path}")
    logger.info(f"  Sharpe: {metrics.get('sharpe', 'N/A'):.3f}  MaxDD: {metrics.get('max_drawdown', 'N/A'):.3f}  Trades: {metrics.get('n_trades', 0)}")
