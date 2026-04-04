import numpy as np
from src.utils.logger import get_logger

logger = get_logger("costs")


def compute_entry_price(mid_price: float, direction: int, slippage_pct: float) -> float:
    # Long entry: buy at mid + slippage; short entry: sell at mid - slippage
    if direction == 1:
        return mid_price * (1.0 + slippage_pct)
    else:
        return mid_price * (1.0 - slippage_pct)


def compute_exit_price(mid_price: float, direction: int, slippage_pct: float) -> float:
    # Long exit (sell): mid - slippage; short exit (buy): mid + slippage
    if direction == 1:
        return mid_price * (1.0 - slippage_pct)
    else:
        return mid_price * (1.0 + slippage_pct)


def compute_commission(notional_usd: float, commission_pct: float) -> float:
    return abs(notional_usd) * commission_pct


def compute_sqrt_market_impact(order_size_usd: float, adv_20d_usd: float, coef: float = 0.1) -> float:
    # Square-root market impact model: slippage_pct = coef * sqrt(order_size / ADV)
    if adv_20d_usd <= 0:
        return 0.0
    slippage_pct = coef * np.sqrt(order_size_usd / (adv_20d_usd + 1e-9))
    return float(slippage_pct)


def compute_funding_cost(funding_rate_8h: float, hold_hours: float) -> float:
    # Funding cost = |funding_rate| * hold_hours / 8
    return abs(funding_rate_8h) * hold_hours / 8.0


def compute_total_trade_cost(
    entry_price: float,
    exit_price: float,
    size_usd: float,
    direction: int,
    adv_usd: float,
    funding_rate: float,
    hold_hours: float,
    cfg,
) -> dict:
    slippage_pct = float(cfg.backtest.slippage_pct)
    commission_pct = float(cfg.backtest.commission_pct)
    impact_coef = float(cfg.backtest.sqrt_impact_coef)

    # Entry costs
    entry_slippage_pct = slippage_pct + compute_sqrt_market_impact(size_usd, adv_usd, impact_coef)
    entry_fill = compute_entry_price(entry_price, direction, entry_slippage_pct)
    slippage_entry = abs(entry_fill - entry_price) / (entry_price + 1e-9) * size_usd

    commission_entry = compute_commission(size_usd, commission_pct)

    # Exit costs
    exit_slippage_pct = slippage_pct + compute_sqrt_market_impact(size_usd, adv_usd, impact_coef)
    exit_fill = compute_exit_price(exit_price, direction, exit_slippage_pct)
    slippage_exit = abs(exit_fill - exit_price) / (exit_price + 1e-9) * size_usd

    commission_exit = compute_commission(size_usd, commission_pct)

    # Funding
    funding = compute_funding_cost(funding_rate, hold_hours) * size_usd

    # Market impact is already included inside entry_slippage_pct and exit_slippage_pct
    # via compute_sqrt_market_impact — do NOT add a separate market_impact term here.
    total_cost_usd = slippage_entry + slippage_exit + commission_entry + commission_exit + funding

    return {
        "slippage_entry": float(slippage_entry),
        "slippage_exit": float(slippage_exit),
        "commission_entry": float(commission_entry),
        "commission_exit": float(commission_exit),
        "funding": float(funding),
        "total_cost_usd": float(total_cost_usd),
    }
