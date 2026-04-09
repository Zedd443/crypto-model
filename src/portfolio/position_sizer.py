import numpy as np
from src.utils.logger import get_logger

logger = get_logger("position_sizer")

# Memecoin max size fraction relative to normal
_MEMECOIN_SIZE_FRACTION = 0.5
_MEMECOIN_LEVERAGE = 1.0


def compute_half_kelly(win_rate: float, avg_win_pct: float, avg_loss_pct: float) -> float:
    # Kelly fraction: f* = (p*b - (1-p)) / b where b = avg_win / avg_loss
    if avg_loss_pct <= 0:
        return 0.0
    b = avg_win_pct / (avg_loss_pct + 1e-9)
    p = win_rate
    kelly_f = (p * b - (1.0 - p)) / (b + 1e-9)
    half_kelly = kelly_f * 0.5
    return float(np.clip(half_kelly, 0.0, 0.25))


def get_growth_gate_limits(equity: float, cfg) -> tuple:
    # Returns (max_active_symbols, max_leverage_a)
    tiers = cfg.growth_gate.tiers
    max_symbols = 1
    max_leverage = 1

    for tier in tiers:
        max_eq = float(tier.get("max_equity", 0))
        if equity <= max_eq:
            max_symbols = int(tier.get("max_symbols", 1))
            max_leverage = int(tier.get("leverage_a_max", 1))
            break
    else:
        # Above all tiers — use last tier values
        last_tier = tiers[-1]
        max_symbols = int(last_tier.get("max_symbols", 1))
        max_leverage = int(last_tier.get("leverage_a_max", 1))

    # Hard override: max_open_positions (0 = use tier value)
    pos_override = int(getattr(cfg.growth_gate, "max_open_positions", 0))
    if pos_override > 0:
        max_symbols = min(max_symbols, pos_override)

    # Hard override: cfg.trading.leverage takes priority, then growth_gate.fixed_leverage (0 = use tier)
    lev_override = int(getattr(getattr(cfg, "trading", cfg), "leverage", 0)) or \
                   int(getattr(cfg.growth_gate, "fixed_leverage", 0))
    if lev_override > 0:
        max_leverage = lev_override

    return max_symbols, max_leverage


def compute_position_size(
    meta_prob: float,
    half_kelly: float,
    equity: float,
    leverage: float,
    cfg,
) -> dict:
    max_position_pct = float(cfg.portfolio.max_position_size)

    # Notional based on confidence-scaled Kelly
    notional = meta_prob * half_kelly * equity * leverage
    margin = notional / leverage

    # Cap margin at max_position_size * equity
    max_margin = max_position_pct * equity
    if margin > max_margin:
        margin = max_margin
        notional = margin * leverage

    return {
        "notional": float(notional),
        "margin": float(margin),
        "leverage_used": float(leverage),
    }


def apply_conformal_scaling(position_size: float, conformal_width: float, cfg) -> float:
    # Same scaling as signal_generator but applied to dollar size
    w_full = float(cfg.model.conformal_width_full)
    w_partial = float(cfg.model.conformal_width_60pct)

    if conformal_width < w_full:
        scale = 1.0
    elif conformal_width < w_partial:
        scale = 0.6
    else:
        scale = 0.3

    return float(position_size * scale)


def check_portfolio_capacity(
    current_positions: dict,
    new_position: dict,
    total_equity: float,
    cfg,
) -> tuple:
    # Returns (scale_factor, should_skip)
    max_margin_pct = float(cfg.portfolio.max_total_margin_pct)
    max_position_pct = float(cfg.portfolio.max_position_size)

    current_margin_total = sum(pos.get("margin", 0) for pos in current_positions.values())
    new_margin = new_position.get("margin", 0)
    hard_limit = max_margin_pct * total_equity
    soft_limit = 0.80 * hard_limit

    # Hard skip
    if current_margin_total + new_margin > hard_limit:
        logger.debug("Portfolio capacity: hard limit reached, skipping new position")
        return 0.0, True

    # Soft limit: scale down
    if current_margin_total + new_margin > soft_limit:
        available = soft_limit - current_margin_total
        if available <= 0:
            return 0.0, True
        scale = min(1.0, available / (new_margin + 1e-9))
        logger.debug(f"Portfolio capacity: scaling new position to {scale:.2f}")
        return float(scale), False

    return 1.0, False


def apply_memecoin_rules(symbol: str, position_size: float, cfg) -> float:
    memecoin_symbols = list(cfg.trading.memecoin_symbols)
    if symbol in memecoin_symbols:
        # Max 50% of normal size, isolated margin, leverage=1x enforced by caller
        position_size = position_size * _MEMECOIN_SIZE_FRACTION
        logger.debug(f"{symbol}: memecoin rules applied — size halved, leverage=1x")
    return float(position_size)
