import numpy as np
from scipy.optimize import minimize
from src.utils.logger import get_logger

logger = get_logger("optimizer")


def optimize_portfolio_weights(
    expected_returns: np.ndarray,
    corr_matrix: np.ndarray,
    cvar_val: float,
    prev_weights: np.ndarray,
    n_assets: int,
    cfg,
) -> np.ndarray:
    if n_assets == 0:
        return np.array([])
    if n_assets == 1:
        return np.array([1.0])

    max_weight = float(cfg.portfolio.max_position_size)
    max_turnover = float(cfg.portfolio.max_turnover)
    cvar_lambda = float(cfg.portfolio.cvar_lambda_penalty)
    cvar_z = float(cfg.portfolio.cvar_z_score)

    # Ensure prev_weights is the right size
    if prev_weights is None or len(prev_weights) != n_assets:
        prev_weights = np.ones(n_assets) / n_assets

    # Covariance from correlation matrix (assume unit vol)
    cov_matrix = corr_matrix.copy() if corr_matrix is not None else np.eye(n_assets)

    def neg_sharpe_minus_cvar(w):
        port_ret = float(w @ expected_returns)
        port_var = float(w @ cov_matrix @ w)
        port_std = np.sqrt(max(port_var, 1e-12))
        sharpe = port_ret / (port_std + 1e-9)
        # CVaR penalty: approximate as z * port_std (parametric, depends on w via port_std)
        cvar_penalty = cvar_lambda * cvar_z * port_std
        return -(sharpe - cvar_penalty)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        # Turnover constraint
        {"type": "ineq", "fun": lambda w: max_turnover - np.sum(np.abs(w - prev_weights))},
    ]

    bounds = [(0.0, max_weight) for _ in range(n_assets)]

    result = minimize(
        neg_sharpe_minus_cvar,
        x0=prev_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500, "disp": False},
    )

    if result.success:
        w = result.x
        # Clip and renormalize for numerical stability
        w = np.clip(w, 0.0, max_weight)
        total = w.sum()
        if total > 1e-9:
            w = w / total
        return w
    else:
        logger.warning(f"Portfolio optimization failed: {result.message} — using equal weights")
        return equal_weight_fallback(n_assets, max_weight)


def equal_weight_fallback(n_assets: int, max_weight: float) -> np.ndarray:
    if n_assets == 0:
        return np.array([])
    equal_w = 1.0 / n_assets
    w = np.full(n_assets, min(equal_w, max_weight))
    # Renormalize after capping
    total = w.sum()
    if total > 0:
        w = w / total
    return w


def rebalance_needed(
    current_weights: np.ndarray,
    signal_strengths: np.ndarray,
    regime_changed: bool,
    cfg,
) -> bool:
    # Trigger rebalance if regime changed OR daily trigger fires
    if regime_changed and cfg.portfolio.rebalance_trigger_regime_change:
        return True

    # Check if signal strengths diverge significantly from current allocation
    if current_weights is None or len(current_weights) == 0:
        return True

    if signal_strengths is not None and len(signal_strengths) == len(current_weights):
        total_strength = float(np.abs(signal_strengths).sum())
        if total_strength > 0:
            target_w = np.abs(signal_strengths) / total_strength
            turnover = float(np.sum(np.abs(current_weights - target_w)))
            if turnover > float(cfg.portfolio.max_turnover):
                return True

    return False
