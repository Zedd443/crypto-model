import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("risk")


def fit_garch_vol_forecasts(returns_df: pd.DataFrame, train_end: str) -> dict:
    # GARCH(1,1) fitted on train, forecast on full period
    from arch import arch_model
    train_end_ts = pd.Timestamp(train_end, tz="UTC") if not isinstance(train_end, pd.Timestamp) else train_end
    vol_forecasts = {}

    for col in returns_df.columns:
        series = returns_df[col].dropna()
        train_series = series[series.index <= train_end_ts]
        if len(train_series) < 50:
            logger.warning(f"GARCH vol: insufficient train data for {col}")
            # Fallback: rolling std
            fallback = series.rolling(20, min_periods=5).std()
            vol_forecasts[col] = fallback
            continue
        try:
            am = arch_model(train_series * 100, vol="Garch", p=1, q=1, dist="normal", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            # Use residual variance from fitted model as vol proxy for full period
            # Forecast 1-step ahead for each bar
            n_out_of_sample = len(series) - len(train_series)
            if n_out_of_sample > 0:
                forecasts = res.forecast(horizon=1, start=len(train_series) - 1, reindex=False)
                forecast_var = forecasts.variance.values[:, 0] / 10000  # unscale
                # Combine fitted conditional vol (train) + forecasted (test)
                fitted_vol = pd.Series(
                    np.sqrt(res.conditional_volatility.values / 100.0),
                    index=train_series.index,
                )
                # Simple rolling std for out-of-sample
                oos_vol = series.iloc[len(train_series):].rolling(20, min_periods=1).std()
                combined = pd.concat([fitted_vol, oos_vol])
                vol_forecasts[col] = combined.reindex(series.index)
            else:
                fitted_vol = pd.Series(
                    np.sqrt(res.conditional_volatility.values / 100.0),
                    index=train_series.index,
                )
                vol_forecasts[col] = fitted_vol.reindex(series.index)
        except Exception as e:
            logger.warning(f"GARCH vol forecast failed for {col}: {e}")
            vol_forecasts[col] = series.rolling(20, min_periods=5).std()

    return vol_forecasts


def compute_portfolio_cvar(
    weights: dict,
    returns_panel: pd.DataFrame,
    vol_forecasts: dict,
    confidence_levels: list,
    n_scenarios: int = 10000,
) -> dict:
    # Historical simulation with GARCH vol scaling
    symbols = list(weights.keys())
    w_arr = np.array([weights.get(s, 0.0) for s in symbols])

    # Align returns
    ret_aligned = returns_panel[symbols].dropna()
    if len(ret_aligned) < 20:
        return {cl: np.nan for cl in confidence_levels}

    hist_vol = ret_aligned.std()

    # Vol-adjusted scenario returns
    scaled_scenarios = ret_aligned.copy()
    for col in symbols:
        if col in vol_forecasts and vol_forecasts[col] is not None:
            # Use last GARCH vol forecast
            garch_vol_last = vol_forecasts[col].dropna().iloc[-1] if len(vol_forecasts[col].dropna()) > 0 else hist_vol.get(col, 1.0)
            hist_v = hist_vol.get(col, 1.0)
            if hist_v > 0:
                scale_factor = garch_vol_last / (hist_v + 1e-12)
                scaled_scenarios[col] = ret_aligned[col] * scale_factor

    # Portfolio PnL for each historical scenario
    port_returns = (scaled_scenarios * w_arr).sum(axis=1).values

    # Bootstrap to get n_scenarios
    rng = np.random.RandomState(42)
    sampled = rng.choice(port_returns, size=n_scenarios, replace=True)

    result = {}
    for cl in confidence_levels:
        cutoff_idx = int((1.0 - cl) * n_scenarios)
        sorted_returns = np.sort(sampled)
        cvar = float(np.mean(sorted_returns[:cutoff_idx])) if cutoff_idx > 0 else float(sorted_returns[0])
        result[cl] = cvar

    return result


def compute_component_var(
    weights: dict,
    returns_panel: pd.DataFrame,
    vol_forecasts: dict,
    confidence: float = 0.95,
) -> dict:
    # Marginal contribution of each position to portfolio VaR
    symbols = list(weights.keys())
    w_arr = np.array([weights.get(s, 0.0) for s in symbols])
    ret_aligned = returns_panel[symbols].dropna()

    if len(ret_aligned) < 20:
        return {s: np.nan for s in symbols}

    # Covariance matrix
    cov = ret_aligned.cov().values

    # Portfolio variance = w^T Σ w
    port_var = float(w_arr @ cov @ w_arr)
    port_std = np.sqrt(max(port_var, 1e-12))

    # z-score for confidence level
    from scipy.stats import norm
    z = norm.ppf(confidence)

    # Marginal VaR per asset = z * (Σw)_i / port_std
    marginal_var = z * (cov @ w_arr) / (port_std + 1e-12)
    # Component VaR = w_i * marginal_var_i
    component_var = {symbols[i]: float(w_arr[i] * marginal_var[i]) for i in range(len(symbols))}
    return component_var


def run_stress_scenarios(
    portfolio_state: dict,
    returns_panel: pd.DataFrame,
) -> dict:
    # Stress scenarios: all_corr_1 and btc_dump_20pct
    weights = portfolio_state.get("weights", {})
    symbols = list(weights.keys())
    w_arr = np.array([weights.get(s, 0.0) for s in symbols])

    results = {}

    # Scenario 1: all correlations = 1 (worst case linear loss)
    if len(symbols) > 0 and returns_panel is not None:
        ret_aligned = returns_panel[symbols].dropna()
        if len(ret_aligned) > 0:
            # Portfolio loss when all assets move together (max loss direction)
            mean_ret = ret_aligned.mean().values
            # Weighted average of expected per-bar returns
            all_corr_1_loss = float(w_arr @ mean_ret)
            # Worst 1% historical bar
            port_ret = (ret_aligned * w_arr).sum(axis=1)
            worst_1pct = float(port_ret.quantile(0.01))
            results["all_corr_1"] = worst_1pct

    # Scenario 2: BTC dumps 20%
    if "BTCUSDT" in returns_panel.columns and len(symbols) > 0:
        ret_aligned = returns_panel[symbols].dropna()
        btc_ret = returns_panel["BTCUSDT"].dropna()
        if len(ret_aligned) > 0 and len(btc_ret) > 0:
            # Find bars where BTC had its largest drawdown
            btc_worst_mask = btc_ret <= btc_ret.quantile(0.01)
            worst_dates = btc_ret[btc_worst_mask].index
            shared = ret_aligned.index.intersection(worst_dates)
            if len(shared) > 0:
                btc_stress_ret = ret_aligned.loc[shared]
                port_loss = float((btc_stress_ret * w_arr).sum(axis=1).mean())
                results["btc_dump_20pct"] = port_loss
            else:
                results["btc_dump_20pct"] = np.nan
    else:
        results["btc_dump_20pct"] = np.nan

    return results
