import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("correlation")


def fit_garch_per_asset(returns_df: pd.DataFrame, train_end: str) -> dict:
    # Fit GARCH(1,1) per column on train data only
    from arch import arch_model
    train_end_ts = pd.Timestamp(train_end, tz="UTC") if not isinstance(train_end, pd.Timestamp) else train_end
    results = {}

    for col in returns_df.columns:
        series = returns_df[col].dropna()
        train_series = series[series.index <= train_end_ts]
        if len(train_series) < 50:
            logger.warning(f"GARCH: insufficient train data for {col} ({len(train_series)} bars)")
            continue
        try:
            am = arch_model(train_series * 100, vol="Garch", p=1, q=1, dist="normal", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            results[col] = res
        except Exception as e:
            logger.warning(f"GARCH fit failed for {col}: {e}")

    return results


def compute_dcc_correlations(returns_df: pd.DataFrame, garch_results: dict, cfg) -> pd.DataFrame:
    try:
        from arch.multivariate import DCC
        cols = list(garch_results.keys())
        if len(cols) < 2:
            raise ImportError("Not enough assets for DCC")
        sub_returns = returns_df[cols].dropna()
        dcc = DCC(sub_returns * 100)
        dcc_result = dcc.fit(disp="off")
        corr_matrix = dcc_result.conditional_correlation
        if hasattr(corr_matrix, "values"):
            return corr_matrix
        # Build DataFrame from numpy array
        n = len(cols)
        corr_rows = {}
        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                corr_rows[f"corr_{c1}_{c2}"] = corr_matrix[:, i, j] if corr_matrix.ndim == 3 else corr_matrix[i, j]
        return pd.DataFrame(corr_rows, index=sub_returns.index)
    except (ImportError, Exception) as e:
        logger.warning(f"DCC failed ({e}), falling back to EWM correlations")
        ewm_span = int(cfg.portfolio.dcc_fallback_ewm_span)
        return compute_ewm_correlations(returns_df, span=ewm_span)


def compute_ewm_correlations(returns_df: pd.DataFrame, span: int = 60) -> pd.DataFrame:
    # Rolling EWM correlation matrix, flattened into columns
    cols = returns_df.columns.tolist()
    ewm_corr_parts = {}

    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i <= j:
                corr_key = f"corr_{c1}_{c2}"
                if i == j:
                    ewm_corr_parts[corr_key] = pd.Series(1.0, index=returns_df.index)
                else:
                    # EWM covariance / (EWM std * EWM std)
                    cov = returns_df[c1].ewm(span=span, adjust=False).cov(returns_df[c2])
                    std1 = returns_df[c1].ewm(span=span, adjust=False).std()
                    std2 = returns_df[c2].ewm(span=span, adjust=False).std()
                    corr = cov / (std1 * std2 + 1e-12)
                    ewm_corr_parts[corr_key] = corr.clip(-1.0, 1.0)

    return pd.DataFrame(ewm_corr_parts, index=returns_df.index)


def ensure_positive_definite(corr_matrix: np.ndarray) -> np.ndarray:
    # Clip negative eigenvalues to 1e-6 and reconstruct
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    eigenvalues_clipped = np.clip(eigenvalues, 1e-6, None)
    pd_matrix = eigenvectors @ np.diag(eigenvalues_clipped) @ eigenvectors.T
    # Normalize to correlation matrix (diagonal = 1)
    d = np.sqrt(np.diag(pd_matrix))
    pd_matrix = pd_matrix / np.outer(d, d)
    return pd_matrix


def check_entry_correlation(
    candidate_returns: pd.Series,
    open_positions_returns: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    cfg,
) -> tuple:
    # Returns (size_multiplier, should_skip)
    skip_threshold = float(cfg.portfolio.correlation_skip_threshold)
    reduce_threshold = float(cfg.portfolio.correlation_reduce_threshold)
    reduce_pct = float(cfg.portfolio.correlation_reduce_pct)

    if open_positions_returns is None or open_positions_returns.empty:
        return 1.0, False

    candidate_name = candidate_returns.name
    max_corr = 0.0

    for pos_symbol in open_positions_returns.columns:
        corr_key = f"corr_{candidate_name}_{pos_symbol}"
        alt_key = f"corr_{pos_symbol}_{candidate_name}"
        if corr_key in corr_matrix.columns:
            c = abs(float(corr_matrix[corr_key].iloc[-1]))
        elif alt_key in corr_matrix.columns:
            c = abs(float(corr_matrix[alt_key].iloc[-1]))
        else:
            # Compute on-the-fly from recent returns
            try:
                cov_data = pd.concat([candidate_returns.tail(60), open_positions_returns[pos_symbol].tail(60)], axis=1).dropna()
                if len(cov_data) > 5:
                    c = abs(float(cov_data.corr().iloc[0, 1]))
                else:
                    c = 0.0
            except Exception:
                c = 0.0
        max_corr = max(max_corr, c)

    if max_corr > skip_threshold:
        logger.debug(f"Correlation skip: max_corr={max_corr:.3f} > {skip_threshold}")
        return 0.0, True

    if max_corr > reduce_threshold:
        scale = 1.0 - reduce_pct
        logger.debug(f"Correlation reduce: max_corr={max_corr:.3f} > {reduce_threshold}, scale={scale:.2f}")
        return scale, False

    return 1.0, False
