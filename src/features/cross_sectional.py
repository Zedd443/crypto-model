import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression
from src.utils.logger import get_logger

logger = get_logger("cross_sectional")


def fit_cross_sectional_stats(train_panel: pd.DataFrame, feature_cols: list, save_path: Path) -> dict:
    # train_panel: wide format with MultiIndex (timestamp, symbol) OR columns per symbol
    # Computes rolling cross-sectional stats on TRAIN data only
    # LEAKAGE GUARD enforced by caller (assert train data only passed in)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {}
    for col in feature_cols:
        if col not in train_panel.columns:
            continue
        series = train_panel[col].dropna()
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "q01": float(series.quantile(0.01)),
            "q99": float(series.quantile(0.99)),
        }

    with open(save_path, "wb") as f:
        pickle.dump(stats, f)
    logger.info(f"Cross-sectional stats fitted and saved to {save_path} for {len(stats)} features")
    return stats


def fit_cross_sectional_stats_from_files(
    parquet_paths: list[Path],
    feature_cols: list,
    train_end_ts,
    save_path: Path,
) -> dict:
    # Ultra-low-memory version: reads ONE column at a time from parquet using pyarrow.
    # Peak RAM = one column × all rows for one file at a time (~1-5 MB per col).
    # LEAKAGE GUARD: only rows with index <= train_end_ts are used.
    import pyarrow.parquet as pq

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure train_end_ts is UTC-aware Timestamp for index comparison
    train_end_ts = pd.Timestamp(train_end_ts, tz="UTC") if not isinstance(train_end_ts, pd.Timestamp) else train_end_ts
    if train_end_ts.tz is None:
        train_end_ts = train_end_ts.tz_localize("UTC")

    accum = {col: {"min": np.inf, "max": -np.inf, "sum": 0.0, "sum_sq": 0.0, "count": 0, "samples": []}
             for col in feature_cols}

    for path in parquet_paths:
        try:
            pf = pq.ParquetFile(path)
            # Read index column to get train mask — index is stored as '__index_level_0__' or similar
            idx_col = pf.schema_arrow.pandas_metadata["index_columns"] if hasattr(pf.schema_arrow, "pandas_metadata") else None
            # Simpler: read only index + one col at a time using row groups
            # Read full index first (cheap — one int64 column)
            idx_table = pf.read(columns=[])  # no data columns, just index
            idx_vals = idx_table.to_pandas().index
            # Normalise to UTC-aware DatetimeIndex for comparison
            if not isinstance(idx_vals, pd.DatetimeIndex):
                idx_vals = pd.to_datetime(idx_vals, utc=True)
            elif idx_vals.tz is None:
                idx_vals = idx_vals.tz_localize("UTC")
            else:
                idx_vals = idx_vals.tz_convert("UTC")
            train_mask = idx_vals <= train_end_ts
            n_train = train_mask.sum()
            if n_train == 0:
                continue

            # Read each feature column individually
            available_cols = set(pf.schema_arrow.names)
            for col in feature_cols:
                if col not in available_cols:
                    continue
                col_table = pf.read(columns=[col])
                vals_all = col_table.column(col).to_pylist()
                vals = np.array([v for v, m in zip(vals_all, train_mask) if m and v is not None], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue
                a = accum[col]
                a["min"] = min(a["min"], float(vals.min()))
                a["max"] = max(a["max"], float(vals.max()))
                a["sum"] += float(vals.sum())
                a["sum_sq"] += float((vals ** 2).sum())
                a["count"] += len(vals)
                # Reservoir: keep up to 10k per file per col for quantile estimation
                if len(a["samples"]) < 10_000:
                    take = min(10_000 - len(a["samples"]), len(vals))
                    a["samples"].append(vals[:take])
                del vals, col_table
        except Exception as e:
            logger.warning(f"Skipping {path.name} in cross-sectional fit: {e}")
            continue

    stats = {}
    for col in feature_cols:
        a = accum[col]
        if a["count"] == 0:
            continue
        n = a["count"]
        mean = a["sum"] / n
        var = max(a["sum_sq"] / n - mean ** 2, 0.0)
        sample_arr = np.concatenate(a["samples"]) if a["samples"] else np.array([mean])
        stats[col] = {
            "mean": mean,
            "std": float(np.sqrt(var)),
            "min": a["min"],
            "max": a["max"],
            "q01": float(np.percentile(sample_arr, 1)),
            "q99": float(np.percentile(sample_arr, 99)),
        }
        a["samples"] = []

    with open(save_path, "wb") as f:
        pickle.dump(stats, f)
    logger.info(f"Cross-sectional stats fitted (column-streaming) for {len(stats)} features")
    return stats


def apply_cross_sectional_ranks(panel_df: pd.DataFrame, stats_path: Path, feature_cols: list) -> pd.DataFrame:
    # Load pre-fitted stats — transform only, never refit
    stats_path = Path(stats_path)
    if not stats_path.exists():
        logger.warning(f"Cross-sectional stats not found at {stats_path}, skipping ranks")
        return panel_df

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    result = panel_df.copy()
    for col in feature_cols:
        if col not in stats or col not in panel_df.columns:
            continue
        s = stats[col]
        rng = s["max"] - s["min"]
        if rng < 1e-12:
            result[f"{col}_rank"] = 0.5
        else:
            result[f"{col}_rank"] = (panel_df[col].clip(s["q01"], s["q99"]) - s["min"]) / rng
    return result


def compute_return_ranks(returns_panel: pd.DataFrame, windows: list = None, stats: dict = None) -> pd.DataFrame:
    if windows is None:
        windows = [1, 4, 24]
    # returns_panel: DataFrame with symbol columns, timestamp index
    # For each window, compute return over that many bars then rank cross-sectionally
    result_parts = {}
    for w in windows:
        ret_w = returns_panel.rolling(w, min_periods=w).sum()
        # Cross-sectional rank at each timestamp using pre-fitted train distribution
        # Rank 0-1 across symbols using scipy percentile rank approach
        col_name = f"return_{w}h_rank"
        # rank along columns (symbols) at each row (timestamp)
        result_parts[col_name] = ret_w.rank(axis=1, pct=True)
    return pd.DataFrame(result_parts, index=returns_panel.index)


def compute_volume_rank(volume_panel: pd.DataFrame) -> pd.DataFrame:
    # Rolling 24-bar volume rank across symbols
    vol_24 = volume_panel.rolling(24, min_periods=24).mean()
    vol_rank = vol_24.rank(axis=1, pct=True)
    return vol_rank.rename(columns={c: f"volume_24h_rank_{c}" for c in vol_rank.columns})


def compute_beta_adj_momentum(returns_df: pd.DataFrame, btc_returns: pd.Series, window: int = 20) -> pd.DataFrame:
    # Rolling OLS regression of each coin on BTC, residual = beta-adjusted momentum
    btc_aligned = btc_returns.reindex(returns_df.index)
    result = pd.DataFrame(index=returns_df.index)

    for col in returns_df.columns:
        coin_ret = returns_df[col]
        residuals = pd.Series(np.nan, index=returns_df.index)

        for i in range(window, len(returns_df)):
            y = coin_ret.iloc[i - window:i].values
            x = btc_aligned.iloc[i - window:i].values
            mask = ~(np.isnan(y) | np.isnan(x))
            if mask.sum() < max(10, window // 2):
                continue
            y_clean, x_clean = y[mask], x[mask]
            x_reshaped = x_clean.reshape(-1, 1)
            try:
                reg = LinearRegression(fit_intercept=True).fit(x_reshaped, y_clean)
                # Residual at current bar
                curr_y = coin_ret.iloc[i]
                curr_x = btc_aligned.iloc[i]
                if not (np.isnan(curr_y) or np.isnan(curr_x)):
                    residuals.iloc[i] = curr_y - reg.predict([[curr_x]])[0]
            except Exception:
                continue

        result[f"{col}_beta_adj_mom"] = residuals

    return result


def flag_reversal_candidates(return_rank_1h: pd.Series) -> pd.Series:
    # 1 if extreme rank (top or bottom 10% cross-sectionally)
    flag = ((return_rank_1h > 0.90) | (return_rank_1h < 0.10)).astype(int)
    return flag.rename("reversal_candidate")
