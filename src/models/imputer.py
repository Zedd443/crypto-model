import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("imputer")


def fit_imputer(X_train: np.ndarray, symbol: str, tf: str, imp_dir, cfg) -> dict:
    # Force float64 — isnan/isfinite do not support object/bool/datetime dtypes.
    # This is the root-level guard: anything non-numeric becomes NaN via 'coerce'.
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.apply(pd.to_numeric, errors="coerce").astype(np.float64).values
    else:
        X_train = np.array(X_train, dtype=np.float64)

    missing_rate = np.isnan(X_train).mean(axis=0)
    col_medians = np.nanmedian(X_train, axis=0)

    # Replace NaN with column median
    nan_mask = np.isnan(X_train)
    X_train[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    imp_dir = Path(imp_dir)
    imp_dir.mkdir(parents=True, exist_ok=True)
    imp_path = imp_dir / f"{symbol}_{tf}_imputer.pkl"

    imputer_data = {
        "col_medians": col_medians,
        "missing_rate": missing_rate,
    }
    with open(imp_path, "wb") as f:
        pickle.dump(imputer_data, f)

    logger.info(
        f"{symbol}: imputer fitted — {int((missing_rate > 0).sum())} cols had missing values, "
        f"saved to {imp_path}"
    )
    return imputer_data


def apply_imputer(X: np.ndarray, imputer_data: dict) -> np.ndarray:
    # Force float64 on inference input as well.
    if isinstance(X, pd.DataFrame):
        X = X.apply(pd.to_numeric, errors="coerce").astype(np.float64).values
    else:
        X = np.array(X, dtype=np.float64)

    col_medians = imputer_data["col_medians"]
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
    return X


def load_imputer(symbol: str, tf: str, imp_dir) -> dict:
    imp_path = Path(imp_dir) / f"{symbol}_{tf}_imputer.pkl"
    if not imp_path.exists():
        raise FileNotFoundError(f"Imputer not found: {imp_path}")
    with open(imp_path, "rb") as f:
        return pickle.load(f)


# For compatibility with stage_04_train.py
def transform_with_imputer(X: np.ndarray, symbol: str, tf: str, imp_dir) -> np.ndarray:
    """Load saved imputer artifact and apply it to X."""
    imputer_data = load_imputer(symbol, tf, imp_dir)
    return apply_imputer(X, imputer_data)


# For compatibility with stage_04_train.py
def fit_robust_scaler(X_train: np.ndarray, symbol: str, tf: str, imp_dir) -> dict:
    """Fit a simple robust scaler: (x - median) / IQR. Saves artifact to {symbol}_{tf}_scaler.pkl in imp_dir."""
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.apply(pd.to_numeric, errors="coerce").astype(np.float64).values
    else:
        X_train = np.array(X_train, dtype=np.float64)

    med = np.nanmedian(X_train, axis=0)
    q25 = np.nanpercentile(X_train, 25, axis=0)
    q75 = np.nanpercentile(X_train, 75, axis=0)
    iqr = q75 - q25
    eps = 1e-12
    scale = np.where(np.abs(iqr) < eps, 1.0, iqr)

    scaler_data = {"median": med, "scale": scale}

    imp_dir = Path(imp_dir)
    imp_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = imp_dir / f"{symbol}_{tf}_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_data, f)

    logger.info(f"{symbol}: robust scaler fitted — saved to {scaler_path}")
    return scaler_data


def transform_with_scaler(X: np.ndarray, symbol: str, tf: str, imp_dir) -> np.ndarray:
    """Load saved scaler artifact and apply it to X."""
    if isinstance(X, pd.DataFrame):
        X = X.apply(pd.to_numeric, errors="coerce").astype(np.float64).values
    else:
        X = np.array(X, dtype=np.float64)

    scaler_path = Path(imp_dir) / f"{symbol}_{tf}_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return (X - scaler["median"]) / scaler["scale"]