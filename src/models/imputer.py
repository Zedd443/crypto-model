import numpy as np
import pickle
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from src.utils.logger import get_logger

logger = get_logger("imputer")


def fit_imputer(X_train: "np.ndarray", symbol: str, tf: str, checkpoint_dir, cfg) -> dict:
    # LEAKAGE GUARD: only fits on X_train, never on val/test
    # Returns dict {"imputer": SimpleImputer, "indicator_cols": np.ndarray}
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Identify columns with >5% missing — missingness itself is informative
    missing_rate = np.isnan(X_train).mean(axis=0)
    indicator_cols = np.where(missing_rate > 0.05)[0]

    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)

    artifact = {"imputer": imputer, "indicator_cols": indicator_cols}
    out_path = checkpoint_dir / f"imputer_{symbol}_{tf}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    logger.info(f"Imputer fitted and saved: {out_path} ({len(indicator_cols)} indicator flags)")
    return artifact


def transform_with_imputer(X: "np.ndarray", symbol: str, tf: str, checkpoint_dir) -> "np.ndarray":
    # Loads pkl, calls transform only — NEVER fits
    # Appends binary missing-indicator flags for columns saved during fit
    checkpoint_dir = Path(checkpoint_dir)
    path = checkpoint_dir / f"imputer_{symbol}_{tf}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Imputer checkpoint not found: {path}")
    with open(path, "rb") as f:
        artifact = pickle.load(f)

    # Support legacy format (bare SimpleImputer or IterativeImputer pickle)
    if isinstance(artifact, dict):
        imputer = artifact["imputer"]
        indicator_cols = artifact["indicator_cols"]
    else:
        imputer = artifact
        indicator_cols = np.array([], dtype=int)

    X_imputed = imputer.transform(X)

    if len(indicator_cols) > 0:
        # Binary flags: 1 where original value was NaN
        flags = np.isnan(X[:, indicator_cols]).astype(np.float32)
        X_imputed = np.hstack([X_imputed, flags])

    return X_imputed


def get_indicator_col_count(symbol: str, tf: str, checkpoint_dir) -> int:
    # Returns number of missing-indicator columns added by transform_with_imputer
    checkpoint_dir = Path(checkpoint_dir)
    path = checkpoint_dir / f"imputer_{symbol}_{tf}.pkl"
    if not path.exists():
        return 0
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    if isinstance(artifact, dict):
        return len(artifact["indicator_cols"])
    return 0


def fit_robust_scaler(X_train: "np.ndarray", symbol: str, tf: str, checkpoint_dir) -> RobustScaler:
    # Fits on train data only, saves to disk
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    scaler = RobustScaler()
    scaler.fit(X_train)

    out_path = checkpoint_dir / f"scaler_{symbol}_{tf}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"RobustScaler fitted and saved: {out_path}")
    return scaler


def transform_with_scaler(X: "np.ndarray", symbol: str, tf: str, checkpoint_dir) -> "np.ndarray":
    # Loads pkl, transforms only — NEVER fits
    checkpoint_dir = Path(checkpoint_dir)
    path = checkpoint_dir / f"scaler_{symbol}_{tf}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Scaler checkpoint not found: {path}")
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    return scaler.transform(X)
