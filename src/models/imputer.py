import numpy as np
import pickle
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import RobustScaler
from src.utils.logger import get_logger

logger = get_logger("imputer")


def fit_imputer(X_train: "np.ndarray", symbol: str, tf: str, checkpoint_dir, cfg) -> IterativeImputer:
    # LEAKAGE GUARD: only fits on X_train, never on val/test
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=10,
        random_state=42,
        skip_complete=True,
    )
    imputer.fit(X_train)

    out_path = checkpoint_dir / f"imputer_{symbol}_{tf}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(imputer, f)
    logger.info(f"Imputer fitted and saved: {out_path}")
    return imputer


def transform_with_imputer(X: "np.ndarray", symbol: str, tf: str, checkpoint_dir) -> "np.ndarray":
    # Loads pkl, calls transform only — NEVER fits
    checkpoint_dir = Path(checkpoint_dir)
    path = checkpoint_dir / f"imputer_{symbol}_{tf}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Imputer checkpoint not found: {path}")
    with open(path, "rb") as f:
        imputer = pickle.load(f)
    return imputer.transform(X)


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
