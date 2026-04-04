import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from xgboost import XGBClassifier
from src.utils.logger import get_logger

logger = get_logger("meta_labeler")


def create_meta_labels(y_true: np.ndarray, oof_proba: np.ndarray) -> np.ndarray:
    # meta_y = 1 if primary prediction was correct, 0 otherwise
    # For binary: correct = (prob_class_1 > 0.5) matches y_true
    if oof_proba.ndim == 2:
        pred_class = (oof_proba[:, 1] > 0.5).astype(int)
    else:
        pred_class = (oof_proba > 0.5).astype(int)
    meta_y = (pred_class == y_true.astype(int)).astype(int)
    return meta_y


def build_meta_features(
    oof_proba: np.ndarray,
    regime_probs_df: pd.DataFrame,
    realized_vol: pd.Series,
    volume_zscore: pd.Series,
    ofi: pd.Series,
) -> pd.DataFrame:
    # ONLY these features to prevent meta overfitting
    if oof_proba.ndim == 2:
        prob_long = oof_proba[:, 1]
        prob_short = oof_proba[:, 0]
    else:
        prob_long = oof_proba
        prob_short = 1.0 - oof_proba

    primary_confidence = np.maximum(prob_long, prob_short)

    meta_df = pd.DataFrame({
        "primary_prob_long": prob_long,
        "primary_prob_short": prob_short,
        "primary_confidence": primary_confidence,
    })

    # Regime probabilities — only raw prob columns, not derived rank/entropy cols
    if regime_probs_df is not None and len(regime_probs_df) > 0:
        prob_cols = [c for c in regime_probs_df.columns if not any(x in c for x in ("_rank", "_entropy", "_diff"))]
        for col in prob_cols:
            meta_df[col] = regime_probs_df[col].values if len(regime_probs_df) == len(meta_df) else np.nan

    # Volatility and flow features
    meta_df["realized_vol"] = realized_vol.values if hasattr(realized_vol, "values") else realized_vol
    meta_df["volume_zscore"] = volume_zscore.values if hasattr(volume_zscore, "values") else volume_zscore
    meta_df["ofi"] = ofi.values if hasattr(ofi, "values") else ofi

    return meta_df


def train_meta_labeler(
    meta_X_train: pd.DataFrame,
    meta_y_train: np.ndarray,
    weights: pd.Series,
    cfg,
) -> XGBClassifier:
    n_estimators = int(cfg.model.meta_n_estimators)
    max_depth = int(cfg.model.meta_max_depth)

    device = os.environ.get("XGB_DEVICE", "cpu")
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        tree_method="hist",
        device=device,
        n_jobs=-1 if device == "cpu" else 1,
        random_state=42,
        verbosity=0,
    )

    w = weights.values if hasattr(weights, "values") else weights
    X = meta_X_train.values if hasattr(meta_X_train, "values") else meta_X_train

    # Fill NaN in meta features
    X = np.nan_to_num(X, nan=0.0)

    model.fit(X, meta_y_train, sample_weight=w)
    logger.info(f"Meta-labeler trained: {n_estimators} estimators, {max_depth} depth")
    return model


def compute_signal_strength(
    primary_proba: np.ndarray,
    meta_proba: np.ndarray,
    cfg,
) -> tuple:
    # primary_proba: (N, 2) or scalar for single bar
    # meta_proba: scalar probability of being correct
    signal_floor = float(cfg.model.meta_signal_floor)

    if primary_proba.ndim == 2:
        prob_long = float(primary_proba[0, 1]) if primary_proba.shape[0] == 1 else primary_proba[:, 1]
    else:
        prob_long = float(primary_proba)

    if np.isscalar(prob_long):
        direction = 1 if prob_long > 0.5 else -1
        primary_conf = prob_long if direction == 1 else (1.0 - prob_long)
        strength = primary_conf * float(meta_proba)
        if strength < signal_floor:
            return 0, 0.0
        return direction, strength
    else:
        # Vectorized
        direction = np.where(prob_long > 0.5, 1, -1)
        primary_conf = np.where(direction == 1, prob_long, 1.0 - prob_long)
        strength = primary_conf * meta_proba
        direction = np.where(strength < signal_floor, 0, direction)
        strength = np.where(strength < signal_floor, 0.0, strength)
        return direction, strength


def save_meta_model(model: XGBClassifier, symbol: str, tf: str, version: str, models_dir) -> None:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{version}_meta.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Meta model saved: {path}")


def load_meta_model(symbol: str, tf: str, version: str, models_dir) -> XGBClassifier:
    models_dir = Path(models_dir)
    path = models_dir / f"{version}_meta.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Meta model not found: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
