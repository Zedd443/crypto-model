import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from xgboost import XGBClassifier
import optuna
from src.utils.logger import get_logger

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = get_logger("meta_labeler")


def create_meta_labels(y_true: np.ndarray, oof_proba: np.ndarray, dead_zone: float = 0.05) -> np.ndarray:
    # meta_y = 1 if primary prediction was correct, 0 otherwise
    # Dead-zone bars (|prob-0.5| < dead_zone): primary is near-random → exclude as "don't trust"
    if oof_proba.ndim == 2:
        prob_long = oof_proba[:, 1]
        pred_class = (prob_long > 0.5).astype(int)
    else:
        prob_long = oof_proba
        pred_class = (prob_long > 0.5).astype(int)
    meta_y = (pred_class == y_true.astype(int)).astype(int)
    in_dead_zone = np.abs(prob_long - 0.5) < dead_zone
    meta_y[in_dead_zone] = 0
    return meta_y


def build_meta_features(
    oof_proba: np.ndarray,
    regime_probs_df: pd.DataFrame,
    realized_vol: pd.Series,
    volume_zscore: pd.Series,
    ofi: pd.Series,
    spread_series: pd.Series = None,
    atr_series: pd.Series = None,
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

    # Time since last signal: bars since |prob_long - 0.5| > 0.1
    signal_active = (np.abs(prob_long - 0.5) > 0.1).astype(float)
    bars_since = np.zeros(len(signal_active))
    counter = len(signal_active)  # large sentinel for "never"
    for i in range(len(signal_active)):
        if signal_active[i] > 0:
            counter = 0
        else:
            counter += 1
        bars_since[i] = counter
    meta_df["time_since_last_signal"] = bars_since

    # Spread-to-ATR ratio (informative about execution cost relative to volatility)
    if spread_series is not None and atr_series is not None:
        spread_vals = spread_series.values if hasattr(spread_series, "values") else np.array(spread_series)
        atr_vals = atr_series.values if hasattr(atr_series, "values") else np.array(atr_series)
        meta_df["spread_to_atr_ratio"] = spread_vals / (atr_vals + 1e-9)

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

    w = weights.values if hasattr(weights, "values") else weights
    X = meta_X_train.values if hasattr(meta_X_train, "values") else meta_X_train
    X = np.nan_to_num(X, nan=0.0)

    # Class imbalance compensation: primary DA ~55% → meta_y=1 for ~55% of bars
    n_meta0 = int((meta_y_train == 0).sum())
    n_meta1 = int((meta_y_train == 1).sum())
    meta_spw = n_meta0 / max(n_meta1, 1)
    logger.info(f"Meta scale_pos_weight: {meta_spw:.3f} (n0={n_meta0}, n1={n_meta1})")

    # 10-trial Optuna mini-study to tune lr and subsample — temporal 80/20 split
    split_idx = int(0.8 * len(X))
    X_ms, X_mv = X[:split_idx], X[split_idx:]
    y_ms, y_mv = meta_y_train[:split_idx], meta_y_train[split_idx:]
    w_ms = w[:split_idx]

    def _meta_objective(trial):
        lr = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        sub = trial.suggest_float("subsample", 0.6, 1.0)
        m = XGBClassifier(
            n_estimators=min(n_estimators, 100),  # faster mini-study
            max_depth=max_depth,
            learning_rate=lr,
            subsample=sub,
            colsample_bytree=0.8,
            scale_pos_weight=meta_spw,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device=device,
            n_jobs=-1 if device == "cpu" else 1,
            random_state=42,
            verbosity=0,
        )
        m.fit(X_ms, y_ms, sample_weight=w_ms)
        proba = m.predict_proba(X_mv)[:, 1]
        # Use log-loss negated as maximize objective
        eps = 1e-9
        logloss = -np.mean(y_mv * np.log(proba + eps) + (1 - y_mv) * np.log(1 - proba + eps))
        return -logloss

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    try:
        study.optimize(_meta_objective, n_trials=10, show_progress_bar=False)
        best_lr = study.best_params["learning_rate"]
        best_sub = study.best_params["subsample"]
        logger.info(f"Meta Optuna best: lr={best_lr:.4f}, subsample={best_sub:.3f}")
    except Exception as e:
        logger.warning(f"Meta Optuna failed: {e} — using defaults")
        best_lr = 0.05
        best_sub = 0.8

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=best_lr,
        subsample=best_sub,
        colsample_bytree=0.8,
        scale_pos_weight=meta_spw,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=device,
        n_jobs=-1 if device == "cpu" else 1,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, meta_y_train, sample_weight=w)
    logger.info(f"Meta-labeler trained: {n_estimators} estimators, depth={max_depth}, spw={meta_spw:.3f}")
    meta_stats = {
        "meta_spw": round(meta_spw, 4),
        "meta_best_lr": round(best_lr, 6),
        "meta_best_subsample": round(best_sub, 4),
        "meta_n0": n_meta0,
        "meta_n1": n_meta1,
    }
    return model, meta_stats


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
