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


def create_meta_labels(y_true: np.ndarray, oof_proba: np.ndarray, dead_zone: float = 0.05) -> tuple:
    # Returns (meta_y, dead_zone_mask).
    # meta_y = 1 if primary prediction was correct, 0 otherwise.
    # dead_zone_mask = True where |prob-0.5| < dead_zone — caller should DROP these rows,
    # not train on them. Setting meta_y=0 there would teach the meta-model to predict
    # "incorrect" on near-random bars, polluting the correctness signal.
    if oof_proba.ndim == 2:
        prob_long = oof_proba[:, 1]
        pred_class = (prob_long > 0.5).astype(int)
    else:
        prob_long = oof_proba
        pred_class = (prob_long > 0.5).astype(int)
    meta_y = (pred_class == y_true.astype(int)).astype(int)
    dead_zone_mask = np.abs(prob_long - 0.5) < dead_zone
    return meta_y, dead_zone_mask


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
    counter = 0  # 0 = "at signal" for the very first bar, not "never seen"
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

    # Triple temporal split: 60% fit / 20% Optuna eval / 20% ES
    # Optuna and final-model early stopping must see different data — using the same
    # held-out set for both lets Optuna leak the ES signal into hyperparameter selection.
    split_60 = int(0.6 * len(X))
    split_80 = int(0.8 * len(X))
    X_ms, y_ms, w_ms = X[:split_60], meta_y_train[:split_60], w[:split_60]
    X_mv, y_mv = X[split_60:split_80], meta_y_train[split_60:split_80]  # Optuna eval
    X_es, y_es = X[split_80:], meta_y_train[split_80:]                  # final ES

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

    # Final model: fit on 60%+20%=80%, early stop on held-out 20% ES set
    es_patience = int(getattr(cfg.model, "meta_early_stopping_rounds", 10))
    X_fit = X[:split_80]
    y_fit = meta_y_train[:split_80]
    w_fit = w[:split_80]
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=best_lr,
        subsample=best_sub,
        colsample_bytree=0.8,
        scale_pos_weight=meta_spw,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=es_patience,
        tree_method="hist",
        device=device,
        n_jobs=-1 if device == "cpu" else 1,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_fit, y_fit, sample_weight=w_fit,
        eval_set=[(X_es, y_es)],
        verbose=False,
    )
    best_n_trees = model.best_iteration + 1 if hasattr(model, "best_iteration") and model.best_iteration else n_estimators
    logger.info(f"Meta-labeler trained: best_trees={best_n_trees}/{n_estimators} depth={max_depth} spw={meta_spw:.3f} es_patience={es_patience}")
    meta_stats = {
        "meta_spw": round(meta_spw, 4),
        "meta_best_lr": round(best_lr, 6),
        "meta_best_subsample": round(best_sub, 4),
        "meta_n0": n_meta0,
        "meta_n1": n_meta1,
        "meta_best_n_trees": best_n_trees,
    }
    return model, meta_stats


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
