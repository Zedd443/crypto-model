import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
import optuna
import shap
from src.utils.logger import get_logger
from src.models.splitter import PurgedTimeSeriesSplit

logger = get_logger("primary_model")

optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_xgb_params(trial_or_dict, cfg) -> dict:
    if hasattr(trial_or_dict, "suggest_float"):
        trial = trial_or_dict
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        }
    else:
        # Fixed dict or cfg-based defaults
        d = trial_or_dict if isinstance(trial_or_dict, dict) else {}
        params = {
            "max_depth": d.get("max_depth", int(cfg.model.xgb_max_depth)),
            "learning_rate": d.get("learning_rate", float(cfg.model.xgb_learning_rate)),
            "subsample": d.get("subsample", float(cfg.model.xgb_subsample)),
            "colsample_bytree": d.get("colsample_bytree", float(cfg.model.xgb_colsample_bytree)),
            "min_child_weight": d.get("min_child_weight", int(cfg.model.xgb_min_child_weight)),
            "reg_alpha": d.get("reg_alpha", float(cfg.model.xgb_reg_alpha)),
            "reg_lambda": d.get("reg_lambda", float(cfg.model.xgb_reg_lambda)),
            "gamma": d.get("gamma", float(cfg.model.xgb_gamma)),
            "n_estimators": d.get("n_estimators", int(cfg.model.xgb_n_estimators)),
        }

    # Pass through scale_pos_weight if provided in the params dict
    if isinstance(trial_or_dict, dict) and "scale_pos_weight" in trial_or_dict:
        params["scale_pos_weight"] = trial_or_dict["scale_pos_weight"]

    # Fixed architecture params — device auto-detected via env var or config
    import os
    device = os.environ.get("XGB_DEVICE", getattr(getattr(cfg, "model", cfg), "xgb_device", "cpu"))
    params.update({
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "tree_method": "hist",
        "device": device,
        "random_state": 42,
        "n_jobs": -1 if device == "cpu" else 1,  # n_jobs ignored on GPU
    })
    return params


def compute_objective(y_true: np.ndarray, y_pred_proba: np.ndarray, returns: np.ndarray) -> float:
    # Combined objective: 0.4*DA + 0.3*IC + 0.3*Sharpe_synthetic
    da_weight = 0.4
    ic_weight = 0.3
    sharpe_weight = 0.3

    # Directional accuracy
    pred_dir = (y_pred_proba[:, 1] > 0.5).astype(int)
    true_dir = y_true.astype(int)
    da = float(np.mean(pred_dir == true_dir))

    # Information coefficient (Spearman rank correlation)
    try:
        ic = spearmanr(y_pred_proba[:, 1], returns).correlation
        if np.isnan(ic):
            ic = 0.0
    except Exception:
        ic = 0.0

    # Synthetic Sharpe from signal-based PnL
    positions = np.where(y_pred_proba[:, 1] > 0.5, 1.0, -1.0)
    pnl = positions * returns
    std_pnl = pnl.std()
    synthetic_sharpe = float(pnl.mean() / (std_pnl + 1e-9)) * np.sqrt(252 * 96)
    # Clip to reasonable range
    synthetic_sharpe = np.clip(synthetic_sharpe, -3.0, 3.0)

    score = da_weight * da + ic_weight * ic + sharpe_weight * (synthetic_sharpe / 3.0)
    return float(score)


def tune_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    weights: pd.Series,
    splitter: PurgedTimeSeriesSplit,
    cfg,
) -> dict:
    n_trials = int(cfg.model.optuna_n_trials)
    patience = int(cfg.model.optuna_patience)
    early_stop_rounds = int(cfg.model.xgb_early_stopping_rounds)

    returns_proxy = y_train.map({0: -1.0, 1: 1.0}).fillna(0.0).values

    best_scores = []

    def objective(trial):
        params = build_xgb_params(trial, cfg)
        n_est = params.pop("n_estimators")

        fold_scores = []
        for train_idx, val_idx in splitter.split(X_train, y_train):
            X_tr = X_train.iloc[train_idx].values
            y_tr = y_train.iloc[train_idx].values
            w_tr = weights.iloc[train_idx].values if hasattr(weights, "iloc") else weights[train_idx]
            X_v = X_train.iloc[val_idx].values
            y_v = y_train.iloc[val_idx].values
            ret_v = returns_proxy[val_idx]

            model = XGBClassifier(
                **params,
                n_estimators=n_est,
                early_stopping_rounds=early_stop_rounds,
                verbosity=0,
            )
            try:
                model.fit(
                    X_tr, y_tr,
                    sample_weight=w_tr,
                    eval_set=[(X_v, y_v)],
                    verbose=False,
                )
                proba = model.predict_proba(X_v)
                score = compute_objective(y_v, proba, ret_v)
                fold_scores.append(score)
            except Exception as e:
                logger.warning(f"Trial {trial.number} fold failed: {e}")
                continue

        if not fold_scores:
            return 0.0
        return float(np.mean(fold_scores))

    def _no_improvement_callback(study, trial):
        # Stop if no improvement in last `patience` trials
        best_scores.append(study.best_value)
        if len(best_scores) >= patience:
            recent = best_scores[-patience:]
            if all(v <= recent[0] for v in recent[1:]):
                study.stop()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, callbacks=[_no_improvement_callback], show_progress_bar=False)

    best_params = study.best_params
    logger.info(f"Optuna best score: {study.best_value:.4f}, params: {best_params}")
    return best_params


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    weights_train: pd.Series,
    params: dict,
    cfg,
) -> tuple:
    early_stop_rounds = int(cfg.model.xgb_early_stopping_rounds)

    xgb_params = build_xgb_params(params, cfg)
    n_estimators = xgb_params.pop("n_estimators")

    model = XGBClassifier(
        **xgb_params,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stop_rounds,
        verbosity=0,
    )

    w_train = weights_train.values if hasattr(weights_train, "values") else weights_train

    model.fit(
        X_train.values if hasattr(X_train, "values") else X_train,
        y_train.values if hasattr(y_train, "values") else y_train,
        sample_weight=w_train,
        eval_set=[(
            X_val.values if hasattr(X_val, "values") else X_val,
            y_val.values if hasattr(y_val, "values") else y_val,
        )],
        verbose=False,
    )

    # Calibrate on val set using IsotonicRegression
    val_proba_raw = model.predict_proba(
        X_val.values if hasattr(X_val, "values") else X_val
    )[:, 1]
    y_val_arr = y_val.values if hasattr(y_val, "values") else y_val

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_proba_raw, y_val_arr)

    best_iter = model.best_iteration
    logger.info(f"XGBoost trained: best_iteration={best_iter}")
    return model, calibrator


def compute_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    splitter: PurgedTimeSeriesSplit,
    params: dict,
    cfg,
) -> np.ndarray:
    # OOF predictions via PurgedTimeSeriesSplit — NEVER in-sample
    oof_proba = np.zeros((len(X), 2))
    oof_proba[:, 0] = 0.5
    oof_proba[:, 1] = 0.5
    early_stop_rounds = int(cfg.model.xgb_early_stopping_rounds)

    for train_idx, val_idx in splitter.split(X, y):
        X_t = X.iloc[train_idx].values
        y_t = y.iloc[train_idx].values
        w_t = weights.iloc[train_idx].values if hasattr(weights, "iloc") else weights[train_idx]
        X_v = X.iloc[val_idx].values
        y_v = y.iloc[val_idx].values

        xgb_params = build_xgb_params(params, cfg)
        n_estimators = xgb_params.pop("n_estimators")

        # Use TAIL of train fold for early stopping — keeps ALL val_idx bars available for OOF.
        # Cap the ES holdout at 200 bars to avoid shrinking small train folds too much.
        train_split = max(int(len(X_t) * 0.8), len(X_t) - 200)
        X_t_fit, y_t_fit = X_t[:train_split], y_t[:train_split]
        w_t_fit = w_t[:train_split]
        X_es, y_es = X_t[train_split:], y_t[train_split:]

        model = XGBClassifier(
            **xgb_params,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stop_rounds,
            verbosity=0,
        )
        try:
            model.fit(
                X_t_fit, y_t_fit,
                sample_weight=w_t_fit,
                eval_set=[(X_es, y_es)],
                verbose=False,
            )
            # ALL val fold bars receive real OOF predictions — no wasted 0.5 defaults
            proba = model.predict_proba(X_v)
            oof_proba[val_idx] = proba
        except Exception as e:
            logger.warning(f"OOF fold failed: {e}")

    return oof_proba


def compute_shap_importance(model, X_train: pd.DataFrame, top_k: int = 20) -> pd.Series:
    explainer = shap.TreeExplainer(model)
    # Use sample for speed if large
    n_sample = min(len(X_train), 2000)
    X_sample = X_train.iloc[:n_sample] if hasattr(X_train, "iloc") else X_train[:n_sample]
    shap_values = explainer.shap_values(X_sample.values if hasattr(X_sample, "values") else X_sample)
    # For binary classifier shap returns array of shape (n, features)
    if isinstance(shap_values, list):
        shap_arr = np.abs(shap_values[1])
    else:
        shap_arr = np.abs(shap_values)
    mean_shap = shap_arr.mean(axis=0)
    feature_names = X_train.columns.tolist() if hasattr(X_train, "columns") else list(range(len(mean_shap)))
    importance = pd.Series(mean_shap, index=feature_names).sort_values(ascending=False)
    return importance.head(top_k)


def compute_conformal_q90(y_cal: np.ndarray, cal_proba: np.ndarray) -> float:
    # Non-conformity scores: |y_true - predicted_prob_class_1|
    if cal_proba.ndim == 2:
        pred_prob = cal_proba[:, 1]
    else:
        pred_prob = cal_proba
    scores = np.abs(y_cal.astype(float) - pred_prob)
    q90 = float(np.quantile(scores, 0.90))
    return q90


def save_model(model, calibrator, symbol: str, tf: str, version: str, models_dir) -> None:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{version}_model.json"
    model.save_model(str(model_path))

    cal_path = models_dir / f"{version}_calibrator.pkl"
    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f)

    logger.info(f"Model saved: {model_path}")


def load_model(symbol: str, tf: str, version: str, models_dir) -> tuple:
    models_dir = Path(models_dir)

    model_path = models_dir / f"{version}_model.json"
    cal_path = models_dir / f"{version}_calibrator.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not cal_path.exists():
        raise FileNotFoundError(f"Calibrator not found: {cal_path}")

    model = XGBClassifier()
    model.load_model(str(model_path))

    with open(cal_path, "rb") as f:
        calibrator = pickle.load(f)

    return model, calibrator
