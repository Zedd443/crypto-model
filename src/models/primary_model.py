import os
import numpy as np
import pandas as pd
import pickle
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


class _SigmoidCalibrator:
    # Thin wrapper: exposes .predict(raw_probs) → calibrated probs, matching IsotonicRegression interface.
    # Internally uses a LogisticRegression fit on (raw_probs.reshape(-1,1), y_true).
    def __init__(self, lr_model):
        self._lr = lr_model

    def predict(self, raw_probs):
        arr = np.asarray(raw_probs).reshape(-1, 1)
        return self._lr.predict_proba(arr)[:, 1]


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
    device = os.environ.get("XGB_DEVICE", getattr(getattr(cfg, "model", cfg), "xgb_device", "cpu"))
    params.update({
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": device,
        "random_state": 42,
        "n_jobs": -1 if device == "cpu" else 1,  # n_jobs ignored on GPU
    })
    return params


def compute_objective(y_true: np.ndarray, y_pred_proba: np.ndarray, returns: np.ndarray, cfg=None) -> float:
    # Weights from config with fallbacks
    da_weight = float(getattr(getattr(cfg, 'model', cfg), 'objective_da_weight', 0.2)) if cfg is not None else 0.2
    ic_weight = float(getattr(getattr(cfg, 'model', cfg), 'objective_ic_weight', 0.3)) if cfg is not None else 0.3
    sharpe_weight = float(getattr(getattr(cfg, 'model', cfg), 'objective_sortino_weight',
                          getattr(getattr(cfg, 'model', cfg), 'objective_sharpe_weight', 0.5))) if cfg is not None else 0.5
    cvar_weight = float(getattr(getattr(cfg, 'model', cfg), 'objective_cvar_weight', 0.1)) if cfg is not None else 0.1
    dead_zone = float(getattr(getattr(cfg, 'model', cfg), 'objective_dead_zone', 0.05)) if cfg is not None else 0.05
    fee_cost = float(getattr(getattr(cfg, 'labels', cfg), 'round_trip_cost_pct', 0.003)) if cfg is not None else 0.003

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

    # Calmar-adjusted Sortino (70% Sortino + 30% Calmar blend)
    # Sortino uses downside deviation instead of std — appropriate for skewed crypto returns
    proba_long = y_pred_proba[:, 1]
    positions = np.where(proba_long > 0.5 + dead_zone, 1.0, np.where(proba_long < 0.5 - dead_zone, -1.0, 0.0))
    active_mask = positions != 0.0
    if active_mask.sum() > 1:
        active_returns = positions[active_mask] * returns[active_mask] - fee_cost  # fee-adjusted
        mean_r = active_returns.mean()
        ann_factor = np.sqrt(252 * 96)
        # Downside deviation: RMS of returns below 0 (target = 0)
        # Fallback when tail < 50 bars: use 0.5 × full std — avoids infinite Sortino in lucky folds
        downside = active_returns[active_returns < 0]
        if len(downside) >= 50:
            downside_dev = float(np.sqrt(np.mean(downside ** 2)))
        else:
            downside_dev = float(np.std(active_returns)) * 0.5
        downside_dev = max(downside_dev, 1e-9)
        sortino = float(mean_r / downside_dev) * ann_factor
        # Calmar: annualized return / max drawdown
        cum = np.cumprod(1 + np.clip(active_returns, -0.5, 0.5))
        running_max = np.maximum.accumulate(cum)
        drawdown = (running_max - cum) / (running_max + 1e-9)
        max_dd = drawdown.max()
        ann_ret = (cum[-1] ** (252 * 96 / max(len(active_returns), 1))) - 1
        calmar = float(ann_ret / (max_dd + 1e-9))
        calmar = np.clip(calmar, -3.0, 3.0)
        calmar_adj_sharpe = 0.7 * np.clip(sortino, -3.0, 3.0) + 0.3 * calmar
        # CVaR 95%: mean of worst 5% returns
        n_tail = max(1, int(0.05 * len(active_returns)))
        cvar_95 = float(np.sort(active_returns)[:n_tail].mean())
        cvar_penalty = np.clip(-cvar_95 * 10, 0.0, 1.0)  # positive when losses, 0 when gains
        # Auto-reduce CVaR weight when tail sample count too small for reliable estimate
        effective_cvar_weight = cvar_weight * min(1.0, n_tail / 50)
    else:
        calmar_adj_sharpe = 0.0
        cvar_penalty = 1.0  # maximum penalty for no active positions
        effective_cvar_weight = cvar_weight

    calmar_adj_sharpe = np.clip(calmar_adj_sharpe, -3.0, 3.0)
    score = (da_weight * da
             + ic_weight * ic
             + sharpe_weight * (calmar_adj_sharpe / 3.0)
             - effective_cvar_weight * cvar_penalty)
    return float(score)


def tune_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    weights: pd.Series,
    splitter: PurgedTimeSeriesSplit,
    cfg,
    price_returns: np.ndarray | None = None,
    warm_start_params: dict | None = None,
    t1: np.ndarray | None = None,
) -> dict:
    n_trials = int(cfg.model.optuna_n_trials)
    patience = int(cfg.model.optuna_patience)
    early_stop_rounds = int(cfg.model.xgb_early_stopping_rounds)

    # Use actual realized price returns if provided; otherwise fall back to binary proxy
    if price_returns is not None and len(price_returns) == len(y_train):
        returns_array = price_returns
    else:
        returns_array = y_train.map({0: -1.0, 1: 1.0}).fillna(0.0).values

    best_scores = []

    def objective(trial):
        params = build_xgb_params(trial, cfg)
        n_est = params.pop("n_estimators")

        fold_scores = []
        for train_idx, val_idx in splitter.split(X_train, y_train, groups=t1):
            X_tr = X_train.iloc[train_idx].values
            y_tr = y_train.iloc[train_idx].values
            w_tr = weights.iloc[train_idx].values if hasattr(weights, "iloc") else weights[train_idx]
            X_v = X_train.iloc[val_idx].values
            y_v = y_train.iloc[val_idx].values
            ret_v = returns_array[val_idx]

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
                score = compute_objective(y_v, proba, ret_v, cfg=cfg)
                fold_scores.append(score)
                # Report intermediate value so HyperbandPruner can kill bad trials early
                trial.report(score, step=len(fold_scores) - 1)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            except optuna.exceptions.TrialPruned:
                raise  # let Optuna handle pruning signal — don't swallow it
            except Exception as e:
                logger.warning(f"Trial {trial.number} fold failed: {e}")
                continue

        if not fold_scores:
            return 0.0
        trial.set_user_attr("fold_scores", fold_scores)
        return float(np.mean(fold_scores))

    def _no_improvement_callback(study, trial):
        # Stop if no improvement in last `patience` trials
        best_scores.append(study.best_value)
        if len(best_scores) >= patience:
            recent = best_scores[-patience:]
            if all(v <= recent[0] for v in recent[1:]):
                study.stop()

    # n_startup_trials: pure random exploration before TPE switches to exploitation.
    # Ensures warm-start doesn't dominate from trial 1 — forces genuine search first.
    n_startup = max(5, n_trials // 4)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=1, n_startup_trials=n_startup),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    if warm_start_params:
        try:
            study.enqueue_trial(warm_start_params)
        except Exception as e:
            logger.warning(f"warm-start enqueue failed: {e}")
    study.optimize(objective, n_trials=n_trials, callbacks=[_no_improvement_callback], show_progress_bar=False)

    best_params = study.best_params
    logger.info(
        f"Optuna best score: {study.best_value:.4f}, params: {best_params} "
        f"[n_estimators={best_params.get('n_estimators','?')} — actual trees determined by early stopping]"
    )
    # Expose per-trial per-fold score matrix for CSCV PBO computation downstream.
    # Each completed trial stores fold_scores as user_attr set inside objective.
    # We re-attach them here so the caller can build the (K_trials × N_folds) matrix.
    trial_fold_scores = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE and "fold_scores" in t.user_attrs:
            trial_fold_scores.append(t.user_attrs["fold_scores"])
    best_params["_trial_fold_scores"] = trial_fold_scores
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

    # Calibrate on val set — method driven by cfg.model.calibration_method
    val_proba_raw = model.predict_proba(
        X_val.values if hasattr(X_val, "values") else X_val
    )[:, 1]
    y_val_arr = y_val.values if hasattr(y_val, "values") else y_val

    cal_method = getattr(getattr(cfg, "model", cfg), "calibration_method", "sigmoid")
    if cal_method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(val_proba_raw, y_val_arr)
    else:
        # Sigmoid / Platt scaling — more robust on small val sets
        # Wrapped so .predict(raw_probs) returns calibrated probabilities (same interface as IsotonicRegression)
        from sklearn.linear_model import LogisticRegression
        _lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        _lr.fit(val_proba_raw.reshape(-1, 1), y_val_arr)
        calibrator = _SigmoidCalibrator(_lr)

    best_iter = model.best_iteration
    # evals_result_ is populated by XGBoost after fit when eval_set is provided
    evals = model.evals_result_ if hasattr(model, "evals_result_") else {}
    val_loss_curve = list(evals.get("validation_0", {}).get("logloss", []))
    best_val_loss = val_loss_curve[best_iter] if val_loss_curve and best_iter < len(val_loss_curve) else float("nan")
    logger.info(f"XGBoost trained: best_iteration={best_iter} val_logloss@best={best_val_loss:.4f}")
    return model, calibrator, val_loss_curve, best_iter


def compute_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    splitter: PurgedTimeSeriesSplit,
    params: dict,
    cfg,
    t1: np.ndarray | None = None,
) -> tuple:
    # OOF predictions via PurgedTimeSeriesSplit — NEVER in-sample
    # t1: barrier end times passed as groups for true purging (not just embargo)
    # Returns: (oof_proba, fold_val_losses, fold_best_iterations)
    oof_proba = np.zeros((len(X), 2))
    oof_proba[:, 0] = 0.5
    oof_proba[:, 1] = 0.5
    early_stop_rounds = int(cfg.model.xgb_early_stopping_rounds)
    fold_val_losses = []
    fold_best_iterations = []

    for train_idx, val_idx in splitter.split(X, y, groups=t1):
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
            # Capture per-fold diagnostics
            best_iter = model.best_iteration
            fold_best_iterations.append(int(best_iter))
            evals = model.evals_result_ if hasattr(model, "evals_result_") else {}
            loss_curve = list(evals.get("validation_0", {}).get("logloss", []))
            best_loss = loss_curve[best_iter] if loss_curve and best_iter < len(loss_curve) else float("nan")
            fold_val_losses.append(float(best_loss))
        except Exception as e:
            logger.warning(f"OOF fold failed: {e}")
            fold_val_losses.append(float("nan"))
            fold_best_iterations.append(0)

    return oof_proba, fold_val_losses, fold_best_iterations


def compute_shap_importance(model, X_train: pd.DataFrame, top_k: int = 20) -> pd.Series:
    explainer = shap.TreeExplainer(model)
    # Sample from the TAIL (most recent bars) — regime stationarity assumption means
    # recent data better represents the distribution the final model will face at inference.
    n_sample = min(len(X_train), 2000)
    X_sample = X_train.iloc[-n_sample:] if hasattr(X_train, "iloc") else X_train[-n_sample:]
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
