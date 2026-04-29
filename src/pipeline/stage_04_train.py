import os
from pathlib import Path
import numpy as np
import pandas as pd
import json
import pickle
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import log_loss

# --- bootstrap imports so `from src...` works when run as a script ---
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../crypto_model
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# --------------------------------------------------------------------

from src.utils.config_loader import get_symbols
from src.utils.state_manager import is_stage_complete, update_project_state, update_completed_symbol
from src.utils.logger import get_logger
from src.utils.io_utils import read_features, write_pipeline_diagnostics
from src.models.splitter import PurgedTimeSeriesSplit, compute_fold_consistency, compute_pbo_cscv
from src.models.imputer import fit_imputer, transform_with_imputer, fit_robust_scaler, transform_with_scaler
from src.models.stability_selection import select_features_pipeline
from src.models.primary_model import (
    tune_hyperparams, train_xgb,
    compute_oof_predictions, compute_shap_importance,
    compute_conformal_q90, save_model,
)
from src.models.model_versioning import generate_version_string, register_model, get_latest_model
from src.visualization.training_diagnostics import generate_all_diagnostics

logger = get_logger("stage_04_train")

_TF = "15m"


def _to_float64_array(X) -> np.ndarray:
    """
    Convert input to float64 numpy array.
    Non-numeric values are coerced to NaN (so they can be imputed).
    """
    if isinstance(X, pd.DataFrame):
        return X.apply(pd.to_numeric, errors="coerce").astype(np.float64).values
    return np.array(X, dtype=np.float64)


# -------------------------


def _load_labels_and_weights(symbol: str, labels_dir: Path) -> tuple:
    labels_path = labels_dir / f"{symbol}_{_TF}_labels.parquet"
    weights_path = labels_dir / f"{symbol}_{_TF}_weights.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")
    labels_df = pd.read_parquet(labels_path)
    weights_df = pd.read_parquet(weights_path) if weights_path.exists() else None
    weights = weights_df["sample_weight"] if weights_df is not None else pd.Series(1.0, index=labels_df.index)
    return labels_df, weights


def _assign_tier(metrics: dict, cfg) -> str:
    edge_val = metrics.get("edge_val", 0.0)
    pct_positive_val = metrics.get("pct_positive_val", 0.5)
    sharpe = metrics.get("synthetic_sharpe", 0.0)
    pbo = metrics.get("pbo", 1.0)
    overfit_ratio = metrics.get("overfit_ratio", float("nan"))
    fold_da_list = metrics.get("fold_da_list", [])

    tier_a_edge = float(cfg.model.tier_A_edge_min)
    tier_a_sharpe = float(cfg.model.tier_A_sharpe_wfo_min)
    tier_a_pbo = float(cfg.model.tier_A_pbo_max)
    tier_a_overfit_max = float(cfg.model.tier_A_overfit_ratio_max)
    tier_a_edge_min_folds = int(cfg.model.tier_A_edge_min_folds)
    tier_b_edge = float(cfg.model.tier_B_edge_min)
    hard_reject_baseline = float(cfg.model.tier_reject_baseline)

    if pct_positive_val > hard_reject_baseline and edge_val < tier_b_edge:
        return "C"

    fold_baseline = max(pct_positive_val, 1.0 - pct_positive_val)
    folds_above_edge = sum(1 for fda in fold_da_list if (fda - fold_baseline) >= tier_a_edge)
    passes_fold_edge = (folds_above_edge >= tier_a_edge_min_folds) if fold_da_list else False

    overfit_ok = (not np.isnan(overfit_ratio)) and (overfit_ratio <= tier_a_overfit_max)

    if (edge_val >= tier_a_edge and sharpe >= tier_a_sharpe and pbo <= tier_a_pbo
            and passes_fold_edge and overfit_ok):
        return "A"
    elif edge_val >= tier_b_edge:
        return "B"
    else:
        return "C"


def _train_symbol(symbol: str, cfg, checkpoints_dir: Path, labels_dir: Path, features_dir: Path, models_dir: Path) -> tuple:
    logger.info(f"Training {symbol}")
    train_end = pd.Timestamp(cfg.data.train_end, tz="UTC")
    val_start = pd.Timestamp(cfg.data.val_start, tz="UTC")
    val_end = pd.Timestamp(cfg.data.val_end, tz="UTC")

    try:
        features_df = read_features(symbol, _TF, features_dir)
    except FileNotFoundError as e:
        return symbol, None, str(e)

    if features_df.columns.duplicated().any():
        features_df = features_df.loc[:, ~features_df.columns.duplicated()]
        logger.warning(f"{symbol}: removed duplicate feature columns from parquet")

    try:
        labels_df, sample_weights = _load_labels_and_weights(symbol, labels_dir)
    except Exception as e:
        return symbol, None, str(e)

    common_idx = features_df.index.intersection(labels_df.index)
    if len(common_idx) < 500:
        return symbol, None, f"Insufficient overlapping bars: {len(common_idx)}"

    features_aligned = features_df.loc[common_idx]
    labels_aligned = labels_df.loc[common_idx]
    weights_aligned = sample_weights.reindex(common_idx).fillna(1.0)

    train_mask = features_aligned.index <= train_end
    val_mask = (features_aligned.index > train_end) & (features_aligned.index <= val_end)

    X_train = features_aligned[train_mask].drop(columns=["is_warmup"], errors="ignore")
    w_train = weights_aligned[train_mask]

    if "is_warmup" in features_aligned.columns:
        warmup_mask_train = features_aligned.loc[train_mask, "is_warmup"] == 1
        X_train = X_train[~warmup_mask_train]
        w_train = w_train[~warmup_mask_train]
        train_label_series = labels_aligned.loc[train_mask, "label"][~warmup_mask_train]
    else:
        train_label_series = labels_aligned.loc[train_mask, "label"]

    train_dir_mask = train_label_series != 0
    X_train = X_train[train_dir_mask]
    w_train = w_train[train_dir_mask]
    y_train_binary = (train_label_series[train_dir_mask] == 1).astype(int)

    X_val = features_aligned[val_mask].drop(columns=["is_warmup"], errors="ignore")
    val_label_series = labels_aligned.loc[val_mask, "label"]
    val_dir_mask = val_label_series != 0
    X_val = X_val[val_dir_mask]
    y_val_binary = (val_label_series[val_dir_mask] == 1).astype(int)

    n_long = int(y_train_binary.sum())
    n_short = int((y_train_binary == 0).sum())
    pct_long = n_long / max(len(y_train_binary), 1)
    logger.info(f"  {symbol}: directional label balance â€” n_long={n_long}, n_short={n_short}, pct_long={pct_long:.3f}")
    scale_pos_weight = max(n_short / max(n_long, 1), 0.05)

    if len(X_train) < 200:
        return symbol, None, f"Insufficient train samples after filtering: {len(X_train)}"

    X_train = X_train.select_dtypes(include=[np.number])
    X_val = X_val.select_dtypes(include=[np.number])

    shared_cols = X_train.columns.intersection(X_val.columns).tolist()
    X_train = X_train[shared_cols]
    X_val = X_val[shared_cols]

    assert X_train.index.max() < val_start, f"Train data bleeds into val for {symbol}"

    splitter = PurgedTimeSeriesSplit(
        n_splits=int(cfg.model.cv_n_splits),
        embargo_pct=float(cfg.model.embargo_pct),
        embargo_bars_min=int(cfg.model.embargo_bars_min),
        val_start_date=str(cfg.data.val_start),
    )

    imp_dir = checkpoints_dir / "imputers"
    imp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  {symbol}: enforcing numeric dtypes before impute")

    _non_numeric_cols = X_train.select_dtypes(
        exclude=[np.floating, np.integer, np.bool_]
    ).columns.tolist()

    if _non_numeric_cols:
        logger.warning(
            f"  {symbol}: dropping {len(_non_numeric_cols)} non-numeric train cols "
            f"before impute: {_non_numeric_cols[:15]}"
            + (" ..." if len(_non_numeric_cols) > 15 else "")
        )
        X_train = X_train.drop(columns=_non_numeric_cols)

    X_train = X_train.apply(pd.to_numeric, errors="coerce").astype(np.float64)

    if _non_numeric_cols:
        _val_drop = [c for c in _non_numeric_cols if c in X_val.columns]
        if _val_drop:
            X_val = X_val.drop(columns=_val_drop)

    X_val = X_val.apply(pd.to_numeric, errors="coerce").astype(np.float64)

    logger.info(f"  {symbol}: fitting imputer and scaler")
    try:
        fit_imputer(X_train.values, symbol, _TF, imp_dir, cfg)
        imputer_data = transform_with_imputer(X_train.values.copy(), symbol, _TF, imp_dir)
        X_train_imp_all = imputer_data
        X_val_imp_all = transform_with_imputer(X_val.values.copy(), symbol, _TF, imp_dir)
    except Exception as e:
        logger.warning(f"  {symbol}: imputer failed ({e}), using median-filled values")
        X_train_imp_all = X_train.values.copy()
        col_medians = np.nanmedian(X_train_imp_all, axis=0)
        nan_mask = np.isnan(X_train_imp_all)
        X_train_imp_all[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

        X_val_imp_all = X_val.values.copy()
        nan_mask_v = np.isnan(X_val_imp_all)
        X_val_imp_all[nan_mask_v] = np.take(col_medians, np.where(nan_mask_v)[1])

    X_train_imp_df = pd.DataFrame(X_train_imp_all[:, :len(X_train.columns)],
                                  index=X_train.index, columns=X_train.columns)
    X_val_imp_df = pd.DataFrame(X_val_imp_all[:, :len(X_val.columns)],
                                index=X_val.index, columns=X_val.columns)

    logger.info(f"  {symbol}: running feature selection")
    try:
        selected_features = select_features_pipeline(X_train_imp_df, y_train_binary, w_train, str(cfg.data.val_start), cfg)
    except Exception as e:
        logger.warning(f"  {symbol}: feature selection failed ({e}), using all features")
        selected_features = X_train.columns.tolist()

    sel_idx = [X_train_imp_df.columns.get_loc(c) for c in selected_features if c in X_train_imp_df.columns]
    X_train_imp = X_train_imp_all[:, sel_idx]
    X_val_imp = X_val_imp_all[:, sel_idx]

    X_train_sel = X_train_imp_df[selected_features]
    X_val_sel = X_val_imp_df[selected_features]

    try:
        fit_imputer(X_train[selected_features].values, symbol, _TF, imp_dir, cfg)
    except Exception as e:
        logger.warning(f"  {symbol}: imputer artifact save failed ({e})")

    try:
        fit_robust_scaler(X_train_imp, symbol, _TF, imp_dir)
        X_train_scaled = transform_with_scaler(X_train_imp, symbol, _TF, imp_dir)
        X_val_scaled = transform_with_scaler(X_val_imp, symbol, _TF, imp_dir)
    except Exception as e:
        logger.warning(f"  {symbol}: scaler failed ({e}), using imputed values")
        X_train_scaled = X_train_imp
        X_val_scaled = X_val_imp

    # Build column names â€” imputer may append missing-indicator flags
    n_indicator = X_train_imp.shape[1] - len(selected_features)
    if n_indicator > 0:
        all_col_names = selected_features + [f"missing_flag_{i}" for i in range(n_indicator)]
    else:
        all_col_names = selected_features

    X_train_final = pd.DataFrame(X_train_scaled, index=X_train_sel.index, columns=all_col_names)
    X_val_final = pd.DataFrame(X_val_scaled, index=X_val_sel.index, columns=all_col_names)

    _labels_for_ret = labels_aligned.loc[train_mask].reindex(X_train_final.index)
    _raw_lbl = _labels_for_ret["label"].values
    _tp = _labels_for_ret["tp_level"].fillna(0.0).values if "tp_level" in _labels_for_ret.columns else np.zeros(len(X_train_final))
    _sl = _labels_for_ret["sl_level"].fillna(0.0).values if "sl_level" in _labels_for_ret.columns else np.zeros(len(X_train_final))
    price_returns = np.where(_raw_lbl == 1, _tp, np.where(_raw_lbl == -1, -_sl, 0.0))

    t1_series = None
    if "t1" in _labels_for_ret.columns:
        t1_series = _labels_for_ret["t1"].values

    prior_params = {}
    try:
        prior_entry = get_latest_model(symbol, _TF, model_type="primary")
        if prior_entry and prior_entry.get("hyperparams"):
            prior_params = {k: v for k, v in prior_entry["hyperparams"].items()
                            if k not in ("scale_pos_weight",)}
            logger.info(f"  {symbol}: warm-starting Optuna from prior params")
    except Exception as e:
        logger.warning(f"  {symbol}: warm-start lookup failed: {e}")

    logger.info(f"  {symbol}: tuning hyperparameters ({cfg.model.optuna_n_trials} trials)")
    try:
        best_params = tune_hyperparams(
            X_train_final, y_train_binary, w_train, splitter, cfg,
            price_returns=price_returns,
            warm_start_params=prior_params if prior_params else None,
            t1=t1_series,
        )
    except Exception as e:
        logger.warning(f"  {symbol}: Optuna failed ({e}), using defaults")
        best_params = {}

    trial_fold_scores = best_params.pop("_trial_fold_scores", [])
    best_params["scale_pos_weight"] = scale_pos_weight

    logger.info(f"  {symbol}: computing OOF predictions")
    fold_val_losses = []
    fold_best_iterations = []
    try:
        oof_proba, fold_val_losses, fold_best_iterations = compute_oof_predictions(
            X_train_final, y_train_binary, w_train, splitter, best_params, cfg, t1=t1_series
        )
    except Exception as e:
        logger.warning(f"  {symbol}: OOF failed ({e})")
        oof_proba = np.full((len(X_train_final), 2), 0.5)

    logger.info(f"  {symbol}: training final model")
    try:
        model, calibrator, val_loss_curve, best_iteration = train_xgb(
            X_train_final, y_train_binary,
            X_val_final, y_val_binary,
            w_train, best_params, cfg
        )
    except Exception as e:
        return symbol, None, f"Training failed: {e}"

    val_proba_raw = model.predict_proba(X_val_scaled)[:, 1]
    val_proba_cal = calibrator.predict(val_proba_raw)
    val_proba_2d = np.column_stack([1 - val_proba_cal, val_proba_cal])
    q90 = compute_conformal_q90(y_val_binary.values, val_proba_2d)

    val_pred_dir = (val_proba_cal > 0.5).astype(int)
    da = float(np.mean(val_pred_dir == y_val_binary.values))
    try:
        ic = float(spearmanr(val_proba_cal, y_val_binary.values).correlation)
    except Exception:
        ic = 0.0

    val_mse = float(np.mean((val_proba_cal - y_val_binary.values) ** 2))

    pct_positive_train = float(y_train_binary.mean())
    pct_positive_val = float(y_val_binary.mean())
    logger.info(
        f"  {symbol}: class balance â€” train_pos={pct_positive_train:.3f} "
        f"val_pos={pct_positive_val:.3f}"
    )

    naive_baseline_val = max(pct_positive_val, 1.0 - pct_positive_val)
    edge_val = da - naive_baseline_val
    val_proba_for_edge = val_proba_cal
    high_conf_mask = (val_proba_for_edge > 0.70) | (val_proba_for_edge < 0.30)
    if high_conf_mask.sum() > 10:
        hc_pred = (val_proba_for_edge[high_conf_mask] > 0.5).astype(int)
        hc_actual = y_val_binary.values[high_conf_mask]
        hc_da = float(np.mean(hc_pred == hc_actual))
        hc_pct = high_conf_mask.mean()
        logger.info(
            f"  {symbol}: naive={naive_baseline_val:.3f} edge={edge_val:+.3f} | "
            f"high_conf_DA={hc_da:.3f} on {high_conf_mask.sum()} samples ({hc_pct:.1%} of val)"
        )
    else:
        logger.info(f"  {symbol}: naive={naive_baseline_val:.3f} edge={edge_val:+.3f}")

    oof_long_proba = oof_proba[:, 1]
    logger.info(
        f"  {symbol}: OOF prob dist â€” mean={oof_long_proba.mean():.3f} "
        f"std={oof_long_proba.std():.3f} "
        f"q05={np.percentile(oof_long_proba, 5):.3f} "
        f"q95={np.percentile(oof_long_proba, 95):.3f}"
    )

    cost = float(cfg.labels.get("round_trip_cost_pct", 0.003))
    max_hold_bars = float(cfg.labels.get("max_hold_bars", 32))
    if "bars_to_exit" in _labels_for_ret.columns:
        actual_avg_hold = float(_labels_for_ret["bars_to_exit"].dropna().mean())
        actual_avg_hold = max(actual_avg_hold, 1.0)
    elif "t1" in _labels_for_ret.columns and _labels_for_ret.index.dtype == "datetime64[ns, UTC]":
        bar_duration_hours = 0.25
        t0 = _labels_for_ret.index.to_series()
        t1_col = pd.to_datetime(_labels_for_ret["t1"], utc=True)
        hold_hours = (t1_col - t0).dt.total_seconds() / 3600.0
        actual_avg_hold = float(hold_hours.dropna().mean() / bar_duration_hours)
        actual_avg_hold = max(actual_avg_hold, 1.0)
    else:
        actual_avg_hold = max_hold_bars / 2.0
    trades_per_year = 252.0 * 96.0 / actual_avg_hold

    fold_sharpes = []
    fold_da_list = []
    for _, val_idx in splitter.split(X_train_final, y_train_binary, groups=t1_series):
        fold_proba = oof_proba[val_idx, 1]
        fold_y = y_train_binary.iloc[val_idx].values
        fold_price_ret = price_returns[val_idx]
        positions = np.where(fold_proba > 0.5, 1.0, -1.0)
        active = positions * fold_price_ret
        if len(active) > 1 and active.std() > 1e-12:
            active_net = active - cost
            if active_net.std() > 1e-12:
                sharpe = float(active_net.mean() / active_net.std() * np.sqrt(trades_per_year))
            else:
                sharpe = 0.0
            fold_sharpes.append(sharpe)
        fold_da = float(np.mean((fold_proba > 0.5).astype(int) == fold_y)) if len(fold_y) > 0 else 0.0
        fold_da_list.append(fold_da)

    fold_consistency = compute_fold_consistency(fold_sharpes)
    pbo = compute_pbo_cscv(trial_fold_scores) if len(trial_fold_scores) >= 2 else 0.5
    synthetic_sharpe = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0

    train_loss_at_best = float("nan")
    val_loss_at_best = val_loss_curve[best_iteration] if val_loss_curve and best_iteration < len(val_loss_curve) else float("nan")
    try:
        train_loss_at_best = float(log_loss(y_train_binary.values, oof_proba[:, 1]))
    except Exception:
        pass
    overfit_ratio = float(train_loss_at_best / val_loss_at_best) if not (np.isnan(train_loss_at_best) or np.isnan(val_loss_at_best) or val_loss_at_best == 0) else float("nan")
    fold_da_std = float(np.std(fold_da_list)) if fold_da_list else float("nan")

    metrics = {
        "da": da,
        "ic": ic,
        "mse": val_mse,
        "pbo": pbo,
        "fold_consistency": fold_consistency,
        "synthetic_sharpe": synthetic_sharpe,
        "conformal_q90": q90,
        "n_train_samples": len(X_train_final),
        "n_val_samples": len(X_val_final),
        "n_features": len(all_col_names),
        "fold_quality_score": round(fold_consistency * max(synthetic_sharpe, 0.0), 4),
        "pct_positive_train": pct_positive_train,
        "pct_positive_val": pct_positive_val,
        "edge_val": round(edge_val, 4),
        "naive_baseline_val": round(naive_baseline_val, 4),
        "fold_da_list": fold_da_list,
        "best_iteration": int(best_iteration),
        "overfit_ratio": overfit_ratio,
        "fold_da_std": fold_da_std,
        "val_logloss": val_loss_at_best,
    }

    tier = _assign_tier(metrics, cfg)
    logger.info(
        f"  {symbol}: DA={da:.3f} MSE={val_mse:.4f} Sharpe(per-trade)={synthetic_sharpe:.3f} "
        f"PBO={pbo:.3f} fold_consistency={fold_consistency:.2f} "
        f"Tier={tier} edge_val={metrics['edge_val']:+.3f} "
        f"folds_above_edge={sum(1 for d in fold_da_list if (d - metrics['naive_baseline_val']) >= float(cfg.model.tier_A_edge_min))}"
    )

    try:
        diagnostics_dir = checkpoints_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = {
            "symbol": symbol,
            "version": "",
            "val_logloss_curve": val_loss_curve,
            "best_iteration": int(best_iteration),
            "fold_val_losses": fold_val_losses,
            "fold_best_iterations": fold_best_iterations,
            "fold_da_list": fold_da_list,
            "overfit_ratio": overfit_ratio,
            "fold_da_std": fold_da_std,
            "val_logloss_best": val_loss_at_best,
            "train_logloss_oof": train_loss_at_best,
        }
        diag_path = diagnostics_dir / f"{symbol}_{_TF}_train_diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f)
    except Exception as e:
        logger.warning(f"  {symbol}: diagnostics save failed: {e}")

    skip_shap = os.environ.get("SKIP_SHAP", "0") == "1"
    if not skip_shap:
        try:
            shap_imp = compute_shap_importance(model, X_train_final, top_k=int(cfg.model.shap_top_k))
            shap_path = models_dir / f"{symbol}_{_TF}_shap_importance.json"
            shap_imp.to_json(shap_path)
        except Exception as e:
            logger.warning(f"  {symbol}: SHAP failed: {e}")
    else:
        logger.info(f"  {symbol}: SHAP skipped (SKIP_SHAP=1)")

    train_start_str = str(X_train_final.index.min().date())
    train_end_str = str(X_train_final.index.max().date())
    version = generate_version_string(symbol, _TF, all_col_names, best_params, train_start_str, train_end_str)

    save_model(model, calibrator, symbol, _TF, version, models_dir)

    try:
        diag_path = checkpoints_dir / "diagnostics" / f"{symbol}_{_TF}_train_diagnostics.json"
        if diag_path.exists():
            with open(diag_path) as f:
                diag_data = json.load(f)
            diag_data["version"] = version
            with open(diag_path, "w") as f:
                json.dump(diag_data, f)
    except Exception:
        pass

    register_model(
        symbol=symbol,
        tf=_TF,
        version=version,
        metrics={**metrics, "tier": tier},
        feature_names=all_col_names,
        hyperparams=best_params,
        train_period=(train_start_str, train_end_str),
        model_path=str(models_dir / f"{version}_model.json"),
        model_type="primary",
        cfg=cfg,
    )

    oof_dir = checkpoints_dir / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    oof_path = oof_dir / f"{symbol}_{_TF}_oof_proba.npy"
    oof_idx_path = oof_dir / f"{symbol}_{_TF}_oof_index.npy"
    np.save(str(oof_path), oof_proba)
    np.save(str(oof_idx_path), X_train_final.index.view("int64"))

    features_sel_path = checkpoints_dir / "feature_selection" / f"{symbol}_{_TF}_selected.json"
    features_sel_path.parent.mkdir(parents=True, exist_ok=True)
    with open(features_sel_path, "w") as f:
        json.dump(all_col_names, f)

    try:
        results_dir = Path(cfg.data.results_dir)
        generate_all_diagnostics(
            symbol=symbol,
            val_loss_curve=val_loss_curve,
            best_iteration=best_iteration,
            fold_da_list=fold_da_list,
            fold_val_losses=fold_val_losses,
            val_proba_cal=val_proba_cal,
            y_val=y_val_binary.values,
            overfit_ratio=overfit_ratio,
            output_dir=results_dir / "diagnostics",
        )
    except Exception as e:
        logger.warning(f"  {symbol}: diagnostic plots failed (non-fatal): {e}")

    try:
        fold_sharpe_std = float(np.std(fold_sharpes)) if fold_sharpes else float("nan")
        fold_sharpe_min = float(np.min(fold_sharpes)) if fold_sharpes else float("nan")
        shap_path = models_dir / f"{symbol}_{_TF}_shap_importance.json"
        top_features_str = ""
        if shap_path.exists():
            try:
                shap_series = pd.read_json(shap_path, typ="series")
                top_features_str = ",".join(shap_series.nlargest(5).index.tolist())
            except Exception:
                pass
        diag_row = {
            "symbol": symbol,
            "stage": "train",
            "tier": tier,
            "da_val": round(da, 4),
            "pertrade_sharpe_mean": round(synthetic_sharpe, 4),
            "fold_sharpe_std": round(fold_sharpe_std, 4) if not np.isnan(fold_sharpe_std) else None,
            "fold_sharpe_min": round(fold_sharpe_min, 4) if not np.isnan(fold_sharpe_min) else None,
            "pbo": round(pbo, 4),
            "overfit_ratio": round(overfit_ratio, 4) if not np.isnan(overfit_ratio) else None,
            "fold_da_std": round(fold_da_std, 4) if not np.isnan(fold_da_std) else None,
            "n_features": len(all_col_names),
            "n_train": len(X_train_final),
            "pct_positive_train": round(pct_positive_train, 4),
            "edge_val": round(edge_val, 4),
            "naive_baseline_val": round(naive_baseline_val, 4),
            "top5_shap_features": top_features_str,
        }
        results_dir = Path(cfg.data.results_dir) if hasattr(cfg.data, "results_dir") else Path("results")
        write_pipeline_diagnostics([diag_row], results_dir)
    except Exception as _diag_exc:
        logger.warning(f"  {symbol}: diagnostics write failed (non-fatal): {_diag_exc}")

    return symbol, {"version": version, "tier": tier, "metrics": metrics}, None


def run(cfg, force: bool = False, symbol_filter: str = None) -> None:
    if not force and is_stage_complete("training"):
        logger.info("Stage 4 already complete, skipping.")
        return

    all_symbols = get_symbols(cfg)
    if symbol_filter:
        _sf = set(symbol_filter) if isinstance(symbol_filter, list) else {symbol_filter}
        all_symbols = [s for s in all_symbols if s.get("name", s.get("symbol")) in _sf]
    symbol_names = [s.get("name", s.get("symbol")) for s in all_symbols]

    if not force:
        from src.utils.state_manager import load_state
        _state = load_state()
        _completed = set(_state.get("stages", {}).get("training", {}).get("completed_symbols", []))
        if _completed:
            _pending = [s for s in symbol_names if s not in _completed]
            logger.info(f"Resuming training â€” {len(_completed)} already done, {len(_pending)} remaining")
            symbol_names = _pending

    checkpoints_dir = Path(cfg.data.checkpoints_dir)
    labels_dir = Path(cfg.data.labels_dir)
    features_dir = Path(cfg.data.features_dir)
    models_dir = Path(cfg.data.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    issues = []
    training_summary = []

    for symbol in tqdm(symbol_names, desc="stage_04", unit="sym"):
        sym, result, err = _train_symbol(symbol, cfg, checkpoints_dir, labels_dir, features_dir, models_dir)
        if err:
            logger.error(f"{sym}: training failed â€” {err}")
            issues.append(f"{sym}: {str(err)[:200]}")
        else:
            update_completed_symbol("training", sym)
            csv_metrics = {k: v for k, v in result["metrics"].items() if k != "fold_da_list"}
            training_summary.append({
                "symbol": sym,
                "version": result["version"],
                "tier": result["tier"],
                **csv_metrics,
            })
            logger.info(f"{sym}: training complete â€” Tier {result['tier']}")

    if training_summary:
        summary_df = pd.DataFrame(training_summary)
        results_dir = Path(cfg.data.results_dir) if hasattr(cfg.data, "results_dir") else Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        summary_path_results = results_dir / "training_summary.csv"
        summary_df.to_csv(summary_path_results, index=False)
        summary_path_models = Path(cfg.data.models_dir) / "training_summary.csv"
        summary_df.to_csv(summary_path_models, index=False)
        logger.info(f"Training summary saved: {summary_path_results} + {summary_path_models}")

    update_project_state("training", "done", issues, output_dir=str(models_dir))
    logger.info(f"Stage 4 complete. {len(training_summary)}/{len(symbol_names)} symbols trained.")
