from pathlib import Path
import numpy as np
import pandas as pd
import json
from src.utils.config_loader import get_symbols
from src.utils.state_manager import is_stage_complete, update_project_state, update_completed_symbol
from src.utils.logger import get_logger
from src.utils.io_utils import read_features
from src.models.splitter import PurgedTimeSeriesSplit, compute_pbo
from src.models.imputer import fit_imputer, transform_with_imputer, fit_robust_scaler, transform_with_scaler
from src.models.stability_selection import select_features_pipeline
from src.models.primary_model import (
    build_xgb_params, tune_hyperparams, train_xgb,
    compute_oof_predictions, compute_shap_importance,
    compute_conformal_q90, save_model,
)
from src.models.model_versioning import generate_version_string, register_model

logger = get_logger("stage_04_train")

_TF = "15m"


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
    da = metrics.get("da", 0.0)
    sharpe = metrics.get("synthetic_sharpe", 0.0)
    pbo = metrics.get("pbo", 1.0)
    dsr = metrics.get("dsr", 0.0)

    tier_a_da = float(cfg.model.tier_A_da_min)
    tier_a_sharpe = float(cfg.model.tier_A_sharpe_wfo_min)
    tier_a_pbo = float(cfg.model.tier_A_pbo_max)
    tier_b_da = float(cfg.model.tier_B_da_min)

    if da >= tier_a_da and sharpe >= tier_a_sharpe and pbo <= tier_a_pbo:
        return "A"
    elif da >= tier_b_da:
        return "B"
    else:
        return "C"


def _train_symbol(symbol: str, cfg, checkpoints_dir: Path, labels_dir: Path, features_dir: Path, models_dir: Path) -> tuple:
    logger.info(f"Training {symbol}")
    train_end = pd.Timestamp(cfg.data.train_end, tz="UTC")
    val_start = pd.Timestamp(cfg.data.val_start, tz="UTC")
    val_end = pd.Timestamp(cfg.data.val_end, tz="UTC")

    # Load features
    try:
        features_df = read_features(symbol, _TF, features_dir)
    except FileNotFoundError as e:
        return symbol, None, str(e)

    # Load labels and weights
    try:
        labels_df, sample_weights = _load_labels_and_weights(symbol, labels_dir)
    except Exception as e:
        return symbol, None, str(e)

    # Align features and labels on common index
    common_idx = features_df.index.intersection(labels_df.index)
    if len(common_idx) < 500:
        return symbol, None, f"Insufficient overlapping bars: {len(common_idx)}"

    features_aligned = features_df.loc[common_idx]
    labels_aligned = labels_df.loc[common_idx]
    weights_aligned = sample_weights.reindex(common_idx).fillna(1.0)

    # Split into train / val / test
    train_mask = features_aligned.index <= train_end
    val_mask = (features_aligned.index > train_end) & (features_aligned.index <= val_end)

    X_train = features_aligned[train_mask].drop(columns=["is_warmup"], errors="ignore")
    y_train = (labels_aligned.loc[train_mask, "label"] + 1) // 2  # convert {-1,0,1} → {0,0,1} binary
    # For binary XGB: 1 = long (label=+1), 0 = short/neutral
    y_train_binary = (labels_aligned.loc[train_mask, "label"] == 1).astype(int)
    w_train = weights_aligned[train_mask]

    X_val = features_aligned[val_mask].drop(columns=["is_warmup"], errors="ignore")
    y_val_binary = (labels_aligned.loc[val_mask, "label"] == 1).astype(int)

    # Filter out warmup rows
    if "is_warmup" in features_aligned.columns:
        warmup_mask_train = features_aligned.loc[train_mask, "is_warmup"] == 1
        X_train = X_train[~warmup_mask_train]
        y_train_binary = y_train_binary[~warmup_mask_train]
        w_train = w_train[~warmup_mask_train]

    if len(X_train) < 200:
        return symbol, None, f"Insufficient train samples after filtering: {len(X_train)}"

    # Drop non-numeric columns
    X_train = X_train.select_dtypes(include=[np.number]).fillna(0)
    X_val = X_val.select_dtypes(include=[np.number]).fillna(0)

    # Ensure same columns
    shared_cols = X_train.columns.intersection(X_val.columns).tolist()
    X_train = X_train[shared_cols]
    X_val = X_val[shared_cols]

    # LEAKAGE GUARD: stability selection only on train data
    assert X_train.index.max() < val_start, f"Train data bleeds into val for {symbol}"

    splitter = PurgedTimeSeriesSplit(
        n_splits=int(cfg.model.cv_n_splits),
        embargo_pct=float(cfg.model.embargo_pct),
        embargo_bars_min=int(cfg.model.embargo_bars_min),
        val_start_date=str(cfg.data.val_start),
    )

    # Step 1: Feature selection on train data only
    logger.info(f"  {symbol}: running feature selection")
    try:
        selected_features = select_features_pipeline(X_train, y_train_binary, w_train, str(cfg.data.val_start), cfg)
    except Exception as e:
        logger.warning(f"  {symbol}: feature selection failed ({e}), using all features")
        selected_features = X_train.columns.tolist()

    X_train_sel = X_train[selected_features]
    X_val_sel = X_val[selected_features] if all(c in X_val.columns for c in selected_features) else X_val

    # Step 2: Fit imputer and scaler on train data only
    logger.info(f"  {symbol}: fitting imputer and scaler")
    imp_dir = checkpoints_dir / "imputers"
    try:
        fit_imputer(X_train_sel.values, symbol, _TF, imp_dir, cfg)
        X_train_imp = transform_with_imputer(X_train_sel.values, symbol, _TF, imp_dir)
        X_val_imp = transform_with_imputer(X_val_sel.values, symbol, _TF, imp_dir)
    except Exception as e:
        logger.warning(f"  {symbol}: imputer failed ({e}), using raw values")
        X_train_imp = X_train_sel.values
        X_val_imp = X_val_sel.values

    try:
        fit_robust_scaler(X_train_imp, symbol, _TF, imp_dir)
        X_train_scaled = transform_with_scaler(X_train_imp, symbol, _TF, imp_dir)
        X_val_scaled = transform_with_scaler(X_val_imp, symbol, _TF, imp_dir)
    except Exception as e:
        logger.warning(f"  {symbol}: scaler failed ({e}), using imputed values")
        X_train_scaled = X_train_imp
        X_val_scaled = X_val_imp

    # Build DataFrames back from scaled arrays for splitter (needs index)
    X_train_final = pd.DataFrame(X_train_scaled, index=X_train_sel.index, columns=selected_features)
    X_val_final = pd.DataFrame(X_val_scaled, index=X_val_sel.index, columns=X_val_sel.columns[:len(selected_features)])

    # Step 3: Hyperparameter tuning with Optuna
    logger.info(f"  {symbol}: tuning hyperparameters ({cfg.model.optuna_n_trials} trials)")
    try:
        best_params = tune_hyperparams(X_train_final, y_train_binary, w_train, splitter, cfg)
    except Exception as e:
        logger.warning(f"  {symbol}: Optuna failed ({e}), using defaults")
        best_params = {}

    # Step 4: OOF predictions for meta-labeling (NEVER in-sample)
    logger.info(f"  {symbol}: computing OOF predictions")
    try:
        oof_proba = compute_oof_predictions(X_train_final, y_train_binary, w_train, splitter, best_params, cfg)
    except Exception as e:
        logger.warning(f"  {symbol}: OOF failed ({e})")
        oof_proba = np.full((len(X_train_final), 2), 0.5)

    # Step 5: Train final model on all train data
    logger.info(f"  {symbol}: training final model")
    try:
        model, calibrator = train_xgb(
            X_train_final, y_train_binary,
            X_val_final, y_val_binary,
            w_train, best_params, cfg
        )
    except Exception as e:
        return symbol, None, f"Training failed: {e}"

    # Step 6: Calibrated val predictions + conformal threshold
    val_proba_raw = model.predict_proba(X_val_scaled)[:, 1]
    val_proba_cal = calibrator.predict(val_proba_raw)
    val_proba_2d = np.column_stack([1 - val_proba_cal, val_proba_cal])
    q90 = compute_conformal_q90(y_val_binary.values, val_proba_2d)

    # Step 7: Compute metrics
    from scipy.stats import spearmanr
    val_pred_dir = (val_proba_cal > 0.5).astype(int)
    da = float(np.mean(val_pred_dir == y_val_binary.values))
    try:
        ic = float(spearmanr(val_proba_cal, y_val_binary.values).correlation)
    except Exception:
        ic = 0.0

    # Compute fold Sharpes from OOF predictions
    fold_sharpes = []
    for _, val_idx in splitter.split(X_train_final, y_train_binary):
        fold_proba = oof_proba[val_idx, 1]
        fold_y = y_train_binary.iloc[val_idx].values
        positions = np.where(fold_proba > 0.5, 1.0, -1.0)
        # Approximate returns from labels
        fold_returns = positions * (fold_y * 2 - 1).astype(float)
        std_ret = fold_returns.std()
        if std_ret > 0:
            sharpe = float(fold_returns.mean() / std_ret * np.sqrt(252 * 96))
            fold_sharpes.append(sharpe)

    pbo = compute_pbo(fold_sharpes)
    synthetic_sharpe = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0

    metrics = {
        "da": da,
        "ic": ic,
        "pbo": pbo,
        "synthetic_sharpe": synthetic_sharpe,
        "conformal_q90": q90,
        "n_train_samples": len(X_train_final),
        "n_val_samples": len(X_val_final),
        "n_features": len(selected_features),
        "dsr": max(synthetic_sharpe - pbo, 0.0),  # simplified DSR proxy
    }

    tier = _assign_tier(metrics, cfg)
    logger.info(f"  {symbol}: DA={da:.3f} Sharpe={synthetic_sharpe:.3f} PBO={pbo:.3f} Tier={tier}")

    # Step 8: SHAP importance
    try:
        shap_imp = compute_shap_importance(model, X_train_final, top_k=int(cfg.model.shap_top_k))
        shap_path = models_dir / f"{symbol}_{_TF}_shap_importance.json"
        shap_imp.to_json(shap_path)
    except Exception as e:
        logger.warning(f"  {symbol}: SHAP failed: {e}")

    # Step 9: Save model and register
    train_start_str = str(X_train_final.index.min().date())
    train_end_str = str(X_train_final.index.max().date())
    version = generate_version_string(symbol, _TF, selected_features, best_params, train_start_str, train_end_str)

    save_model(model, calibrator, symbol, _TF, version, models_dir)
    register_model(
        symbol=symbol,
        tf=_TF,
        version=version,
        metrics={**metrics, "tier": tier},
        feature_names=selected_features,
        hyperparams=best_params,
        train_period=(train_start_str, train_end_str),
        model_path=str(models_dir / f"{version}_model.json"),
        model_type="primary",
    )

    # Save OOF predictions for meta-labeler (stage 05)
    oof_path = checkpoints_dir / "oof" / f"{symbol}_{_TF}_oof_proba.npy"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(oof_path), oof_proba)

    # Save feature selection result
    features_sel_path = checkpoints_dir / "feature_selection" / f"{symbol}_{_TF}_selected.json"
    features_sel_path.parent.mkdir(parents=True, exist_ok=True)
    with open(features_sel_path, "w") as f:
        json.dump(selected_features, f)

    return symbol, {"version": version, "tier": tier, "metrics": metrics}, None


def run(cfg, force: bool = False, symbol_filter: str = None) -> None:
    if not force and is_stage_complete("training"):
        logger.info("Stage 4 already complete, skipping.")
        return

    all_symbols = get_symbols(cfg)
    if symbol_filter:
        all_symbols = [s for s in all_symbols if s.get("name", s.get("symbol")) == symbol_filter]
    symbol_names = [s.get("name", s.get("symbol")) for s in all_symbols]

    checkpoints_dir = Path(cfg.data.checkpoints_dir)
    labels_dir = Path(cfg.data.labels_dir)
    features_dir = Path(cfg.data.features_dir)
    models_dir = Path(cfg.data.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    issues = []
    training_summary = []

    for symbol in symbol_names:
        sym, result, err = _train_symbol(symbol, cfg, checkpoints_dir, labels_dir, features_dir, models_dir)
        if err:
            logger.error(f"{sym}: training failed — {err}")
            issues.append(f"{sym}: {str(err)[:200]}")
        else:
            update_completed_symbol("training", sym)
            training_summary.append({
                "symbol": sym,
                "version": result["version"],
                "tier": result["tier"],
                **result["metrics"],
            })
            logger.info(f"{sym}: training complete — Tier {result['tier']}")

    # Save training summary CSV
    if training_summary:
        summary_df = pd.DataFrame(training_summary)
        summary_path = Path(cfg.data.models_dir) / "training_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Training summary saved: {summary_path}")

    update_project_state("training", "done", issues, output_dir=str(models_dir))
    logger.info(f"Stage 4 complete. {len(training_summary)}/{len(symbol_names)} symbols trained.")
