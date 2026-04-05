import os
from pathlib import Path
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
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
    fold_da_list = metrics.get("fold_da_list", [])

    tier_a_da = float(cfg.model.tier_A_da_min)
    tier_a_sharpe = float(cfg.model.tier_A_sharpe_wfo_min)
    tier_a_pbo = float(cfg.model.tier_A_pbo_max)
    tier_a_da_min_folds = int(cfg.model.tier_A_da_min_folds)
    tier_b_da = float(cfg.model.tier_B_da_min)

    # Count how many CV folds exceed the DA threshold
    folds_above_da = sum(1 for fda in fold_da_list if fda >= tier_a_da)
    passes_fold_da = (folds_above_da >= tier_a_da_min_folds) if fold_da_list else False

    if da >= tier_a_da and sharpe >= tier_a_sharpe and pbo <= tier_a_pbo and passes_fold_da:
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
    w_train = weights_aligned[train_mask]

    # Filter warmup rows first
    if "is_warmup" in features_aligned.columns:
        warmup_mask_train = features_aligned.loc[train_mask, "is_warmup"] == 1
        X_train = X_train[~warmup_mask_train]
        w_train = w_train[~warmup_mask_train]
        train_label_series = labels_aligned.loc[train_mask, "label"][~warmup_mask_train]
    else:
        train_label_series = labels_aligned.loc[train_mask, "label"]

    # Drop neutral (time-barrier) labels — keep only directional {-1, +1}
    train_dir_mask = train_label_series != 0
    X_train = X_train[train_dir_mask]
    w_train = w_train[train_dir_mask]
    y_train_binary = (train_label_series[train_dir_mask] == 1).astype(int)  # -1→0 (short), +1→1 (long)

    X_val = features_aligned[val_mask].drop(columns=["is_warmup"], errors="ignore")
    val_label_series = labels_aligned.loc[val_mask, "label"]
    val_dir_mask = val_label_series != 0
    X_val = X_val[val_dir_mask]
    y_val_binary = (val_label_series[val_dir_mask] == 1).astype(int)

    n_long = int(y_train_binary.sum())
    n_short = int((y_train_binary == 0).sum())
    pct_long = n_long / max(len(y_train_binary), 1)
    logger.info(f"  {symbol}: directional label balance — n_long={n_long}, n_short={n_short}, pct_long={pct_long:.3f}")
    scale_pos_weight = n_short / max(n_long, 1)

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
    X_val_final = pd.DataFrame(X_val_scaled, index=X_val_sel.index, columns=selected_features)

    # Step 3: Hyperparameter tuning with Optuna
    logger.info(f"  {symbol}: tuning hyperparameters ({cfg.model.optuna_n_trials} trials)")
    try:
        best_params = tune_hyperparams(X_train_final, y_train_binary, w_train, splitter, cfg)
    except Exception as e:
        logger.warning(f"  {symbol}: Optuna failed ({e}), using defaults")
        best_params = {}

    best_params["scale_pos_weight"] = scale_pos_weight

    # Step 4: OOF predictions for meta-labeling (NEVER in-sample)
    logger.info(f"  {symbol}: computing OOF predictions")
    fold_val_losses = []
    fold_best_iterations = []
    try:
        oof_proba, fold_val_losses, fold_best_iterations = compute_oof_predictions(
            X_train_final, y_train_binary, w_train, splitter, best_params, cfg
        )
    except Exception as e:
        logger.warning(f"  {symbol}: OOF failed ({e})")
        oof_proba = np.full((len(X_train_final), 2), 0.5)

    # Step 5: Train final model on all train data
    logger.info(f"  {symbol}: training final model")
    try:
        model, calibrator, val_loss_curve, best_iteration = train_xgb(
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

    # MSE: mean squared error between calibrated val probabilities and binary labels
    val_mse = float(np.mean((val_proba_cal - y_val_binary.values) ** 2))

    # Class balance on train and val sets
    pct_positive_train = float(y_train_binary.mean())
    pct_positive_val = float(y_val_binary.mean())
    logger.info(
        f"  {symbol}: class balance — train_pos={pct_positive_train:.3f} "
        f"val_pos={pct_positive_val:.3f}"
    )

    # Build a price-return series for train samples indexed identically to X_train_final
    # label=+1 → tp_level (long trade hits TP), label=-1 → -sl_level (hits SL), label=0 → 0
    # This gives the actual realized return per bar under the triple-barrier scheme
    train_labels_for_sharpe = labels_aligned.loc[train_mask]
    # Reindex to X_train_final.index — this drops warmup rows automatically since they were
    # already filtered out of X_train before building X_train_final
    train_labels_for_sharpe = train_labels_for_sharpe.reindex(X_train_final.index)
    raw_label_col = train_labels_for_sharpe["label"].values
    tp_col = train_labels_for_sharpe["tp_level"].fillna(0.0).values if "tp_level" in train_labels_for_sharpe.columns else np.zeros(len(X_train_final))
    sl_col = train_labels_for_sharpe["sl_level"].fillna(0.0).values if "sl_level" in train_labels_for_sharpe.columns else np.zeros(len(X_train_final))
    # Realized pct return: +tp_level for TP hit, -sl_level for SL hit, 0 for time barrier
    price_returns = np.where(raw_label_col == 1, tp_col,
                    np.where(raw_label_col == -1, -sl_col, 0.0))

    # Compute fold Sharpes using actual price returns (not binary label proxies)
    fold_sharpes = []
    fold_da_list = []
    for _, val_idx in splitter.split(X_train_final, y_train_binary):
        fold_proba = oof_proba[val_idx, 1]
        fold_y = y_train_binary.iloc[val_idx].values
        fold_price_ret = price_returns[val_idx]
        # Direction: go long when model says prob > 0.5, else flat (no forced short for binary model)
        positions = np.where(fold_proba > 0.5, 1.0, 0.0)
        # Only count bars where we have a position; return = position × realized_price_return
        fold_returns = positions * fold_price_ret
        # Use only active-position bars to avoid diluting Sharpe with flat periods
        active = fold_returns[positions > 0]
        if len(active) > 1 and active.std() > 1e-12:
            sharpe = float(active.mean() / active.std() * np.sqrt(252 * 96))
            fold_sharpes.append(sharpe)
        fold_da = float(np.mean((fold_proba > 0.5).astype(int) == fold_y)) if len(fold_y) > 0 else 0.0
        fold_da_list.append(fold_da)

    pbo = compute_pbo(fold_sharpes)
    synthetic_sharpe = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0

    # overfit_ratio: train_logloss / val_logloss at best_iteration
    # Closer to 1.0 = less overfit. < 0.85 = model likely memorizing train set.
    train_loss_at_best = float("nan")
    val_loss_at_best = val_loss_curve[best_iteration] if val_loss_curve and best_iteration < len(val_loss_curve) else float("nan")
    # Train logloss: approximate from OOF predictions on train set
    try:
        from sklearn.metrics import log_loss
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
        "synthetic_sharpe": synthetic_sharpe,
        "conformal_q90": q90,
        "n_train_samples": len(X_train_final),
        "n_val_samples": len(X_val_final),
        "n_features": len(selected_features),
        "dsr": max(synthetic_sharpe - pbo, 0.0),  # simplified DSR proxy
        "pct_positive_train": pct_positive_train,
        "pct_positive_val": pct_positive_val,
        "fold_da_list": fold_da_list,
        "best_iteration": int(best_iteration),
        "overfit_ratio": overfit_ratio,
        "fold_da_std": fold_da_std,
        "val_logloss": val_loss_at_best,
    }

    tier = _assign_tier(metrics, cfg)
    logger.info(
        f"  {symbol}: DA={da:.3f} MSE={val_mse:.4f} Sharpe={synthetic_sharpe:.3f} PBO={pbo:.3f} "
        f"Tier={tier} folds_above_da={sum(1 for d in fold_da_list if d >= float(cfg.model.tier_A_da_min))}"
    )

    # Step 8a: Save training diagnostics JSON for overfitting analysis
    try:
        diagnostics_dir = checkpoints_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = {
            "symbol": symbol,
            "version": "",  # filled after version is generated below
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

    # Step 8b: SHAP importance — skip on Kaggle (CPU-only, ~2-5 min/symbol) via SKIP_SHAP=1
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

    # Step 9: Save model and register
    train_start_str = str(X_train_final.index.min().date())
    train_end_str = str(X_train_final.index.max().date())
    version = generate_version_string(symbol, _TF, selected_features, best_params, train_start_str, train_end_str)

    save_model(model, calibrator, symbol, _TF, version, models_dir)

    # Update version in diagnostics file now that we have it
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
        feature_names=selected_features,
        hyperparams=best_params,
        train_period=(train_start_str, train_end_str),
        model_path=str(models_dir / f"{version}_model.json"),
        model_type="primary",
    )

    # Save OOF predictions for meta-labeler (stage 05)
    # Keep .npy for speed; save index separately so stage_05 can align by timestamp.
    oof_dir = checkpoints_dir / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    oof_path = oof_dir / f"{symbol}_{_TF}_oof_proba.npy"
    oof_idx_path = oof_dir / f"{symbol}_{_TF}_oof_index.npy"
    np.save(str(oof_path), oof_proba)
    np.save(str(oof_idx_path), X_train_final.index.view("int64"))  # store as ns-since-epoch int64

    # Save feature selection result
    features_sel_path = checkpoints_dir / "feature_selection" / f"{symbol}_{_TF}_selected.json"
    features_sel_path.parent.mkdir(parents=True, exist_ok=True)
    with open(features_sel_path, "w") as f:
        json.dump(selected_features, f)

    # Generate diagnostic plots (non-fatal — skip on error)
    try:
        from src.visualization.training_diagnostics import generate_all_diagnostics
        diag_dir = checkpoints_dir / "diagnostics"
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

    return symbol, {"version": version, "tier": tier, "metrics": metrics}, None


def run(cfg, force: bool = False, symbol_filter: str = None) -> None:
    if not force and is_stage_complete("training"):
        logger.info("Stage 4 already complete, skipping.")
        return

    all_symbols = get_symbols(cfg)
    if symbol_filter:
        all_symbols = [s for s in all_symbols if s.get("name", s.get("symbol")) == symbol_filter]
    symbol_names = [s.get("name", s.get("symbol")) for s in all_symbols]

    # Skip symbols already trained in a prior run (pipeline is resumable)
    # --force bypasses resume: retrain all symbols in symbol_names
    if not force:
        from src.utils.state_manager import load_state
        _state = load_state()
        _completed = set(_state.get("stages", {}).get("training", {}).get("completed_symbols", []))
        if _completed:
            _pending = [s for s in symbol_names if s not in _completed]
            logger.info(f"Resuming training — {len(_completed)} already done, {len(_pending)} remaining")
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
            logger.error(f"{sym}: training failed — {err}")
            issues.append(f"{sym}: {str(err)[:200]}")
        else:
            update_completed_symbol("training", sym)
            # Exclude fold_da_list (list type) from CSV — it's used only for tier assignment
            csv_metrics = {k: v for k, v in result["metrics"].items() if k != "fold_da_list"}
            training_summary.append({
                "symbol": sym,
                "version": result["version"],
                "tier": result["tier"],
                **csv_metrics,
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
