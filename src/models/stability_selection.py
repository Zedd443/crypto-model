import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from src.utils.logger import get_logger

logger = get_logger("stability_selection")


def run_stability_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    weights: pd.Series,
    val_start_date: str,
    cfg,
) -> list:
    # CRITICAL: assert all training data precedes validation period
    val_start_ts = pd.Timestamp(val_start_date, tz="UTC")
    assert X_train.index.max() < val_start_ts, (
        f"Stability selection window bleeds into val: train max={X_train.index.max()} >= val_start={val_start_ts}"
    )

    n_bootstrap = int(cfg.model.stability_n_bootstrap)
    threshold = float(cfg.model.stability_threshold)
    n_features = X_train.shape[1]
    feature_names = X_train.columns.tolist()

    selection_freq = np.zeros(n_features)

    rng = np.random.RandomState(42)
    n_samples = len(X_train)
    # Cap subsample at 20k rows for speed — stability selection only needs relative importance ranks
    subsample_size = min(n_samples // 2, 20000)

    X_arr = X_train.values
    y_arr = y_train.values
    w_arr = weights.reindex(X_train.index).fillna(1.0).values

    for b in range(n_bootstrap):
        idx = rng.choice(n_samples, size=subsample_size, replace=False)
        X_b = X_arr[idx]
        y_b = y_arr[idx]
        w_b = w_arr[idx]

        rf = RandomForestClassifier(
            n_estimators=int(getattr(cfg.model, 'stability_rf_n_estimators', 50)),
            max_depth=int(getattr(cfg.model, 'stability_rf_max_depth', 8)),
            max_features="sqrt",
            n_jobs=-1,
            random_state=b,
            class_weight="balanced",
        )
        try:
            rf.fit(X_b, y_b, sample_weight=w_b)
            # Mark features selected by this bootstrap (top 50% by importance)
            importances = rf.feature_importances_
            cutoff = np.median(importances)
            selected = importances >= cutoff
            selection_freq += selected.astype(float)
        except Exception as e:
            logger.warning(f"Bootstrap {b} failed: {e}")
            continue

    selection_freq /= n_bootstrap

    # MI tiebreaker: for borderline features (within 0.5/n_bootstrap of threshold), rank by mutual info
    borderline_margin = 0.5 / n_bootstrap
    borderline_mask = np.abs(selection_freq - threshold) <= borderline_margin
    if borderline_mask.sum() > 0:
        try:
            mi_scores = mutual_info_classif(X_arr, y_arr, random_state=42)
            # Among borderline features, bump MI-superior ones above threshold, drop MI-inferior
            borderline_idx = np.where(borderline_mask)[0]
            mi_border = mi_scores[borderline_idx]
            mi_median = np.median(mi_border)
            for i in borderline_idx:
                if selection_freq[i] < threshold and mi_scores[i] >= mi_median:
                    selection_freq[i] = threshold  # promote
                elif selection_freq[i] >= threshold and mi_scores[i] < mi_median:
                    selection_freq[i] = threshold - borderline_margin * 0.5  # demote
        except Exception as e:
            logger.warning(f"MI tiebreaker failed: {e}")

    selected_features = [
        feature_names[i] for i in range(n_features)
        if selection_freq[i] >= threshold
    ]

    logger.info(
        f"Stability selection: {len(selected_features)}/{n_features} features selected "
        f"(threshold={threshold}, n_bootstrap={n_bootstrap})"
    )
    return selected_features


def variance_threshold_filter(X: pd.DataFrame, threshold: float) -> list:
    # Remove features with variance below threshold
    variances = X.var()
    selected = variances[variances > threshold].index.tolist()
    n_removed = len(X.columns) - len(selected)
    if n_removed > 0:
        logger.info(f"Variance threshold removed {n_removed} low-variance features")
    return selected


def select_features_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    weights: pd.Series,
    val_start_date: str,
    cfg,
) -> list:
    # Step 1: variance threshold filter
    variance_thresh = float(cfg.model.variance_threshold)
    after_var = variance_threshold_filter(X_train, variance_thresh)
    X_filtered = X_train[after_var]

    # Step 2: stability selection
    selected = run_stability_selection(X_filtered, y_train, weights, val_start_date, cfg)

    # Fallback: if stability selection returns too few features, use all post-variance features
    min_features = 10
    if len(selected) < min_features:
        logger.warning(
            f"Stability selection returned only {len(selected)} features — "
            f"falling back to all {len(after_var)} variance-filtered features"
        )
        selected = after_var

    logger.info(f"Final selected features: {len(selected)}")
    return selected
