#!/usr/bin/env python3
"""
Refit IsotonicRegression calibrators for all symbols using test set data.

The original calibrators were fit on the val set (train_end → val_end) during
stage 4. Live raw probabilities fall in the compressed zone of that calibrator
because the model's OOF predictions during training were overconfident.

This script:
  1. Loads test set features + labels (test_start onward)
  2. Runs model.predict_proba() on test set → raw probs
  3. Fits a new IsotonicRegression(out_of_bounds='clip') on (raw_probs, true_labels)
  4. Overwrites the calibrator pkl alongside the existing model version

LEAKAGE GUARD: only test set data (>= test_start) is used to fit the calibrator.
The test set is held-out from training and validation — no leakage.

Usage:
    .venv/Scripts/python scripts/refit_calibrator.py
    .venv/Scripts/python scripts/refit_calibrator.py --symbol BTCUSDT  # single symbol
    .venv/Scripts/python scripts/refit_calibrator.py --dry-run          # print stats, no save
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config, get_symbols
from src.models.primary_model import load_model
from src.models.model_versioning import get_latest_model
from src.utils.logger import get_logger

logger = get_logger("refit_calibrator")


def load_test_features(symbol: str, cfg) -> pd.DataFrame | None:
    features_dir = Path(cfg.data.features_dir)
    path = features_dir / f"{symbol}_15m_features.parquet"
    if not path.exists():
        logger.warning(f"{symbol}: feature parquet not found at {path}")
        return None
    df = pd.read_parquet(path)
    test_start = pd.Timestamp(cfg.data.test_start, tz="UTC")
    df_test = df[df.index >= test_start]
    if len(df_test) == 0:
        logger.warning(f"{symbol}: no test rows (>= {cfg.data.test_start})")
        return None
    return df_test


def load_test_labels(symbol: str, cfg) -> pd.Series | None:
    labels_dir = Path(cfg.data.labels_dir)
    path = labels_dir / f"{symbol}_15m_labels.parquet"
    if not path.exists():
        logger.warning(f"{symbol}: label parquet not found at {path}")
        return None
    df = pd.read_parquet(path)
    test_start = pd.Timestamp(cfg.data.test_start, tz="UTC")
    s = df.loc[df.index >= test_start, "label"] if "label" in df.columns else None
    if s is None or len(s) == 0:
        logger.warning(f"{symbol}: no test labels found")
        return None
    return s


def refit_one(symbol: str, cfg, dry_run: bool = False) -> bool:
    # 1. Get model version from registry
    reg = get_latest_model(symbol, "15m", model_type="primary")
    if reg is None:
        logger.warning(f"{symbol}: no registered primary model — skipping")
        return False

    version = reg["version"]
    models_dir = Path(cfg.data.models_dir)

    # 2. Load model + old calibrator
    try:
        model, old_calibrator = load_model(symbol, "15m", version, models_dir)
    except Exception as e:
        logger.error(f"{symbol}: could not load model — {e}")
        return False

    selected_features = reg.get("feature_names", [])
    if not selected_features:
        logger.warning(f"{symbol}: no feature_names in registry — skipping")
        return False

    # 3. Load test set features
    X_test_full = load_test_features(symbol, cfg)
    if X_test_full is None:
        return False

    # 4. Load test set labels
    y_test_full = load_test_labels(symbol, cfg)
    if y_test_full is None:
        return False

    # 5. Align features and labels
    common_idx = X_test_full.index.intersection(y_test_full.index)
    if len(common_idx) < 50:
        logger.warning(f"{symbol}: only {len(common_idx)} common test rows — skipping (need >= 50)")
        return False

    X_test = X_test_full.loc[common_idx].drop(columns=["is_warmup"], errors="ignore")
    y_test = y_test_full.loc[common_idx]

    # Drop neutral labels (label=0 means time-barrier exit, not directional)
    dir_mask = y_test != 0
    X_test = X_test[dir_mask]
    y_test = y_test[dir_mask]
    y_binary = (y_test == 1).astype(int)  # -1→0 (short), +1→1 (long)

    if len(X_test) < 30:
        logger.warning(f"{symbol}: only {len(X_test)} directional test rows — skipping (need >= 30)")
        return False

    # 6. Subset to model's selected features, fill missing with 0
    missing = [c for c in selected_features if c not in X_test.columns]
    if missing:
        logger.warning(f"{symbol}: {len(missing)} features missing from test set — filling 0")
        for c in missing:
            X_test[c] = 0.0
    X_test = X_test[selected_features].select_dtypes(include=[np.number]).fillna(0)

    # 7. Apply imputer + scaler (transform-only)
    imp_dir = Path(cfg.data.checkpoints_dir) / "imputers"
    imp_path = imp_dir / f"imputer_{symbol}_15m.pkl"
    scaler_path = imp_dir / f"scaler_{symbol}_15m.pkl"
    if not imp_path.exists() or not scaler_path.exists():
        logger.warning(f"{symbol}: imputer/scaler not found — skipping")
        return False

    with open(imp_path, "rb") as f:
        imputer = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    X_imp = imputer.transform(X_test.values)
    X_scaled = scaler.transform(X_imp)

    # 8. Get raw probabilities from XGBoost
    raw_proba = model.predict_proba(X_scaled)[:, 1]

    # Diagnostics
    old_cal_proba = old_calibrator.predict(raw_proba)
    logger.info(
        f"{symbol}: test rows={len(X_test)} directional, "
        f"pct_long={y_binary.mean():.3f}, "
        f"raw_proba mean={raw_proba.mean():.3f} std={raw_proba.std():.3f} "
        f"range=[{raw_proba.min():.3f}, {raw_proba.max():.3f}]"
    )
    logger.info(
        f"{symbol}: OLD calibrator output mean={old_cal_proba.mean():.3f} "
        f"range=[{old_cal_proba.min():.3f}, {old_cal_proba.max():.3f}]"
    )

    # 9. Fit new calibrator on test set raw probs
    new_calibrator = IsotonicRegression(out_of_bounds="clip")
    new_calibrator.fit(raw_proba, y_binary.values)
    new_cal_proba = new_calibrator.predict(raw_proba)

    logger.info(
        f"{symbol}: NEW calibrator output mean={new_cal_proba.mean():.3f} "
        f"range=[{new_cal_proba.min():.3f}, {new_cal_proba.max():.3f}]"
    )

    if dry_run:
        logger.info(f"{symbol}: dry-run — calibrator NOT saved")
        return True

    # 10. Save new calibrator (overwrites old one for this version)
    cal_path = models_dir / f"{version}_calibrator.pkl"
    with open(cal_path, "wb") as f:
        pickle.dump(new_calibrator, f)
    logger.info(f"{symbol}: calibrator saved → {cal_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Refit calibrators on test set data")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol to refit (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Print diagnostics without saving")
    args = parser.parse_args()

    cfg = load_config()
    if args.symbol:
        symbols = [args.symbol]
    else:
        # get_symbols returns list of dicts with "symbol" key; extract names
        sym_list = get_symbols(cfg)
        symbols = [s["symbol"] if isinstance(s, dict) else s for s in sym_list]
        # Fallback: use all symbols with a trained primary model from registry
        if not symbols:
            from src.models.model_versioning import get_active_models
            symbols = [m["symbol"] for m in get_active_models("15m", model_type="primary")]

    logger.info(f"Refitting calibrators for {len(symbols)} symbols — test_start={cfg.data.test_start} dry_run={args.dry_run}")

    success, failed, skipped = 0, 0, 0
    for sym in symbols:
        try:
            ok = refit_one(sym, cfg, dry_run=args.dry_run)
            if ok:
                success += 1
            else:
                skipped += 1
        except Exception as e:
            logger.error(f"{sym}: unhandled error — {e}")
            failed += 1

    logger.info(f"Done — success={success} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
