"""
crypto-model Kaggle training script — GPU training on Kaggle T4/P100.

Stage 2 (feature generation) has 3 modes, detected automatically:

  Mode A — pre-built features uploaded as batch datasets (FASTEST, ~1h total)
            Attach: crypto-model-features-1, -2, -3, -4
            Stage 2 is SKIPPED, all parquets symlinked from datasets.

  Mode B — features not attached, raw data available (SLOW, ~3-4h total)
            Attach: crypto-model-raw-data only (no feature datasets)
            Stage 2 runs and generates features from scratch on CPU.

  Mode C — local stage 2 already done, upload data/features/ as batch datasets
            Run locally: python kaggle_notebook/upload_features.py
            Then re-run notebook -> auto-detected as Mode A.

Stages 3-7 always run on GPU (XGB_DEVICE=cuda).

Dataset mount paths on Kaggle:
  /kaggle/input/crypto-model-raw-data/          <- raw OHLCV
  /kaggle/input/crypto-model-checkpoints/       <- checkpoints + models
  /kaggle/input/crypto-model-features-1/ .. -4/ <- pre-built features (Mode A)
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO     = Path("/kaggle/working/repo")
RAW      = Path("/kaggle/input/crypto-model-raw-data")
CKPT_IN  = Path("/kaggle/input/crypto-model-checkpoints")
WORK     = Path("/kaggle/working")

# ── 1. Clone latest code from GitHub ──────────────────────────────────────────
subprocess.run([
    "git", "clone", "--depth=1",
    "https://github.com/Zedd443/crypto-model.git",
    str(REPO)
], check=True)

sys.path.insert(0, str(REPO))
os.chdir(REPO)

# ── 2. Install dependencies ────────────────────────────────────────────────────
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "--no-deps",
    "xgboost>=2.0.0", "lightgbm>=4.0.0", "optuna>=3.0.0",
    "hmmlearn>=0.3.0", "ruptures>=1.1.0", "arch>=6.0.0",
    "omegaconf>=2.3.0", "loguru>=0.7.0", "pyarrow>=14.0.0",
    "dcor>=0.6.0", "shap>=0.44.0", "scikit-learn==1.6.1",
], check=True)

# ── 3. Wire data directories ───────────────────────────────────────────────────
data_dir = REPO / "data"
data_dir.mkdir(exist_ok=True)

# Raw data: symlink (read-only, no copy needed)
raw_link = data_dir / "raw"
if not raw_link.exists():
    raw_link.symlink_to(RAW)

# Mode A: symlink pre-built feature parquets from batch datasets
# Debug: print all available input dirs so we can verify mount paths
print("Available /kaggle/input/ directories:")
for _p in sorted(Path("/kaggle/input").iterdir()):
    print(f"  {_p}")

feat_dst = data_dir / "features"
feat_dst.mkdir(exist_ok=True)

_feat_count = 0
for _batch_num in range(1, 5):
    # Try both known Kaggle mount path formats
    _candidates = [
        Path(f"/kaggle/input/crypto-model-features-{_batch_num}"),
        Path(f"/kaggle/input/datasets/irfandragneel/crypto-model-features-{_batch_num}"),
    ]
    _batch_dir = next((p for p in _candidates if p.exists()), None)
    if _batch_dir is None:
        print(f"  features-{_batch_num}: not attached (tried {[str(c) for c in _candidates]})")
        continue
    print(f"  features-{_batch_num}: found at {_batch_dir}")
    for _pq in _batch_dir.glob("**/*.parquet"):  # ** to catch any subdir
        _link = feat_dst / _pq.name
        if not _link.exists():
            _link.symlink_to(_pq)
            _feat_count += 1

if _feat_count:
    print(f"Mode A: {_feat_count} feature files linked — stage 2 will be skipped")
else:
    print("Mode B: no feature datasets attached — stage 2 will generate features from raw data")

# Checkpoints: copy to working dir (need write access during pipeline)
ckpt_dst = data_dir / "checkpoints"
ckpt_dst.mkdir(exist_ok=True)
if CKPT_IN.exists():
    shutil.copytree(CKPT_IN, ckpt_dst, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("dataset-metadata.json"))
    print(f"Checkpoints copied from {CKPT_IN}")

models_dst = REPO / "models"
models_dst.mkdir(exist_ok=True)
if (CKPT_IN / "models").exists():
    shutil.copytree(CKPT_IN / "models", models_dst, dirs_exist_ok=True)
    print("Models restored from checkpoints dataset")

if (CKPT_IN / "model_registry.json").exists():
    shutil.copy(CKPT_IN / "model_registry.json", REPO / "model_registry.json")

# Restore labels if present in checkpoints dataset
if (CKPT_IN / "labels").exists():
    labels_dst = data_dir / "labels"
    labels_dst.mkdir(exist_ok=True)
    shutil.copytree(CKPT_IN / "labels", labels_dst, dirs_exist_ok=True)
    print("Labels restored from checkpoints dataset")

# Restore project_state.json so pipeline can resume prior run
if (CKPT_IN / "project_state.json").exists():
    shutil.copy(CKPT_IN / "project_state.json", REPO / "project_state.json")
    print("Restored project_state.json — pipeline will skip completed stages")

# Create dirs pipeline needs
for d in ["data/features", "data/processed", "data/labels", "results", "logs", "monitoring"]:
    (REPO / d).mkdir(parents=True, exist_ok=True)

# ── 4. Enable GPU for XGBoost ─────────────────────────────────────────────────
os.environ["XGB_DEVICE"] = "cuda"
print("XGB_DEVICE=cuda — XGBoost will use GPU")

os.environ["LABEL_WORKERS"] = "2"   # stage 3: 2 parallel CPU label workers
os.environ["META_WORKERS"] = "1"    # stage 5: GPU forces serial anyway
os.environ["SKIP_SHAP"] = "1"       # SHAP is CPU-only, skip to save ~2-4h
print("LABEL_WORKERS=2, META_WORKERS=1, SKIP_SHAP=1")

# ── 5. Load config ────────────────────────────────────────────────────────────
from src.utils.config_loader import load_config
from omegaconf import OmegaConf
cfg = load_config()

OmegaConf.update(cfg, "model.optuna_n_trials", 25, merge=True)
OmegaConf.update(cfg, "model.stability_n_bootstrap", 20, merge=True)
print(f"Kaggle overrides: optuna_n_trials=25, stability_n_bootstrap=20")

# ── 6. Run pipeline stages ────────────────────────────────────────────────────
from src.pipeline import (
    stage_02_features,
    stage_03_labels,
    stage_04_train,
    stage_05_meta,
    stage_06_portfolio,
    stage_07_backtest,
)

import sys as _sys
import traceback as _tb

_sys.stdout.reconfigure(line_buffering=True)
import logging as _logging
_logging.basicConfig(stream=_sys.stdout, level=_logging.INFO, force=True)

OUT = WORK / "output"
OUT.mkdir(exist_ok=True)

def _save_checkpoint(stage_name: str):
    """Save incremental checkpoint after each stage so a mid-run crash loses minimal work."""
    try:
        if (REPO / "data/checkpoints").exists():
            shutil.copytree(REPO / "data/checkpoints", OUT / "checkpoints", dirs_exist_ok=True)
        if (REPO / "models").exists():
            shutil.copytree(REPO / "models", OUT / "models", dirs_exist_ok=True)
        if (REPO / "data/labels").exists():
            shutil.copytree(REPO / "data/labels", OUT / "labels", dirs_exist_ok=True)
        for f in ["model_registry.json", "project_state.json"]:
            src = REPO / f
            if src.exists():
                shutil.copy(src, OUT / f)
        print(f"  [checkpoint saved after {stage_name}]", flush=True)
    except Exception as _e:
        print(f"  [checkpoint save failed: {_e}]", flush=True)

def _run_stage(name, fn, *args, **kwargs):
    print(f"\n=== {name} ===", flush=True)
    try:
        fn(*args, **kwargs)
        print(f"=== {name} DONE ===", flush=True)
    except Exception:
        print(f"\n!!! {name} FAILED !!!", flush=True)
        _tb.print_exc()
        _save_checkpoint(f"{name} (failed)")
        _sys.exit(1)

# Stage 2: Mode A -> skip, Mode B -> generate from raw
_feat_files = list((REPO / "data/features").glob("*.parquet"))
if _feat_files:
    print(f"\n=== Stage 2: SKIPPED — {len(_feat_files)} feature files ready (Mode A) ===", flush=True)
else:
    _run_stage("Stage 2: features (Mode B)", stage_02_features.run, cfg, force=False)
    _save_checkpoint("stage_02")

_run_stage("Stage 3: labels",         stage_03_labels.run,    cfg)
_save_checkpoint("stage_03")

_run_stage("Stage 4: training (GPU)", stage_04_train.run,     cfg)
_save_checkpoint("stage_04")

_run_stage("Stage 5: meta-labeling",  stage_05_meta.run,      cfg)
_save_checkpoint("stage_05")

_run_stage("Stage 6: portfolio",      stage_06_portfolio.run, cfg)
_run_stage("Stage 7: backtest",       stage_07_backtest.run,  cfg)

# ── 7. Package final output ───────────────────────────────────────────────────
print("\n=== Packaging output ===")
_save_checkpoint("final")

shutil.make_archive(str(WORK / "crypto_model_output"), "zip", OUT)
print(f"Output zipped: {WORK}/crypto_model_output.zip")
print("Done.")
