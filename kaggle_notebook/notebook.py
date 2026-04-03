"""
crypto-model Kaggle training script — GPU training on Kaggle T4/P100.

Stage 2 (feature generation) has 3 modes, detected automatically:

  Mode A — pre-built features uploaded as batch datasets (FASTEST, ~1h total)
            Attach: crypto-model-features-1, -2, -3, -4
            Stage 2 is SKIPPED, all 59 parquets symlinked from datasets.

  Mode B — features not attached, raw data available (SLOW, ~3-4h total)
            Attach: crypto-model-raw-data only (no feature datasets)
            Stage 2 runs and generates features from scratch on CPU.

  Mode C — local stage 2 already done, upload data/features/ as batch datasets
            Run locally: python kaggle_notebook/upload_features.py
            Then re-run notebook → auto-detected as Mode A.

Stages 3-7 always run on GPU (XGB_DEVICE=cuda).
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO      = Path("/kaggle/working/repo")
RAW       = Path("/kaggle/input/crypto-model-raw-data")
CKPT_IN   = Path("/kaggle/input/crypto-model-checkpoints")
FEAT_IN   = Path("/kaggle/input")  # features spread across -1/-2/-3/-4 subdirs
WORK      = Path("/kaggle/working")

# ── 1. Clone latest code from GitHub ─────────────────────────────────────────
subprocess.run([
    "git", "clone", "--depth=1",
    "https://github.com/Zedd443/crypto-model.git",
    str(REPO)
], check=True)

sys.path.insert(0, str(REPO))
os.chdir(REPO)

# ── 2. Install dependencies ───────────────────────────────────────────────────
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "--no-deps",
    "xgboost>=2.0.0", "lightgbm>=4.0.0", "optuna>=3.0.0",
    "hmmlearn>=0.3.0", "ruptures>=1.1.0", "arch>=6.0.0",
    "omegaconf>=2.3.0", "loguru>=0.7.0", "pyarrow>=14.0.0",
    "dcor>=0.6.0", "shap>=0.44.0",
], check=True)

# ── 3. Wire data directories ──────────────────────────────────────────────────
data_dir = REPO / "data"
data_dir.mkdir(exist_ok=True)

# Raw data: symlink (read-only, no copy needed)
raw_link = data_dir / "raw"
if not raw_link.exists():
    raw_link.symlink_to(RAW)

# Mode A: symlink pre-built feature parquets from batch datasets (-1 to -4)
# Mode B: no datasets attached → stage 2 generates features from raw data
feat_dst = data_dir / "features"
feat_dst.mkdir(exist_ok=True)

_feat_count = 0
for _batch_num in range(1, 5):
    # User datasets mount at /kaggle/input/datasets/{user}/{slug}
    _batch_dir = Path(f"/kaggle/input/datasets/irfandragneel/crypto-model-features-{_batch_num}")
    if not _batch_dir.exists():
        continue
    for _pq in _batch_dir.glob("*.parquet"):
        _link = feat_dst / _pq.name
        if not _link.exists():
            _link.symlink_to(_pq)
            _feat_count += 1

if _feat_count:
    print(f"Mode A: {_feat_count} feature files linked — stage 2 will be skipped")
else:
    print("Mode B: no feature datasets attached — stage 2 will generate features from raw data")

# Checkpoints: dataset root IS the checkpoints dir (no subfolder)
ckpt_dst = data_dir / "checkpoints"
if CKPT_IN.exists() and not ckpt_dst.exists():
    shutil.copytree(CKPT_IN, ckpt_dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("dataset-metadata.json"))

models_dst = REPO / "models"
models_dst.mkdir(exist_ok=True)
# models/ subfolder exists inside the dataset root
if (CKPT_IN / "models").exists():
    shutil.copytree(CKPT_IN / "models", models_dst, dirs_exist_ok=True)

if (CKPT_IN / "model_registry.json").exists():
    shutil.copy(CKPT_IN / "model_registry.json", REPO / "model_registry.json")

# Restore project_state.json from checkpoints so pipeline can resume prior run
if (CKPT_IN / "project_state.json").exists():
    shutil.copy(CKPT_IN / "project_state.json", REPO / "project_state.json")
    print("Restored project_state.json from checkpoints — pipeline will skip completed stages")

# Create dirs pipeline needs
for d in ["data/features", "data/processed", "data/labels", "results", "logs", "monitoring"]:
    (REPO / d).mkdir(parents=True, exist_ok=True)

# ── 4. Enable GPU for XGBoost ────────────────────────────────────────────────
os.environ["XGB_DEVICE"] = "cuda"
print("XGB_DEVICE=cuda — XGBoost will use GPU")

# ── 5. Load config ────────────────────────────────────────────────────────────
from src.utils.config_loader import load_config
cfg = load_config()

# ── 6. Run pipeline stages ───────────────────────────────────────────────────
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

# Flush stdout after every write so Kaggle captures logs in real-time
_sys.stdout.reconfigure(line_buffering=True)
import logging as _logging
_logging.basicConfig(stream=_sys.stdout, level=_logging.INFO, force=True)

def _run_stage(name, fn, *args, **kwargs):
    print(f"\n=== {name} ===", flush=True)
    try:
        fn(*args, **kwargs)
        print(f"=== {name} DONE ===", flush=True)
    except Exception:
        print(f"\n!!! {name} FAILED !!!", flush=True)
        _tb.print_exc()
        _sys.exit(1)

# Stage 2: Mode A → skip (features already linked), Mode B → generate from raw
_feat_files = list((REPO / "data/features").glob("*.parquet"))
if _feat_files:
    print(f"\n=== Stage 2: SKIPPED — {len(_feat_files)} feature files ready (Mode A) ===", flush=True)
else:
    _run_stage("Stage 2: features (Mode B — generating from raw)", stage_02_features.run, cfg, force=False)
_run_stage("Stage 3: labels",        stage_03_labels.run,   cfg)
_run_stage("Stage 4: training (GPU)",stage_04_train.run,    cfg)
_run_stage("Stage 5: meta-labeling", stage_05_meta.run,     cfg)
_run_stage("Stage 6: portfolio",     stage_06_portfolio.run,cfg)
_run_stage("Stage 7: backtest",      stage_07_backtest.run, cfg)

# ── 7. Package output for download ───────────────────────────────────────────
print("\n=== Packaging output ===")
out = WORK / "output"
out.mkdir(exist_ok=True)

shutil.copytree(REPO / "data/checkpoints", out / "checkpoints", dirs_exist_ok=True)
shutil.copytree(REPO / "models",           out / "models",      dirs_exist_ok=True)
shutil.copy(REPO / "model_registry.json",  out / "model_registry.json")
shutil.copy(REPO / "project_state.json",   out / "project_state.json")

# Zip for easy download
shutil.make_archive(str(WORK / "crypto_model_output"), "zip", out)
print(f"Output zipped: {WORK}/crypto_model_output.zip")
print("Done.")
