"""
crypto-model GPU notebook — Stages 4-7 (train → backtest). GPU required.

Run this AFTER notebook_cpu.py output has been uploaded to crypto-model-checkpoints.

What this does:
  1. Clone latest code from GitHub
  2. Install all dependencies (XGBoost GPU, LightGBM, Optuna, etc.)
  3. Link features via Mode A (pre-built, skips stage 2)
  4. Restore labels + checkpoints from dataset
  5. Run Stage 4: XGBoost training with GPU (all symbols, fresh OOF .npy + index)
  6. Run Stage 5: meta-labeling (uses new OOF index files)
  7. Run Stage 6: portfolio signal generation
  8. Run Stage 7: backtest
  9. Save checkpoint after each stage (crash-safe)
 10. Auto-push output to crypto-model-checkpoints Kaggle dataset

Required datasets:
  - crypto-model-raw-data       (raw OHLCV)
  - crypto-model-checkpoints    (labels from CPU run + prior models)
  - crypto-model-features-1..4  (pre-built features, Mode A)

Auto-push requires Kaggle secrets:
  KAGGLE_USERNAME and KAGGLE_KEY set in notebook "Add-ons → Secrets"

Dataset mount paths:
  /kaggle/input/crypto-model-raw-data/
  /kaggle/input/crypto-model-checkpoints/
  /kaggle/input/datasets/irfandragneel/crypto-model-features-{n}/
"""
import os
import sys
import shutil
import subprocess
import json
from pathlib import Path
from datetime import datetime, timezone

REPO    = Path("/kaggle/working/repo")
RAW     = Path("/kaggle/input/crypto-model-raw-data")
CKPT_IN = Path("/kaggle/input/crypto-model-checkpoints")
WORK    = Path("/kaggle/working")
OUT     = WORK / "output"
OUT.mkdir(exist_ok=True)

# ── 1. Clone latest code ───────────────────────────────────────────────────────
if REPO.exists():
    subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"], check=True)
else:
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
    "dcor>=0.6.0", "shap>=0.44.0", "scikit-learn==1.6.1", "rich>=13.0.0",
], check=True)

# ── 3. Wire data directories ───────────────────────────────────────────────────
data_dir = REPO / "data"
data_dir.mkdir(exist_ok=True)

raw_link = data_dir / "raw"
if not raw_link.exists():
    raw_link.symlink_to(RAW)

# Mode A: symlink pre-built feature parquets
print("Available /kaggle/input/ directories:")
for _p in sorted(Path("/kaggle/input").iterdir()):
    print(f"  {_p}")

feat_dst = data_dir / "features"
feat_dst.mkdir(exist_ok=True)
_feat_count = 0
for _batch_num in range(1, 5):
    _batch_dir = Path(f"/kaggle/input/datasets/irfandragneel/crypto-model-features-{_batch_num}")
    if not _batch_dir.exists():
        print(f"  features-{_batch_num}: not attached")
        continue
    print(f"  features-{_batch_num}: found at {_batch_dir}")
    for _pq in _batch_dir.glob("**/*.parquet"):
        _link = feat_dst / _pq.name
        if not _link.exists():
            _link.symlink_to(_pq)
            _feat_count += 1

if _feat_count:
    print(f"Mode A: {_feat_count} feature files linked — stage 2 will be skipped")
else:
    print("WARNING: No feature datasets found — stage 2 would need to run but is not in this notebook")
    print("Attach crypto-model-features-1..4 datasets to use Mode A")

# Checkpoints (contains labels from CPU run)
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

if (CKPT_IN / "labels").exists():
    labels_dst = data_dir / "labels"
    labels_dst.mkdir(exist_ok=True)
    shutil.copytree(CKPT_IN / "labels", labels_dst, dirs_exist_ok=True)
    print("Labels restored from checkpoints dataset")

# Restore project_state.json but RESET training + meta_labeling so all symbols retrain
if (CKPT_IN / "project_state.json").exists():
    shutil.copy(CKPT_IN / "project_state.json", REPO / "project_state.json")
    # Reset training stages so stage_04/05 reruns all symbols with fresh OOF
    with open(REPO / "project_state.json") as _f:
        _state = json.load(_f)
    _state["stages"]["training"]["status"] = "pending"
    _state["stages"]["training"]["completed_symbols"] = []
    _state["stages"]["meta_labeling"]["status"] = "pending"
    _state["stages"]["meta_labeling"]["completed_symbols"] = []
    with open(REPO / "project_state.json", "w") as _f:
        json.dump(_state, _f, indent=2)
    print("project_state.json restored — training/meta_labeling reset for full retrain")

for d in ["data/features", "data/processed", "data/labels", "results", "logs", "monitoring"]:
    (REPO / d).mkdir(parents=True, exist_ok=True)

# ── 4. GPU + env vars ─────────────────────────────────────────────────────────
os.environ["XGB_DEVICE"] = "cuda"
os.environ["META_WORKERS"] = "1"
os.environ["SKIP_SHAP"] = "1"
print("XGB_DEVICE=cuda, META_WORKERS=1, SKIP_SHAP=1")

# ── 5. Load config ────────────────────────────────────────────────────────────
from src.utils.config_loader import load_config
from omegaconf import OmegaConf
cfg = load_config()

OmegaConf.update(cfg, "model.optuna_n_trials", 25, merge=True)
OmegaConf.update(cfg, "model.stability_n_bootstrap", 20, merge=True)
print(f"Config: train_end={cfg.data.train_end} val_end={cfg.data.val_end} test_start={cfg.data.test_start}")
print(f"Kaggle overrides: optuna_n_trials=25, stability_n_bootstrap=20")

# ── 6. Run pipeline stages ────────────────────────────────────────────────────
from src.pipeline import (
    stage_04_train,
    stage_05_meta,
    stage_06_portfolio,
    stage_07_backtest,
)

import sys as _sys
import traceback as _tb
try:
    _sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass  # Kaggle OutStream doesn't support reconfigure
import logging as _logging
_logging.basicConfig(stream=_sys.stdout, level=_logging.INFO, force=True)

def _save_checkpoint(stage_name: str):
    try:
        if (REPO / "data/checkpoints").exists():
            shutil.copytree(REPO / "data/checkpoints", OUT / "checkpoints", dirs_exist_ok=True)
        if (REPO / "models").exists():
            shutil.copytree(REPO / "models", OUT / "models", dirs_exist_ok=True)
        if (REPO / "data/labels").exists():
            shutil.copytree(REPO / "data/labels", OUT / "labels", dirs_exist_ok=True)
        if (REPO / "results").exists():
            shutil.copytree(REPO / "results", OUT / "results", dirs_exist_ok=True)
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

# Stage 2 always skipped — features linked via Mode A
_feat_files = list((REPO / "data/features").glob("*.parquet"))
print(f"\n=== Stage 2: SKIPPED — {len(_feat_files)} feature files ready (Mode A) ===", flush=True)

# Stage 3 already done in CPU notebook — skip
print(f"\n=== Stage 3: SKIPPED — labels restored from checkpoints dataset ===", flush=True)

_run_stage("Stage 4: training (GPU)", stage_04_train.run, cfg)
_save_checkpoint("stage_04")

_run_stage("Stage 5: meta-labeling",  stage_05_meta.run,      cfg)
_save_checkpoint("stage_05")

_run_stage("Stage 6: portfolio",      stage_06_portfolio.run, cfg)
_save_checkpoint("stage_06")

_run_stage("Stage 7: backtest",       stage_07_backtest.run,  cfg)

# ── 7. Package final output ───────────────────────────────────────────────────
print("\n=== Packaging output ===")
_save_checkpoint("final")

shutil.make_archive(str(WORK / "gpu_output"), "zip", OUT)
print(f"Output zipped: {WORK}/gpu_output.zip")

# ── 8. Auto-push to crypto-model-checkpoints dataset ─────────────────────────
# Requires KAGGLE_USERNAME and KAGGLE_KEY set in notebook Secrets (Add-ons menu)
_kaggle_user = os.environ.get("KAGGLE_USERNAME", "")
_kaggle_key  = os.environ.get("KAGGLE_KEY", "")

if _kaggle_user and _kaggle_key:
    # Write kaggle.json for CLI auth
    kaggle_cfg_dir = Path("/root/.kaggle")
    kaggle_cfg_dir.mkdir(exist_ok=True)
    with open(kaggle_cfg_dir / "kaggle.json", "w") as _kf:
        json.dump({"username": _kaggle_user, "key": _kaggle_key}, _kf)
    (kaggle_cfg_dir / "kaggle.json").chmod(0o600)

    _version_msg = f"retrain {datetime.now(timezone.utc).strftime('%Y-%m-%d')} — stages 4-7 GPU"
    print(f"\n=== Auto-pushing output to crypto-model-checkpoints: '{_version_msg}' ===")

    # Write dataset-metadata.json for the checkpoints dataset
    with open(OUT / "dataset-metadata.json", "w") as _mf:
        json.dump({
            "id": "irfandragneel/crypto-model-checkpoints",
            "licenses": [{"name": "CC0-1.0"}]
        }, _mf)

    _push_result = subprocess.run([
        "kaggle", "datasets", "version",
        "-p", str(OUT),
        "-m", _version_msg,
        "--dir-mode", "zip",
    ], capture_output=True, text=True)

    if _push_result.returncode == 0:
        print("Auto-push SUCCESSFUL — crypto-model-checkpoints updated")
        print(_push_result.stdout)
    else:
        print("Auto-push FAILED — download output manually")
        print(_push_result.stderr)
else:
    print("\nNo KAGGLE_USERNAME/KAGGLE_KEY secrets found — skipping auto-push")
    print("Manually download /kaggle/working/output/ or gpu_output.zip")

print("\nDone.")
