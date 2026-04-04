"""
crypto-model CPU notebook — Stage 3 (labels) only. No GPU required.

Run this FIRST before the GPU notebook.

What this does:
  1. Clone latest code from GitHub
  2. Install dependencies (CPU only — no xgboost GPU build needed)
  3. Link raw OHLCV data from dataset
  4. Run Stage 3: triple-barrier labeling for all symbols
  5. Save output (labels + checkpoints) for upload to crypto-model-checkpoints dataset

After this completes:
  - Download /kaggle/working/output/ or version it as a new crypto-model-checkpoints dataset
  - Then run notebook_gpu.py which picks up from Stage 4

Dataset mount paths:
  /kaggle/input/crypto-model-raw-data/   <- raw OHLCV (required)
  /kaggle/input/crypto-model-checkpoints/ <- optional, restore prior state if attached
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

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

# ── 2. Install dependencies (CPU only) ────────────────────────────────────────
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "--no-deps",
    "hmmlearn>=0.3.0", "ruptures>=1.1.0",
    "omegaconf>=2.3.0", "loguru>=0.7.0", "pyarrow>=14.0.0",
    "scikit-learn==1.6.1",
], check=True)

# ── 3. Wire data directories ───────────────────────────────────────────────────
data_dir = REPO / "data"
data_dir.mkdir(exist_ok=True)

raw_link = data_dir / "raw"
if not raw_link.exists():
    raw_link.symlink_to(RAW)

# Restore prior checkpoints if attached (resume labeling if partially done)
ckpt_dst = data_dir / "checkpoints"
ckpt_dst.mkdir(exist_ok=True)
if CKPT_IN.exists():
    shutil.copytree(CKPT_IN, ckpt_dst, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("dataset-metadata.json"))
    print("Checkpoints restored from dataset")

if (CKPT_IN / "labels").exists():
    labels_dst = data_dir / "labels"
    labels_dst.mkdir(exist_ok=True)
    shutil.copytree(CKPT_IN / "labels", labels_dst, dirs_exist_ok=True)
    print("Labels restored from checkpoints dataset")

if (CKPT_IN / "project_state.json").exists():
    shutil.copy(CKPT_IN / "project_state.json", REPO / "project_state.json")
    print("Restored project_state.json")

for d in ["data/processed", "data/labels", "results", "logs", "monitoring"]:
    (REPO / d).mkdir(parents=True, exist_ok=True)

# ── 4. Config + env ───────────────────────────────────────────────────────────
os.environ["LABEL_WORKERS"] = "4"   # CPU notebook: use more workers
print("LABEL_WORKERS=4")

from src.utils.config_loader import load_config
cfg = load_config()

# ── 5. Run Stage 3 ────────────────────────────────────────────────────────────
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
        if (REPO / "data/labels").exists():
            shutil.copytree(REPO / "data/labels", OUT / "labels", dirs_exist_ok=True)
        for f in ["project_state.json"]:
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

from src.pipeline import stage_03_labels
_run_stage("Stage 3: labels (CPU)", stage_03_labels.run, cfg)
_save_checkpoint("stage_03")

# ── 6. Package output ─────────────────────────────────────────────────────────
print("\n=== Packaging output ===")
_save_checkpoint("final")
shutil.make_archive(str(WORK / "cpu_output"), "zip", OUT)
print(f"Output zipped: {WORK}/cpu_output.zip")
print(f"Done. Upload /kaggle/working/output/ as new version of crypto-model-checkpoints dataset.")
print(f"Then run notebook_gpu.py for stages 4-7.")
