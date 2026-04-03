"""
crypto-model Kaggle training script
Runs stages 3-7 (features already in checkpoints dataset, or re-run stage 2 if needed)
GPU: XGB_DEVICE=cuda is set automatically when GPU is enabled
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

print("\n=== Stage 2: features ===")
stage_02_features.run(cfg, force=False)  # skips if already in checkpoints

print("\n=== Stage 3: labels ===")
stage_03_labels.run(cfg)

print("\n=== Stage 4: training (GPU) ===")
stage_04_train.run(cfg)

print("\n=== Stage 5: meta-labeling ===")
stage_05_meta.run(cfg)

print("\n=== Stage 6: portfolio ===")
stage_06_portfolio.run(cfg)

print("\n=== Stage 7: backtest ===")
stage_07_backtest.run(cfg)

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
