"""
Split feature parquets into N batches and upload each as a separate Kaggle dataset.
Datasets: crypto-model-features-1, -2, -3, -4
Usage: python kaggle_notebook/upload_features.py [--batch 1]  (1-indexed, omit = all)
"""
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

FEATURES_DIR = Path("data/features")
KAGGLE_USER  = "irfandragneel"
N_BATCHES    = 4
BATCH_LIMIT_GB = 5.5  # max per dataset

# ── Collect files sorted by size descending ──────────────────────────────────
files = sorted(FEATURES_DIR.glob("*.parquet"), key=lambda p: p.stat().st_size, reverse=True)

# ── Greedy bin-packing into N_BATCHES ────────────────────────────────────────
batches: list[list[Path]] = [[] for _ in range(N_BATCHES)]
totals  = [0.0] * N_BATCHES

for f in files:
    size_gb = f.stat().st_size / 1024**3
    # Pick bin with most room that still fits
    idx = min(range(N_BATCHES), key=lambda i: totals[i] if totals[i] + size_gb <= BATCH_LIMIT_GB else float("inf"))
    batches[idx].append(f)
    totals[idx] += size_gb

for i, (batch, total) in enumerate(zip(batches, totals), 1):
    print(f"Batch {i}: {len(batch)} files, {total:.2f} GB — {[f.name for f in batch]}")

# ── Which batches to upload ───────────────────────────────────────────────────
if "--batch" in sys.argv:
    run_batches = [int(sys.argv[sys.argv.index("--batch") + 1])]
else:
    run_batches = list(range(1, N_BATCHES + 1))

# ── Upload each batch ─────────────────────────────────────────────────────────
kaggle = str(Path(".venv/Scripts/kaggle"))

for batch_num in run_batches:
    batch = batches[batch_num - 1]
    if not batch:
        print(f"Batch {batch_num}: empty, skipping")
        continue

    slug    = f"crypto-model-features-{batch_num}"
    staging = Path(f"C:/kgl_feat_{batch_num}")
    staging.mkdir(parents=True, exist_ok=True)

    # Write metadata
    meta = {"title": slug, "id": f"{KAGGLE_USER}/{slug}", "licenses": [{"name": "other"}]}
    (staging / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    # Copy files
    for f in batch:
        dst = staging / f.name
        if not dst.exists():
            print(f"  Copying {f.name} ({f.stat().st_size/1024**2:.0f} MB)...")
            shutil.copy2(f, dst)

    total_gb = sum(f.stat().st_size for f in batch) / 1024**3
    print(f"\nUploading batch {batch_num} ({len(batch)} files, {total_gb:.2f} GB)...")

    # Try create first, fall back to version if already exists
    result = subprocess.run(
        [kaggle, "datasets", "create", "-p", str(staging), "--dir-mode", "zip"],
        capture_output=True, text=True
    )
    if "already exists" in result.stderr or "already exists" in result.stdout or result.returncode != 0:
        print(f"  Dataset exists, creating new version...")
        result = subprocess.run(
            [kaggle, "datasets", "version", "-p", str(staging), "-m", "feature update"],
            capture_output=True, text=True
        )
    print(result.stdout or result.stderr)
    if result.returncode == 0:
        print(f"Batch {batch_num} upload complete.")
    else:
        print(f"Batch {batch_num} FAILED. Re-run with: python kaggle_notebook/upload_features.py --batch {batch_num}")
        sys.exit(1)

print("\nAll batches done.")
