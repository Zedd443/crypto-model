"""
Split feature parquets into N batches and upload each as a separate Kaggle dataset.
Datasets: crypto-model-features-1, -2, -3, -4

Usage:
  python kaggle_notebook/upload_features.py            # upload all 4 batches
  python kaggle_notebook/upload_features.py --batch 3  # upload batch 3 only (1-indexed)

If a batch upload is interrupted, re-run with --batch N to resume that batch only.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    # Fallback: tqdm not installed — wrap with a no-op
    def tqdm(it, **kwargs):
        desc = kwargs.get("desc", "")
        total = kwargs.get("total", None)
        print(f"{desc} ({total} items)" if total else desc)
        return it

FEATURES_DIR   = Path("data/features")
KAGGLE_USER    = "irfandragneel"
N_BATCHES      = 4
BATCH_LIMIT_GB = 5.5  # soft cap per dataset (Kaggle limit is 100 GB, we stay well under)

# ── Collect all feature parquets, largest first (greedy bin-packing works better) ──
files = sorted(FEATURES_DIR.glob("*.parquet"), key=lambda p: p.stat().st_size, reverse=True)

# ── Greedy bin-packing: assign each file to the batch with most remaining space ──
batches: list[list[Path]] = [[] for _ in range(N_BATCHES)]
totals  = [0.0] * N_BATCHES

for f in files:
    size_gb = f.stat().st_size / 1024 ** 3
    idx = min(
        range(N_BATCHES),
        key=lambda i: totals[i] if totals[i] + size_gb <= BATCH_LIMIT_GB else float("inf"),
    )
    batches[idx].append(f)
    totals[idx] += size_gb

print("Batch plan:")
for i, (batch, total) in enumerate(zip(batches, totals), 1):
    print(f"  Batch {i}: {len(batch):2d} files  {total:.2f} GB")

# ── Determine which batches to run ───────────────────────────────────────────────
if "--batch" in sys.argv:
    run_batches = [int(sys.argv[sys.argv.index("--batch") + 1])]
else:
    run_batches = list(range(1, N_BATCHES + 1))

# ── Locate kaggle CLI inside the venv ────────────────────────────────────────────
kaggle = str(Path(".venv/Scripts/kaggle"))

# ── Upload each batch ────────────────────────────────────────────────────────────
for batch_num in run_batches:
    batch = batches[batch_num - 1]
    if not batch:
        print(f"\nBatch {batch_num}: empty — skipping")
        continue

    slug    = f"crypto-model-features-{batch_num}"
    staging = Path(f"C:/kgl_feat_{batch_num}")
    staging.mkdir(parents=True, exist_ok=True)

    # Dataset metadata required by kaggle CLI
    meta = {
        "title": slug,
        "id": f"{KAGGLE_USER}/{slug}",
        "licenses": [{"name": "other"}],
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    # Copy files with per-file progress bar
    total_gb = sum(f.stat().st_size for f in batch) / 1024 ** 3
    print(f"\nBatch {batch_num} — {len(batch)} files  {total_gb:.2f} GB  →  {staging}")

    total_bytes = sum(f.stat().st_size for f in batch)
    copied_bytes = 0

    with tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024,
              desc=f"  Copying batch {batch_num}", ncols=90) as pbar:
        for f in batch:
            dst = staging / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
            file_bytes = f.stat().st_size
            pbar.update(file_bytes)
            copied_bytes += file_bytes

    # Upload: create new dataset or bump version if it already exists
    print(f"  Uploading to Kaggle as '{slug}'...")
    result = subprocess.run(
        [kaggle, "datasets", "create", "-p", str(staging), "--dir-mode", "zip"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or "already exists" in (result.stderr + result.stdout):
        print("  Dataset already exists — creating new version...")
        result = subprocess.run(
            [kaggle, "datasets", "version", "-p", str(staging), "-m", "feature update"],
            capture_output=True, text=True,
        )

    output = (result.stdout or result.stderr).strip()
    if output:
        print(f"  {output}")

    if result.returncode == 0:
        print(f"  Batch {batch_num} upload complete.")
    else:
        print(f"\n  Batch {batch_num} FAILED.")
        print(f"  Retry with: python kaggle_notebook/upload_features.py --batch {batch_num}")
        sys.exit(1)

print("\nAll batches done.")
