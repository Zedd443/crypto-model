"""
Delete all pipeline-generated artifacts (stage 3+). Raw data and features are NEVER touched.

Usage:
    .venv/Scripts/python.exe scripts/clean_generated.py              # dry-run
    .venv/Scripts/python.exe scripts/clean_generated.py --confirm    # actually delete
    .venv/Scripts/python.exe scripts/clean_generated.py --stage 3    # only labels
    .venv/Scripts/python.exe scripts/clean_generated.py --stage 4    # labels + checkpoints + models
    .venv/Scripts/python.exe scripts/clean_generated.py --stage 5    # + meta
    .venv/Scripts/python.exe scripts/clean_generated.py --stage 6    # + signals
    .venv/Scripts/python.exe scripts/clean_generated.py --stage 7    # + results/backtest
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger("clean_generated")

# Safe dirs — never deleted, only listed for safety check
_SAFE = {
    ROOT / "data" / "raw",
    ROOT / "data" / "features",
}

# What each stage produces. Each entry: (path, glob_pattern, is_dir)
# is_dir=True means delete the whole directory tree
_STAGE_ARTIFACTS: dict[int, list[tuple[Path, str, bool]]] = {
    3: [
        (ROOT / "data" / "labels",                          "*.parquet",  False),
    ],
    4: [
        (ROOT / "data" / "checkpoints" / "imputers",        "*",          True),
        (ROOT / "data" / "checkpoints" / "fracdiff",        "*",          True),
        (ROOT / "data" / "checkpoints" / "hmm",             "*",          True),
        (ROOT / "data" / "checkpoints" / "oof",             "*",          True),
        (ROOT / "data" / "checkpoints" / "feature_selection","*",         True),
        (ROOT / "data" / "checkpoints" / "diagnostics",     "*",          True),
        (ROOT / "data" / "checkpoints" / "cross_sectional_stats.pkl", "", False),
        (ROOT / "models",                                   "*.json",     False),
        (ROOT / "models",                                   "*.pkl",      False),
        (ROOT / "results",                                  "training_summary.csv", False),
        (ROOT / "results",                                  "pipeline_diagnostics.csv", False),
    ],
    5: [
        (ROOT / "results",                                  "meta_summary.csv", False),
        (ROOT / "data" / "checkpoints" / "portfolio_weights.json", "",    False),
    ],
    6: [
        (ROOT / "data" / "checkpoints" / "signals",         "*",          True),
    ],
    7: [
        (ROOT / "results",                                  "backtest_summary.csv", False),
        (ROOT / "results",                                  "trade_log.csv",        False),
        (ROOT / "results",                                  "equity_curve.csv",     False),
        (ROOT / "results",                                  "*_nav.parquet",        False),
        (ROOT / "results",                                  "*_trades.csv",         False),
    ],
}


def _collect(from_stage: int) -> list[Path]:
    targets: list[Path] = []
    for stage in range(from_stage, max(_STAGE_ARTIFACTS) + 1):
        for entry in _STAGE_ARTIFACTS.get(stage, []):
            path, pattern, is_dir = entry

            # Single explicit file (no glob)
            if pattern == "":
                if path.exists():
                    # Safety: never touch raw/features
                    for safe in _SAFE:
                        if path == safe or safe in path.parents:
                            logger.error(f"SAFETY BLOCK: {path} is inside {safe}")
                            continue
                    targets.append(path)
                continue

            if is_dir:
                if path.exists() and path.is_dir():
                    for safe in _SAFE:
                        if path == safe or safe in path.parents:
                            logger.error(f"SAFETY BLOCK: {path} is inside {safe}")
                            break
                    else:
                        targets.append(path)
            else:
                if path.exists() and path.is_dir():
                    for p in path.glob(pattern):
                        for safe in _SAFE:
                            if p == safe or safe in p.parents:
                                logger.error(f"SAFETY BLOCK: {p} is inside {safe}")
                                break
                        else:
                            targets.append(p)
    return targets


def _human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f}{unit}"
        n_bytes //= 1024
    return f"{n_bytes:.1f}TB"


def _total_size(paths: list[Path]) -> int:
    total = 0
    for p in paths:
        try:
            if p.is_dir():
                total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            elif p.exists():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true", help="actually delete (default: dry-run)")
    ap.add_argument("--stage", type=int, default=3, choices=range(3, 8),
                    help="delete artifacts from this stage onwards (default: 3 = everything)")
    args = ap.parse_args()

    targets = _collect(args.stage)

    if not targets:
        logger.info("Nothing to clean.")
        return

    total = _total_size(targets)
    logger.info(f"Stage {args.stage}+ artifacts to remove: {len(targets)} paths, ~{_human_size(total)}")
    for p in targets[:20]:
        label = "[DIR]" if p.is_dir() else "     "
        logger.info(f"  {label} {p.relative_to(ROOT)}")
    if len(targets) > 20:
        logger.info(f"  ... and {len(targets) - 20} more")

    if not args.confirm:
        logger.info("Dry-run — pass --confirm to actually delete.")
        return

    deleted = 0
    for p in targets:
        try:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
            deleted += 1
        except Exception as exc:
            logger.warning(f"Failed to delete {p}: {exc}")

    logger.info(f"Deleted {deleted} paths, freed ~{_human_size(total)}")

    # Reset project_state.json stage statuses
    state_path = ROOT / "project_state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            stage_map = {3: "labels", 4: "training", 5: "meta_labeling", 6: "signals", 7: "backtest"}
            for s in range(args.stage, max(stage_map) + 1):
                key = stage_map.get(s)
                if key and key in state.get("stages", {}):
                    state["stages"][key]["status"] = "pending"
                    state["stages"][key]["hash"] = None
                    state["stages"][key]["last_run"] = None
                    if "completed_symbols" in state["stages"][key]:
                        state["stages"][key]["completed_symbols"] = []
                    if "failed_symbols" in state["stages"][key]:
                        state["stages"][key]["failed_symbols"] = {}
            state_path.write_text(json.dumps(state, indent=2))
            logger.info("project_state.json stage statuses reset.")
        except Exception as exc:
            logger.warning(f"Could not update project_state.json: {exc}")


if __name__ == "__main__":
    main()
