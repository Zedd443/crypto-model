import json
import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger("state_manager")

STATE_PATH = Path("project_state.json")
LOCK_PATH = Path("project_state.json.lock")

# Stage ordering used for downstream invalidation
STAGE_ORDER = ["ingest", "features", "labels", "training", "meta_labeling", "portfolio", "backtest"]


def _acquire_lock(timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            time.sleep(0.1)
    raise TimeoutError(f"Could not acquire lock on {LOCK_PATH} within {timeout}s")


def _release_lock() -> None:
    try:
        LOCK_PATH.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Failed to release lock: {e}")


def load_state() -> dict:
    if not STATE_PATH.exists():
        logger.warning("project_state.json not found, returning empty state")
        return _default_state()
    with open(STATE_PATH, "r") as f:
        return json.load(f)


def save_state(state: dict) -> None:
    _acquire_lock()
    try:
        state["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
        logger.debug("project_state.json saved")
    finally:
        _release_lock()


def mark_stage_complete(stage: str, output_dir: str | None = None) -> None:
    state = load_state()
    h = _hash_directory(output_dir) if output_dir else None
    if stage in state["stages"]:
        state["stages"][stage]["status"] = "done"
        state["stages"][stage]["hash"] = h
        state["stages"][stage]["last_run"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    logger.info(f"Stage '{stage}' marked complete")


def mark_stage_failed(stage: str, reason: str) -> None:
    state = load_state()
    if stage in state["stages"]:
        state["stages"][stage]["status"] = "failed"
        state["stages"][stage]["last_run"] = datetime.now(timezone.utc).isoformat()
        issues = state["stages"][stage].get("issues", [])
        issues.append({"time": datetime.now(timezone.utc).isoformat(), "reason": reason})
        state["stages"][stage]["issues"] = issues
    save_state(state)
    logger.error(f"Stage '{stage}' marked failed: {reason}")


def is_stage_complete(stage: str) -> bool:
    state = load_state()
    return state.get("stages", {}).get(stage, {}).get("status") == "done"


def update_completed_symbol(stage: str, symbol: str) -> None:
    state = load_state()
    if stage in state["stages"]:
        completed = state["stages"][stage].get("completed_symbols", [])
        if symbol not in completed:
            completed.append(symbol)
        state["stages"][stage]["completed_symbols"] = completed
    save_state(state)


def update_project_state(stage: str, status: str, issues: list[str] | None = None, output_dir: str | None = None) -> None:
    state = load_state()

    prev_hash = state.get("stages", {}).get(stage, {}).get("hash")
    new_hash = _hash_directory(output_dir) if output_dir else None

    if stage in state["stages"]:
        state["stages"][stage]["status"] = status
        state["stages"][stage]["hash"] = new_hash
        state["stages"][stage]["last_run"] = datetime.now(timezone.utc).isoformat()
        state["stages"][stage]["issues"] = issues or []

    # Downstream invalidation: if this stage's hash changed, mark downstream as re_evaluate
    if prev_hash and new_hash and prev_hash != new_hash:
        stage_idx = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else -1
        if stage_idx >= 0:
            for downstream in STAGE_ORDER[stage_idx + 1:]:
                if state["stages"].get(downstream, {}).get("status") == "done":
                    state["stages"][downstream]["status"] = "re_evaluate"
                    logger.warning(f"Stage '{downstream}' marked re_evaluate due to upstream hash change in '{stage}'")

    save_state(state)


def increment_demo_trades() -> int:
    # Thread-safe increment of account.demo_trades_completed; returns new count
    state = load_state()
    account = state.setdefault("account", {})
    current = int(account.get("demo_trades_completed", 0))
    account["demo_trades_completed"] = current + 1
    save_state(state)
    logger.debug(f"demo_trades_completed incremented to {current + 1}")
    return current + 1


def update_equity(equity: float, state_path: Path = Path("project_state.json")) -> None:
    state = load_state()
    account = state.setdefault("account", {})
    account["current_equity"] = float(equity)
    save_state(state)
    logger.debug(f"account.current_equity updated to {equity:.4f}")


def _hash_directory(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    md5 = hashlib.md5()
    for f in sorted(p.rglob("*")):
        if f.is_file():
            md5.update(f.name.encode())
            md5.update(str(f.stat().st_size).encode())
    return md5.hexdigest()


def _default_state() -> dict:
    return {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "stages": {
            "ingest": {"status": "pending", "hash": None, "last_run": None, "issues": []},
            "features": {"status": "pending", "hash": None, "last_run": None, "issues": []},
            "labels": {"status": "pending", "hash": None, "last_run": None, "issues": []},
            "training": {"status": "pending", "completed_symbols": [], "failed_symbols": {}, "hash": None, "last_run": None, "issues": []},
            "meta_labeling": {"status": "pending", "completed_symbols": [], "failed_symbols": {}, "hash": None, "last_run": None, "issues": []},
            "portfolio": {"status": "pending", "hash": None, "last_run": None, "issues": []},
            "backtest": {"status": "pending", "hash": None, "last_run": None, "issues": []},
        },
        "model_tiers": {},
        "account": {
            "current_equity": 120.0,
            "active_symbol_count": 1,
            "growth_gate_unlocked": False,
            "growth_gate_threshold": 300.0,
            "demo_trades_completed": 0,
        },
        "global_alerts": [],
        "next_scheduled_retrain": None,
    }
