import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("model_versioning")

_REGISTRY_PATH = Path("model_registry.json")
_LOCK_PATH = Path("model_registry.json.lock")


def _acquire_lock(timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            fd = os.open(str(_LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            time.sleep(0.05)
    raise TimeoutError(f"Could not acquire registry lock within {timeout}s")


def _release_lock() -> None:
    try:
        _LOCK_PATH.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Failed to release registry lock: {e}")


def generate_version_string(
    symbol: str,
    tf: str,
    feature_names: list,
    hyperparams: dict,
    train_start: str,
    train_end: str,
) -> str:
    # SHA256 of sorted feature names + sorted hyperparams + train dates
    hash_input = (
        str(sorted(feature_names))
        + str(sorted(hyperparams.items()))
        + str(train_start)
        + str(train_end)
    )
    h = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{symbol}_{tf}_{ts}_{h}"


def register_model(
    symbol: str,
    tf: str,
    version: str,
    metrics: dict,
    feature_names: list,
    hyperparams: dict,
    train_period: tuple,
    model_path: str,
    model_type: str = "primary",
) -> None:
    entry = {
        "version": version,
        "symbol": symbol,
        "tf": tf,
        "model_type": model_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "train_start": str(train_period[0]),
        "train_end": str(train_period[1]),
        "model_path": str(model_path),
        "hyperparams": hyperparams,
    }

    _acquire_lock()
    try:
        if _REGISTRY_PATH.exists():
            with open(_REGISTRY_PATH) as f:
                registry = json.load(f)
        else:
            registry = {"models": []}

        registry["models"].append(entry)

        with open(_REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=2)
        logger.info(f"Model registered: {version} ({model_type}) for {symbol} {tf}")
    finally:
        _release_lock()


def get_latest_model(symbol: str, tf: str, model_type: str = "primary") -> dict | None:
    if not _REGISTRY_PATH.exists():
        return None

    # Acquire lock to prevent reading while concurrent write is in progress
    _acquire_lock()
    try:
        with open(_REGISTRY_PATH) as f:
            registry = json.load(f)

        matching = [
            m for m in registry.get("models", [])
            if m["symbol"] == symbol and m["tf"] == tf and m["model_type"] == model_type
        ]

        if not matching:
            return None

        # Sort by created_at descending
        matching.sort(key=lambda x: x["created_at"], reverse=True)
        return matching[0]
    finally:
        _release_lock()


def get_active_models() -> dict:
    if not _REGISTRY_PATH.exists():
        return {}

    # Acquire lock to prevent reading while concurrent write is in progress
    _acquire_lock()
    try:
        with open(_REGISTRY_PATH) as f:
            registry = json.load(f)

        # Group by (symbol, tf) and return latest primary model per pair
        latest = {}
        for m in registry.get("models", []):
            if m["model_type"] != "primary":
                continue
            key = f"{m['symbol']}_{m['tf']}"
            if key not in latest or m["created_at"] > latest[key]["created_at"]:
                latest[key] = m

        return latest
    finally:
        _release_lock()
