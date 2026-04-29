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


_CONFIG_HASH_KEYS = [
    # Model hyperparams / objective — changes invalidate trained models
    "model.objective_sortino_weight",
    "model.objective_cvar_weight",
    "model.objective_dead_zone",
    "model.embargo_bars",
    "model.cv_n_splits",
    "model.meta_n_estimators",
    "model.meta_max_depth",
    "model.meta_early_stopping_rounds",
    # Label geometry — changes invalidate labels AND models
    "labels.tp_atr_mult",
    "labels.sl_atr_mult",
    "labels.max_hold_bars",
    "labels.min_return_pct",
    # HMM regime — changes invalidate HMM and downstream models
    "hmm.n_states",
    # Training dates — changes invalidate models trained on old splits
    "data.train_end",
    "data.val_end",
]


def _get_nested(cfg, dotted_key: str):
    """Resolve 'a.b.c' → cfg.a.b.c, returning None if any level is missing."""
    parts = dotted_key.split(".")
    obj = cfg
    for p in parts:
        obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj


def compute_config_hash(cfg) -> str:
    """SHA256 of the model-relevant config keys. Use to detect stale models."""
    snapshot = {}
    for key in _CONFIG_HASH_KEYS:
        val = _get_nested(cfg, key)
        snapshot[key] = str(val) if val is not None else "__missing__"
    payload = json.dumps(snapshot, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def generate_version_string(
    symbol: str,
    tf: str,
    feature_names: list,
    hyperparams: dict,
    train_start: str,
    train_end: str,
) -> str:
    # Fixed filename — retrain always overwrites the same file on disk.
    # History is preserved in the registry (config_hash + created_at per entry).
    return f"{symbol}_{tf}"


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
    cfg=None,
) -> None:
    config_hash = compute_config_hash(cfg) if cfg is not None else "__no_cfg__"
    entry = {
        "version": version,
        "symbol": symbol,
        "tf": tf,
        "model_type": model_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash,
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
        registry = {"models": []}
        if _REGISTRY_PATH.exists():
            try:
                with open(_REGISTRY_PATH) as f:
                    loaded = json.load(f)
                # Handle both {"models": [...]} dict and legacy bare-list formats
                if isinstance(loaded, dict) and "models" in loaded:
                    registry = loaded
                elif isinstance(loaded, list):
                    logger.warning("model_registry.json was a bare list — converting to {models: []} format")
                    registry = {"models": loaded}
                else:
                    logger.warning(f"model_registry.json has unexpected format ({type(loaded).__name__}) — starting fresh")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not read model_registry.json ({e}) — starting fresh")

        # Upsert: replace existing entry for (symbol, tf, model_type) to avoid stale history bloat
        key = (symbol, tf, model_type)
        registry["models"] = [
            m for m in registry["models"]
            if not (m["symbol"] == symbol and m["tf"] == tf and m["model_type"] == model_type)
        ]
        registry["models"].append(entry)

        with open(_REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=2)
        logger.info(f"Model registered: {version} ({model_type}) for {symbol} {tf}")
    finally:
        _release_lock()


def _read_registry() -> dict:
    # Read registry without holding the write lock — reads are safe to race with other reads.
    # If the file is being written concurrently, retry up to 3 times with a short backoff.
    for attempt in range(3):
        try:
            with open(_REGISTRY_PATH) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict) and "models" in loaded:
                return loaded
            if isinstance(loaded, list):
                return {"models": loaded}
            return {"models": []}
        except (json.JSONDecodeError, OSError):
            if attempt < 2:
                time.sleep(0.1)
    return {"models": []}


def get_latest_model(symbol: str, tf: str, model_type: str = "primary", cfg=None) -> dict | None:
    if not _REGISTRY_PATH.exists():
        return None

    registry = _read_registry()
    matching = [
        m for m in registry.get("models", [])
        if m["symbol"] == symbol and m["tf"] == tf and m["model_type"] == model_type
    ]

    if not matching:
        return None

    # Sort by created_at descending
    matching.sort(key=lambda x: x["created_at"], reverse=True)
    best = matching[0]

    # Warn if model was trained with a different config
    if cfg is not None:
        current_hash = compute_config_hash(cfg)
        stored_hash = best.get("config_hash", "__missing__")
        if stored_hash not in ("__missing__", "__no_cfg__") and stored_hash != current_hash:
            logger.warning(
                f"STALE MODEL: {symbol} {tf} {model_type} — config has changed since training. "
                f"stored_hash={stored_hash} current_hash={current_hash}. "
                f"Retrain recommended (--stage 4 --force)."
            )

    return best


def get_active_models() -> dict:
    if not _REGISTRY_PATH.exists():
        return {}

    registry = _read_registry()
    # Group by (symbol, tf) and return latest primary model per pair
    latest = {}
    for m in registry.get("models", []):
        if m["model_type"] != "primary":
            continue
        key = f"{m['symbol']}_{m['tf']}"
        if key not in latest or m["created_at"] > latest[key]["created_at"]:
            latest[key] = m

    return latest
