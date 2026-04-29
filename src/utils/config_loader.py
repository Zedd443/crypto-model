from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from src.utils.logger import get_logger

logger = get_logger("config_loader")

_config_cache: DictConfig | None = None


def load_config(config_path: str = "config/base.yaml") -> DictConfig:
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    base_path = Path(config_path)
    symbols_path = Path("config/symbols.yaml")

    if not base_path.exists():
        raise FileNotFoundError(f"Config not found: {base_path}")

    base_cfg = OmegaConf.load(base_path)

    if symbols_path.exists():
        symbols_cfg = OmegaConf.load(symbols_path)
        cfg = OmegaConf.merge(base_cfg, symbols_cfg)
    else:
        logger.warning(f"symbols.yaml not found at {symbols_path}, proceeding without symbol metadata")
        cfg = base_cfg

    # Env var overrides for date splits — used by Kaggle weekly retrain to shift window forward.
    # TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE, TEST_START_DATE must be YYYY-MM-DD strings.
    import os
    _date_overrides = {
        "train_end":   os.environ.get("TRAIN_END_DATE"),
        "val_start":   os.environ.get("VAL_START_DATE"),
        "val_end":     os.environ.get("VAL_END_DATE"),
        "test_start":  os.environ.get("TEST_START_DATE"),
    }
    for key, val in _date_overrides.items():
        if val:
            OmegaConf.update(cfg, f"data.{key}", val, merge=True)
            logger.info(f"Config override: data.{key} = {val} (from env)")

    _validate_config(cfg)
    _config_cache = cfg
    logger.info(f"Config loaded from {base_path}")
    return cfg


def _validate_config(cfg: DictConfig) -> None:
    import pandas as pd

    train_end = pd.Timestamp(cfg.data.train_end)
    val_start = pd.Timestamp(cfg.data.val_start)
    val_end = pd.Timestamp(cfg.data.val_end)
    test_start = pd.Timestamp(cfg.data.test_start)

    if not (train_end < val_start):
        raise ValueError(f"train_end ({train_end}) must be < val_start ({val_start})")
    if not (val_start < val_end):
        raise ValueError(f"val_start ({val_start}) must be < val_end ({val_end})")
    if not (val_end < test_start):
        raise ValueError(f"val_end ({val_end}) must be < test_start ({test_start})")

    logger.debug("Config date validation passed")


def get_symbols(cfg: DictConfig) -> list[dict]:
    if not hasattr(cfg, "symbols"):
        logger.warning("No symbols section in config")
        return []

    result = []
    for symbol, meta in cfg.symbols.items():
        meta_dict = OmegaConf.to_container(meta, resolve=True)
        is_active = meta_dict.get("is_active", True)
        if is_active:
            result.append({"symbol": symbol, **meta_dict})

    return result
