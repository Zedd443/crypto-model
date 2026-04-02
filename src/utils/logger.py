from pathlib import Path
from loguru import logger as _logger
import sys

_configured = False

def get_logger(name: str):
    global _configured
    if not _configured:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        _logger.remove()
        _logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
        _logger.add(log_dir / f"{name}.log", rotation="50 MB", retention="30 days", level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")
        _configured = True
    return _logger.bind(name=name)
