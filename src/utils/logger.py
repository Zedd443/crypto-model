from pathlib import Path
from loguru import logger as _logger
import sys

_stderr_added = False
_file_sinks: set = set()

def get_logger(name: str):
    global _stderr_added
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    if not _stderr_added:
        _logger.remove()
        _logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
        _stderr_added = True

    if name not in _file_sinks:
        _logger.add(log_dir / f"{name}.log", rotation="50 MB", retention="30 days", level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")
        _file_sinks.add(name)

    return _logger.bind(name=name)
