"""Logging configuration via loguru."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure loguru for the application.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to write logs to disk.
    """
    logger.remove()
    fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
    logger.add(sys.stderr, level=level, format=fmt)
    if log_file is not None:
        logger.add(log_file, level=level, format=fmt, rotation="10 MB")


def get_logger(name: str) -> Logger:
    """Return a loguru logger bound to the given name.

    Args:
        name: Module or component name to bind.

    Returns:
        A bound loguru logger instance.
    """
    return logger.bind(name=name)
