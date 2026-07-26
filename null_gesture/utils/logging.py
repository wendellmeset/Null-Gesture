"""Shared logging setup for Null-Gesture."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from null_gesture.config import PROJECT_ROOT

LOG_DIR: Path = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE: Path = LOG_DIR / "null_gesture.log"


def setup_logging(*, verbose: bool = False) -> logging.Logger:
    """Configure and return the root logger for the package."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger("null_gesture")
    logger.setLevel(level)

    # File handler (always)
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger
