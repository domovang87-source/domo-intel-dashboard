"""Logging: colourless, timestamped, mirrored to data/logs/creator_intel.log."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from .config import settings

_CONFIGURED = False

_FMT = "%(asctime)s  %(levelname)-7s  %(name)-28s  %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, (level or settings.log_level), logging.INFO))

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    root.addHandler(stream)

    file_handler = RotatingFileHandler(
        settings.logs_dir / "creator_intel.log",
        maxBytes=8 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(_FMT.replace("%(asctime)s", "%(asctime)s")))
    root.addHandler(file_handler)

    # Third-party chatter we do not want at INFO.
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "instaloader", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
