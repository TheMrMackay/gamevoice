"""GameVoice - read any game's on-screen dialogue aloud, in character."""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

__version__ = "1.0.0"

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-18s %(message)s"


def setup_logging(level: int = logging.INFO, to_file: bool = True) -> Path | None:
    """Configure root logging once. Returns the log file path, if any."""
    root = logging.getLogger()
    if root.handlers:
        return None

    root.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S")

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    if not to_file:
        return None

    from .config import user_data_dir

    try:
        log_dir = user_data_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "gamevoice.log"
        rotating = logging.handlers.RotatingFileHandler(
            path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        rotating.setFormatter(formatter)
        root.addHandler(rotating)
        return path
    except OSError as exc:
        root.warning("file logging disabled: %s", exc)
        return None
