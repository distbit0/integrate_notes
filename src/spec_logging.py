from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from spec_config import LOG_FILE_ROTATION_BYTES


def configure_logging(log_path: Path) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", enqueue=False)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RuntimeError(f"Failed to prepare log directory {log_path.parent}: {error}") from error
    logger.add(
        log_path,
        level="DEBUG",
        rotation=LOG_FILE_ROTATION_BYTES,
        enqueue=False,
        encoding="utf-8",
    )
