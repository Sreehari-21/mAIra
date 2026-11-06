from __future__ import annotations

import logging
import sys
from typing import Any

from app.core.config import get_settings


def setup_logger(name: str = "genai_platform") -> logging.Logger:
    """Configures and returns a structured logger for the platform."""
    settings = get_settings()

    logger = logging.getLogger(name)

    if not logger.handlers:
        log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(log_level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


logger = setup_logger()

