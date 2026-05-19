"""Logging helpers."""

from __future__ import annotations

import logging


def get_logger(name: str = "biosound_cluster") -> logging.Logger:
    """Return a package logger."""
    return logging.getLogger(name)


def configure_logging(verbose: bool = False) -> None:
    """Configure basic console logging for CLI use."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy_logger in ("matplotlib", "numba"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
