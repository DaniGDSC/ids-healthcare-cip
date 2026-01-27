"""Logging utilities with structured JSON support and metric helpers."""

import logging
import sys
import time
from contextlib import contextmanager
from typing import Optional, Any, Dict

from pythonjsonlogger import jsonlogger


def _build_formatter(structured: bool) -> logging.Formatter:
    if structured:
        return jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    return logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def get_logger(
    name: str,
    level: str = 'INFO',
    log_file: Optional[str] = None,
    structured: bool = False
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = _build_formatter(structured)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def log_metric(logger: logging.Logger, name: str, value: Any, **labels: Any) -> None:
    logger.info({"event": "metric", "metric": name, "value": value, **labels})


def log_alert(logger: logging.Logger, message: str, severity: str = "error", **fields: Any) -> None:
    logger.error({"event": "alert", "severity": severity, "message": message, **fields})


@contextmanager
def time_block(logger: logging.Logger, name: str, **fields: Any):
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        log_metric(logger, f"duration.{name}", duration, **fields)