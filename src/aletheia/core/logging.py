"""Structured logging.

One log line per event, JSON in deployed environments and human-readable locally. Every request binds a
request id into the context so a log line can be traced back to the response the user saw.
"""

from __future__ import annotations

import logging
import sys

import structlog
import structlog.dev

_ALLOWED_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def configure_logging(level: str = "INFO", *, json: bool = False) -> None:
    """Configure stdlib logging and structlog once, at process start."""
    normalised = level.upper()
    numeric = getattr(logging, normalised, None)
    if normalised not in _ALLOWED_LEVELS or not isinstance(numeric, int):
        raise ValueError(f"unknown log level: {level!r}")

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric,
        force=True,
    )

    renderer: structlog.types.Processor
    renderer = (
        structlog.processors.JSONRenderer()
        if json
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
