"""Structured logging configuration using structlog."""

import logging
import sys
from typing import Literal

import orjson
import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars, unbind_contextvars


def configure_logging(
    level: str = "INFO",
    format: Literal["json", "console"] = "console",
) -> None:
    """Configure structlog for production-ready logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format - "json" for production, "console" for development
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Common processors
    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if format == "json":
        # Production: JSON output to stdout
        processors.append(
            structlog.processors.JSONRenderer(serializer=orjson.dumps)
        )
        logger_factory = structlog.BytesLoggerFactory(file=sys.stdout.buffer)
    else:
        # Development: colored console output
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
        logger_factory = structlog.WriteLoggerFactory()

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=logger_factory,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger instance with the given name.

    Args:
        name: Logger name, typically __name__

    Returns:
        Configured structlog BoundLogger
    """
    return structlog.get_logger(name)


# Re-export context management functions
__all__ = [
    "configure_logging",
    "get_logger",
    "bind_contextvars",
    "clear_contextvars",
    "unbind_contextvars",
]
