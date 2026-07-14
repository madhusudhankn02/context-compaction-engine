"""
Structured logging setup.

Logs are emitted as single-line JSON objects so that, in Phase 4, they can
be shipped directly into the OpenTelemetry collector / PostgreSQL audit
trail without a separate parsing step. This module has zero dependency on
the rest of the engine so it can be imported first, before anything else.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """Renders each LogRecord as a single-line JSON document."""

    RESERVED_ATTRS = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Pull in any structured `extra=` fields the caller attached.
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_logging: bool = True) -> None:
    """Idempotently configure the root logger for this process."""
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Avoid duplicate handlers if called more than once (e.g. in tests).
    if root.handlers:
        root.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    if json_logging:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
