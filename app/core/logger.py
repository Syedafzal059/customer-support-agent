"""JSON-line structured logging for the application."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonLogFormatter(logging.Formatter):
    """Emit one JSON object per log line for easy parsing."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extra = getattr(record, "structured", None)
        if isinstance(extra, dict):
            payload["extra"] = extra
        return json.dumps(payload, ensure_ascii=False)


APP_LOG_NAMESPACE = "app"


def setup_logging(level: str = "INFO") -> None:
    """Attach JSON formatting to the ``app.*`` loggers only (does not strip uvicorn/root handlers)."""
    pkg = logging.getLogger(APP_LOG_NAMESPACE)
    if pkg.handlers:
        pkg.setLevel(level.upper())
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    pkg.addHandler(handler)
    pkg.setLevel(level.upper())
    pkg.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
