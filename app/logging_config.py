"""Structured JSON logging.

Every log line is one JSON object with a stable field set so it can be
indexed (ELK, Loki, Datadog) without a custom parser.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

# Fields we are willing to index. Anything else on the LogRecord is ignored
# so a caller cannot accidentally dump a transcript into the log stream.
_EXTRA_KEYS = (
    "request_id",
    "transcript_id",
    "model",
    "attempts",
    "latency_ms",
    "status",
    "error",
    "validation_errors",
    "matched_pii",
    "injection_suspected",
    "injection_patterns",
    "flagged_for_review",
    "review_reason",
)


def set_request_id(rid: str) -> contextvars.Token:
    return _request_id.set(rid)


def reset_request_id(token: contextvars.Token) -> None:
    _request_id.reset(token)


def get_request_id() -> str | None:
    return _request_id.get()


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
        }
        for key in _EXTRA_KEYS:
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val

        if "request_id" not in payload:
            rid = get_request_id()
            if rid:
                payload["request_id"] = rid
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)[:4000]
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: int | str = logging.INFO) -> None:
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    for noisy in ("httpx", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str = "harness") -> logging.Logger:
    return logging.getLogger(name)
