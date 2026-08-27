"""Logging configuration with a hard redaction backstop.

The proxy's own call sites are written never to log sensitive material. This
module adds a second line of defence: a filter that drops any record whose
formatted text looks like an authorization header or a bearer token, so an
accidental future log statement cannot leak one.
"""

from __future__ import annotations

import logging
import re

LOGGER_NAME = "cli_proxy"

_REDACTION_PATTERNS = (
    re.compile(r"authorization", re.IGNORECASE),
    re.compile(r"\bbearer\s+\S+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\b[0-9a-f]{32,}\b", re.IGNORECASE),
)

_REDACTED = "[redacted by cli-proxy]"


class RedactingFilter(logging.Filter):
    """Replace a record's message entirely if it may contain a secret."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
        except Exception:  # noqa: BLE001 - a broken record must not crash logging
            record.msg = _REDACTED
            record.args = ()
            return True

        for pattern in _REDACTION_PATTERNS:
            if pattern.search(rendered):
                record.msg = _REDACTED
                record.args = ()
                break
        return True


def configure_logging(level: str) -> logging.Logger:
    """Configure and return the proxy logger."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        handler.addFilter(RedactingFilter())
        logger.addHandler(handler)
    else:
        for existing in logger.handlers:
            if not any(isinstance(f, RedactingFilter) for f in existing.filters):
                existing.addFilter(RedactingFilter())

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)
