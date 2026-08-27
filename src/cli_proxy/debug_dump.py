"""Full-fidelity request/response dumps for debugging an editor integration.

This module deliberately breaks the project's normal logging hygiene, so it is
off unless ``CLI_PROXY_DEBUG_DUMP`` is set. When on, it writes the *entire*
exchange -- inbound body, serialised prompt, Claude's raw stdout, the outgoing
body -- to a file in cleartext. That includes any source code the editor sent.

Two rules make it safe enough to ship:

* Files are written **directly** with :func:`os.open`, never through the
  ``logging`` module. ``RedactingFilter`` blanks any record containing a long
  hex run, which would silently destroy most dumps.
* A small set of request headers is redacted even here, plus a final pass that
  scrubs the proxy's own bearer token from the serialised document. Those values
  have no debugging value and are pure secret.

A dump failure must never affect the HTTP response, so every write is wrapped.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import threading
import uuid
from datetime import UTC, datetime
from typing import Any

from .config import Settings
from .logging_setup import get_logger

_LOG = get_logger()

#: Per-field ceiling. One oversized editor payload must not fill the disk.
MAX_FIELD_BYTES = 2 * 1024 * 1024

#: Aggregate ceiling for the captured SSE chunk list.
MAX_SSE_BYTES = MAX_FIELD_BYTES

REDACTED = "[redacted]"

#: Headers whose values are always removed, dump mode or not.
SECRET_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "api-key",
    }
)


def _cap(value: str) -> Any:
    """Return ``value``, or a truncation marker when it is too large."""
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= MAX_FIELD_BYTES:
        return value
    kept = encoded[:MAX_FIELD_BYTES].decode("utf-8", errors="replace")
    return {
        "_truncated": True,
        "_original_bytes": len(encoded),
        "_kept_bytes": MAX_FIELD_BYTES,
        "text": kept,
    }


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def redact_headers(headers: Any) -> dict[str, str]:
    """Copy a header mapping with all secret-bearing values replaced."""
    try:
        items = list(headers.items())
    except AttributeError:
        items = list(headers or [])
    out: dict[str, str] = {}
    for key, value in items:
        name = str(key)
        out[name] = REDACTED if name.lower() in SECRET_HEADERS else str(value)
    return out


class Exchange:
    """No-op recorder. Call sites always hold one of these, enabled or not."""

    request_id = ""

    def record_inbound(
        self, *, method: str, path: str, query: str, headers: Any, body: bytes
    ) -> None:
        """Record the inbound HTTP request."""

    def record_inbound_body(self, body: bytes) -> None:
        """Attach the raw body once it has been read and size-checked."""

    def record_normalized(
        self, *, flavor: str, model_alias: str, model_id: str, stream: bool, **extra: Any
    ) -> None:
        """Record what the request normalised to."""

    def record_prompt(self, prompt: str) -> None:
        """Record the full document written to Claude's stdin."""

    def record_argv(self, argv: list[str]) -> None:
        """Record the exact argv used to spawn the CLI."""

    def record_claude_result(
        self, *, stdout: bytes, stderr: bytes, returncode: int | None
    ) -> None:
        """Record raw subprocess output."""

    def record_decision(self, decision: Any) -> None:
        """Record the parsed decision."""

    def record_response(self, *, status: int, body: Any) -> None:
        """Record a non-streaming outgoing response."""

    def record_sse_chunk(self, chunk: str) -> None:
        """Record one emitted server-sent-event chunk."""

    def record_note(self, note: str) -> None:
        """Record a free-form marker, e.g. a stream lifecycle event."""

    def record_error(self, exc: BaseException) -> None:
        """Record a failure."""

    def close(self) -> None:
        """Write the dump. Safe to call more than once."""


NULL_EXCHANGE = Exchange()


class _RecordingExchange(Exchange):
    """Collects one exchange and writes it out on :meth:`close`."""

    def __init__(self, dumper: DebugDumper, sequence: int) -> None:
        self._dumper = dumper
        self._sequence = sequence
        self.request_id = uuid.uuid4().hex[:8]
        self._started = datetime.now(UTC)
        self._closed = False
        self._sse_bytes = 0
        self._sse_capped = False
        # Insertion order is the record order, which is the point.
        self._data: dict[str, Any] = {
            "dump_sequence": sequence,
            "request_id": self.request_id,
            "started_utc": self._started.isoformat(),
        }

    # -- recording ---------------------------------------------------------

    def record_inbound(
        self, *, method: str, path: str, query: str, headers: Any, body: bytes
    ) -> None:
        self._data["inbound"] = {
            "method": method,
            "path": path,
            "query": query,
            "headers": redact_headers(headers),
            "body_bytes": len(body),
            "body": _cap(_decode(body)),
        }

    def record_inbound_body(self, body: bytes) -> None:
        inbound = self._data.setdefault("inbound", {})
        inbound["body_bytes"] = len(body)
        inbound["body"] = _cap(_decode(body))

    def record_normalized(
        self, *, flavor: str, model_alias: str, model_id: str, stream: bool, **extra: Any
    ) -> None:
        self._data["normalized"] = {
            "api_flavor": flavor,
            "model_alias": model_alias,
            "model_id": model_id,
            "stream": stream,
            **extra,
        }

    def record_prompt(self, prompt: str) -> None:
        self._data["claude_stdin_prompt"] = _cap(prompt)

    def record_argv(self, argv: list[str]) -> None:
        self._data["claude_argv"] = [_cap(str(item)) for item in argv]

    def record_claude_result(
        self, *, stdout: bytes, stderr: bytes, returncode: int | None
    ) -> None:
        self._data["claude_result"] = {
            "returncode": returncode,
            "stdout": _cap(_decode(stdout)),
            "stderr": _cap(_decode(stderr)),
        }

    def record_decision(self, decision: Any) -> None:
        self._data["decision"] = {
            "kind": getattr(decision, "kind", None),
            "content": _cap(str(getattr(decision, "content", "") or "")),
            "tool_calls": getattr(decision, "tool_calls", []),
            "error": getattr(decision, "error", ""),
            "model_name": getattr(decision, "model_name", ""),
            "input_tokens": getattr(decision, "input_tokens", 0),
            "output_tokens": getattr(decision, "output_tokens", 0),
        }

    def record_response(self, *, status: int, body: Any) -> None:
        self._data["response"] = {
            "status": status,
            "body": body if isinstance(body, (dict, list)) else _cap(str(body)),
        }

    def record_sse_chunk(self, chunk: str) -> None:
        chunks = self._data.setdefault("sse_chunks", [])
        if self._sse_capped:
            return
        self._sse_bytes += len(chunk.encode("utf-8", errors="replace"))
        if self._sse_bytes > MAX_SSE_BYTES:
            self._sse_capped = True
            chunks.append({"_truncated": True, "_note": "SSE capture cap reached"})
            return
        chunks.append(chunk)

    def record_note(self, note: str) -> None:
        self._data.setdefault("notes", []).append(
            f"{datetime.now(UTC).isoformat()} {note}"
        )

    def record_error(self, exc: BaseException) -> None:
        self._data.setdefault("errors", []).append(
            {"type": type(exc).__name__, "detail": _cap(str(exc))}
        )

    # -- output ------------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        finished = datetime.now(UTC)
        self._data["finished_utc"] = finished.isoformat()
        self._data["elapsed_seconds"] = round(
            (finished - self._started).total_seconds(), 3
        )
        self._dumper._write(self._sequence, self.request_id, self._started, self._data)


class DebugDumper:
    """Factory for :class:`Exchange` recorders."""

    def __init__(self, settings: Settings) -> None:
        self._enabled = bool(settings.debug_dump and settings.debug_dump_dir)
        self._directory = settings.debug_dump_dir
        self._console = bool(settings.debug_dump_console)
        self._token = settings.token or ""
        self._counter = itertools.count(1)
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def directory(self) -> str:
        return self._directory

    def begin(self) -> Exchange:
        """Start recording an exchange, or hand back the no-op recorder."""
        if not self._enabled:
            return NULL_EXCHANGE
        return _RecordingExchange(self, next(self._counter))

    def _serialize(self, data: dict[str, Any]) -> str:
        try:
            text = json.dumps(data, indent=2, ensure_ascii=False, default=repr)
        except (TypeError, ValueError):
            text = json.dumps({"_unserialisable": repr(data)[:4096]}, indent=2)
        # Final safety pass: the proxy's own bearer token must not survive into
        # a dump even if it turned up somewhere unexpected.
        if self._token:
            text = text.replace(self._token, REDACTED)
        return text

    def _write(
        self,
        sequence: int,
        request_id: str,
        started: datetime,
        data: dict[str, Any],
    ) -> None:
        try:
            text = self._serialize(data)
            stamp = started.strftime("%Y%m%dT%H%M%S%fZ")
            name = f"{sequence:06d}-{stamp}-{request_id}.json"
            path = os.path.join(self._directory, name)
            with self._lock:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                try:
                    os.write(fd, text.encode("utf-8", errors="replace"))
                finally:
                    os.close(fd)
            if self._console:
                sys.stdout.write(
                    f"\n===== cli-proxy debug dump {name} =====\n{text}\n"
                    f"===== end debug dump {name} =====\n"
                )
                sys.stdout.flush()
        except Exception as exc:  # noqa: BLE001 - a dump must never fail a request
            _LOG.warning("debug dump could not be written: %s", type(exc).__name__)
