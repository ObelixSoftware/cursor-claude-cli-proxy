"""Non-interactive Claude Code CLI invocation.

Design rules enforced here:

* ``asyncio.create_subprocess_exec`` only -- never ``shell=True``, and never a
  command string built by concatenation. The argv list is fixed; the only
  variable-length elements are the model alias, the schema and the system
  prompt, each passed as a single discrete argv element.
* The conversation itself goes over **stdin**, never argv. That keeps prompt
  text out of the process table and out of any argv-length limit.
* stdout and stderr are captured separately. stderr is classified, never echoed.
* No Anthropic HTTP call is made from this process; the CLI uses whatever
  authenticated state it already has. Nothing here reads, copies, prints or
  otherwise touches OAuth credentials, keychain entries or token files.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .debug_dump import NULL_EXCHANGE, Exchange
from .errors import (
    ClaudeAuthError,
    ClaudeOutputError,
    ClaudeProcessError,
    ClaudeTimeoutError,
    ClaudeUnavailableError,
    ResponseTooLargeError,
)
from .logging_setup import get_logger
from .schema import ADAPTER_OUTPUT_SCHEMA, SchemaViolation, validate_adapter_output

_LOG = get_logger()

#: Seconds to wait after SIGTERM before escalating to SIGKILL.
GRACE_PERIOD_SECONDS = 5.0

#: Short timeout for the cheap health probes.
PROBE_TIMEOUT_SECONDS = 20.0

#: The single built-in Claude tool that must stay enabled.
#:
#: ``--json-schema`` is implemented inside Claude Code as a built-in tool named
#: ``StructuredOutput``. Passing ``--tools ""`` therefore disables structured
#: output too: the CLI reports the schema call under ``permission_denials`` and
#: falls back to prose. ``StructuredOutput`` only returns a JSON object to the
#: caller -- it cannot read files, write files or run commands -- so allowlisting
#: exactly this one tool is what "all internal tools disabled" has to mean in
#: practice. Verified against Claude Code 2.1.231.
STRUCTURED_OUTPUT_TOOL = "StructuredOutput"

#: Belt-and-braces explicit denials. ``--tools`` already restricts the built-in
#: set to ``StructuredOutput`` alone; this second list means a future CLI change
#: to ``--tools`` semantics still cannot hand Claude filesystem or shell access.
DENIED_TOOLS = (
    "Agent",
    "Bash",
    "BashOutput",
    "Edit",
    "ExitPlanMode",
    "Glob",
    "Grep",
    "KillShell",
    "MultiEdit",
    "NotebookEdit",
    "NotebookRead",
    "Read",
    "SlashCommand",
    "Task",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
    "Write",
)

#: Environment variables removed from the subprocess environment. The proxy's
#: own bearer token must never be visible to the CLI or to anything it spawns.
_STRIPPED_ENV_VARS = (
    "CLI_PROXY_TOKEN",
    "CLI_PROXY_HOST",
    "CLI_PROXY_PORT",
    "CLI_PROXY_MAX_REQUEST_BYTES",
    "CLI_PROXY_MAX_RESPONSE_BYTES",
    "CLI_PROXY_LOG_LEVEL",
)

_AUTH_FAILURE_MARKERS = (
    "not logged in",
    "please log in",
    "please run /login",
    "run `claude login`",
    "authentication_error",
    "invalid api key",
    "oauth token has expired",
    "unauthorized",
    "401",
)


@dataclass(frozen=True)
class ClaudeDecision:
    """A validated decision extracted from Claude's structured output."""

    kind: str
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    model_name: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class HealthProbe:
    """Result of the cheap CLI probes used by ``/health``."""

    executable_path: str
    executable_available: bool
    version: str | None
    authenticated: bool
    detail: str


def _subprocess_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV_VARS}
    # Keep the CLI in non-interactive mode regardless of the operator's shell.
    env["CI"] = env.get("CI", "1")
    env["TERM"] = env.get("TERM", "dumb")
    return env


def _looks_like_auth_failure(stderr_text: str) -> bool:
    lowered = stderr_text.lower()
    return any(marker in lowered for marker in _AUTH_FAILURE_MARKERS)


async def _terminate_process(proc: asyncio.subprocess.Process) -> None:
    """SIGTERM, wait ``GRACE_PERIOD_SECONDS``, then SIGKILL."""
    if proc.returncode is not None:
        return

    with contextlib.suppress(ProcessLookupError):
        proc.terminate()

    try:
        await asyncio.wait_for(proc.wait(), timeout=GRACE_PERIOD_SECONDS)
        return
    except (TimeoutError, asyncio.TimeoutError):
        _LOG.warning(
            "claude subprocess pid=%s ignored SIGTERM after %.0fs; sending SIGKILL",
            proc.pid,
            GRACE_PERIOD_SECONDS,
        )
    except ProcessLookupError:
        return

    with contextlib.suppress(ProcessLookupError):
        proc.send_signal(signal.SIGKILL)
    with contextlib.suppress(Exception):
        await proc.wait()


class ClaudeRunner:
    """Owns concurrency limiting and argv construction for the CLI."""

    def __init__(self, settings: Settings, system_prompt: str) -> None:
        self._settings = settings
        self._system_prompt = system_prompt
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)

    @property
    def settings(self) -> Settings:
        return self._settings

    def build_argv(self, model_alias: str) -> list[str]:
        """Build the fixed argv for one invocation.

        Every element is a discrete list entry, so no value can be reinterpreted
        as a shell token.
        """
        return [
            self._settings.claude_executable,
            "--print",
            "--safe-mode",
            "--tools",
            STRUCTURED_OUTPUT_TOOL,
            "--disallowedTools",
            ",".join(DENIED_TOOLS),
            "--disable-slash-commands",
            "--no-session-persistence",
            "--output-format",
            "json",
            "--model",
            model_alias,
            "--system-prompt",
            self._system_prompt,
            "--json-schema",
            json.dumps(ADAPTER_OUTPUT_SCHEMA, separators=(",", ":")),
        ]

    async def run(
        self,
        prompt: str,
        model_alias: str,
        exchange: Exchange | None = None,
    ) -> ClaudeDecision:
        """Invoke the CLI once and return its validated decision.

        ``exchange`` is the optional debug recorder. It defaults to the no-op
        one, so existing callers need no change.
        """
        recorder = exchange or NULL_EXCHANGE
        argv = self.build_argv(model_alias)
        recorder.record_argv(argv)
        payload = prompt.encode("utf-8")

        async with self._semaphore:
            stdout, stderr, returncode = await self._exec(argv, payload, recorder)

        recorder.record_claude_result(
            stdout=stdout, stderr=stderr, returncode=returncode
        )
        decision = self._parse_envelope(stdout, stderr, returncode)
        recorder.record_decision(decision)
        return decision

    async def _exec(
        self,
        argv: list[str],
        payload: bytes,
        exchange: Exchange | None = None,
    ) -> tuple[bytes, bytes, int | None]:
        recorder = exchange or NULL_EXCHANGE
        limit = self._settings.max_response_bytes + (1024 * 1024)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._settings.working_dir,
                env=_subprocess_env(),
                limit=limit,
            )
        except FileNotFoundError as exc:
            raise ClaudeUnavailableError(log_hint="claude executable not found") from exc
        except PermissionError as exc:
            raise ClaudeUnavailableError(
                log_hint="claude executable not executable"
            ) from exc
        except OSError as exc:
            raise ClaudeUnavailableError(log_hint="claude spawn failed") from exc

        _LOG.info(
            "claude subprocess started pid=%s model=%s timeout=%.0fs",
            proc.pid,
            argv[argv.index("--model") + 1],
            self._settings.timeout_seconds,
        )
        recorder.record_note(f"claude subprocess started pid={proc.pid}")

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=payload),
                timeout=self._settings.timeout_seconds,
            )
        except (TimeoutError, asyncio.TimeoutError) as exc:
            recorder.record_note(f"claude subprocess pid={proc.pid} timed out")
            await _terminate_process(proc)
            _LOG.warning("claude subprocess pid=%s timed out", proc.pid)
            raise ClaudeTimeoutError(log_hint="claude timed out") from exc
        except asyncio.CancelledError:
            # Client disconnected, or the server is shutting down.
            recorder.record_note(f"claude subprocess pid={proc.pid} cancelled")
            await _terminate_process(proc)
            _LOG.info("claude subprocess pid=%s cancelled; process reaped", proc.pid)
            raise
        except ValueError as exc:
            # asyncio raises this when a stream exceeds its buffer limit.
            recorder.record_note(f"claude subprocess pid={proc.pid} exceeded stream limit")
            await _terminate_process(proc)
            raise ResponseTooLargeError(log_hint="claude stdout over stream limit") from exc

        _LOG.info(
            "claude subprocess pid=%s exited rc=%s stdout_bytes=%d stderr_bytes=%d",
            proc.pid,
            proc.returncode,
            len(stdout),
            len(stderr),
        )
        return stdout, stderr, proc.returncode

    def _parse_envelope(
        self, stdout: bytes, stderr: bytes, returncode: int | None
    ) -> ClaudeDecision:
        if len(stdout) > self._settings.max_response_bytes:
            raise ResponseTooLargeError(log_hint="claude stdout over configured limit")

        # stderr is only ever *classified*. It is never returned or logged
        # verbatim, because it can carry environment and prompt fragments.
        stderr_text = stderr.decode("utf-8", errors="replace")

        if returncode != 0:
            if _looks_like_auth_failure(stderr_text):
                raise ClaudeAuthError(log_hint="claude reported an auth failure")
            _LOG.error(
                "claude exited rc=%s (stderr suppressed, %d bytes)",
                returncode,
                len(stderr),
            )
            raise ClaudeProcessError(log_hint=f"claude rc={returncode}")

        if not stdout.strip():
            raise ClaudeOutputError(log_hint="claude produced empty stdout")

        try:
            envelope = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ClaudeOutputError(log_hint="claude stdout was not valid JSON") from exc

        if not isinstance(envelope, dict):
            raise ClaudeOutputError(log_hint="claude envelope was not an object")

        if envelope.get("is_error") or envelope.get("subtype") not in (None, "success"):
            if _looks_like_auth_failure(str(envelope.get("api_error_status") or "")):
                raise ClaudeAuthError(log_hint="claude envelope reported auth failure")
            _LOG.error(
                "claude envelope reported failure subtype=%s",
                _safe_subtype(envelope.get("subtype")),
            )
            raise ClaudeProcessError(log_hint="claude envelope is_error")

        structured = envelope.get("structured_output")

        if structured is None:
            # Most common cause: the StructuredOutput tool was denied, so the
            # CLI answered in prose. Guessing at prose is explicitly out of
            # scope, so fail loudly instead.
            if _structured_output_was_denied(envelope):
                raise ClaudeOutputError(
                    "The Claude Code CLI denied its own structured-output tool, so no "
                    "machine-readable reply was produced. This proxy will not parse "
                    "free-form text.",
                    log_hint="StructuredOutput denied by permissions",
                )
            structured = _structured_from_result(envelope.get("result"))

        if structured is None:
            raise ClaudeOutputError(log_hint="no structured_output in claude envelope")

        try:
            decision = validate_adapter_output(structured)
        except SchemaViolation as exc:
            _LOG.error("claude structured output violated the contract: %s", exc)
            raise ClaudeOutputError(log_hint="structured output violated contract") from exc

        model_name, input_tokens, output_tokens = _usage_from_envelope(envelope)

        return ClaudeDecision(
            kind=decision["kind"],
            content=decision.get("content", ""),
            tool_calls=decision.get("tool_calls", []),
            error=decision.get("error", ""),
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def probe_health(self) -> HealthProbe:
        """Run ``--version`` and ``auth status``, reporting booleans only.

        ``claude auth status`` prints account information. That output is parsed
        for a single boolean and then discarded -- it is never logged, stored or
        returned to a client.
        """
        exe = self._settings.claude_executable
        version: str | None = None
        authenticated = False
        details: list[str] = []

        rc, out, _ = await self._probe([exe, "--version"])
        if rc is None:
            return HealthProbe(
                executable_path=exe,
                executable_available=False,
                version=None,
                authenticated=False,
                detail="claude executable could not be started",
            )
        if rc == 0:
            version = out.decode("utf-8", errors="replace").strip() or None
        else:
            details.append("claude --version exited non-zero")

        rc_auth, out_auth, _ = await self._probe([exe, "auth", "status"])
        if rc_auth == 0:
            authenticated = _auth_status_is_logged_in(out_auth)
            if not authenticated:
                details.append("claude auth status reports not signed in")
        elif rc_auth is None:
            details.append("claude auth status could not be started")
        else:
            details.append("claude auth status exited non-zero")

        return HealthProbe(
            executable_path=exe,
            executable_available=version is not None,
            version=version,
            authenticated=authenticated,
            detail="; ".join(details) if details else "ok",
        )

    async def _probe(self, argv: list[str]) -> tuple[int | None, bytes, bytes]:
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._settings.working_dir,
                env=_subprocess_env(),
            )
        except (OSError, ValueError):
            return None, b"", b""

        try:
            out, err = await asyncio.wait_for(
                proc.communicate(), timeout=PROBE_TIMEOUT_SECONDS
            )
        except (TimeoutError, asyncio.TimeoutError):
            await _terminate_process(proc)
            return None, b"", b""
        except asyncio.CancelledError:
            await _terminate_process(proc)
            raise

        return proc.returncode, out, err


def _safe_subtype(value: Any) -> str:
    """Allow only a short alphanumeric subtype into logs."""
    text = str(value or "")[:40]
    return "".join(ch for ch in text if ch.isalnum() or ch in "_-") or "unknown"


def _structured_output_was_denied(envelope: dict[str, Any]) -> bool:
    denials = envelope.get("permission_denials")
    if not isinstance(denials, list):
        return False
    return any(
        isinstance(entry, dict) and entry.get("tool_name") == STRUCTURED_OUTPUT_TOOL
        for entry in denials
    )


def _structured_from_result(result: Any) -> dict[str, Any] | None:
    """Recover the decision from ``result`` when it is exactly a JSON object.

    Claude Code duplicates structured output into ``result`` as a JSON string.
    This accepts that single well-defined case and nothing else: no fenced-block
    extraction, no substring scanning, no prose interpretation.
    """
    if isinstance(result, dict):
        return result
    if not isinstance(result, str):
        return None
    text = result.strip()
    if not text.startswith("{") or not text.endswith("}"):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _usage_from_envelope(envelope: dict[str, Any]) -> tuple[str, int, int]:
    model_name = ""
    model_usage = envelope.get("modelUsage")
    if isinstance(model_usage, dict):
        for name, stats in model_usage.items():
            if isinstance(stats, dict) and stats.get("outputTokens"):
                model_name = str(name)
                break
        if not model_name and model_usage:
            model_name = str(next(iter(model_usage)))

    usage = envelope.get("usage")
    input_tokens = 0
    output_tokens = 0
    if isinstance(usage, dict):
        for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                input_tokens += value
        value = usage.get("output_tokens")
        if isinstance(value, int):
            output_tokens = value

    return model_name, input_tokens, output_tokens


def _auth_status_is_logged_in(raw: bytes) -> bool:
    """Extract only the boolean login state from ``claude auth status``.

    The rest of that payload (email, org, subscription) is intentionally
    dropped on the floor and never leaves this function.
    """
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return False
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        lowered = text.lower()
        if "not logged in" in lowered or "logged out" in lowered:
            return False
        return "logged in" in lowered or "authenticated" in lowered
    if isinstance(parsed, dict):
        return bool(parsed.get("loggedIn"))
    return False
