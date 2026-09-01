"""Environment-driven configuration.

Nothing here reads Claude credentials, keychain entries or token files. The
only secret this module handles is the proxy's own bearer token, which is
supplied by the operator via ``CLI_PROXY_TOKEN``.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
DEFAULT_MODEL_ALIAS = "sonnet"
DEFAULT_TIMEOUT_SECONDS = 600.0
#: Cursor's ``/multitask`` fans one turn out into several concurrent requests.
#: Four slots lets a typical fan-out run in parallel; each one is a full process
#: launch that re-pays the whole prompt, so this is not free.
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_MAX_REQUEST_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
DEFAULT_LOG_LEVEL = "INFO"

# Minimum token length we will accept. `openssl rand -hex 32` produces 64.
MIN_TOKEN_LENGTH = 16

#: Public model ids advertised on ``/v1/models`` mapped to Claude CLI aliases.
#: ``None`` means "use the configured default alias".
MODEL_ALIAS_MAP: dict[str, str | None] = {
    "claude-cli-proxy": None,
    "claude-cli-sonnet": "sonnet",
    "claude-cli-opus": "opus",
}


class ConfigError(RuntimeError):
    """Raised when the environment cannot produce a usable configuration."""


def _env_str(name: str, default: str | None = None) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw else default


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    return value


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ConfigError(f"{name} must be one of 1/0/true/false/yes/no/on/off")


def _env_float(name: str, default: float, *, minimum: float) -> float:
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    return value


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings."""

    token: str
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    claude_executable: str = "claude"
    default_model_alias: str = DEFAULT_MODEL_ALIAS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    log_level: str = DEFAULT_LOG_LEVEL
    working_dir: str = field(default_factory=lambda: str(Path(tempfile.gettempdir())))

    # -- debugging -------------------------------------------------------
    # These three switch OFF the project's normal logging hygiene. When
    # ``debug_dump`` is on, full request bodies, the serialised prompt, raw
    # Claude output and full response bodies are written out. Intended for
    # working out what an editor is actually sending. Off by default.
    debug_dump: bool = False
    debug_dump_dir: str = ""
    debug_dump_console: bool = False

    def resolve_model_alias(self, requested: str | None) -> str:
        """Map a public model id onto a Claude CLI model alias.

        Unknown ids fall back to the configured default so that an editor
        sending an unexpected model name still gets a working response.
        """
        if not requested:
            return self.default_model_alias
        if requested in MODEL_ALIAS_MAP:
            return MODEL_ALIAS_MAP[requested] or self.default_model_alias
        return self.default_model_alias

    @property
    def advertised_models(self) -> tuple[str, ...]:
        return tuple(MODEL_ALIAS_MAP)


def _resolve_executable(raw: str | None) -> str:
    """Resolve the Claude executable to an absolute path."""
    candidate = raw or "claude"
    if os.path.sep in candidate:
        return str(Path(candidate).expanduser())
    found = shutil.which(candidate)
    return found or candidate


def _resolve_working_dir(raw: str | None) -> str:
    """Pick a working directory for the Claude subprocess.

    Defaults to a private empty scratch directory. Claude is launched with no
    filesystem tools, but an empty cwd removes any chance of it picking up
    project context and keeps the blast radius at zero.
    """
    if raw:
        path = Path(raw).expanduser()
        if not path.is_dir():
            raise ConfigError(f"CLAUDE_WORKING_DIR is not a directory: {path}")
        return str(path)

    scratch = Path(tempfile.gettempdir()) / "cli-proxy-scratch"
    scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
    return str(scratch)


def _resolve_debug_dump_dir(raw: str | None) -> str:
    """Create the dump directory, owner-readable only."""
    path = Path(raw).expanduser() if raw else Path.cwd() / "debug-dumps"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    # mkdir's mode is ignored when the directory already exists.
    with contextlib.suppress(OSError):
        path.chmod(0o700)
    return str(path)


def load_settings(*, require_token: bool = True) -> Settings:
    """Build :class:`Settings` from the process environment."""
    token = _env_str("CLI_PROXY_TOKEN")
    if require_token:
        if not token:
            raise ConfigError(
                "CLI_PROXY_TOKEN is not set. Generate one with: openssl rand -hex 32"
            )
        if len(token) < MIN_TOKEN_LENGTH:
            raise ConfigError(
                f"CLI_PROXY_TOKEN must be at least {MIN_TOKEN_LENGTH} characters. "
                "Generate one with: openssl rand -hex 32"
            )
        if token == "replace-me-with-openssl-rand-hex-32":
            raise ConfigError(
                "CLI_PROXY_TOKEN still holds the .env.example placeholder. "
                "Generate a real one with: openssl rand -hex 32"
            )

    log_level = (_env_str("CLI_PROXY_LOG_LEVEL", DEFAULT_LOG_LEVEL) or "").upper()
    if log_level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        raise ConfigError(f"CLI_PROXY_LOG_LEVEL is not a valid level: {log_level}")

    port = _env_int("CLI_PROXY_PORT", DEFAULT_PORT, minimum=1)
    if port > 65535:
        raise ConfigError("CLI_PROXY_PORT must be <= 65535")

    debug_dump = _env_bool("CLI_PROXY_DEBUG_DUMP")

    return Settings(
        token=token or "",
        host=_env_str("CLI_PROXY_HOST", DEFAULT_HOST) or DEFAULT_HOST,
        port=port,
        claude_executable=_resolve_executable(_env_str("CLAUDE_EXECUTABLE")),
        default_model_alias=_env_str("CLAUDE_DEFAULT_MODEL", DEFAULT_MODEL_ALIAS)
        or DEFAULT_MODEL_ALIAS,
        timeout_seconds=_env_float(
            "CLAUDE_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS, minimum=1.0
        ),
        max_concurrency=_env_int(
            "CLAUDE_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY, minimum=1
        ),
        max_request_bytes=_env_int(
            "CLI_PROXY_MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES, minimum=1024
        ),
        max_response_bytes=_env_int(
            "CLI_PROXY_MAX_RESPONSE_BYTES", DEFAULT_MAX_RESPONSE_BYTES, minimum=1024
        ),
        log_level=log_level,
        working_dir=_resolve_working_dir(_env_str("CLAUDE_WORKING_DIR")),
        debug_dump=debug_dump,
        debug_dump_dir=(
            _resolve_debug_dump_dir(_env_str("CLI_PROXY_DEBUG_DUMP_DIR"))
            if debug_dump
            else ""
        ),
        debug_dump_console=_env_bool("CLI_PROXY_DEBUG_DUMP_CONSOLE"),
    )
