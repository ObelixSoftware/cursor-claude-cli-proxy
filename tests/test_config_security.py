"""Unit tests for configuration, model mapping, logging redaction and auth."""

from __future__ import annotations

import logging

import pytest

from cli_proxy.config import ConfigError, Settings, load_settings
from cli_proxy.errors import AuthenticationError
from cli_proxy.logging_setup import RedactingFilter
from cli_proxy.security import extract_bearer_token, require_bearer, tokens_match

GOOD_TOKEN = "a" * 64


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "CLI_PROXY_TOKEN",
        "CLI_PROXY_HOST",
        "CLI_PROXY_PORT",
        "CLAUDE_EXECUTABLE",
        "CLAUDE_DEFAULT_MODEL",
        "CLAUDE_TIMEOUT_SECONDS",
        "CLAUDE_MAX_CONCURRENCY",
        "CLAUDE_WORKING_DIR",
        "CLI_PROXY_MAX_REQUEST_BYTES",
        "CLI_PROXY_MAX_RESPONSE_BYTES",
        "CLI_PROXY_LOG_LEVEL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_missing_token_is_rejected():
    with pytest.raises(ConfigError, match="CLI_PROXY_TOKEN is not set"):
        load_settings()


def test_short_token_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLI_PROXY_TOKEN", "tooshort")
    with pytest.raises(ConfigError, match="at least"):
        load_settings()


def test_example_placeholder_token_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLI_PROXY_TOKEN", "replace-me-with-openssl-rand-hex-32")
    with pytest.raises(ConfigError, match="placeholder"):
        load_settings()


def test_defaults_bind_to_loopback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLI_PROXY_TOKEN", GOOD_TOKEN)
    resolved = load_settings()
    assert resolved.host == "127.0.0.1"
    assert resolved.port == 8787
    assert resolved.timeout_seconds == 600.0
    assert resolved.max_concurrency == 1


def test_invalid_numeric_env_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLI_PROXY_TOKEN", GOOD_TOKEN)
    monkeypatch.setenv("CLAUDE_MAX_CONCURRENCY", "zero")
    with pytest.raises(ConfigError, match="must be an integer"):
        load_settings()


def test_invalid_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLI_PROXY_TOKEN", GOOD_TOKEN)
    monkeypatch.setenv("CLI_PROXY_LOG_LEVEL", "CHATTY")
    with pytest.raises(ConfigError, match="valid level"):
        load_settings()


def test_working_dir_must_exist(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("CLI_PROXY_TOKEN", GOOD_TOKEN)
    monkeypatch.setenv("CLAUDE_WORKING_DIR", str(tmp_path / "nope"))
    with pytest.raises(ConfigError, match="not a directory"):
        load_settings()


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("claude-cli-proxy", "sonnet"),
        ("claude-cli-sonnet", "sonnet"),
        ("claude-cli-opus", "opus"),
        ("something-cursor-invented", "sonnet"),
        (None, "sonnet"),
    ],
)
def test_model_alias_mapping(model_id, expected):
    resolved = Settings(token=GOOD_TOKEN, default_model_alias="sonnet")
    assert resolved.resolve_model_alias(model_id) == expected


def test_default_alias_is_configurable():
    resolved = Settings(token=GOOD_TOKEN, default_model_alias="opus")
    assert resolved.resolve_model_alias("claude-cli-proxy") == "opus"
    assert resolved.resolve_model_alias("claude-cli-sonnet") == "sonnet"


def test_advertised_models():
    resolved = Settings(token=GOOD_TOKEN)
    assert resolved.advertised_models == (
        "claude-cli-proxy",
        "claude-cli-sonnet",
        "claude-cli-opus",
    )


# -- bearer token ----------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Bearer abc123", "abc123"),
        ("bearer abc123", "abc123"),
        ("BEARER   abc123  ", "abc123"),
        ("Basic abc123", None),
        ("abc123", None),
        ("Bearer ", None),
        ("", None),
        (None, None),
    ],
)
def test_extract_bearer_token(header, expected):
    assert extract_bearer_token(header) == expected


def test_tokens_match_is_exact():
    assert tokens_match(GOOD_TOKEN, GOOD_TOKEN)
    assert not tokens_match(GOOD_TOKEN[:-1] + "b", GOOD_TOKEN)
    assert not tokens_match(None, GOOD_TOKEN)
    assert not tokens_match("", GOOD_TOKEN)


def test_tokens_match_uses_constant_time_comparison(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[bytes, bytes]] = []
    import hmac as hmac_module

    real = hmac_module.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr("cli_proxy.security.hmac.compare_digest", spy)
    tokens_match("nope", GOOD_TOKEN)
    assert calls, "security.tokens_match must go through hmac.compare_digest"


def test_require_bearer_raises_without_token():
    with pytest.raises(AuthenticationError):
        require_bearer(None, GOOD_TOKEN)
    with pytest.raises(AuthenticationError):
        require_bearer("Bearer wrong", GOOD_TOKEN)
    require_bearer(f"Bearer {GOOD_TOKEN}", GOOD_TOKEN)


def test_require_bearer_refuses_when_proxy_has_no_token():
    with pytest.raises(AuthenticationError, match="no configured bearer token"):
        require_bearer(f"Bearer {GOOD_TOKEN}", "")


# -- log redaction ---------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "Authorization: Bearer abc",
        "authorization header seen",
        "bearer sk-abcdefghijkl",
        "token=" + "f" * 40,
        "sk-abcdefghijklmnop",
    ],
)
def test_redacting_filter_scrubs_secret_shaped_messages(message):
    record = logging.LogRecord("t", logging.INFO, __file__, 1, message, None, None)
    RedactingFilter().filter(record)
    assert record.getMessage() == "[redacted by cli-proxy]"


def test_redacting_filter_keeps_benign_messages():
    record = logging.LogRecord(
        "t", logging.INFO, __file__, 1, "claude subprocess pid=42 rc=0", None, None
    )
    RedactingFilter().filter(record)
    assert record.getMessage() == "claude subprocess pid=42 rc=0"
