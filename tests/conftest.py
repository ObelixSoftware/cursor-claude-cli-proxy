"""Shared fixtures.

Every test in this suite runs against ``scripts/fake_claude.py`` or a mocked
runner. Nothing here can reach the real Claude CLI or the Anthropic API.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from cli_proxy.app import create_app, load_adapter_system_prompt
from cli_proxy.claude_runner import ClaudeRunner
from cli_proxy.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_CLAUDE = REPO_ROOT / "scripts" / "fake_claude.py"

TEST_TOKEN = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
AUTH_HEADERS = {"Authorization": f"Bearer {TEST_TOKEN}"}


@pytest.fixture
def fake_claude_wrapper(tmp_path: Path) -> Path:
    """A tiny executable shim that runs the fake CLI under this interpreter."""
    wrapper = tmp_path / "claude"
    wrapper.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_CLAUDE}" "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    path = tmp_path / "scratch"
    path.mkdir()
    return path


@pytest.fixture
def settings(fake_claude_wrapper: Path, work_dir: Path) -> Settings:
    return Settings(
        token=TEST_TOKEN,
        host="127.0.0.1",
        port=8787,
        claude_executable=str(fake_claude_wrapper),
        default_model_alias="sonnet",
        timeout_seconds=30.0,
        max_concurrency=1,
        max_request_bytes=64 * 1024,
        max_response_bytes=4 * 1024 * 1024,
        log_level="CRITICAL",
        working_dir=str(work_dir),
    )


@pytest.fixture
def runner(settings: Settings) -> ClaudeRunner:
    return ClaudeRunner(settings, load_adapter_system_prompt())


@pytest.fixture
def app(settings: Settings, runner: ClaudeRunner):
    return create_app(settings=settings, runner=runner)


@pytest.fixture
async def client(app) -> Iterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://proxy.test"
    ) as async_client:
        yield async_client


@pytest.fixture
def fake_mode(monkeypatch: pytest.MonkeyPatch):
    """Set ``FAKE_CLAUDE_*`` variables for the child process to inherit."""

    def _set(mode: str, **extra: str) -> None:
        monkeypatch.setenv("FAKE_CLAUDE_MODE", mode)
        for key, value in extra.items():
            monkeypatch.setenv(f"FAKE_CLAUDE_{key.upper()}", value)

    yield _set

    for key in list(os.environ):
        if key.startswith("FAKE_CLAUDE_"):
            os.environ.pop(key, None)
