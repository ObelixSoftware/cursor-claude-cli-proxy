"""The opt-in debug dump.

Dump mode deliberately writes full request and response bodies to disk, so
these tests exist mainly to pin down what it must *never* write: the
authorization header and the proxy's own bearer token.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from cli_proxy import debug_dump as debug_dump_module
from cli_proxy.app import create_app, load_adapter_system_prompt
from cli_proxy.claude_runner import ClaudeRunner
from cli_proxy.config import Settings, load_settings
from tests.conftest import AUTH_HEADERS, TEST_TOKEN

CHAT_BODY = {
    "model": "claude-cli-sonnet",
    "messages": [{"role": "user", "content": "please read src/secret.py"}],
}


@pytest.fixture
def dump_dir(tmp_path: Path) -> Path:
    path = tmp_path / "debug-dumps"
    path.mkdir(mode=0o700)
    return path


@pytest.fixture
def dump_settings(
    fake_claude_wrapper: Path, work_dir: Path, dump_dir: Path
) -> Settings:
    return Settings(
        token=TEST_TOKEN,
        claude_executable=str(fake_claude_wrapper),
        default_model_alias="sonnet",
        timeout_seconds=30.0,
        max_request_bytes=64 * 1024,
        log_level="CRITICAL",
        working_dir=str(work_dir),
        debug_dump=True,
        debug_dump_dir=str(dump_dir),
        debug_dump_console=False,
    )


@pytest.fixture
def dump_app(dump_settings: Settings):
    runner = ClaudeRunner(dump_settings, load_adapter_system_prompt())
    return create_app(settings=dump_settings, runner=runner)


@pytest.fixture
async def dump_client(dump_app) -> Iterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=dump_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://proxy.test"
    ) as client:
        yield client


def dumps(directory: Path) -> list[dict]:
    files = sorted(directory.glob("*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


# -- defaults --------------------------------------------------------------


def test_dumping_is_off_by_default_in_settings() -> None:
    resolved = Settings(token=TEST_TOKEN)
    assert resolved.debug_dump is False
    assert resolved.debug_dump_dir == ""
    assert resolved.debug_dump_console is False


def test_dumping_is_off_by_default_in_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in (
        "CLI_PROXY_DEBUG_DUMP",
        "CLI_PROXY_DEBUG_DUMP_DIR",
        "CLI_PROXY_DEBUG_DUMP_CONSOLE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CLI_PROXY_TOKEN", TEST_TOKEN)
    monkeypatch.chdir(tmp_path)

    resolved = load_settings()
    assert resolved.debug_dump is False
    assert resolved.debug_dump_dir == ""
    assert not (tmp_path / "debug-dumps").exists(), (
        "the dump directory must not be created unless dumping is on"
    )


async def test_no_dump_is_written_when_disabled(
    client: httpx.AsyncClient, fake_mode, dump_dir: Path
) -> None:
    fake_mode("message")
    response = await client.post(
        "/v1/chat/completions", json=CHAT_BODY, headers=AUTH_HEADERS
    )
    assert response.status_code == 200
    assert dumps(dump_dir) == []


# -- redaction -------------------------------------------------------------


async def test_authorization_header_is_redacted(
    dump_client: httpx.AsyncClient, fake_mode, dump_dir: Path
) -> None:
    fake_mode("message")
    response = await dump_client.post(
        "/v1/chat/completions",
        json=CHAT_BODY,
        headers={**AUTH_HEADERS, "Cookie": "session=abc", "X-Api-Key": "sk-nope"},
    )
    assert response.status_code == 200

    (record,) = dumps(dump_dir)
    headers = record["inbound"]["headers"]
    assert headers["authorization"] == "[redacted]"
    assert headers["cookie"] == "[redacted]"
    assert headers["x-api-key"] == "[redacted]"
    # Ordinary headers stay readable -- that is the point of the feature.
    assert headers["content-type"] == "application/json"


async def test_the_bearer_token_never_appears_in_a_dump_file(
    dump_client: httpx.AsyncClient, fake_mode, dump_dir: Path
) -> None:
    fake_mode("message")
    body = dict(CHAT_BODY)
    # Worst case: the token is echoed inside the request body itself.
    body["messages"] = [{"role": "user", "content": f"my token is {TEST_TOKEN}"}]

    response = await dump_client.post(
        "/v1/chat/completions", json=body, headers=AUTH_HEADERS
    )
    assert response.status_code == 200

    files = sorted(dump_dir.glob("*.json"))
    assert files
    for path in files:
        raw = path.read_text(encoding="utf-8")
        assert TEST_TOKEN not in raw
        assert "[redacted]" in raw


# -- contents --------------------------------------------------------------


async def test_a_dump_is_written_for_a_non_streaming_request(
    dump_client: httpx.AsyncClient, fake_mode, dump_dir: Path
) -> None:
    fake_mode("message", text="Dumped answer.")
    response = await dump_client.post(
        "/v1/chat/completions", json=CHAT_BODY, headers=AUTH_HEADERS
    )
    assert response.status_code == 200

    (record,) = dumps(dump_dir)
    assert record["inbound"]["method"] == "POST"
    assert record["inbound"]["path"] == "/v1/chat/completions"
    assert "please read src/secret.py" in record["inbound"]["body"]

    assert record["normalized"]["api_flavor"] == "chat_completions"
    assert record["normalized"]["model_alias"] == "sonnet"

    # The full prompt and the exact argv are the whole point of the dump.
    assert "# CONVERSATION" in record["claude_stdin_prompt"]
    assert "please read src/secret.py" in record["claude_stdin_prompt"]
    assert "--json-schema" in record["claude_argv"]
    assert "StructuredOutput" in record["claude_argv"]

    assert record["claude_result"]["returncode"] == 0
    assert "Dumped answer." in record["claude_result"]["stdout"]

    assert record["decision"]["kind"] == "message"
    assert record["response"]["status"] == 200
    assert record["response"]["body"]["choices"][0]["message"]["content"] == (
        "Dumped answer."
    )


async def test_a_dump_is_written_for_a_streaming_request_with_every_chunk(
    dump_client: httpx.AsyncClient, fake_mode, dump_dir: Path
) -> None:
    fake_mode("message", text="Streamed and dumped.")
    body = {**CHAT_BODY, "stream": True}
    response = await dump_client.post(
        "/v1/chat/completions", json=body, headers=AUTH_HEADERS
    )
    assert response.status_code == 200
    assert response.text.endswith("data: [DONE]\n\n")

    (record,) = dumps(dump_dir)
    assert record["normalized"]["stream"] is True

    chunks = record["sse_chunks"]
    assert chunks[-1] == "data: [DONE]\n\n"
    assert "".join(chunks) == response.text
    assert any("Streamed and dumped." in chunk for chunk in chunks)


async def test_a_dump_records_a_rejected_request(
    dump_client: httpx.AsyncClient, dump_dir: Path
) -> None:
    response = await dump_client.post(
        "/v1/chat/completions", json=CHAT_BODY, headers={"Authorization": "Bearer wrong"}
    )
    assert response.status_code == 401

    (record,) = dumps(dump_dir)
    assert record["inbound"]["headers"]["authorization"] == "[redacted]"
    assert record["response"]["status"] == 401
    assert record["errors"][0]["type"] == "AuthenticationError"


async def test_dump_files_are_owner_only(
    dump_client: httpx.AsyncClient, fake_mode, dump_dir: Path
) -> None:
    fake_mode("message")
    await dump_client.post("/v1/chat/completions", json=CHAT_BODY, headers=AUTH_HEADERS)

    (path,) = sorted(dump_dir.glob("*.json"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


async def test_large_fields_are_truncated_rather_than_written_whole(
    dump_client: httpx.AsyncClient, fake_mode, dump_dir: Path, monkeypatch
) -> None:
    monkeypatch.setattr(debug_dump_module, "MAX_FIELD_BYTES", 512)
    fake_mode("message")
    body = {
        "model": "claude-cli-sonnet",
        "messages": [{"role": "user", "content": "y" * 4000}],
    }
    response = await dump_client.post(
        "/v1/chat/completions", json=body, headers=AUTH_HEADERS
    )
    assert response.status_code == 200

    (record,) = dumps(dump_dir)
    captured = record["inbound"]["body"]
    assert captured["_truncated"] is True
    assert captured["_original_bytes"] > 4000
    assert len(captured["text"]) == 512


# -- failure tolerance -----------------------------------------------------


async def test_a_dump_write_failure_does_not_break_the_response(
    dump_client: httpx.AsyncClient, fake_mode, dump_dir: Path, monkeypatch
) -> None:
    fake_mode("message", text="Still delivered.")

    def explode(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(os, "open", explode)

    response = await dump_client.post(
        "/v1/chat/completions", json=CHAT_BODY, headers=AUTH_HEADERS
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Still delivered."
    assert dumps(dump_dir) == []


async def test_a_dump_write_failure_does_not_break_a_stream(
    dump_client: httpx.AsyncClient, fake_mode, monkeypatch
) -> None:
    fake_mode("message", text="Still streamed.")

    def explode(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(os, "open", explode)

    body = {**CHAT_BODY, "stream": True}
    response = await dump_client.post(
        "/v1/chat/completions", json=body, headers=AUTH_HEADERS
    )
    assert response.status_code == 200
    assert "Still streamed." in response.text
    assert response.text.endswith("data: [DONE]\n\n")
