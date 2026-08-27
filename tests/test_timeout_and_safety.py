"""Timeout, concurrency, disconnection and end-to-end safety invariants."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from cli_proxy.app import create_app, load_adapter_system_prompt
from cli_proxy.claude_runner import ClaudeRunner
from cli_proxy.logging_setup import LOGGER_NAME
from tests.conftest import AUTH_HEADERS

CHAT_BODY = {
    "model": "claude-cli-proxy",
    "messages": [{"role": "user", "content": "hi"}],
}


# -- timeout ---------------------------------------------------------------


async def test_timeout_maps_to_504(settings, fake_mode):
    fake_mode("hang")
    impatient = replace(settings, timeout_seconds=1.0)
    app = create_app(
        settings=impatient,
        runner=ClaudeRunner(impatient, load_adapter_system_prompt()),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        response = await client.post(
            "/v1/chat/completions", json=CHAT_BODY, headers=AUTH_HEADERS, timeout=60
        )

    assert response.status_code == 504
    assert "timeout" in response.json()["error"]["message"].lower()


async def test_timeout_error_reveals_nothing_sensitive(settings, fake_mode):
    fake_mode("hang")
    impatient = replace(settings, timeout_seconds=1.0)
    app = create_app(
        settings=impatient,
        runner=ClaudeRunner(impatient, load_adapter_system_prompt()),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        response = await client.post(
            "/v1/chat/completions", json=CHAT_BODY, headers=AUTH_HEADERS, timeout=60
        )

    body = response.text
    assert settings.token not in body
    assert "/Users/" not in body
    assert "Traceback" not in body


async def test_proxy_recovers_after_a_timeout(settings, fake_mode):
    """A timed-out subprocess must not hold the concurrency slot."""
    fake_mode("hang")
    impatient = replace(settings, timeout_seconds=1.0)
    app = create_app(
        settings=impatient,
        runner=ClaudeRunner(impatient, load_adapter_system_prompt()),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        first = await client.post(
            "/v1/chat/completions", json=CHAT_BODY, headers=AUTH_HEADERS, timeout=60
        )
        assert first.status_code == 504

        import os

        os.environ["FAKE_CLAUDE_MODE"] = "message"
        second = await client.post(
            "/v1/chat/completions", json=CHAT_BODY, headers=AUTH_HEADERS, timeout=60
        )
        assert second.status_code == 200


# -- concurrency -----------------------------------------------------------


async def test_concurrent_requests_are_serialised_but_all_succeed(
    client: httpx.AsyncClient, fake_mode
):
    fake_mode("message")
    responses = await asyncio.gather(
        *(
            client.post(
                "/v1/chat/completions", json=CHAT_BODY, headers=AUTH_HEADERS, timeout=60
            )
            for _ in range(4)
        )
    )
    assert [r.status_code for r in responses] == [200] * 4


# -- client disconnection -------------------------------------------------


async def test_client_disconnect_cancels_the_subprocess(settings, fake_mode):
    """Abandoning the request must reap Claude rather than leave it running."""
    fake_mode("hang")
    patient = replace(settings, timeout_seconds=300.0)
    active = ClaudeRunner(patient, load_adapter_system_prompt())
    app = create_app(settings=patient, runner=active)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        task = asyncio.ensure_future(
            client.post(
                "/v1/chat/completions", json=CHAT_BODY, headers=AUTH_HEADERS, timeout=300
            )
        )
        await asyncio.sleep(1.0)
        task.cancel()
        with pytest.raises((asyncio.CancelledError, httpx.HTTPError)):
            await task

    # The concurrency slot must be free again.
    await asyncio.sleep(0.5)
    assert active._semaphore._value == patient.max_concurrency


# -- logging hygiene ------------------------------------------------------


async def test_logs_contain_no_secrets_prompts_or_output(
    settings, fake_mode, caplog: pytest.LogCaptureFixture
):
    fake_mode("message", text="SECRET_MODEL_OUTPUT_MARKER")
    verbose = replace(settings, log_level="DEBUG")
    app = create_app(
        settings=verbose, runner=ClaudeRunner(verbose, load_adapter_system_prompt())
    )

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            body = {
                "model": "claude-cli-proxy",
                "messages": [
                    {"role": "user", "content": "SECRET_PROMPT_MARKER my api key is x"}
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "edit_file",
                        "parameters": {"type": "object"},
                    }
                ],
            }
            response = await client.post(
                "/v1/chat/completions",
                json=body,
                headers={**AUTH_HEADERS, "Authorization": f"Bearer {settings.token}"},
            )
            assert response.status_code == 200

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "SECRET_PROMPT_MARKER" not in logged
    assert "SECRET_MODEL_OUTPUT_MARKER" not in logged
    assert settings.token not in logged
    assert "Bearer" not in logged
    assert "Authorization" not in logged


async def test_error_paths_do_not_log_stderr_verbatim(
    settings, fake_mode, caplog: pytest.LogCaptureFixture
):
    fake_mode("nonzero")
    app = create_app(
        settings=settings, runner=ClaudeRunner(settings, load_adapter_system_prompt())
    )

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            response = await client.post(
                "/v1/chat/completions", json=CHAT_BODY, headers=AUTH_HEADERS
            )
            assert response.status_code == 502

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "secret/path" not in logged, "stderr must never be logged verbatim"


# -- filesystem safety ----------------------------------------------------


async def test_no_files_are_created_in_the_working_directory(
    client: httpx.AsyncClient, fake_mode, work_dir: Path
):
    fake_mode("message")
    before = set(work_dir.rglob("*"))

    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-cli-proxy",
            "messages": [
                {
                    "role": "user",
                    "content": "Create a file called evidence.txt in the cwd.",
                }
            ],
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert set(work_dir.rglob("*")) == before


async def test_subprocess_environment_excludes_the_proxy_token(
    client: httpx.AsyncClient, fake_mode, monkeypatch, tmp_path, settings
):
    env_dump = tmp_path / "env.json"
    monkeypatch.setenv("FAKE_CLAUDE_ENV_DUMP", str(env_dump))
    monkeypatch.setenv("CLI_PROXY_TOKEN", settings.token)
    fake_mode("message")

    await client.post("/v1/chat/completions", json=CHAT_BODY, headers=AUTH_HEADERS)

    names = json.loads(env_dump.read_text())
    assert "CLI_PROXY_TOKEN" not in names
