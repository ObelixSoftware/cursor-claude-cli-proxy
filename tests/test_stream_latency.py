"""Time-to-first-byte behaviour of the event stream.

``httpx.ASGITransport`` buffers the whole response before returning it, so it
cannot observe when the first byte was sent. These tests drive the ASGI
application directly and timestamp every outbound message instead.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from cli_proxy import app as app_module
from tests.conftest import TEST_TOKEN

SentMessage = tuple[float, dict[str, Any]]


def _scope(path: str, *, body_present: bool = True) -> dict[str, Any]:
    headers = [
        (b"host", b"proxy.test"),
        (b"content-type", b"application/json"),
        (b"authorization", f"Bearer {TEST_TOKEN}".encode()),
    ]
    if body_present:
        headers.append((b"accept", b"text/event-stream"))
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "headers": headers,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "server": ("proxy.test", 80),
        "client": ("127.0.0.1", 12345),
        "root_path": "",
    }


async def drive(
    app: Any,
    path: str,
    body: dict[str, Any],
    *,
    disconnect_after: float | None = None,
) -> tuple[float, list[SentMessage]]:
    """Run one request against the raw ASGI app, timestamping each send."""
    payload = json.dumps(body).encode()
    sent: list[SentMessage] = []
    body_delivered = asyncio.Event()
    started = time.monotonic()

    async def receive() -> dict[str, Any]:
        if not body_delivered.is_set():
            body_delivered.set()
            return {"type": "http.request", "body": payload, "more_body": False}
        if disconnect_after is not None:
            await asyncio.sleep(disconnect_after)
            return {"type": "http.disconnect"}
        # Mirror a live server: no further client messages until it hangs up.
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append((time.monotonic() - started, dict(message)))

    await app(_scope(path), receive, send)
    return started, sent


def data_chunks(sent: list[SentMessage]) -> list[tuple[float, str]]:
    return [
        (elapsed, message["body"].decode())
        for elapsed, message in sent
        if message["type"] == "http.response.body" and message.get("body")
    ]


# -- chat completions ------------------------------------------------------


async def test_chat_stream_first_byte_precedes_the_model_answer(
    app, fake_mode, monkeypatch: pytest.MonkeyPatch
):
    """The opening chunk must not wait for the CLI to finish."""
    monkeypatch.setattr(app_module, "_HEARTBEAT_SECONDS", 0.2)
    fake_mode("message", text="Late answer.", delay_seconds="1.5")

    _, sent = await drive(
        app,
        "/v1/chat/completions",
        {
            "model": "claude-cli-sonnet",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )

    start = [message for _, message in sent if message["type"] == "http.response.start"]
    assert start and start[0]["status"] == 200

    chunks = data_chunks(sent)
    assert chunks, "the stream produced no bytes"

    first_elapsed, first_body = chunks[0]
    assert first_elapsed < 0.75, f"first byte took {first_elapsed:.2f}s"
    first = json.loads(first_body.removeprefix("data: "))
    assert first["choices"][0]["delta"]["role"] == "assistant"

    keepalives = [body for _, body in chunks if body.startswith(":")]
    assert keepalives, "expected keepalives while the CLI was running"
    assert all(body == ": keepalive\n\n" for body in keepalives)

    content = [
        json.loads(body.removeprefix("data: "))
        for _, body in chunks
        if body.startswith("data: ") and not body.startswith("data: [DONE]")
    ]
    text = "".join(
        chunk["choices"][0]["delta"].get("content") or ""
        for chunk in content
        if chunk.get("choices")
    )
    assert text == "Late answer."

    last_elapsed, last_body = chunks[-1]
    assert last_body == "data: [DONE]\n\n"
    assert last_elapsed > 1.4, "the answer should still arrive only when ready"


async def test_responses_stream_first_byte_precedes_the_model_answer(
    app, fake_mode, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(app_module, "_HEARTBEAT_SECONDS", 0.2)
    fake_mode("message", text="Late answer.", delay_seconds="1.5")

    _, sent = await drive(
        app,
        "/v1/responses",
        {"model": "claude-cli-sonnet", "input": "hi", "stream": True},
    )

    chunks = data_chunks(sent)
    first_elapsed, first_body = chunks[0]
    assert first_elapsed < 0.75, f"first byte took {first_elapsed:.2f}s"
    assert first_body.startswith("event: response.created")

    assert any(body == ": keepalive\n\n" for _, body in chunks)
    assert chunks[-1][1] == "data: [DONE]\n\n"
    assert chunks[-1][0] > 1.4


async def test_streaming_respects_payload_shape_not_the_url(app, fake_mode):
    """A Responses-shaped streaming body on /v1/chat/completions streams events."""
    fake_mode("message", text="Shape detected.")

    _, sent = await drive(
        app,
        "/v1/chat/completions",
        {"model": "claude-cli-sonnet", "input": "hi", "stream": True},
    )

    bodies = [body for _, body in data_chunks(sent)]
    assert bodies[0].startswith("event: response.created")
    assert any(body.startswith("event: response.completed") for body in bodies)
    assert bodies[-1] == "data: [DONE]\n\n"


async def test_client_disconnect_stops_the_stream_early(
    app, fake_mode, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(app_module, "_HEARTBEAT_SECONDS", 0.2)
    fake_mode("message", text="Never delivered.", delay_seconds="10")

    _, sent = await drive(
        app,
        "/v1/chat/completions",
        {
            "model": "claude-cli-sonnet",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        disconnect_after=0.6,
    )

    bodies = [body for _, body in data_chunks(sent)]
    assert bodies, "the opening chunk should still have been sent"
    assert "data: [DONE]\n\n" not in bodies
    assert not any("Never delivered." in body for body in bodies)
    # The subprocess is cancelled and reaped by ClaudeRunner; give the reaper a
    # moment so the test does not leave a task mid-cleanup.
    await asyncio.sleep(0.2)
