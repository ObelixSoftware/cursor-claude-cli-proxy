"""ASGI-level streaming tests: disconnect and delayed CLI completion.

``httpx.ASGITransport`` buffers the whole response before returning it, so it
cannot observe mid-request disconnects. These tests drive the ASGI application
directly and timestamp every outbound message instead.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

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
    # Starlette's ``is_disconnected`` only sees a message that is already
    # waiting (it peeks with a cancelled scope). Queue the disconnect the
    # way a real ASGI server does, rather than sleeping inside receive().
    disconnect_ready = asyncio.Event()
    if disconnect_after is not None:
        asyncio.get_running_loop().call_later(
            disconnect_after, disconnect_ready.set
        )

    async def receive() -> dict[str, Any]:
        if not body_delivered.is_set():
            body_delivered.set()
            return {"type": "http.request", "body": payload, "more_body": False}
        if disconnect_after is not None:
            if not disconnect_ready.is_set():
                await disconnect_ready.wait()
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


def _start_status(sent: list[SentMessage]) -> int:
    start = [message for _, message in sent if message["type"] == "http.response.start"]
    assert start, "the application sent no HTTP response start"
    return start[0]["status"]


# -- chat completions ------------------------------------------------------


async def test_chat_stream_opens_only_after_the_cli_finishes(app, fake_mode):
    """SSE headers wait for the CLI envelope so a 429 can still be HTTP 429."""
    fake_mode("message", text="Late answer.", delay_seconds="0.4")

    _, sent = await drive(
        app,
        "/v1/chat/completions",
        {
            "model": "claude-cli-sonnet",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )

    assert _start_status(sent) == 200

    chunks = data_chunks(sent)
    assert chunks, "the stream produced no bytes"

    first_elapsed, first_body = chunks[0]
    assert first_elapsed >= 0.3, f"SSE opened before the CLI finished ({first_elapsed:.2f}s)"
    first = json.loads(first_body.removeprefix("data: "))
    assert first["choices"][0]["delta"]["role"] == "assistant"

    assert not any(body.startswith(":") for _, body in chunks)

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
    assert last_elapsed >= 0.3


async def test_responses_stream_opens_only_after_the_cli_finishes(app, fake_mode):
    fake_mode("message", text="Late answer.", delay_seconds="0.4")

    _, sent = await drive(
        app,
        "/v1/responses",
        {"model": "claude-cli-sonnet", "input": "hi", "stream": True},
    )

    assert _start_status(sent) == 200
    chunks = data_chunks(sent)
    first_elapsed, first_body = chunks[0]
    assert first_elapsed >= 0.3, f"SSE opened before the CLI finished ({first_elapsed:.2f}s)"
    assert first_body.startswith("event: response.created")
    assert not any(body.startswith(":") for _, body in chunks)
    assert chunks[-1][1] == "data: [DONE]\n\n"
    assert chunks[-1][0] >= 0.3


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


async def test_client_disconnect_cancels_before_the_stream_opens(app, fake_mode):
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

    assert _start_status(sent) == 499
    bodies = [body for _, body in data_chunks(sent)]
    assert not any("Never delivered." in body for body in bodies)
    assert not any("text/event-stream" in body for body in bodies)
    # The subprocess is cancelled and reaped by ClaudeRunner; give the reaper a
    # moment so the test does not leave a task mid-cleanup.
    await asyncio.sleep(0.2)
