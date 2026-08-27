"""Streaming (server-sent events) tests for both API formats.

Streaming is buffered: Claude runs to completion, then the finished answer is
emitted as a valid event stream. These tests assert the stream is syntactically
correct and terminated properly, not that tokens arrive incrementally.
"""

from __future__ import annotations

import json

import httpx

from tests.conftest import AUTH_HEADERS


def parse_sse(raw: str) -> list[tuple[str | None, str]]:
    """Split an SSE body into (event, data) pairs."""
    events: list[tuple[str | None, str]] = []
    for block in raw.split("\n\n"):
        block = block.strip("\n")
        if not block:
            continue
        event_name: str | None = None
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if data_lines:
            events.append((event_name, "\n".join(data_lines)))
    return events


# -- chat completions ------------------------------------------------------


async def test_chat_stream_is_well_formed(client: httpx.AsyncClient, fake_mode):
    fake_mode("message", text="Streamed answer.")
    body = {
        "model": "claude-cli-sonnet",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(response.text)
    assert events[-1][1] == "[DONE]", "stream must end with the [DONE] marker"

    chunks = [json.loads(data) for _, data in events[:-1]]
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert len({chunk["id"] for chunk in chunks}) == 1, "one id for the whole stream"
    assert all(chunk["model"] == "claude-cli-sonnet" for chunk in chunks)

    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"

    text = "".join(
        chunk["choices"][0]["delta"].get("content") or ""
        for chunk in chunks
        if chunk["choices"]
    )
    assert text == "Streamed answer."

    finish_reasons = [
        chunk["choices"][0]["finish_reason"] for chunk in chunks if chunk["choices"]
    ]
    assert finish_reasons[-1] == "stop"
    assert finish_reasons.count("stop") == 1

    usage_chunks = [chunk for chunk in chunks if chunk.get("usage")]
    assert usage_chunks and usage_chunks[-1]["usage"]["total_tokens"] == 23


async def test_chat_stream_carries_tool_calls(client: httpx.AsyncClient, fake_mode):
    fake_mode(
        "tool_calls",
        tool_calls=json.dumps(
            [{"name": "read_file", "arguments": {"target_file": "src/main.py"}}]
        ),
    )
    body = {
        "model": "claude-cli-proxy",
        "messages": [{"role": "user", "content": "read it"}],
        "tools": [{"type": "function", "name": "read_file", "parameters": {}}],
        "stream": True,
    }
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 200

    events = parse_sse(response.text)
    assert events[-1][1] == "[DONE]"
    chunks = [json.loads(data) for _, data in events[:-1]]

    tool_deltas = [
        chunk["choices"][0]["delta"]["tool_calls"]
        for chunk in chunks
        if chunk["choices"] and "tool_calls" in chunk["choices"][0]["delta"]
    ]
    assert len(tool_deltas) == 1

    entry = tool_deltas[0][0]
    assert entry["index"] == 0
    assert entry["type"] == "function"
    assert entry["id"].startswith("call_")
    assert entry["function"]["name"] == "read_file"
    assert json.loads(entry["function"]["arguments"]) == {"target_file": "src/main.py"}

    finish_reasons = [
        chunk["choices"][0]["finish_reason"] for chunk in chunks if chunk["choices"]
    ]
    assert finish_reasons[-1] == "tool_calls"


# -- responses -------------------------------------------------------------


async def test_responses_stream_is_well_formed(client: httpx.AsyncClient, fake_mode):
    fake_mode("message", text="Responses stream.")
    body = {"model": "claude-cli-sonnet", "input": "hi", "stream": True}
    response = await client.post("/v1/responses", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(response.text)
    assert events[-1][1] == "[DONE]"

    named = [(name, json.loads(data)) for name, data in events[:-1]]
    types = [name for name, _ in named]

    assert types[0] == "response.created"
    assert types[-1] == "response.completed"
    for required in (
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
    ):
        assert required in types, f"missing {required}"

    # Every event names its own type and carries a monotonic sequence number.
    for index, (name, payload) in enumerate(named):
        assert payload["type"] == name
        assert payload["sequence_number"] == index

    deltas = [p["delta"] for n, p in named if n == "response.output_text.delta"]
    assert "".join(deltas) == "Responses stream."

    final = named[-1][1]["response"]
    assert final["status"] == "completed"
    assert final["output_text"] == "Responses stream."
    assert final["output"][0]["content"][0]["text"] == "Responses stream."

    response_ids = {p["response"]["id"] for n, p in named if "response" in p}
    assert len(response_ids) == 1


async def test_responses_stream_carries_function_call_items(
    client: httpx.AsyncClient, fake_mode
):
    fake_mode(
        "tool_calls",
        preamble="Reading.",
        tool_calls=json.dumps([{"name": "read_file", "arguments": {"target_file": "a"}}]),
    )
    body = {
        "model": "claude-cli-proxy",
        "input": "read a",
        "tools": [{"type": "function", "name": "read_file", "parameters": {}}],
        "stream": True,
    }
    response = await client.post("/v1/responses", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 200

    events = parse_sse(response.text)
    assert events[-1][1] == "[DONE]"
    named = [(name, json.loads(data)) for name, data in events[:-1]]
    types = [name for name, _ in named]

    assert "response.function_call_arguments.delta" in types
    assert "response.function_call_arguments.done" in types

    done_args = [
        payload["arguments"]
        for name, payload in named
        if name == "response.function_call_arguments.done"
    ]
    assert json.loads(done_args[0]) == {"target_file": "a"}

    final_items = named[-1][1]["response"]["output"]
    assert [item["type"] for item in final_items] == ["message", "function_call"]


# -- error handling while streaming ---------------------------------------


async def test_chat_stream_reports_a_late_failure_in_band(
    client: httpx.AsyncClient, fake_mode
):
    """The stream opens before Claude runs, so a CLI failure arrives in band.

    An HTTP error status is no longer available once the 200 has been sent, so
    the stream is closed cleanly and carries an ``error`` object instead.
    """
    fake_mode("nonzero")
    body = {
        "model": "claude-cli-proxy",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(response.text)
    assert events[-1][1] == "[DONE]"

    payloads = [json.loads(data) for _, data in events[:-1]]
    errors = [payload["error"] for payload in payloads if "error" in payload]
    assert len(errors) == 1
    assert errors[0]["type"] == "api_error"
    assert "exited unsuccessfully" in errors[0]["message"]

    finish_reasons = [
        payload["choices"][0]["finish_reason"]
        for payload in payloads
        if payload.get("choices")
    ]
    assert finish_reasons[-1] == "stop"


async def test_responses_stream_reports_a_late_failure_as_response_failed(
    client: httpx.AsyncClient, fake_mode
):
    fake_mode("nonzero")
    body = {"model": "claude-cli-proxy", "input": "hi", "stream": True}
    response = await client.post("/v1/responses", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 200

    events = parse_sse(response.text)
    assert events[-1][1] == "[DONE]"

    named = [(name, json.loads(data)) for name, data in events[:-1]]
    types = [name for name, _ in named]
    assert types[0] == "response.created"
    assert types[-1] == "response.failed"

    failed = named[-1][1]["response"]
    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "api_error"
    assert "exited unsuccessfully" in failed["error"]["message"]


async def test_model_structured_error_is_reported_in_band_when_streaming(
    client: httpx.AsyncClient, fake_mode
):
    fake_mode("error", error="I will not do that")
    body = {
        "model": "claude-cli-proxy",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 200

    events = parse_sse(response.text)
    payloads = [json.loads(data) for _, data in events[:-1]]
    errors = [payload["error"] for payload in payloads if "error" in payload]
    assert errors and "I will not do that" in errors[0]["message"]


async def test_non_streaming_failure_still_returns_a_json_http_error(
    client: httpx.AsyncClient, fake_mode
):
    """Only the streaming path changed; buffered requests keep their status."""
    fake_mode("nonzero")
    body = {
        "model": "claude-cli-proxy",
        "messages": [{"role": "user", "content": "hi"}],
    }
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 502
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["type"] == "api_error"


async def test_failures_detected_before_the_stream_opens_are_http_errors(
    client: httpx.AsyncClient, fake_mode
):
    """Auth, size and shape problems are known early, so they stay HTTP errors."""
    fake_mode("message")

    oversized = {
        "model": "claude-cli-proxy",
        "messages": [{"role": "user", "content": "x" * 200_000}],
        "stream": True,
    }
    response = await client.post(
        "/v1/chat/completions", json=oversized, headers=AUTH_HEADERS
    )
    assert response.status_code == 413

    malformed = await client.post(
        "/v1/chat/completions",
        content=b"{not json",
        headers={**AUTH_HEADERS, "Content-Type": "application/json"},
    )
    assert malformed.status_code == 400

    image = {
        "model": "claude-cli-proxy",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,A"}}
                ],
            }
        ],
        "stream": True,
    }
    rejected = await client.post(
        "/v1/chat/completions", json=image, headers=AUTH_HEADERS
    )
    assert rejected.status_code == 400


async def test_stream_requires_authentication(client: httpx.AsyncClient):
    body = {
        "model": "claude-cli-proxy",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 401
