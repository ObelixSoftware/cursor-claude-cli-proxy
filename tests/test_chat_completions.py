"""OpenAI Chat Completions compatibility tests."""

from __future__ import annotations

import json

import httpx
import pytest

from cli_proxy.errors import ClaudeAuthError, ClaudeRateLimitError
from tests.conftest import AUTH_HEADERS


async def test_plain_text_completion_is_openai_shaped(
    client: httpx.AsyncClient, fake_mode
):
    fake_mode("message", text="The parser rejects empty input.")
    body = {
        "model": "claude-cli-sonnet",
        "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Why does the parser fail?"},
        ],
    }
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 200

    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "claude-cli-sonnet"
    assert isinstance(payload["id"], str) and payload["id"].startswith("chatcmpl-")
    assert isinstance(payload["created"], int)

    assert len(payload["choices"]) == 1
    choice = payload["choices"][0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "The parser rejects empty input."
    assert "tool_calls" not in choice["message"]

    usage = payload["usage"]
    assert usage["prompt_tokens"] == 16
    assert usage["completion_tokens"] == 7
    assert usage["total_tokens"] == 23


@pytest.mark.parametrize(
    ("model_id", "expected_alias"),
    [
        ("claude-cli-proxy", "sonnet"),
        ("claude-cli-sonnet", "sonnet"),
        ("claude-cli-opus", "opus"),
    ],
)
async def test_model_id_selects_the_cli_alias(
    client: httpx.AsyncClient, fake_mode, monkeypatch, tmp_path, model_id, expected_alias
):
    argv_dump = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_CLAUDE_ARGV_DUMP", str(argv_dump))
    fake_mode("message")

    body = {"model": model_id, "messages": [{"role": "user", "content": "hi"}]}
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 200
    assert response.json()["model"] == model_id

    argv = json.loads(argv_dump.read_text())
    assert argv[argv.index("--model") + 1] == expected_alias


async def test_optional_parameters_are_accepted(client: httpx.AsyncClient, fake_mode):
    fake_mode("message")
    body = {
        "model": "claude-cli-proxy",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.4,
        "max_tokens": 256,
        "top_p": 1,
        "presence_penalty": 0,
        "n": 1,
        "user": "someone",
        "stream": False,
    }
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 200


async def test_generation_hints_reach_claude(
    client: httpx.AsyncClient, fake_mode, monkeypatch, tmp_path
):
    prompt_dump = tmp_path / "prompt.txt"
    monkeypatch.setenv("FAKE_CLAUDE_PROMPT_DUMP", str(prompt_dump))
    fake_mode("message")

    body = {
        "model": "claude-cli-proxy",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.4,
        "max_tokens": 256,
    }
    await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)

    prompt = prompt_dump.read_text()
    assert "Requested temperature: 0.4" in prompt
    assert "maximum output tokens: 256" in prompt


async def test_multi_turn_conversation_is_passed_whole(
    client: httpx.AsyncClient, fake_mode, monkeypatch, tmp_path
):
    """Cursor owns conversation state, so every turn must reach the CLI."""
    prompt_dump = tmp_path / "prompt.txt"
    monkeypatch.setenv("FAKE_CLAUDE_PROMPT_DUMP", str(prompt_dump))
    fake_mode("message")

    body = {
        "model": "claude-cli-proxy",
        "messages": [
            {"role": "system", "content": "MARK_SYSTEM"},
            {"role": "user", "content": "MARK_USER_1"},
            {"role": "assistant", "content": "MARK_ASSISTANT_1"},
            {"role": "user", "content": "MARK_USER_2"},
        ],
    }
    await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)

    prompt = prompt_dump.read_text()
    for marker in ("MARK_SYSTEM", "MARK_USER_1", "MARK_ASSISTANT_1", "MARK_USER_2"):
        assert marker in prompt


async def test_stateless_invocation_never_uses_continue(
    client: httpx.AsyncClient, fake_mode, monkeypatch, tmp_path
):
    argv_dump = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_CLAUDE_ARGV_DUMP", str(argv_dump))
    fake_mode("message")

    body = {"model": "claude-cli-proxy", "messages": [{"role": "user", "content": "hi"}]}
    await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)

    argv = json.loads(argv_dump.read_text())
    assert "--continue" not in argv
    assert "-c" not in argv
    assert "--resume" not in argv
    assert "--session-id" not in argv
    assert "--no-session-persistence" in argv


RATE_LIMIT_LEAKS = ("1:10pm", "Johannesburg", "You've hit your", "Africa/")


async def test_rate_limit_maps_to_http_429(client: httpx.AsyncClient, fake_mode):
    fake_mode("rate_limit")
    body = {"model": "claude-cli-proxy", "messages": [{"role": "user", "content": "hi"}]}
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 429
    error = response.json()["error"]
    assert error["type"] == "rate_limit_error"
    assert error["code"] == "rate_limit_error"
    assert error["message"] == ClaudeRateLimitError.client_message
    for leak in RATE_LIMIT_LEAKS:
        assert leak not in error["message"]


async def test_auth_failure_is_still_502(client: httpx.AsyncClient, fake_mode):
    fake_mode("auth_fail")
    body = {"model": "claude-cli-proxy", "messages": [{"role": "user", "content": "hi"}]}
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 502
    error = response.json()["error"]
    assert error["type"] == "api_error"
    assert error["code"] == "api_error"
    assert error["message"] == ClaudeAuthError.client_message


async def test_structured_model_error_maps_to_502(client: httpx.AsyncClient, fake_mode):
    fake_mode("error", error="no suitable tool is available")
    body = {"model": "claude-cli-proxy", "messages": [{"role": "user", "content": "hi"}]}
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 502
    assert "no suitable tool is available" in response.json()["error"]["message"]


async def test_unparseable_output_maps_to_502(client: httpx.AsyncClient, fake_mode):
    fake_mode("prose")
    body = {"model": "claude-cli-proxy", "messages": [{"role": "user", "content": "hi"}]}
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 502
    assert "structured output" in response.json()["error"]["message"]


async def test_error_bodies_are_openai_shaped(client: httpx.AsyncClient, fake_mode):
    fake_mode("nonzero")
    body = {"model": "claude-cli-proxy", "messages": [{"role": "user", "content": "hi"}]}
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)

    error = response.json()["error"]
    assert set(error) == {"message", "type", "code", "param"}
    assert isinstance(error["message"], str)


async def test_responses_style_body_posted_to_chat_completions(
    client: httpx.AsyncClient, fake_mode
):
    """Some Cursor builds post 'input' plus flat tools to /v1/chat/completions.

    Detection is by shape, so this must be served as a Responses reply.
    """
    fake_mode("message", text="Detected by shape.")
    body = {
        "model": "claude-cli-proxy",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
        "tools": [
            {"type": "function", "name": "read_file", "parameters": {"type": "object"}}
        ],
    }
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 200

    payload = response.json()
    assert payload["object"] == "response", "shape detection must win over the URL"
    assert payload["output_text"] == "Detected by shape."


async def test_image_url_is_forwarded_to_claude(
    client: httpx.AsyncClient, fake_mode, monkeypatch, tmp_path
):
    argv_dump = tmp_path / "argv.json"
    prompt_dump = tmp_path / "prompt.txt"
    monkeypatch.setenv("FAKE_CLAUDE_ARGV_DUMP", str(argv_dump))
    monkeypatch.setenv("FAKE_CLAUDE_PROMPT_DUMP", str(prompt_dump))
    fake_mode("message", text="A red pixel.")

    body = {
        "model": "claude-cli-sonnet",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what colour is this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        ],
    }
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "A red pixel."

    argv = json.loads(argv_dump.read_text())
    assert argv[argv.index("--input-format") + 1] == "stream-json"

    stdin_payload = json.loads(prompt_dump.read_text())
    blocks = stdin_payload["message"]["content"]
    text_blocks = [
        block.get("text", "")
        for block in blocks
        if block.get("type") == "text"
    ]
    assert any("what colour is this" in text for text in text_blocks)
    assert {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
    } in blocks
