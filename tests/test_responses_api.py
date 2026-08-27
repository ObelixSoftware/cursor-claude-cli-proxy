"""OpenAI Responses API compatibility tests."""

from __future__ import annotations

import json

import httpx

from tests.conftest import AUTH_HEADERS


async def test_text_input_returns_a_response_object(
    client: httpx.AsyncClient, fake_mode
):
    fake_mode("message", text="Responses API works.")
    body = {"model": "claude-cli-sonnet", "input": "Say something."}
    response = await client.post("/v1/responses", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 200

    payload = response.json()
    assert payload["object"] == "response"
    assert payload["id"].startswith("resp_")
    assert payload["status"] == "completed"
    assert payload["model"] == "claude-cli-sonnet"
    assert payload["error"] is None
    assert payload["output_text"] == "Responses API works."

    assert len(payload["output"]) == 1
    item = payload["output"][0]
    assert item["type"] == "message"
    assert item["role"] == "assistant"
    assert item["status"] == "completed"
    assert item["content"][0]["type"] == "output_text"
    assert item["content"][0]["text"] == "Responses API works."

    usage = payload["usage"]
    assert usage["input_tokens"] == 16
    assert usage["output_tokens"] == 7
    assert usage["total_tokens"] == 23


async def test_instructions_are_forwarded_as_system_text(
    client: httpx.AsyncClient, fake_mode, monkeypatch, tmp_path
):
    prompt_dump = tmp_path / "prompt.txt"
    monkeypatch.setenv("FAKE_CLAUDE_PROMPT_DUMP", str(prompt_dump))
    fake_mode("message")

    body = {
        "model": "claude-cli-proxy",
        "instructions": "MARK_INSTRUCTIONS",
        "input": "MARK_INPUT",
    }
    await client.post("/v1/responses", json=body, headers=AUTH_HEADERS)

    prompt = prompt_dump.read_text()
    assert "MARK_INSTRUCTIONS" in prompt
    assert "MARK_INPUT" in prompt
    assert "# EDITOR INSTRUCTIONS" in prompt


async def test_flat_tool_definitions_reach_the_catalogue(
    client: httpx.AsyncClient, fake_mode, monkeypatch, tmp_path
):
    prompt_dump = tmp_path / "prompt.txt"
    monkeypatch.setenv("FAKE_CLAUDE_PROMPT_DUMP", str(prompt_dump))
    fake_mode("message")

    body = {
        "model": "claude-cli-proxy",
        "input": "list the files",
        "tools": [
            {
                "type": "function",
                "name": "list_dir",
                "description": "MARK_TOOL_DESCRIPTION",
                "parameters": {
                    "type": "object",
                    "properties": {"relative_workspace_path": {"type": "string"}},
                    "required": ["relative_workspace_path"],
                },
            }
        ],
    }
    await client.post("/v1/responses", json=body, headers=AUTH_HEADERS)

    prompt = prompt_dump.read_text()
    assert "list_dir" in prompt
    assert "MARK_TOOL_DESCRIPTION" in prompt
    assert "relative_workspace_path" in prompt


async def test_response_echoes_the_tool_catalogue(client: httpx.AsyncClient, fake_mode):
    fake_mode("message")
    body = {
        "model": "claude-cli-proxy",
        "input": "hi",
        "tools": [
            {"type": "function", "name": "grep", "parameters": {"type": "object"}}
        ],
        "temperature": 0.5,
        "max_output_tokens": 400,
    }
    response = await client.post("/v1/responses", json=body, headers=AUTH_HEADERS)
    payload = response.json()

    assert payload["tools"][0]["name"] == "grep"
    assert payload["tools"][0]["type"] == "function"
    assert payload["temperature"] == 0.5
    assert payload["max_output_tokens"] == 400


async def test_previous_tool_calls_and_results_are_replayed(
    client: httpx.AsyncClient, fake_mode, monkeypatch, tmp_path
):
    prompt_dump = tmp_path / "prompt.txt"
    monkeypatch.setenv("FAKE_CLAUDE_PROMPT_DUMP", str(prompt_dump))
    fake_mode("message", text="It prints hi.")

    body = {
        "model": "claude-cli-proxy",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "What does main.py do?"}],
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_previous_1",
                "name": "read_file",
                "arguments": '{"target_file":"main.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_previous_1",
                "output": "MARK_TOOL_RESULT print('hi')",
            },
        ],
        "tools": [
            {"type": "function", "name": "read_file", "parameters": {"type": "object"}}
        ],
    }
    response = await client.post("/v1/responses", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 200
    assert response.json()["output_text"] == "It prints hi."

    prompt = prompt_dump.read_text()
    assert "MARK_TOOL_RESULT" in prompt
    assert "call_previous_1" in prompt
    assert "read_file" in prompt
    assert "TOOL RESULT" in prompt


async def test_chat_body_posted_to_responses_endpoint_is_honoured(
    client: httpx.AsyncClient, fake_mode
):
    fake_mode("message", text="Chat body accepted.")
    body = {
        "model": "claude-cli-proxy",
        "messages": [{"role": "user", "content": "hello"}],
    }
    response = await client.post("/v1/responses", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 200
    assert response.json()["object"] == "chat.completion"


async def test_image_input_is_rejected(client: httpx.AsyncClient):
    body = {
        "model": "claude-cli-proxy",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "data:image/png;base64,AA"}
                ],
            }
        ],
    }
    response = await client.post("/v1/responses", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 400
    assert "text only" in response.json()["error"]["message"]


async def test_empty_input_is_rejected(client: httpx.AsyncClient):
    response = await client.post(
        "/v1/responses", json={"model": "claude-cli-proxy", "input": []},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 400


async def test_tool_choice_constraint_reaches_claude(
    client: httpx.AsyncClient, fake_mode, monkeypatch, tmp_path
):
    prompt_dump = tmp_path / "prompt.txt"
    monkeypatch.setenv("FAKE_CLAUDE_PROMPT_DUMP", str(prompt_dump))
    fake_mode("tool_calls", tool_calls=json.dumps([{"name": "grep", "arguments": {}}]))

    body = {
        "model": "claude-cli-proxy",
        "input": "find main",
        "tools": [
            {"type": "function", "name": "grep", "parameters": {"type": "object"}}
        ],
        "tool_choice": {"type": "function", "name": "grep"},
    }
    await client.post("/v1/responses", json=body, headers=AUTH_HEADERS)

    prompt = prompt_dump.read_text()
    assert "# TOOL CHOICE" in prompt
    assert "MUST call the tool 'grep'" in prompt
