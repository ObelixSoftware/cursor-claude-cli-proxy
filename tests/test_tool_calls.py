"""Tool-call translation and round-trip tests.

A round trip is: the proxy asks for a tool call, the editor executes it, the
editor sends the result back, and the proxy feeds that result to a fresh
stateless Claude invocation.
"""

from __future__ import annotations

import json

import httpx

from tests.conftest import AUTH_HEADERS

TOOLS_CHAT = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_file": {"type": "string"},
                    "should_read_entire_file": {"type": "boolean"},
                },
                "required": ["target_file"],
            },
        },
    }
]


async def test_tool_call_is_translated_to_openai_format(
    client: httpx.AsyncClient, fake_mode
):
    fake_mode(
        "tool_calls",
        tool_calls=json.dumps(
            [
                {
                    "name": "read_file",
                    "arguments": {
                        "target_file": "src/main.py",
                        "should_read_entire_file": True,
                    },
                }
            ]
        ),
    )
    body = {
        "model": "claude-cli-sonnet",
        "messages": [{"role": "user", "content": "Read src/main.py"}],
        "tools": TOOLS_CHAT,
    }
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 200

    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"

    calls = choice["message"]["tool_calls"]
    assert len(calls) == 1
    call = calls[0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "read_file"
    assert isinstance(call["id"], str) and call["id"].startswith("call_")

    # OpenAI requires 'arguments' to be a JSON *string*.
    assert isinstance(call["function"]["arguments"], str)
    assert json.loads(call["function"]["arguments"]) == {
        "target_file": "src/main.py",
        "should_read_entire_file": True,
    }


async def test_tool_names_are_preserved_verbatim(client: httpx.AsyncClient, fake_mode):
    odd_name = "mcp_some-server_do.thing_v2"
    fake_mode("tool_calls", tool_calls=json.dumps([{"name": odd_name, "arguments": {}}]))
    body = {
        "model": "claude-cli-proxy",
        "messages": [{"role": "user", "content": "go"}],
        "tools": [{"type": "function", "name": odd_name, "parameters": {}}],
    }
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    calls = response.json()["choices"][0]["message"]["tool_calls"]
    assert calls[0]["function"]["name"] == odd_name


async def test_call_ids_are_unique_across_parallel_calls(
    client: httpx.AsyncClient, fake_mode
):
    fake_mode(
        "tool_calls",
        tool_calls=json.dumps(
            [
                {"name": "read_file", "arguments": {"target_file": "a.py"}},
                {"name": "read_file", "arguments": {"target_file": "b.py"}},
                {"name": "read_file", "arguments": {"target_file": "c.py"}},
            ]
        ),
    )
    body = {
        "model": "claude-cli-proxy",
        "messages": [{"role": "user", "content": "read all three"}],
        "tools": TOOLS_CHAT,
    }
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)

    calls = response.json()["choices"][0]["message"]["tool_calls"]
    assert len(calls) == 3
    ids = [call["id"] for call in calls]
    assert len(set(ids)) == 3, "call ids must be unique"
    assert all(call_id.startswith("call_") for call_id in ids)


async def test_call_ids_are_unique_across_requests(client: httpx.AsyncClient, fake_mode):
    fake_mode(
        "tool_calls",
        tool_calls=json.dumps([{"name": "read_file", "arguments": {"target_file": "a"}}]),
    )
    body = {
        "model": "claude-cli-proxy",
        "messages": [{"role": "user", "content": "read"}],
        "tools": TOOLS_CHAT,
    }
    first = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    second = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)

    id_one = first.json()["choices"][0]["message"]["tool_calls"][0]["id"]
    id_two = second.json()["choices"][0]["message"]["tool_calls"][0]["id"]
    assert id_one != id_two


async def test_preamble_text_accompanies_tool_calls(
    client: httpx.AsyncClient, fake_mode
):
    fake_mode(
        "tool_calls",
        preamble="Let me read that file.",
        tool_calls=json.dumps([{"name": "read_file", "arguments": {"target_file": "a"}}]),
    )
    body = {
        "model": "claude-cli-proxy",
        "messages": [{"role": "user", "content": "read"}],
        "tools": TOOLS_CHAT,
    }
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)

    message = response.json()["choices"][0]["message"]
    assert message["content"] == "Let me read that file."
    assert len(message["tool_calls"]) == 1


async def test_full_round_trip_feeds_results_back_to_claude(
    client: httpx.AsyncClient, fake_mode, monkeypatch, tmp_path
):
    # Turn 1: the model asks for a tool call.
    fake_mode(
        "tool_calls",
        tool_calls=json.dumps(
            [{"name": "read_file", "arguments": {"target_file": "src/main.py"}}]
        ),
    )
    first_body = {
        "model": "claude-cli-sonnet",
        "messages": [{"role": "user", "content": "What does src/main.py do?"}],
        "tools": TOOLS_CHAT,
    }
    first = await client.post(
        "/v1/chat/completions", json=first_body, headers=AUTH_HEADERS
    )
    assert first.status_code == 200

    call = first.json()["choices"][0]["message"]["tool_calls"][0]
    call_id = call["id"]

    # Turn 2: the editor executed the tool and returns its output.
    prompt_dump = tmp_path / "prompt2.txt"
    monkeypatch.setenv("FAKE_CLAUDE_PROMPT_DUMP", str(prompt_dump))
    fake_mode("message", text="It prints a greeting.")

    second_body = {
        "model": "claude-cli-sonnet",
        "messages": [
            {"role": "user", "content": "What does src/main.py do?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": call["function"]["arguments"],
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": "read_file",
                "content": "MARK_FILE_CONTENT print('hello')",
            },
        ],
        "tools": TOOLS_CHAT,
    }
    second = await client.post(
        "/v1/chat/completions", json=second_body, headers=AUTH_HEADERS
    )
    assert second.status_code == 200

    choice = second.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "It prints a greeting."

    # The tool result and the original call id must both have reached the CLI.
    prompt = prompt_dump.read_text()
    assert "MARK_FILE_CONTENT" in prompt
    assert call_id in prompt
    assert "You previously requested these tool calls" in prompt


async def test_responses_round_trip_emits_function_call_items(
    client: httpx.AsyncClient, fake_mode
):
    fake_mode(
        "tool_calls",
        preamble="Reading now.",
        tool_calls=json.dumps(
            [{"name": "read_file", "arguments": {"target_file": "src/main.py"}}]
        ),
    )
    body = {
        "model": "claude-cli-sonnet",
        "input": "Read src/main.py",
        "tools": [
            {"type": "function", "name": "read_file", "parameters": {"type": "object"}}
        ],
    }
    response = await client.post("/v1/responses", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 200

    output = response.json()["output"]
    kinds = [item["type"] for item in output]
    assert kinds == ["message", "function_call"]

    call_item = output[1]
    assert call_item["name"] == "read_file"
    assert call_item["status"] == "completed"
    assert call_item["id"].startswith("fc_")
    assert call_item["call_id"].startswith("call_")
    assert json.loads(call_item["arguments"]) == {"target_file": "src/main.py"}


async def test_tool_arguments_survive_code_content_without_corruption(
    client: httpx.AsyncClient, fake_mode
):
    """Arguments routinely carry code with quotes, braces and newlines."""
    tricky = {
        "target_file": "src/a b/main.py",
        "code_edit": 'def f():\n    return {"k": "v\\"quoted\\""}\n',
        "instructions": "Line1\nLine2\ttabbed \\ backslash",
    }
    fake_mode(
        "tool_calls",
        tool_calls=json.dumps([{"name": "edit_file", "arguments": tricky}]),
    )
    body = {
        "model": "claude-cli-proxy",
        "messages": [{"role": "user", "content": "edit"}],
        "tools": [{"type": "function", "name": "edit_file", "parameters": {}}],
    }
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)

    call = response.json()["choices"][0]["message"]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == tricky


async def test_tool_call_with_empty_arguments(client: httpx.AsyncClient, fake_mode):
    fake_mode("tool_calls", tool_calls=json.dumps([{"name": "list_dir", "arguments": {}}]))
    body = {
        "model": "claude-cli-proxy",
        "messages": [{"role": "user", "content": "list"}],
        "tools": [{"type": "function", "name": "list_dir", "parameters": {}}],
    }
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    call = response.json()["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["arguments"] == "{}"
