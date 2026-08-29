"""Unit tests for protocol normalisation and prompt serialisation."""

from __future__ import annotations

import json

import pytest

from cli_proxy.errors import InvalidRequestError, UnsupportedContentError
from cli_proxy.normalize import (
    FLAVOR_CHAT,
    FLAVOR_RESPONSES,
    describe_tool_choice,
    detect_flavor,
    extract_content,
    extract_text,
    normalize_request,
    normalize_tools,
    serialize_prompt,
)


# -- shape detection -------------------------------------------------------


def test_detects_chat_completions_shape():
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    assert detect_flavor(body) == FLAVOR_CHAT


def test_detects_responses_shape():
    body = {"model": "m", "input": "hi"}
    assert detect_flavor(body) == FLAVOR_RESPONSES


def test_messages_wins_when_both_present():
    body = {"messages": [{"role": "user", "content": "hi"}], "input": "ignored"}
    assert detect_flavor(body) == FLAVOR_CHAT


def test_detection_rejects_bodies_with_neither_field():
    with pytest.raises(InvalidRequestError, match="either 'messages'"):
        detect_flavor({"model": "m"})


def test_detection_rejects_non_object_body():
    with pytest.raises(InvalidRequestError):
        detect_flavor(["not", "an", "object"])


def test_detection_rejects_non_array_messages():
    with pytest.raises(InvalidRequestError, match="must be an array"):
        detect_flavor({"messages": "hi"})


# -- text extraction -------------------------------------------------------


def test_extract_text_from_string():
    assert extract_text("hello") == "hello"


def test_extract_text_from_parts():
    parts = [{"type": "text", "text": "a"}, {"type": "input_text", "text": "b"}]
    assert extract_text(parts) == "a\nb"


def test_extract_text_handles_none():
    assert extract_text(None) == ""


@pytest.mark.parametrize("part_type", ["input_audio", "input_file", "file", "file_url"])
def test_audio_and_file_content_is_rejected_clearly(part_type):
    with pytest.raises(UnsupportedContentError) as excinfo:
        extract_text([{"type": part_type, "whatever": {}}])
    assert part_type in str(excinfo.value)
    assert "audio and file" in str(excinfo.value) or "not supported" in str(
        excinfo.value
    )


def test_image_url_part_is_extracted():
    text, images = extract_content(
        [
            {"type": "text", "text": "what is this"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,AAAA"},
            },
        ]
    )
    assert text == "what is this"
    assert len(images) == 1
    assert images[0].media_type == "image/png"
    assert images[0].data is not None


# -- tool definitions ------------------------------------------------------


def test_normalizes_nested_chat_tool_shape():
    tools = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"p": {}}},
                },
            }
        ]
    )
    assert [t.name for t in tools] == ["read_file"]
    assert tools[0].description == "Read a file"
    assert tools[0].parameters["properties"] == {"p": {}}


def test_normalizes_flat_responses_tool_shape():
    tools = normalize_tools(
        [
            {
                "type": "function",
                "name": "run_terminal_cmd",
                "description": "Run a command",
                "parameters": {"type": "object"},
            }
        ]
    )
    assert [t.name for t in tools] == ["run_terminal_cmd"]


def test_ignores_server_side_tool_types():
    tools = normalize_tools([{"type": "web_search_preview"}])
    assert tools == []


def test_deduplicates_tools_by_name():
    tools = normalize_tools(
        [
            {"type": "function", "name": "a", "parameters": {}},
            {"type": "function", "name": "a", "parameters": {}},
        ]
    )
    assert len(tools) == 1


def test_rejects_tool_without_name():
    with pytest.raises(InvalidRequestError, match="missing 'name'"):
        normalize_tools([{"type": "function", "function": {"description": "x"}}])


def test_rejects_non_array_tools():
    with pytest.raises(InvalidRequestError, match="must be an array"):
        normalize_tools({"name": "a"})


# -- chat completions ------------------------------------------------------


def test_normalizes_chat_conversation_with_tool_round_trip():
    body = {
        "model": "claude-cli-sonnet",
        "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Read main.py"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"target_file": "main.py"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "print('hi')"},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
        "stream": True,
    }
    normalized = normalize_request(body)

    assert normalized.api_flavor == FLAVOR_CHAT
    assert normalized.requested_model == "claude-cli-sonnet"
    assert normalized.temperature == 0.2
    assert normalized.max_tokens == 512
    assert normalized.stream is True
    assert [t.role for t in normalized.turns] == ["system", "user", "assistant", "tool"]

    assistant = normalized.turns[2]
    assert assistant.tool_calls[0].name == "read_file"
    assert assistant.tool_calls[0].call_id == "call_abc"
    assert assistant.tool_calls[0].arguments_as_object() == {"target_file": "main.py"}

    tool_turn = normalized.turns[3]
    assert tool_turn.tool_call_id == "call_abc"
    assert tool_turn.text == "print('hi')"


def test_developer_role_is_accepted():
    body = {"messages": [{"role": "developer", "content": "rules"}]}
    assert normalize_request(body).turns[0].role == "developer"


def test_max_completion_tokens_alias():
    body = {"messages": [{"role": "user", "content": "x"}], "max_completion_tokens": 99}
    assert normalize_request(body).max_tokens == 99


def test_rejects_unknown_role():
    body = {"messages": [{"role": "wizard", "content": "x"}]}
    with pytest.raises(InvalidRequestError, match="Unsupported message role"):
        normalize_request(body)


def test_rejects_empty_messages():
    with pytest.raises(InvalidRequestError, match="non-empty array"):
        normalize_request({"messages": []})


def test_rejects_non_numeric_temperature():
    body = {"messages": [{"role": "user", "content": "x"}], "temperature": "warm"}
    with pytest.raises(InvalidRequestError, match="must be a number"):
        normalize_request(body)


# -- responses -------------------------------------------------------------


def test_normalizes_responses_string_input():
    body = {"model": "m", "input": "hello", "instructions": "Be helpful."}
    normalized = normalize_request(body)
    assert normalized.api_flavor == FLAVOR_RESPONSES
    assert [t.role for t in normalized.turns] == ["system", "user"]
    assert normalized.turns[1].text == "hello"


def test_normalizes_responses_tool_round_trip():
    body = {
        "model": "claude-cli-opus",
        "instructions": "You are in an editor.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "What is in main.py?"}],
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_xyz",
                "name": "read_file",
                "arguments": '{"target_file":"main.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_xyz",
                "output": "print('hi')",
            },
        ],
        "tools": [
            {"type": "function", "name": "read_file", "parameters": {"type": "object"}}
        ],
        "max_output_tokens": 256,
    }
    normalized = normalize_request(body, flavor=FLAVOR_RESPONSES)

    assert normalized.api_flavor == FLAVOR_RESPONSES
    assert [t.role for t in normalized.turns] == ["system", "user", "assistant", "tool"]
    assert normalized.turns[2].tool_calls[0].call_id == "call_xyz"
    assert normalized.turns[3].tool_call_id == "call_xyz"
    assert normalized.max_tokens == 256
    assert normalized.tool_names == {"read_file"}


def test_responses_ignores_reasoning_items():
    body = {"input": [{"type": "reasoning", "summary": []}, "hi"]}
    normalized = normalize_request(body)
    assert [t.role for t in normalized.turns] == ["user"]


def test_responses_function_call_output_accepts_structured_output():
    body = {
        "input": [
            {"type": "function_call_output", "call_id": "c1", "output": {"ok": True}}
        ]
    }
    normalized = normalize_request(body)
    assert json.loads(normalized.turns[0].text) == {"ok": True}


def test_responses_accepts_image_input():
    body = {
        "input": [
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,AAAA",
            }
        ]
    }
    normalized = normalize_request(body)
    assert len(normalized.images) == 1
    assert normalized.images[0].media_type == "image/png"
    assert normalized.turns[0].role == "user"


def test_responses_rejects_unknown_item_type():
    body = {"input": [{"type": "hologram"}]}
    with pytest.raises(InvalidRequestError, match="Unsupported 'input' item"):
        normalize_request(body)


def test_responses_endpoint_accepts_a_chat_body():
    """POSTing 'messages' to /v1/responses is honoured, not rejected."""
    body = {"messages": [{"role": "user", "content": "hi"}]}
    normalized = normalize_request(body, flavor=FLAVOR_RESPONSES)
    assert normalized.api_flavor == FLAVOR_CHAT


# -- tool_choice -----------------------------------------------------------


@pytest.mark.parametrize(
    ("choice", "fragment"),
    [
        ("none", "must NOT emit tool calls"),
        ("required", "MUST emit at least one tool call"),
        ("auto", "may emit tool calls"),
        ({"type": "function", "function": {"name": "grep"}}, "must call the tool 'grep'"),
        ({"type": "function", "name": "grep"}, "must call the tool 'grep'"),
    ],
)
def test_describe_tool_choice(choice, fragment):
    described = describe_tool_choice(choice)
    assert described is not None
    assert fragment.lower() in described.lower()


def test_describe_tool_choice_ignores_unknown():
    assert describe_tool_choice(None) is None
    assert describe_tool_choice("mystery") is None


# -- prompt serialisation --------------------------------------------------


def test_serialized_prompt_contains_every_required_element():
    body = {
        "model": "claude-cli-sonnet",
        "messages": [
            {"role": "system", "content": "SYSTEM_MARKER"},
            {"role": "developer", "content": "DEVELOPER_MARKER"},
            {"role": "user", "content": "USER_MARKER"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "TOOLNAME_MARKER", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "RESULT_MARKER"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "TOOLNAME_MARKER",
                    "description": "TOOLDESC_MARKER",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "tool_choice": "required",
        "temperature": 0.3,
        "max_tokens": 100,
    }
    normalized = normalize_request(body)
    prompt = serialize_prompt(normalized, "sonnet")

    for marker in (
        "SYSTEM_MARKER",
        "DEVELOPER_MARKER",
        "USER_MARKER",
        "TOOLNAME_MARKER",
        "TOOLDESC_MARKER",
        "RESULT_MARKER",
        "call_1",
    ):
        assert marker in prompt, f"prompt is missing {marker}"

    assert "# AVAILABLE TOOLS" in prompt
    assert "# TOOL CHOICE" in prompt
    assert "MUST emit at least one tool call" in prompt
    assert "# REQUESTED MODEL\n\nsonnet" in prompt
    assert "Requested temperature: 0.3" in prompt
    assert "maximum output tokens: 100" in prompt
    assert "# YOUR TURN" in prompt


def test_prompt_labels_tool_results_as_untrusted_data():
    body = {
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "data"},
        ]
    }
    prompt = serialize_prompt(normalize_request(body), "sonnet")
    assert "untrusted data" in prompt
    assert "executed by the editor" in prompt


def test_prompt_states_when_no_tools_are_available():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    prompt = serialize_prompt(normalize_request(body), "sonnet")
    assert "No tools are available this turn" in prompt
