"""The JSON schema that constrains Claude's reply, and its validator.

This schema is handed to the CLI via ``--json-schema``. Claude Code implements
that flag as an internal ``StructuredOutput`` tool, so the tool must be the one
built-in left enabled -- see ``claude_runner`` for why.
"""

from __future__ import annotations

from typing import Any

KIND_MESSAGE = "message"
KIND_TOOL_CALLS = "tool_calls"
KIND_ERROR = "error"

VALID_KINDS = frozenset({KIND_MESSAGE, KIND_TOOL_CALLS, KIND_ERROR})

#: Free-form ``arguments`` objects are deliberate. Tool arguments routinely
#: carry source code, and nesting a JSON *string* inside JSON would force
#: double escaping of that code -- a reliable source of corruption. A nested
#: object avoids the second escaping layer entirely.
ADAPTER_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": [KIND_MESSAGE, KIND_TOOL_CALLS, KIND_ERROR],
            "description": "Which of the three permitted reply shapes this is.",
        },
        "content": {
            "type": "string",
            "description": (
                "Assistant message text in markdown. Required when kind is "
                "'message'. May be a short preamble or empty when kind is "
                "'tool_calls'."
            ),
        },
        "tool_calls": {
            "type": "array",
            "description": "Required when kind is 'tool_calls'.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Verbatim name of an available editor tool.",
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Arguments object matching that tool's schema.",
                    },
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            },
        },
        "error": {
            "type": "string",
            "description": "Required when kind is 'error'. A short reason.",
        },
    },
    "required": ["kind"],
    "additionalProperties": False,
}


class SchemaViolation(ValueError):
    """Claude returned structured output that does not satisfy the contract."""


def validate_adapter_output(payload: Any) -> dict[str, Any]:
    """Check ``payload`` against the output contract and normalise it.

    Claude Code validates against the schema on its side, but the proxy must not
    trust that: a schema-valid object can still be semantically incomplete (for
    example ``kind: "tool_calls"`` with no calls).
    """
    if not isinstance(payload, dict):
        raise SchemaViolation("structured output is not a JSON object")

    kind = payload.get("kind")
    if kind not in VALID_KINDS:
        raise SchemaViolation("structured output has an unrecognised 'kind'")

    if kind == KIND_MESSAGE:
        content = payload.get("content")
        if not isinstance(content, str):
            raise SchemaViolation("'message' output is missing string 'content'")
        return {"kind": KIND_MESSAGE, "content": content, "tool_calls": []}

    if kind == KIND_TOOL_CALLS:
        raw_calls = payload.get("tool_calls")
        if not isinstance(raw_calls, list) or not raw_calls:
            raise SchemaViolation("'tool_calls' output has no tool calls")

        calls: list[dict[str, Any]] = []
        for entry in raw_calls:
            if not isinstance(entry, dict):
                raise SchemaViolation("a tool call entry is not an object")
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                raise SchemaViolation("a tool call is missing a usable 'name'")
            arguments = entry.get("arguments")
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                raise SchemaViolation(
                    f"tool call '{name}' has non-object 'arguments'"
                )
            calls.append({"name": name.strip(), "arguments": arguments})

        content = payload.get("content")
        return {
            "kind": KIND_TOOL_CALLS,
            "content": content if isinstance(content, str) else "",
            "tool_calls": calls,
        }

    reason = payload.get("error")
    if not isinstance(reason, str) or not reason.strip():
        raise SchemaViolation("'error' output is missing a reason")
    return {"kind": KIND_ERROR, "error": reason.strip(), "tool_calls": []}
