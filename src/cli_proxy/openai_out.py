"""Construction of OpenAI-shaped responses and server-sent event streams.

Streaming here is **buffered in content but not in connection**: the stream is
opened and its first events are emitted before Claude is invoked, then the
finished answer arrives in one burst rather than token by token. The two stream
builders in this module split that into ``opening()``, ``heartbeat()`` and
``body()`` so the caller can hold the connection open while the CLI runs.
See ``LIMITATIONS.md``.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from typing import Any

from .claude_runner import ClaudeDecision
from .normalize import NormalizedRequest, new_call_id, new_item_id
from .schema import KIND_MESSAGE, KIND_TOOL_CALLS

SSE_DONE = "data: [DONE]\n\n"

#: A server-sent-event comment. The SSE specification requires clients to
#: ignore any line beginning with a colon, so this keeps the connection and any
#: intermediary alive without inserting a pseudo-chunk that a strictly typed
#: OpenAI client might try to interpret as model output. Every OpenAI SDK
#: stream parser skips comment lines.
SSE_KEEPALIVE = ": keepalive\n\n"

#: Stand-in used to shape the opening events, before Claude has answered.
_PENDING_DECISION = ClaudeDecision(kind=KIND_MESSAGE)


def _now() -> int:
    return int(time.time())


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:24]}"


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _usage_block(decision: ClaudeDecision, *, responses_shape: bool) -> dict[str, Any]:
    prompt_tokens = max(decision.input_tokens, 0)
    completion_tokens = max(decision.output_tokens, 0)
    total = prompt_tokens + completion_tokens
    if responses_shape:
        return {
            "input_tokens": prompt_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": completion_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": total,
        }
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total,
        # Present in every current OpenAI response. Cheap insurance against a
        # strictly typed client that treats them as required.
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
        "completion_tokens_details": {
            "reasoning_tokens": 0,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    }


def build_tool_calls(decision: ClaudeDecision) -> list[dict[str, Any]]:
    """Translate Claude's decision into OpenAI ``tool_calls`` entries.

    Tool names are preserved verbatim. ``arguments`` is re-serialised to the
    JSON *string* the OpenAI wire format requires.
    """
    entries: list[dict[str, Any]] = []
    for call in decision.tool_calls:
        entries.append(
            {
                "id": new_call_id(),
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(
                        call.get("arguments") or {}, ensure_ascii=False
                    ),
                },
            }
        )
    return entries


# ---------------------------------------------------------------------------
# Chat Completions
# ---------------------------------------------------------------------------


def build_chat_completion(
    decision: ClaudeDecision, *, model_id: str
) -> dict[str, Any]:
    """Build a ``chat.completion`` body."""
    message: dict[str, Any] = {"role": "assistant", "content": None, "refusal": None}
    finish_reason = "stop"

    if decision.kind == KIND_TOOL_CALLS:
        tool_calls = build_tool_calls(decision)
        message["tool_calls"] = tool_calls
        message["content"] = decision.content or None
        finish_reason = "tool_calls"
    else:
        message["content"] = decision.content

    return {
        "id": new_completion_id(),
        "object": "chat.completion",
        "created": _now(),
        "model": model_id,
        "system_fingerprint": None,
        "service_tier": None,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": _usage_block(decision, responses_shape=False),
    }


class ChatStreamBuilder:
    """Builds a ``chat.completion.chunk`` stream in three separable stages.

    ``opening()`` can be emitted before Claude has been asked anything, which is
    what lets the HTTP response start immediately. All stages share one
    completion id and one ``created`` timestamp.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.completion_id = new_completion_id()
        self.created = _now()

    def _chunk(self, delta: dict[str, Any], finish_reason: str | None) -> str:
        payload = {
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model_id,
            "system_fingerprint": None,
            "service_tier": None,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            # Real OpenAI chunks carry an explicit null here until the final
            # usage-only chunk.
            "usage": None,
        }
        return f"data: {_dumps(payload)}\n\n"

    def opening(self) -> Iterator[str]:
        """The role delta, safe to send before any model work has happened."""
        yield self._chunk({"role": "assistant", "content": ""}, None)

    def heartbeat(self) -> str:
        return SSE_KEEPALIVE

    def body(self, decision: ClaudeDecision) -> Iterator[str]:
        """The content, the finish chunk, the usage chunk and ``[DONE]``."""
        if decision.kind == KIND_TOOL_CALLS:
            if decision.content:
                yield self._chunk({"content": decision.content}, None)
            streamed_calls = [
                {
                    "index": index,
                    "id": entry["id"],
                    "type": "function",
                    "function": entry["function"],
                }
                for index, entry in enumerate(build_tool_calls(decision))
            ]
            yield self._chunk({"tool_calls": streamed_calls}, None)
            finish_reason = "tool_calls"
        else:
            if decision.content:
                yield self._chunk({"content": decision.content}, None)
            finish_reason = "stop"

        yield self._chunk({}, finish_reason)

        final = {
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model_id,
            "choices": [],
            "usage": _usage_block(decision, responses_shape=False),
        }
        yield f"data: {_dumps(final)}\n\n"
        yield SSE_DONE

    def failure(self, message: str, *, error_type: str = "api_error") -> Iterator[str]:
        """Terminate an already-open stream with an in-band error.

        Once the 200 has gone out there is no way to send an HTTP error status,
        so the stream is closed cleanly and an ``error`` object is sent in the
        shape the OpenAI clients recognise.
        """
        yield self._chunk({}, "stop")
        payload = {
            "error": {
                "message": message,
                "type": error_type,
                "code": error_type,
                "param": None,
            }
        }
        yield f"data: {_dumps(payload)}\n\n"
        yield SSE_DONE


def stream_chat_completion(
    decision: ClaudeDecision, *, model_id: str
) -> Iterator[str]:
    """Emit a complete ``chat.completion.chunk`` stream for a finished decision."""
    builder = ChatStreamBuilder(model_id)
    yield from builder.opening()
    yield from builder.body(decision)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


def _response_output_items(decision: ClaudeDecision) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    if decision.content:
        items.append(
            {
                "type": "message",
                "id": new_message_id(),
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": decision.content,
                        "annotations": [],
                    }
                ],
            }
        )

    if decision.kind == KIND_TOOL_CALLS:
        for call in decision.tool_calls:
            items.append(
                {
                    "type": "function_call",
                    "id": new_item_id(),
                    "call_id": new_call_id(),
                    "name": call["name"],
                    "arguments": json.dumps(
                        call.get("arguments") or {}, ensure_ascii=False
                    ),
                    "status": "completed",
                }
            )

    if not items:
        items.append(
            {
                "type": "message",
                "id": new_message_id(),
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "", "annotations": []}],
            }
        )

    return items


def build_response(
    decision: ClaudeDecision,
    *,
    model_id: str,
    request: NormalizedRequest | None = None,
    response_id: str | None = None,
    output_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a Responses API ``response`` body."""
    items = output_items if output_items is not None else _response_output_items(decision)
    text_parts = [
        part.get("text", "")
        for item in items
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    ]

    return {
        "id": response_id or new_response_id(),
        "object": "response",
        "created_at": _now(),
        "status": "completed",
        "model": model_id,
        "output": items,
        "output_text": "".join(text_parts),
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": False,
        "temperature": request.temperature if request else None,
        "tool_choice": (request.tool_choice if request else None) or "auto",
        "tools": (
            [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "strict": False,
                }
                for tool in request.tools
            ]
            if request
            else []
        ),
        "top_p": None,
        "max_output_tokens": request.max_tokens if request else None,
        "truncation": "disabled",
        "usage": _usage_block(decision, responses_shape=True),
        "user": None,
    }


class ResponseStreamBuilder:
    """Builds a Responses API event stream in three separable stages.

    ``opening()`` emits ``response.created`` and ``response.in_progress``, both
    of which are well defined before any output exists, so they can go out
    before Claude is invoked.
    """

    def __init__(
        self, model_id: str, request: NormalizedRequest | None = None
    ) -> None:
        self.model_id = model_id
        self.response_id = new_response_id()
        self._request = request
        self._sequence = 0

    def _event(self, event_type: str, payload: dict[str, Any]) -> str:
        body = dict(payload)
        body["type"] = event_type
        body["sequence_number"] = self._sequence
        self._sequence += 1
        return f"event: {event_type}\ndata: {_dumps(body)}\n\n"

    def _snapshot(
        self,
        status: str,
        decision: ClaudeDecision,
        items: list[dict[str, Any]] | None = None,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        shell = build_response(
            decision,
            model_id=self.model_id,
            request=self._request,
            response_id=self.response_id,
            output_items=items if items is not None else [],
        )
        shell["status"] = status
        if error is not None:
            shell["error"] = error
        return shell

    def opening(self) -> Iterator[str]:
        yield self._event(
            "response.created",
            {"response": self._snapshot("in_progress", _PENDING_DECISION)},
        )
        yield self._event(
            "response.in_progress",
            {"response": self._snapshot("in_progress", _PENDING_DECISION)},
        )

    def heartbeat(self) -> str:
        return SSE_KEEPALIVE

    def body(self, decision: ClaudeDecision) -> Iterator[str]:
        items = _response_output_items(decision)

        for output_index, item in enumerate(items):
            if item["type"] == "message":
                skeleton = {**item, "status": "in_progress", "content": []}
                yield self._event(
                    "response.output_item.added",
                    {"output_index": output_index, "item": skeleton},
                )
                part = {"type": "output_text", "text": "", "annotations": []}
                yield self._event(
                    "response.content_part.added",
                    {
                        "item_id": item["id"],
                        "output_index": output_index,
                        "content_index": 0,
                        "part": part,
                    },
                )
                text = item["content"][0]["text"]
                if text:
                    yield self._event(
                        "response.output_text.delta",
                        {
                            "item_id": item["id"],
                            "output_index": output_index,
                            "content_index": 0,
                            "delta": text,
                            "logprobs": [],
                        },
                    )
                yield self._event(
                    "response.output_text.done",
                    {
                        "item_id": item["id"],
                        "output_index": output_index,
                        "content_index": 0,
                        "text": text,
                        "logprobs": [],
                    },
                )
                yield self._event(
                    "response.content_part.done",
                    {
                        "item_id": item["id"],
                        "output_index": output_index,
                        "content_index": 0,
                        "part": item["content"][0],
                    },
                )
                yield self._event(
                    "response.output_item.done",
                    {"output_index": output_index, "item": item},
                )
                continue

            skeleton = {**item, "status": "in_progress", "arguments": ""}
            yield self._event(
                "response.output_item.added",
                {"output_index": output_index, "item": skeleton},
            )
            yield self._event(
                "response.function_call_arguments.delta",
                {
                    "item_id": item["id"],
                    "output_index": output_index,
                    "delta": item["arguments"],
                },
            )
            yield self._event(
                "response.function_call_arguments.done",
                {
                    "item_id": item["id"],
                    "output_index": output_index,
                    "arguments": item["arguments"],
                },
            )
            yield self._event(
                "response.output_item.done",
                {"output_index": output_index, "item": item},
            )

        yield self._event(
            "response.completed",
            {"response": self._snapshot("completed", decision, items=items)},
        )
        yield SSE_DONE

    def failure(self, message: str, *, error_type: str = "api_error") -> Iterator[str]:
        """Terminate an already-open stream with ``response.failed``."""
        error = {"code": error_type, "message": message}
        snapshot = self._snapshot("failed", _PENDING_DECISION, error=error)
        yield self._event("response.failed", {"response": snapshot})
        yield SSE_DONE


def stream_response(
    decision: ClaudeDecision,
    *,
    model_id: str,
    request: NormalizedRequest | None = None,
) -> Iterator[str]:
    """Emit a complete Responses API event stream for a finished decision."""
    builder = ResponseStreamBuilder(model_id, request)
    yield from builder.opening()
    yield from builder.body(decision)
