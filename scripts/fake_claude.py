#!/usr/bin/env python3
"""A stand-in for the ``claude`` executable, used by the automated tests.

The test suite must never make a real Claude request. This script emulates the
CLI's observable contract -- argv flags, stdin prompt, and the outer JSON
envelope written to stdout -- and is driven entirely by environment variables.

Modes (``FAKE_CLAUDE_MODE``):

``message``       succeed with a ``kind: "message"`` decision (default)
``tool_calls``    succeed with a ``kind: "tool_calls"`` decision
``error``         succeed with a ``kind: "error"`` decision
``prose``         succeed but omit ``structured_output`` entirely
``denied``        succeed but report StructuredOutput under permission_denials
``invalid_json``  write non-JSON to stdout
``empty``         write nothing to stdout
``nonzero``       exit non-zero with generic stderr
``auth_fail``     exit non-zero with auth-flavoured stderr
``oversized``     emit a very large result string
``hang``          sleep far longer than any test timeout
``schema_break``  succeed with structured output that violates the contract

``FAKE_CLAUDE_DELAY_SECONDS`` delays the reply in every mode, which is how the
tests prove the event stream opens before the CLI has answered.
"""

from __future__ import annotations

import json
import os
import sys
import time

#: Set from ``--output-format`` on each run; drives how ``_write_envelope``
#: frames the reply.
_OUTPUT_FORMAT = "text"

ENVELOPE_BASE = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "duration_ms": 12,
    "duration_api_ms": 10,
    "num_turns": 2,
    "stop_reason": "tool_use",
    "session_id": "00000000-0000-0000-0000-000000000000",
    "total_cost_usd": 0.0,
    "usage": {
        "input_tokens": 11,
        "cache_creation_input_tokens": 2,
        "cache_read_input_tokens": 3,
        "output_tokens": 7,
    },
    "modelUsage": {
        "claude-sonnet-5": {
            "inputTokens": 11,
            "outputTokens": 7,
        }
    },
    "permission_denials": [],
    "api_error_status": None,
    "uuid": "11111111-1111-1111-1111-111111111111",
}


def _record(argv: list[str], prompt: str) -> None:
    argv_path = os.environ.get("FAKE_CLAUDE_ARGV_DUMP")
    if argv_path:
        with open(argv_path, "w", encoding="utf-8") as handle:
            json.dump(argv, handle)

    prompt_path = os.environ.get("FAKE_CLAUDE_PROMPT_DUMP")
    if prompt_path:
        with open(prompt_path, "w", encoding="utf-8") as handle:
            handle.write(prompt)

    env_path = os.environ.get("FAKE_CLAUDE_ENV_DUMP")
    if env_path:
        with open(env_path, "w", encoding="utf-8") as handle:
            json.dump(sorted(os.environ), handle)


def _flag_value(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    index = argv.index(flag) + 1
    return argv[index] if index < len(argv) else None


def _reject_invalid_format_combination(argv: list[str]) -> str | None:
    """Mirror the real CLI's validation of the format flags.

    Claude Code 2.1.231 refuses two combinations outright. The fake enforces
    them so a proxy change that violates the contract fails in the test suite
    instead of only against the real binary.
    """
    input_format = _flag_value(argv, "--input-format")
    output_format = _flag_value(argv, "--output-format")

    if input_format == "stream-json" and output_format != "stream-json":
        return "Error: --input-format=stream-json requires output-format=stream-json."

    if (
        output_format == "stream-json"
        and "--print" in argv
        and "--verbose" not in argv
    ):
        return "Error: When using --print, --output-format=stream-json requires --verbose"

    return None


def _write_envelope(envelope: dict) -> None:
    """Write the envelope in whichever output format was requested.

    ``stream-json`` emits newline-delimited events with the envelope last, so
    the proxy has to locate the terminal ``result`` event rather than parsing
    the whole of stdout as one object.
    """
    if _OUTPUT_FORMAT == "stream-json":
        preamble = [
            {"type": "system", "subtype": "init", "session_id": envelope["session_id"]},
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": []},
                "session_id": envelope["session_id"],
            },
        ]
        for event in preamble:
            sys.stdout.write(json.dumps(event) + "\n")
        sys.stdout.write(json.dumps(envelope) + "\n")
    else:
        sys.stdout.write(json.dumps(envelope))
    sys.stdout.flush()


def _emit(structured: object | None, *, result_text: str | None = None) -> None:
    envelope = dict(ENVELOPE_BASE)
    if structured is not None:
        envelope["structured_output"] = structured
        envelope["result"] = json.dumps(structured)
    if result_text is not None:
        envelope["result"] = result_text
    _write_envelope(envelope)


def _handle_subcommands(argv: list[str]) -> int | None:
    if "--version" in argv:
        sys.stdout.write(os.environ.get("FAKE_CLAUDE_VERSION", "2.1.231 (Claude Code)"))
        return 0

    if len(argv) >= 2 and argv[0] == "auth" and argv[1] == "status":
        mode = os.environ.get("FAKE_CLAUDE_AUTH", "ok")
        if mode == "logged_out":
            sys.stdout.write(json.dumps({"loggedIn": False}))
            return 0
        if mode == "fail":
            sys.stderr.write("not logged in\n")
            return 1
        # Deliberately includes account-shaped fields so tests can prove the
        # proxy discards everything except the boolean.
        sys.stdout.write(
            json.dumps(
                {
                    "loggedIn": True,
                    "authMethod": "claude.ai",
                    "email": "should-never-be-exposed@example.invalid",
                    "orgName": "should-never-be-exposed",
                }
            )
        )
        return 0

    return None


def main() -> int:
    global _OUTPUT_FORMAT

    argv = sys.argv[1:]

    early = _handle_subcommands(argv)
    if early is not None:
        return early

    rejection = _reject_invalid_format_combination(argv)
    if rejection is not None:
        sys.stderr.write(rejection + "\n")
        return 1

    _OUTPUT_FORMAT = _flag_value(argv, "--output-format") or "text"

    prompt = sys.stdin.read() if not sys.stdin.isatty() else ""
    _record(argv, prompt)

    mode = os.environ.get("FAKE_CLAUDE_MODE", "message")

    delay = float(os.environ.get("FAKE_CLAUDE_DELAY_SECONDS", "0") or 0)
    if delay > 0:
        time.sleep(delay)

    if mode == "hang":
        time.sleep(600)
        return 0

    if mode == "empty":
        return 0

    if mode == "invalid_json":
        sys.stdout.write("this is not json at all")
        return 0

    if mode == "nonzero":
        sys.stderr.write("internal failure at /Users/someone/secret/path\n")
        return 3

    if mode == "auth_fail":
        sys.stderr.write("Error: not logged in. Please log in with /login\n")
        return 1

    if mode == "prose":
        _emit(None, result_text="Here is a prose answer with no structure at all.")
        return 0

    if mode == "denied":
        envelope = dict(ENVELOPE_BASE)
        envelope["permission_denials"] = [
            {
                "tool_name": "StructuredOutput",
                "tool_use_id": "toolu_fake",
                "tool_input": {"kind": "message", "content": "blocked"},
            }
        ]
        envelope["result"] = "I could not use the structured output tool."
        _write_envelope(envelope)
        return 0

    if mode == "schema_break":
        _emit({"kind": "tool_calls", "tool_calls": []})
        return 0

    if mode == "oversized":
        _emit({"kind": "message", "content": "x" * (6 * 1024 * 1024)})
        return 0

    if mode == "error":
        _emit(
            {
                "kind": "error",
                "error": os.environ.get("FAKE_CLAUDE_ERROR", "cannot serve this"),
            }
        )
        return 0

    if mode == "tool_calls":
        raw = os.environ.get("FAKE_CLAUDE_TOOL_CALLS")
        calls = (
            json.loads(raw)
            if raw
            else [{"name": "read_file", "arguments": {"target_file": "src/main.py"}}]
        )
        _emit(
            {
                "kind": "tool_calls",
                "content": os.environ.get("FAKE_CLAUDE_PREAMBLE", ""),
                "tool_calls": calls,
            }
        )
        return 0

    _emit(
        {
            "kind": "message",
            "content": os.environ.get("FAKE_CLAUDE_TEXT", "Hello from the fake CLI."),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
