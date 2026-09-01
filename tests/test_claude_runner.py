"""Tests for the Claude subprocess layer.

These run against ``scripts/fake_claude.py``. They assert the safety invariants
that matter most: no shell, prompt over stdin, correct flags, timeout escalation
and refusal to interpret unstructured output.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from cli_proxy.claude_runner import (
    DENIED_TOOLS,
    STRUCTURED_OUTPUT_TOOL,
    ClaudeRunner,
    _auth_status_is_logged_in,
    _looks_like_auth_failure,
    _looks_like_rate_limit,
    _status_is_rate_limit,
    _structured_from_result,
    _subprocess_env,
)
from cli_proxy.errors import (
    ClaudeAuthError,
    ClaudeOutputError,
    ClaudeProcessError,
    ClaudeRateLimitError,
    ClaudeTimeoutError,
    ClaudeUnavailableError,
    ResponseTooLargeError,
)
from cli_proxy.images import ImageAttachment
from cli_proxy.schema import KIND_ERROR, KIND_MESSAGE, KIND_TOOL_CALLS

PROMPT = "# CONVERSATION\n\nhello"


# -- argv construction -----------------------------------------------------


def test_argv_uses_required_safety_flags(runner: ClaudeRunner):
    argv = runner.build_argv("sonnet")

    assert argv[1] == "--print"
    assert "--safe-mode" in argv
    assert "--disable-slash-commands" in argv
    assert "--no-session-persistence" in argv
    assert argv[argv.index("--output-format") + 1] == "json"
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert "--json-schema" in argv
    assert "--system-prompt" in argv


def test_argv_uses_stream_json_input_only_when_images_are_present(runner: ClaudeRunner):
    plain = runner.build_argv("sonnet")
    assert "--input-format" not in plain
    assert "--verbose" not in plain

    with_images = runner.build_argv("sonnet", has_images=True)
    assert with_images[with_images.index("--input-format") + 1] == "stream-json"
    assert "--safe-mode" in with_images
    assert with_images[with_images.index("--tools") + 1] == STRUCTURED_OUTPUT_TOOL


def test_argv_satisfies_the_cli_stream_json_flag_constraints(runner: ClaudeRunner):
    """The CLI rejects stream-json input unless the output format matches and
    ``--verbose`` is set. Both are chained, so both are asserted here."""
    argv = runner.build_argv("sonnet", has_images=True)

    assert argv[argv.index("--input-format") + 1] == "stream-json"
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in argv


def test_argv_never_contains_dangerous_flags(runner: ClaudeRunner):
    argv = runner.build_argv("sonnet")
    assert "--dangerously-skip-permissions" not in argv
    assert "--allow-dangerously-skip-permissions" not in argv
    assert "--bare" not in argv
    assert "--continue" not in argv
    assert "-c" not in argv
    assert "--resume" not in argv
    assert "--permission-mode" not in argv


def test_argv_enables_only_the_structured_output_tool(runner: ClaudeRunner):
    argv = runner.build_argv("sonnet")
    assert argv[argv.index("--tools") + 1] == STRUCTURED_OUTPUT_TOOL

    denied = argv[argv.index("--disallowedTools") + 1].split(",")
    for dangerous in ("Bash", "Edit", "Write", "Read", "Task", "WebFetch"):
        assert dangerous in denied, f"{dangerous} must be explicitly denied"
    assert STRUCTURED_OUTPUT_TOOL not in denied


def test_denied_tool_list_has_no_duplicates():
    assert len(DENIED_TOOLS) == len(set(DENIED_TOOLS))


def test_argv_json_schema_is_valid_json(runner: ClaudeRunner):
    argv = runner.build_argv("sonnet")
    schema = json.loads(argv[argv.index("--json-schema") + 1])
    assert schema["properties"]["kind"]["enum"] == ["message", "tool_calls", "error"]


def test_argv_is_a_list_so_no_shell_is_possible(runner: ClaudeRunner):
    argv = runner.build_argv("sonnet")
    assert isinstance(argv, list)
    assert all(isinstance(item, str) for item in argv)


def test_proxy_token_is_stripped_from_subprocess_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLI_PROXY_TOKEN", "super-secret-token-value")
    env = _subprocess_env()
    assert "CLI_PROXY_TOKEN" not in env
    assert "super-secret-token-value" not in env.values()


# -- happy paths -----------------------------------------------------------


async def test_returns_message_decision(runner: ClaudeRunner, fake_mode):
    fake_mode("message", text="Structured hello")
    decision = await runner.run(PROMPT, "sonnet")
    assert decision.kind == KIND_MESSAGE
    assert decision.content == "Structured hello"
    assert decision.input_tokens == 16
    assert decision.output_tokens == 7


async def test_returns_tool_call_decision(runner: ClaudeRunner, fake_mode):
    fake_mode(
        "tool_calls",
        tool_calls=json.dumps(
            [{"name": "grep", "arguments": {"pattern": "def main", "n": 3}}]
        ),
    )
    decision = await runner.run(PROMPT, "sonnet")
    assert decision.kind == KIND_TOOL_CALLS
    assert decision.tool_calls == [
        {"name": "grep", "arguments": {"pattern": "def main", "n": 3}}
    ]


async def test_returns_error_decision(runner: ClaudeRunner, fake_mode):
    fake_mode("error", error="no tool for that")
    decision = await runner.run(PROMPT, "sonnet")
    assert decision.kind == KIND_ERROR
    assert decision.error == "no tool for that"


async def test_prompt_travels_over_stdin_not_argv(
    runner: ClaudeRunner, fake_mode, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    argv_dump = tmp_path / "argv.json"
    prompt_dump = tmp_path / "prompt.txt"
    monkeypatch.setenv("FAKE_CLAUDE_ARGV_DUMP", str(argv_dump))
    monkeypatch.setenv("FAKE_CLAUDE_PROMPT_DUMP", str(prompt_dump))
    fake_mode("message")

    secret_marker = "UNIQUE_PROMPT_MARKER_9f3a"
    await runner.run(f"{PROMPT}\n{secret_marker}", "sonnet")

    recorded_argv = json.loads(argv_dump.read_text())
    assert secret_marker not in " ".join(recorded_argv), (
        "prompt text must never appear in argv"
    )
    assert secret_marker in prompt_dump.read_text()


async def test_images_travel_as_stream_json_on_stdin(
    runner: ClaudeRunner, fake_mode, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    argv_dump = tmp_path / "argv.json"
    prompt_dump = tmp_path / "prompt.txt"
    monkeypatch.setenv("FAKE_CLAUDE_ARGV_DUMP", str(argv_dump))
    monkeypatch.setenv("FAKE_CLAUDE_PROMPT_DUMP", str(prompt_dump))
    fake_mode("message")

    image = ImageAttachment(media_type="image/png", data="AAAA")
    decision = await runner.run(PROMPT, "sonnet", images=[image])

    # The fake emits a newline-delimited stream for this output format, so a
    # decision coming back at all proves the terminal result event was found.
    assert decision.kind == KIND_MESSAGE

    recorded_argv = json.loads(argv_dump.read_text())
    assert recorded_argv[recorded_argv.index("--input-format") + 1] == "stream-json"
    assert "AAAA" not in " ".join(recorded_argv)

    stdin_payload = json.loads(prompt_dump.read_text())
    assert stdin_payload["type"] == "user"
    blocks = stdin_payload["message"]["content"]
    assert blocks[0]["type"] == "text"
    assert PROMPT in blocks[0]["text"]
    assert blocks[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
    }


# -- failure mapping -------------------------------------------------------


async def test_missing_executable_maps_to_503(settings, tmp_path: Path):
    broken = replace(settings, claude_executable=str(tmp_path / "no-such-claude"))
    with pytest.raises(ClaudeUnavailableError):
        await ClaudeRunner(broken, "prompt").run(PROMPT, "sonnet")


async def test_nonzero_exit_maps_to_502(runner: ClaudeRunner, fake_mode):
    fake_mode("nonzero")
    with pytest.raises(ClaudeProcessError):
        await runner.run(PROMPT, "sonnet")


async def test_stderr_is_never_echoed_to_the_client(runner: ClaudeRunner, fake_mode):
    fake_mode("nonzero")
    with pytest.raises(ClaudeProcessError) as excinfo:
        await runner.run(PROMPT, "sonnet")
    message = excinfo.value.client_message
    assert "secret" not in message
    assert "/Users/" not in message
    assert message == "The Claude Code CLI exited unsuccessfully."


async def test_auth_failure_maps_to_502_with_guidance(runner: ClaudeRunner, fake_mode):
    fake_mode("auth_fail")
    with pytest.raises(ClaudeAuthError) as excinfo:
        await runner.run(PROMPT, "sonnet")
    assert "claude auth status" in excinfo.value.client_message
    assert excinfo.value.status_code == 502
    assert not isinstance(excinfo.value, ClaudeRateLimitError)


async def test_rate_limit_maps_to_429_with_fixed_message(
    runner: ClaudeRunner, fake_mode
):
    fake_mode("rate_limit")
    with pytest.raises(ClaudeRateLimitError) as excinfo:
        await runner.run(PROMPT, "sonnet")
    error = excinfo.value
    assert error.status_code == 429
    assert error.error_type == "rate_limit_error"
    assert error.client_message == ClaudeRateLimitError.client_message
    assert error.log_hint == "claude rate limited (429)"
    assert "1:10pm" not in error.client_message
    assert "Johannesburg" not in error.client_message
    assert "You've hit your" not in error.client_message


async def test_rate_limit_is_not_classified_as_auth(runner: ClaudeRunner, fake_mode):
    fake_mode("rate_limit")
    with pytest.raises(ClaudeRateLimitError):
        await runner.run(PROMPT, "sonnet")


async def test_rate_limit_stream_json_envelope_is_detected(runner: ClaudeRunner):
    """Image requests parse the terminal stream-json result, not one JSON object."""
    result = {
        "type": "result",
        "subtype": "success",
        "is_error": True,
        "api_error_status": "429",
        "terminal_reason": "api_error",
        "result": "You've hit your session limit · resets 1:10pm (Africa/Johannesburg)",
    }
    stdout = (
        json.dumps({"type": "system", "subtype": "init"})
        + "\n"
        + json.dumps(result)
        + "\n"
    ).encode()
    with pytest.raises(ClaudeRateLimitError) as excinfo:
        runner._parse_envelope(stdout, b"", 1, stream_json=True)
    assert excinfo.value.client_message == ClaudeRateLimitError.client_message
    assert "Johannesburg" not in excinfo.value.client_message


async def test_rate_limit_image_request_uses_stream_json_envelope(
    runner: ClaudeRunner, fake_mode
):
    fake_mode("rate_limit")
    image = ImageAttachment(media_type="image/png", data="AAAA")
    with pytest.raises(ClaudeRateLimitError) as excinfo:
        await runner.run(PROMPT, "sonnet", images=[image])
    assert excinfo.value.client_message == ClaudeRateLimitError.client_message
    assert "Johannesburg" not in excinfo.value.client_message


async def test_rate_limit_json_envelope_is_detected_on_zero_exit(runner: ClaudeRunner):
    envelope = {
        "type": "result",
        "subtype": "success",
        "is_error": True,
        "api_error_status": 429,
        "result": "You've hit your session limit · resets 1:10pm (Africa/Johannesburg)",
    }
    with pytest.raises(ClaudeRateLimitError):
        runner._parse_envelope(json.dumps(envelope).encode(), b"", 0)


async def test_invalid_json_maps_to_output_error(runner: ClaudeRunner, fake_mode):
    fake_mode("invalid_json")
    with pytest.raises(ClaudeOutputError):
        await runner.run(PROMPT, "sonnet")


async def test_empty_stdout_maps_to_output_error(runner: ClaudeRunner, fake_mode):
    fake_mode("empty")
    with pytest.raises(ClaudeOutputError):
        await runner.run(PROMPT, "sonnet")


async def test_prose_output_is_refused_not_guessed_at(runner: ClaudeRunner, fake_mode):
    """No structured output means a clear failure, never prose parsing."""
    fake_mode("prose")
    with pytest.raises(ClaudeOutputError):
        await runner.run(PROMPT, "sonnet")


async def test_denied_structured_output_is_reported_precisely(
    runner: ClaudeRunner, fake_mode
):
    fake_mode("denied")
    with pytest.raises(ClaudeOutputError) as excinfo:
        await runner.run(PROMPT, "sonnet")
    assert "structured-output tool" in excinfo.value.client_message
    assert "will not parse" in excinfo.value.client_message


async def test_schema_violation_maps_to_output_error(runner: ClaudeRunner, fake_mode):
    fake_mode("schema_break")
    with pytest.raises(ClaudeOutputError):
        await runner.run(PROMPT, "sonnet")


async def test_oversized_output_is_refused(settings, fake_mode):
    fake_mode("oversized")
    tight = replace(settings, max_response_bytes=64 * 1024)
    with pytest.raises(ResponseTooLargeError):
        await ClaudeRunner(tight, "prompt").run(PROMPT, "sonnet")


# -- timeout and cancellation ---------------------------------------------


async def test_timeout_terminates_the_subprocess(settings, fake_mode):
    fake_mode("hang")
    impatient = replace(settings, timeout_seconds=1.0)
    with pytest.raises(ClaudeTimeoutError):
        await ClaudeRunner(impatient, "prompt").run(PROMPT, "sonnet")


async def test_timeout_does_not_leak_the_process(settings, fake_mode):
    fake_mode("hang")
    impatient = replace(settings, timeout_seconds=1.0)
    active = ClaudeRunner(impatient, "prompt")

    with pytest.raises(ClaudeTimeoutError):
        await active.run(PROMPT, "sonnet")

    # The semaphore must have been released, so a second call still works.
    import os

    os.environ["FAKE_CLAUDE_MODE"] = "message"
    decision = await active.run(PROMPT, "sonnet")
    assert decision.kind == KIND_MESSAGE


async def test_cancellation_reaps_the_subprocess(settings, fake_mode):
    fake_mode("hang")
    active = ClaudeRunner(settings, "prompt")

    task = asyncio.ensure_future(active.run(PROMPT, "sonnet"))
    await asyncio.sleep(0.8)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    import os

    os.environ["FAKE_CLAUDE_MODE"] = "message"
    decision = await asyncio.wait_for(active.run(PROMPT, "sonnet"), timeout=20)
    assert decision.kind == KIND_MESSAGE


async def test_concurrency_is_limited_by_the_semaphore(settings, fake_mode):
    fake_mode("message")
    serial = replace(settings, max_concurrency=1)
    active = ClaudeRunner(serial, "prompt")

    results = await asyncio.gather(
        *(active.run(PROMPT, "sonnet") for _ in range(3))
    )
    assert all(r.kind == KIND_MESSAGE for r in results)
    assert active._semaphore._value == 1


# -- health probe ----------------------------------------------------------


async def test_health_probe_reports_availability(runner: ClaudeRunner):
    probe = await runner.probe_health()
    assert probe.executable_available is True
    assert probe.version == "2.1.231 (Claude Code)"
    assert probe.authenticated is True


async def test_health_probe_leaks_no_account_details(runner: ClaudeRunner):
    """The fake CLI emits email/org fields; none may survive the probe."""
    probe = await runner.probe_health()
    rendered = repr(probe)
    assert "should-never-be-exposed" not in rendered
    assert "example.invalid" not in rendered
    assert "claude.ai" not in rendered


async def test_health_probe_reports_logged_out(
    runner: ClaudeRunner, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("FAKE_CLAUDE_AUTH", "logged_out")
    probe = await runner.probe_health()
    assert probe.executable_available is True
    assert probe.authenticated is False
    assert "not signed in" in probe.detail


async def test_health_probe_handles_missing_executable(settings, tmp_path: Path):
    broken = replace(settings, claude_executable=str(tmp_path / "absent"))
    probe = await ClaudeRunner(broken, "prompt").probe_health()
    assert probe.executable_available is False
    assert probe.authenticated is False


# -- helpers ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["Not logged in", "please log in", "Unauthorized", "invalid API key", "401 error"],
)
def test_auth_failure_detection(text):
    assert _looks_like_auth_failure(text)


def test_auth_failure_detection_ignores_unrelated_errors():
    assert not _looks_like_auth_failure("ENOENT: no such file")
    assert not _looks_like_auth_failure("429")
    assert not _looks_like_auth_failure(
        "You've hit your session limit · resets 1:10pm (Africa/Johannesburg)"
    )


@pytest.mark.parametrize(
    "text",
    [
        "You've hit your session limit · resets 1:10pm (Africa/Johannesburg)",
        "session limit",
        "rate_limit_event rejected",
        "rate limit exceeded",
    ],
)
def test_rate_limit_detection(text):
    assert _looks_like_rate_limit(text)
    assert not _looks_like_auth_failure(text)


def test_rate_limit_detection_ignores_unrelated_errors():
    assert not _looks_like_rate_limit("ENOENT: no such file")
    assert not _looks_like_rate_limit("Error: not logged in. Please log in with /login")


@pytest.mark.parametrize("value", [429, "429", " 429 "])
def test_status_is_rate_limit_accepts_int_or_string(value):
    assert _status_is_rate_limit(value)


@pytest.mark.parametrize("value", [None, "", 401, "401", "ok", 500])
def test_status_is_rate_limit_rejects_other_values(value):
    assert not _status_is_rate_limit(value)


def test_auth_status_parsing_returns_only_a_boolean():
    payload = json.dumps({"loggedIn": True, "email": "a@b.c", "orgName": "Corp"}).encode()
    assert _auth_status_is_logged_in(payload) is True
    assert _auth_status_is_logged_in(json.dumps({"loggedIn": False}).encode()) is False
    assert _auth_status_is_logged_in(b"") is False
    assert _auth_status_is_logged_in(b"not logged in") is False


def test_structured_from_result_accepts_only_whole_json_objects():
    assert _structured_from_result('{"kind":"message","content":"x"}') == {
        "kind": "message",
        "content": "x",
    }
    assert _structured_from_result({"kind": "message"}) == {"kind": "message"}
    assert _structured_from_result("Here you go: {\"kind\":\"message\"}") is None
    assert _structured_from_result("```json\n{\"kind\":\"message\"}\n```") is None
    assert _structured_from_result("plain prose") is None
    assert _structured_from_result(None) is None
