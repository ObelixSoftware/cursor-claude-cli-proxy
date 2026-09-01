"""Parallel agents, prompt-edit supersede, agent counting and process reaping.

These are the invariants Cursor's ``/multitask`` depends on. Everything here
runs against ``scripts/fake_claude.py``; no real Claude call is ever made.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from cli_proxy.app import create_app, load_adapter_system_prompt
from cli_proxy.claude_runner import ClaudeRunner
from cli_proxy.errors import AgentSupersededError
from cli_proxy.logging_setup import LOGGER_NAME
from cli_proxy.normalize import derive_conversation_identity, normalize_request
from cli_proxy.schema import KIND_MESSAGE
from tests.conftest import AUTH_HEADERS

PROMPT = "# CONVERSATION\n\nhello"

CHAT_BODY = {
    "model": "claude-cli-proxy",
    "messages": [{"role": "user", "content": "hi"}],
}


def conversation(last_user: str, *, opening: str = "the original question") -> dict:
    """A body with earlier turns, so it has a prefix to be identified by."""
    return {
        "model": "claude-cli-sonnet",
        "messages": [
            {"role": "user", "content": opening},
            {"role": "assistant", "content": "the earlier answer"},
            {"role": "user", "content": last_user},
        ],
    }


def identity_for(body: dict):
    return derive_conversation_identity(body, normalize_request(body))


async def wait_for_agents(active: ClaudeRunner, count: int, timeout: float = 10.0):
    """Block until ``count`` subprocesses are running, or fail the test."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if active.agents_running == count:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(
        f"expected {count} agents running, saw {active.agents_running}"
    )


def running_pid(active: ClaudeRunner, key: str) -> int:
    agent = active._agents_by_key[key]
    assert agent.process is not None
    return agent.process.pid


def process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


async def wait_until_dead(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_is_alive(pid):
            return True
        await asyncio.sleep(0.02)
    return False


# -- parallelism -----------------------------------------------------------


async def test_three_agents_run_at_the_same_time(settings, fake_mode):
    """Three slots must overlap, not queue: the point of /multitask."""
    fake_mode("message", delay_seconds="0.5")
    parallel = replace(settings, max_concurrency=3)
    active = ClaudeRunner(parallel, "prompt")

    started = time.monotonic()
    results = await asyncio.gather(*(active.run(PROMPT, "sonnet") for _ in range(3)))
    elapsed = time.monotonic() - started

    assert all(r.kind == KIND_MESSAGE for r in results)
    # Serialised, this would be at least 1.5 s of sleeping alone.
    assert elapsed < 1.4, f"three agents took {elapsed:.2f}s; they were serialised"
    assert active.agents_running == 0


async def test_one_slot_still_serialises(settings, fake_mode):
    fake_mode("message", delay_seconds="0.4")
    serial = replace(settings, max_concurrency=1)
    active = ClaudeRunner(serial, "prompt")

    started = time.monotonic()
    results = await asyncio.gather(*(active.run(PROMPT, "sonnet") for _ in range(3)))
    elapsed = time.monotonic() - started

    assert all(r.kind == KIND_MESSAGE for r in results)
    assert elapsed >= 1.2, f"three serialised agents took only {elapsed:.2f}s"
    assert active._semaphore._value == 1


async def test_concurrent_http_requests_all_succeed(settings, fake_mode):
    fake_mode("message")
    parallel = replace(settings, max_concurrency=2)
    app = create_app(
        settings=parallel,
        runner=ClaudeRunner(parallel, load_adapter_system_prompt()),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        responses = await asyncio.gather(
            *(
                client.post(
                    "/v1/chat/completions",
                    json=CHAT_BODY,
                    headers=AUTH_HEADERS,
                    timeout=60,
                )
                for _ in range(4)
            )
        )

    assert [r.status_code for r in responses] == [200] * 4


# -- prompt edits ----------------------------------------------------------


def test_an_edited_prompt_keeps_the_key_and_changes_the_signature():
    first = identity_for(conversation("first version"))
    second = identity_for(conversation("second version"))

    assert first is not None and second is not None
    assert first.key == second.key
    assert first.turn_signature != second.turn_signature


def test_a_different_conversation_gets_a_different_key():
    mine = identity_for(conversation("same question", opening="my thread"))
    theirs = identity_for(conversation("same question", opening="another thread"))

    assert mine is not None and theirs is not None
    assert mine.key != theirs.key


def test_a_first_turn_request_has_no_identity():
    """Nothing to match on, so two /multitask siblings cannot collide."""
    assert identity_for(CHAT_BODY) is None


def test_an_explicit_conversation_id_identifies_even_a_first_turn():
    body = dict(CHAT_BODY, conversation_id="cursor-thread-7")
    found = identity_for(body)

    assert found is not None
    assert found.key.startswith("id:")


def test_identity_digests_do_not_leak_prompt_text():
    found = identity_for(conversation("a very secret instruction"))

    assert found is not None
    assert "secret" not in found.key
    assert "secret" not in found.turn_signature


async def test_editing_a_prompt_kills_the_running_agent(settings, fake_mode):
    fake_mode("hang")
    active = ClaudeRunner(replace(settings, max_concurrency=4), "prompt")

    original = identity_for(conversation("first version"))
    edited = identity_for(conversation("second version"))
    assert original is not None and edited is not None

    stale = asyncio.ensure_future(
        active.run(PROMPT, "sonnet", identity=original)
    )
    await wait_for_agents(active, 1)
    old_pid = running_pid(active, original.key)

    os.environ["FAKE_CLAUDE_MODE"] = "message"
    decision = await asyncio.wait_for(
        active.run(PROMPT, "sonnet", identity=edited), timeout=30
    )

    assert decision.kind == KIND_MESSAGE
    with pytest.raises(AgentSupersededError):
        await asyncio.wait_for(stale, timeout=10)
    assert await wait_until_dead(old_pid), "the superseded process is still alive"
    assert active.agents_running == 0
    assert active._agents_by_key == {}


async def test_a_superseded_agent_does_not_evict_its_replacement(settings, fake_mode):
    """The old agent exits last, so its cleanup must leave the new one registered."""
    fake_mode("hang")
    active = ClaudeRunner(replace(settings, max_concurrency=4), "prompt")

    original = identity_for(conversation("first version"))
    edited = identity_for(conversation("second version"))
    assert original is not None and edited is not None

    stale = asyncio.ensure_future(active.run(PROMPT, "sonnet", identity=original))
    await wait_for_agents(active, 1)

    replacement = asyncio.ensure_future(active.run(PROMPT, "sonnet", identity=edited))
    with pytest.raises(AgentSupersededError):
        await asyncio.wait_for(stale, timeout=10)
    await wait_for_agents(active, 1)

    registered = active._agents_by_key.get(edited.key)
    assert registered is not None, "the replacement was evicted by the agent it replaced"
    assert registered.turn_signature == edited.turn_signature

    replacement.cancel()
    await asyncio.gather(replacement, return_exceptions=True)
    await wait_for_agents(active, 0)


async def test_a_sibling_conversation_is_left_running(settings, fake_mode):
    """Two /multitask agents on different threads must not kill each other."""
    fake_mode("hang")
    active = ClaudeRunner(replace(settings, max_concurrency=4), "prompt")

    mine = identity_for(conversation("question", opening="thread one"))
    theirs = identity_for(conversation("question", opening="thread two"))
    assert mine is not None and theirs is not None

    first = asyncio.ensure_future(active.run(PROMPT, "sonnet", identity=mine))
    await wait_for_agents(active, 1)
    first_pid = running_pid(active, mine.key)

    second = asyncio.ensure_future(active.run(PROMPT, "sonnet", identity=theirs))
    await wait_for_agents(active, 2)

    assert process_is_alive(first_pid)
    assert not first.done()

    for task in (first, second):
        task.cancel()
    await asyncio.gather(first, second, return_exceptions=True)


async def test_resending_the_same_prompt_does_not_supersede(settings, fake_mode):
    """A duplicate is not an edit, so the first agent keeps going."""
    fake_mode("hang")
    active = ClaudeRunner(replace(settings, max_concurrency=4), "prompt")

    same = identity_for(conversation("unchanged question"))
    assert same is not None

    first = asyncio.ensure_future(active.run(PROMPT, "sonnet", identity=same))
    await wait_for_agents(active, 1)
    first_pid = running_pid(active, same.key)

    second = asyncio.ensure_future(active.run(PROMPT, "sonnet", identity=same))
    await wait_for_agents(active, 2)

    assert process_is_alive(first_pid)
    assert not first.done()

    for task in (first, second):
        task.cancel()
    await asyncio.gather(first, second, return_exceptions=True)


async def test_an_agent_waiting_for_a_slot_is_superseded_before_it_spawns(
    settings, fake_mode
):
    fake_mode("hang")
    active = ClaudeRunner(replace(settings, max_concurrency=1), "prompt")

    blocker = asyncio.ensure_future(active.run(PROMPT, "sonnet"))
    await wait_for_agents(active, 1)

    original = identity_for(conversation("first version"))
    edited = identity_for(conversation("second version"))
    assert original is not None and edited is not None

    queued = asyncio.ensure_future(active.run(PROMPT, "sonnet", identity=original))
    await asyncio.sleep(0.2)
    assert active.agents_running == 1, "the queued agent should not have spawned"

    superseding = asyncio.ensure_future(
        active.run(PROMPT, "sonnet", identity=edited)
    )
    with pytest.raises(AgentSupersededError):
        await asyncio.wait_for(queued, timeout=10)

    for task in (blocker, superseding):
        task.cancel()
    await asyncio.gather(blocker, superseding, return_exceptions=True)


# -- process reaping -------------------------------------------------------


async def test_the_process_is_gone_after_a_successful_run(settings, fake_mode):
    fake_mode("message", delay_seconds="0.5")
    active = ClaudeRunner(settings, "prompt")

    named = identity_for(conversation("a question"))
    assert named is not None

    # The pid is only observable while the run is in flight, so read it from
    # the registry while a second task does the work.
    work = asyncio.ensure_future(active.run(PROMPT, "sonnet", identity=named))
    await wait_for_agents(active, 1)
    pid = running_pid(active, named.key)

    decision = await asyncio.wait_for(work, timeout=30)

    assert decision.kind == KIND_MESSAGE
    assert await wait_until_dead(pid), "the finished process is still alive"
    assert active.agents_running == 0
    assert active._agents_by_key == {}
    assert active._semaphore._value == settings.max_concurrency


async def test_cancelling_reaps_the_whole_process_group(settings, fake_mode):
    """A child spawned by the CLI must not outlive the request."""
    child_pid_file = Path(settings.working_dir) / "child.pid"
    fake_mode("hang_with_child", child_pid_file=str(child_pid_file))
    active = ClaudeRunner(settings, "prompt")

    work = asyncio.ensure_future(active.run(PROMPT, "sonnet"))
    await wait_for_agents(active, 1)

    deadline = time.monotonic() + 10
    while not child_pid_file.exists() and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    child_pid = int(child_pid_file.read_text().strip())
    assert process_is_alive(child_pid)

    work.cancel()
    await asyncio.gather(work, return_exceptions=True)

    assert await wait_until_dead(child_pid), "the CLI's child outlived the request"
    assert active.agents_running == 0


async def test_a_timeout_reaps_the_whole_process_group(settings, fake_mode):
    child_pid_file = Path(settings.working_dir) / "child.pid"
    fake_mode("hang_with_child", child_pid_file=str(child_pid_file))
    impatient = replace(settings, timeout_seconds=1.0)
    active = ClaudeRunner(impatient, "prompt")

    with pytest.raises(Exception):
        await active.run(PROMPT, "sonnet")

    child_pid = int(child_pid_file.read_text().strip())
    assert await wait_until_dead(child_pid), "the CLI's child outlived the timeout"
    assert active.agents_running == 0


async def test_superseding_reaps_the_whole_process_group(settings, fake_mode):
    child_pid_file = Path(settings.working_dir) / "child.pid"
    fake_mode("hang_with_child", child_pid_file=str(child_pid_file))
    active = ClaudeRunner(replace(settings, max_concurrency=4), "prompt")

    original = identity_for(conversation("first version"))
    edited = identity_for(conversation("second version"))
    assert original is not None and edited is not None

    stale = asyncio.ensure_future(active.run(PROMPT, "sonnet", identity=original))
    await wait_for_agents(active, 1)

    deadline = time.monotonic() + 10
    while not child_pid_file.exists() and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    child_pid = int(child_pid_file.read_text().strip())

    os.environ["FAKE_CLAUDE_MODE"] = "message"
    await asyncio.wait_for(active.run(PROMPT, "sonnet", identity=edited), timeout=30)
    await asyncio.gather(stale, return_exceptions=True)

    assert await wait_until_dead(child_pid), "the superseded CLI's child survived"


async def test_a_client_disconnect_reaps_the_process(settings, fake_mode):
    fake_mode("hang")
    patient = replace(settings, timeout_seconds=300.0)
    active = ClaudeRunner(patient, load_adapter_system_prompt())
    app = create_app(settings=patient, runner=active)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        task = asyncio.ensure_future(
            client.post(
                "/v1/chat/completions",
                json=CHAT_BODY,
                headers=AUTH_HEADERS,
                timeout=300,
            )
        )
        await wait_for_agents(active, 1)
        # An anonymous first-turn request has no identity, so it is never
        # registered and can never supersede a /multitask sibling.
        assert active._agents_by_key == {}
        task.cancel()
        with pytest.raises((asyncio.CancelledError, httpx.HTTPError)):
            await task

    await asyncio.sleep(0.5)
    assert active.agents_running == 0
    assert active._semaphore._value == patient.max_concurrency


# -- the console count -----------------------------------------------------


async def test_the_agent_count_is_logged_on_start_and_finish(
    settings, fake_mode, caplog: pytest.LogCaptureFixture
):
    fake_mode("message")
    active = ClaudeRunner(replace(settings, max_concurrency=4), "prompt")

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        await active.run(PROMPT, "sonnet")

    lines = [
        record.getMessage()
        for record in caplog.records
        if "agents running" in record.getMessage()
    ]
    assert "cli-proxy agents running: 1/4" in lines
    assert "cli-proxy agents running: 0/4" in lines


async def test_the_count_reaches_the_configured_maximum(settings, fake_mode):
    fake_mode("hang")
    active = ClaudeRunner(replace(settings, max_concurrency=3), "prompt")

    tasks = [asyncio.ensure_future(active.run(PROMPT, "sonnet")) for _ in range(3)]
    await wait_for_agents(active, 3)

    assert active.agents_running == 3

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await wait_for_agents(active, 0)


async def test_health_reports_the_agent_count(settings, fake_mode):
    fake_mode("message")
    parallel = replace(settings, max_concurrency=4)
    app = create_app(
        settings=parallel,
        runner=ClaudeRunner(parallel, load_adapter_system_prompt()),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        body = (await client.get("/health")).json()

    assert body["max_concurrency"] == 4
    assert body["agents_running"] == 0
    assert body["proxy_version"] == "1.2.0"
