# v1.2.0 Release Notes

## Parallel agents for Cursor `/multitask`

This release lets Cursor's `/multitask` fan out across several Claude Code CLI
processes at once, shows how many agents are running, replaces a running agent
when you edit its prompt, guarantees every finished agent's process tree is
killed, and maps a Claude Code session / usage limit to a Cursor-visible HTTP
429 instead of a generic 502.

### What's New

#### Several Claude CLIs at once
`CLAUDE_MAX_CONCURRENCY` now defaults to **4**. Cursor `/multitask` opens
several independent `POST /v1/chat/completions` (or `/v1/responses`) requests,
and each one now gets its own `claude --print` process group instead of queuing
behind a single slot. A fifth concurrent request still queues.

#### A live agent count on the console
The proxy logs the count on startup and on every change:

```
cli-proxy agents running: 3/4
```

`GET /health` reports the same numbers as `agents_running` and
`max_concurrency`.

#### Prompt edits supersede a running agent
`claude --print` reads stdin once and cannot be handed a new prompt mid-run. If
a later request carries the same conversation prefix with a different last user
message, the proxy kills that agent's process group and starts a fresh
invocation with the updated conversation. Independent `/multitask` siblings are
left alone, and two brand-new first messages are never treated as edits of each
other.

#### Process-group reaping
Every invocation is now spawned with `start_new_session=True`, so the CLI leads
its own process group. On success, timeout, cancellation, client disconnect and
supersede, the proxy signals the whole group (SIGTERM, 5 s grace, then SIGKILL).
Children spawned by the CLI can no longer outlive their request.

#### Claude session limits are HTTP 429, not a generic 502
When Claude Code is over its five-hour session limit it still writes a
json / stream-json envelope (`api_error_status` 429, `is_error`) and exits
rc=1. That used to become `ClaudeProcessError` — HTTP 502, "The Claude Code
CLI exited unsuccessfully" — so Cursor showed a blank or failed turn with no
useful reason.

It is now `ClaudeRateLimitError`: HTTP **429**, OpenAI type / code
`rate_limit_error`, with a fixed client message Cursor can display. The CLI's
reset clock and other dynamic result text are classified, never echoed.
Non-streaming and Cursor-bound stream requests both return that 429 JSON via
the existing `ProxyError` handler when the CLI errors before any tokens —
streaming does not open SSE 200 with an in-band error Cursor ignores. Auth
failures stay **502** `ClaudeAuthError`; a 429 is not treated as auth.

### Version Bump
- **From:** 1.1.0 (text + images)
- **To:** 1.2.0 (parallel `/multitask` agents, prompt-edit supersede, console
  agent count, process-group reap, Cursor-visible 429 session-limit errors)

`src/cli_proxy/__init__.py` had drifted to `0.1.0` while `pyproject.toml` said
`1.1.0`; both now report `1.2.0`, as does `/health`.

### Breaking Changes
None to the HTTP API.

**Behaviour change:** `CLAUDE_MAX_CONCURRENCY` defaults to `4` instead of `1`,
so the proxy may now run up to four Claude invocations simultaneously. Each is a
full process launch that re-pays the whole prompt, which multiplies memory and
token cost. Set `CLAUDE_MAX_CONCURRENCY=1` in `.env` to restore the old
serialised behaviour.

### Known Limitations
- **Supersede is a restart, not a resume.** `--continue` and `--resume` remain
  unused. Applying an edited prompt means killing the old process and paying for
  the whole conversation again.
- **First-turn edits are not matched.** A conversation with no prior turns has
  no prefix to match on, so two first-turn siblings never cancel each other. A
  first-message edit relies on the editor aborting the previous HTTP stream,
  which the existing disconnect handling already reaps.
- **Streaming is still buffered.** Unchanged from 1.1.0.
- **Session-limit reset times are not forwarded.** Cursor sees a fixed
  message, not Claude's "resets 1:10pm" clock.

### Files Changed
- `pyproject.toml`, `src/cli_proxy/__init__.py` - version 1.2.0
- `src/cli_proxy/config.py` - `DEFAULT_MAX_CONCURRENCY` 1 to 4
- `src/cli_proxy/claude_runner.py` - agent registry, conversation keys,
  supersede, process-group reaping, agent-count logging, 429 session-limit
  classification
- `src/cli_proxy/errors.py` - `ClaudeRateLimitError`
- `src/cli_proxy/app.py` - pass the conversation key through, report
  `agents_running` on `/health`
- `src/cli_proxy/normalize.py` - conversation key derived from the request
- `README.md`, `LIMITATIONS.md`, `.env.example` - documentation
- `tests/test_multitask.py` and related test updates

### Upgrade Path
1. Update your installation: `pip install -e ".[dev]"` (or rebuild your venv)
2. Optionally set `CLAUDE_MAX_CONCURRENCY` in `.env`; the new default is 4
3. Restart the proxy and watch for `cli-proxy agents running: 0/4`

---

**Status:** Experimental (unchanged)
**Compatibility:** Requires Claude Code CLI 2.1.231 or later
**Python:** 3.12+

For questions or issues, refer to the [Troubleshooting](README.md#13-troubleshooting) section in the README.
