# Cursor To Claude CLI Proxy

An experimental, OpenAI-compatible HTTP proxy that lets **Cursor's native Agent
interface** use your **locally installed, already-authenticated Claude Code
CLI** as its model endpoint.

The proxy accepts an OpenAI Chat Completions or Responses request, turns it into
a single non-interactive `claude` subprocess invocation, and translates the
result back into an OpenAI-shaped response. Cursor keeps ownership of everything
else: tool execution, file edits, diffs and terminal commands all stay inside
Cursor.

> **Status: experimental and unsupported.** It has been confirmed working
> end-to-end against real Cursor traffic, but Cursor's model endpoint is a
> private integration surface and can change without notice. Read
> [Scope, compatibility and authorisation](#15-scope-compatibility-and-authorisation)
> before you use it.

---

## Contents

1. [What this is, and what it is not](#1-what-this-is-and-what-it-is-not)
2. [Architecture](#2-architecture)
3. [Limitations](#3-limitations)
4. [Requirements and installation](#4-requirements-and-installation)
5. [Authentication: two separate layers](#5-authentication-two-separate-layers)
6. [Configuration](#6-configuration)
7. [Running it locally](#7-running-it-locally)
8. [Endpoint reference](#8-endpoint-reference)
9. [Testing with curl](#9-testing-with-curl)
10. [Configuring Cursor](#10-configuring-cursor)
11. [Exposing it over HTTPS with a tunnel](#11-exposing-it-over-https-with-a-tunnel)
12. [Running at login with a user LaunchAgent](#12-running-at-login-with-a-user-launchagent)
13. [Troubleshooting](#13-troubleshooting)
14. [Uninstalling everything](#14-uninstalling-everything)
15. [Scope, compatibility and authorisation](#15-scope-compatibility-and-authorisation)

---

## 1. What this is, and what it is not

**It is a model endpoint.** To Cursor it looks like an OpenAI-compatible API
server. You point Cursor's "custom OpenAI base URL" at it, Cursor posts
`/v1/chat/completions`, and the proxy answers in OpenAI's response format.

**It is not an MCP server.** MCP servers give a model extra tools. This does the
opposite: it *is* the model, and it has no tools of its own. It cannot read your
files, write your files or run commands.

Concretely:

| Concern | Who owns it |
| --- | --- |
| Deciding what to do next | Claude, via this proxy |
| Reading files, editing files, applying diffs | Cursor |
| Running terminal commands | Cursor |
| Codebase search and indexing | Cursor |
| Approving and reverting changes | You, in Cursor's UI |

The Claude subprocess is launched with every filesystem and shell tool
explicitly denied. It is handed the conversation on stdin and can reply with one
of exactly three things: a message, a list of tool calls for Cursor to execute,
or a structured error. It never touches your project. Verified behaviour: under
this configuration Claude reports that it has no file-writing tool and creates
no files.

The proxy also never reads, copies, stores or forwards your Claude credentials.
It shells out to `claude`, which uses whatever login state it already has.

---

## 2. Architecture

### Request flow

```
  Cursor
    │  POST /v1/chat/completions        (Cursor appends /v1 itself)
    │  Authorization: Bearer <proxy token>
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ cli_proxy.app  (FastAPI / uvicorn, bound to 127.0.0.1 by default)   │
│                                                                     │
│  1. size guard    Content-Length and streamed body vs               │
│                   CLI_PROXY_MAX_REQUEST_BYTES        → 413          │
│  2. auth          constant-time bearer compare       → 401          │
│  3. normalise     cli_proxy.normalize                → 400          │
│                   detects Chat Completions vs Responses from the    │
│                   PAYLOAD SHAPE, not the URL                        │
│  4. serialise     one text document: editor instructions,           │
│                   tool catalogue, tool choice, conversation turns   │
└─────────────────────────────────────────────────────────────────────┘
    │  prompt document on STDIN (never argv, never a shell string)
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ cli_proxy.claude_runner                                             │
│                                                                     │
│   claude --print --safe-mode                                        │
│          --tools StructuredOutput                                   │
│          --disallowedTools Agent,Bash,...,Write                     │
│          --disable-slash-commands --no-session-persistence          │
│          --output-format json                                       │
│          --model <alias>                                            │
│          --system-prompt <adapter prompt>                           │
│          --json-schema <output contract>                            │
│                                                                     │
│   asyncio.create_subprocess_exec, never shell=True.                 │
│   Timeout → SIGTERM → 5 s grace → SIGKILL.                          │
└─────────────────────────────────────────────────────────────────────┘
    │  JSON envelope on stdout; `structured_output` is the decision
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ cli_proxy.schema      validate the three-way contract   → 502       │
│ cli_proxy.openai_out  build chat.completion / response,             │
│                       or the SSE event stream                       │
└─────────────────────────────────────────────────────────────────────┘
    │  200 JSON, or text/event-stream
    ▼
  Cursor executes any tool calls and posts the results back as a new request
```

### The three-way output contract

Claude is constrained by `--json-schema` to return exactly one of:

| `kind` | Meaning | Becomes |
| --- | --- | --- |
| `message` | A normal assistant reply. `content` is markdown. | `finish_reason: "stop"` with `content` |
| `tool_calls` | One or more editor tools to run, each with a verbatim `name` and an `arguments` object. | `finish_reason: "tool_calls"` with OpenAI `tool_calls` |
| `error` | Claude cannot serve the request at all. | HTTP 502 `UpstreamModelError`, or an in-band stream error |

**Unstructured prose is refused, not guessed at.** If Claude answers in free text
instead of calling the structured-output tool, the proxy raises
`ClaudeOutputError` (HTTP 502) rather than trying to scrape an answer out of
markdown. There is no fenced-block extraction and no substring scanning. The one
concession is that Claude Code duplicates its structured output into the
envelope's `result` field as a JSON string; if `result` is *exactly* a JSON
object, that is accepted.

### Why `StructuredOutput` is the one enabled tool

`--json-schema` is implemented inside Claude Code as an internal tool named
`StructuredOutput`. Passing `--tools ""` or `--disallowedTools "*"` therefore
denies structured output too — the call shows up under `permission_denials` and
the CLI falls back to prose. The shipped configuration is `--tools
"StructuredOutput"` plus an explicit `--disallowedTools` deny list covering
`Agent, Bash, BashOutput, Edit, ExitPlanMode, Glob, Grep, KillShell, MultiEdit,
NotebookEdit, NotebookRead, Read, SlashCommand, Task, TodoWrite, WebFetch,
WebSearch, Write`.

`StructuredOutput` only hands a JSON object back to its caller — it cannot read
files, write files, run commands or reach the network. This is a deliberate
documented deviation from the original design intent; see
[`LIMITATIONS.md`](LIMITATIONS.md) §1 for the full reasoning.

### Deliberate non-features

| Not used | Why |
| --- | --- |
| `--bare` | Bare mode ignores subscription OAuth, so a claude.ai subscription would stop working. |
| `--dangerously-skip-permissions` | Never used, under any configuration. |
| `--continue` / `--resume` | Every request is a fresh, stateless invocation. Unrelated Cursor conversations cannot mix. Session persistence is off. |
| `shell=True` | argv is a fixed list; no value can be reinterpreted as a shell token. |
| Prompt on argv | The conversation goes over **stdin**, keeping it out of the process table and away from argv length limits. |

### Module layout

| Module | Responsibility |
| --- | --- |
| `src/cli_proxy/app.py` | FastAPI routes, auth, size limits, the streaming generator, disconnect handling |
| `src/cli_proxy/config.py` | Environment parsing, validation, model alias map |
| `src/cli_proxy/normalize.py` | Both payload shapes → one conversation model; prompt serialisation |
| `src/cli_proxy/claude_runner.py` | argv construction, subprocess lifecycle, timeout/kill escalation, health probes |
| `src/cli_proxy/schema.py` | The output contract and its validator |
| `src/cli_proxy/openai_out.py` | Chat Completions / Responses bodies and SSE stream builders |
| `src/cli_proxy/errors.py` | Error taxonomy → HTTP status and client-safe message |
| `src/cli_proxy/security.py` | Constant-time bearer comparison |
| `src/cli_proxy/logging_setup.py` | Logging plus a hard redaction backstop |
| `src/cli_proxy/debug_dump.py` | Opt-in full-fidelity request/response dumps |
| `src/cli_proxy/prompts/adapter_system_prompt.md` | The system prompt that establishes the editor-model role |

---

## 3. Limitations

Full detail is in [`LIMITATIONS.md`](LIMITATIONS.md). The headlines:

- **Streaming is buffered.** The SSE connection opens immediately —
  time-to-first-byte measured at roughly 0.001–0.003 s — and emits a
  `: keepalive` SSE comment every 7 seconds while Claude runs. But the model text
  still arrives in **one burst at the end**. There is no token-by-token
  streaming, because `claude --print` writes its JSON envelope only on
  completion. (Before the stream was made to open early, time-to-first-byte
  equalled full model latency — 4.35 s, 15.76 s and 37.2 s were measured — which
  is exactly what made Cursor appear to hang with no response at all.)
- **The 7-second heartbeat is a guess.** It is comfortably under any normal HTTP
  idle timeout, but Cursor's actual tolerance for a silent stream is not
  documented and has not been measured. If Cursor gives up on long turns, this is
  the first constant to try lowering (`_HEARTBEAT_SECONDS` in `app.py`).
- **Text input only.** `image_url`, `input_image`, `image`, `input_audio`,
  `audio`, `input_file`, `file` and `file_url` content parts are rejected with
  HTTP 400. Pasting a screenshot into a chat turn fails the whole request.
- **Stateless per request, so cost grows steeply.** Cursor resends the entire
  conversation every turn and every invocation re-pays for the whole prompt. Two
  observed real Cursor turns reported **214,250** and **430,367** prompt tokens.
  Observed per-turn latency was around **11.4 s**. A long agent session gets
  slower and more expensive with every turn, and nothing in the proxy can
  amortise that.
- **Cursor's model picker does not choose the model.** Cursor sends its own model
  name (observed: `gpt-5.6-sol`), which is not one of the ids this proxy
  advertises. Unknown ids fall back to `CLAUDE_DEFAULT_MODEL`. See
  [§10](#10-configuring-cursor).
- **One request at a time by default.** `CLAUDE_MAX_CONCURRENCY` defaults to 1;
  a second request queues behind the first.
- **`temperature` and `max_tokens` are hints, not limits.** They are described to
  the model in the prompt. The CLI exposes no flags for them.
- **Debug dumps are cleartext.** See [§6](#6-configuration) and
  [§7](#7-running-it-locally).

---

## 4. Requirements and installation

| Requirement | Notes |
| --- | --- |
| macOS | Developed and verified on macOS. Nothing is macOS-specific except the LaunchAgent section. |
| **Python 3.12** | `pyproject.toml` requires `>=3.12`. If your default `python3` is newer (3.14, for example) the venv must still be built with 3.12. |
| Claude Code CLI, authenticated | Verified against **2.1.231** at `/opt/homebrew/bin/claude`, signed in with a **claude.ai subscription** (OAuth), not an API key. |
| `openssl` | For generating the proxy's bearer token. Ships with macOS. |
| `curl`, optionally `jq` | For the smoke test and the examples below. |

### Install

```bash
cd /path/to/cli-proxy

# Build the venv with 3.12 explicitly.
/opt/homebrew/bin/python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e ".[dev]"
```

`./scripts/run.sh` will do this for you on first run. It uses `python3.12` and
honours a `PYTHON_BIN` override:

```bash
PYTHON_BIN=/opt/homebrew/bin/python3.12 ./scripts/run.sh
```

### Confirm the Claude CLI

```bash
which claude          # e.g. /opt/homebrew/bin/claude
claude --version      # e.g. 2.1.231 (Claude Code)
claude auth status    # must report that you are signed in
```

### Create your configuration

```bash
cp .env.example .env
chmod 600 .env
```

Then edit `.env` — at minimum set `CLI_PROXY_TOKEN` and `CLAUDE_EXECUTABLE`. See
[§5](#5-authentication-two-separate-layers) and [§6](#6-configuration).

> `.env` is gitignored, along with `debug-dumps/`, `logs/`, `cert.pem` and the
> obvious Cloudflare credential filenames.

### Run the tests

```bash
cd /path/to/cli-proxy
./.venv/bin/python -m pytest -q
```

**208 tests, all passing** as of this writing. The suite injects a fake runner
and **never calls Claude**, so it is fast, free and offline. For real end-to-end
calls use `./scripts/smoke-test.sh` — see [§9](#9-testing-with-curl).

---

## 5. Authentication: two separate layers

These are completely independent. Do not conflate them.

### Layer A — Claude's own authentication (the proxy never touches it)

The `claude` CLI is already signed in to your Anthropic account. The proxy
**never reads, copies, stores, logs or forwards** that credential. It has no
Anthropic API key, makes no Anthropic HTTP calls, and does not read the keychain,
`~/.claude`, or any token file. It simply spawns `claude`, which uses whatever
login state it already has.

Verify it independently of the proxy:

```bash
claude auth status
```

The proxy's `/health` endpoint runs the same probe and reduces the answer to a
single boolean (`claude_authenticated`). The account details in that output are
discarded inside the function that parses them and never leave the process.

Consequence worth knowing: because the CLI runs as *your* user with *your* login,
the proxy must run as your user too. That is why [§12](#12-running-at-login-with-a-user-launchagent)
uses a user LaunchAgent and not a root LaunchDaemon.

### Layer B — the proxy's own bearer token (you create this)

Every `/v1` endpoint requires `Authorization: Bearer <token>`. Generate one:

```bash
openssl rand -hex 32
```

Put it in `.env` as `CLI_PROXY_TOKEN`. Requirements enforced at startup:

- must be set, or the process exits with a configuration error;
- must be at least 16 characters (`openssl rand -hex 32` gives 64);
- must not still be the `.env.example` placeholder.

Handling guarantees:

- Comparison is **constant time** (`hmac.compare_digest`), including when no
  token was presented, so there is no secret-dependent branch.
- The token is **never logged**. The logging layer additionally has a redaction
  filter that blanks any record containing an authorization-like string or a long
  hex run.
- The token is **stripped from the subprocess environment**, along with the other
  `CLI_PROXY_*` variables, so the Claude CLI and anything it spawns cannot see it.
- The token is **scrubbed from debug dumps**: the `authorization` header is always
  written as `[redacted]`, and a final pass replaces any occurrence of the token
  itself anywhere in the serialised document. Both verified.

Rotate it by generating a new one, updating `.env` and restarting the proxy.

---

## 6. Configuration

All configuration is environment variables, read once at startup by
`src/cli_proxy/config.py`. `./scripts/run.sh` sources `.env` if present.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLI_PROXY_TOKEN` | *(required)* | Bearer token for every `/v1` endpoint. Minimum 16 characters. The `.env.example` placeholder is rejected. |
| `CLI_PROXY_HOST` | `127.0.0.1` | Listen address. Binding anything other than loopback logs a warning at startup; do not do it without an authenticating layer in front. |
| `CLI_PROXY_PORT` | `8787` | Listen port. Must be 1–65535. |
| `CLAUDE_EXECUTABLE` | `claude` resolved on `PATH` | Path to the Claude Code CLI. A value containing a `/` is used as-is (expanding `~`); a bare name is resolved with `which`. **Its existence is not checked at startup** — see [§13](#13-troubleshooting). |
| `CLAUDE_DEFAULT_MODEL` | `sonnet` | Claude model alias used for the `claude-cli-proxy` id and for any unrecognised model id. `sonnet`, `opus`, `haiku`, or a full model name. **In practice this is the only real control over which model serves Cursor.** |
| `CLAUDE_TIMEOUT_SECONDS` | `600` | Wall-clock limit for one CLI invocation. On expiry: SIGTERM, 5 s grace, then SIGKILL, and HTTP 504. Minimum 1. |
| `CLAUDE_MAX_CONCURRENCY` | `1` | Concurrent `claude` subprocesses. Each is a full process launch, so raising this multiplies memory and token cost. Minimum 1. |
| `CLAUDE_WORKING_DIR` | *(blank)* | Working directory for the subprocess. Blank creates and uses a private empty scratch directory (`$TMPDIR/cli-proxy-scratch`, mode 0700). If set, it must already exist and be a directory or startup fails. Pointing it at a real project is not recommended. |
| `CLI_PROXY_MAX_REQUEST_BYTES` | `4194304` (4 MiB) | Maximum accepted request body. `Content-Length` is checked first, then the streamed size. Over the limit → HTTP 413. Minimum 1024. |
| `CLI_PROXY_MAX_RESPONSE_BYTES` | `4194304` (4 MiB) | Maximum accepted CLI stdout. Over the limit → HTTP 502. Minimum 1024. |
| `CLI_PROXY_LOG_LEVEL` | `INFO` | One of `CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`. No level ever logs authorization headers, prompts, source code, tool arguments or Claude output. |
| `CLI_PROXY_DEBUG_DUMP` | `0` (but see below) | **Dangerous.** Writes one JSON file per request containing the full inbound body, the full prompt, the exact argv, the CLI's raw stdout and stderr, and the complete outgoing response or event stream — in cleartext. |
| `CLI_PROXY_DEBUG_DUMP_DIR` | `./debug-dumps` | Where dumps go. Created mode 0700 if missing; files are written mode 0600. Gitignored. |
| `CLI_PROXY_DEBUG_DUMP_CONSOLE` | `0` (but see below) | Also echo each dump to stdout. Only meaningful when dumping is on. |

Booleans accept `1/true/yes/on` and `0/false/no/off`; anything else is a
configuration error.

### About the debug dumps

`./scripts/run.sh` sets `CLI_PROXY_DEBUG_DUMP=1`, `CLI_PROXY_DEBUG_DUMP_CONSOLE=1`
and `CLI_PROXY_DEBUG_DUMP_DIR=<repo>/debug-dumps` **as defaults**, because the
script exists to work out what an editor is actually sending. It prints a large
warning banner when it does.

An explicit `CLI_PROXY_DEBUG_DUMP=0` in `.env` or in the environment overrides the
script default. (`.env.example` ships with `CLI_PROXY_DEBUG_DUMP=0`, so if you
copied it verbatim, dumping is already off.)

**What ends up on disk when it is on:** the entire conversation Cursor sent, the
entire prompt handed to Claude, and **any source code the editor included** — all
in plain text. Files are mode 0600 inside a 0700 directory. The `authorization`
header is always `[redacted]` and the proxy's bearer token is scrubbed from the
document. *Nothing else is redacted.* Turn it off for normal use and delete the
directory when you are done:

```bash
rm -rf debug-dumps
```

---

## 7. Running it locally

```bash
cd /path/to/cli-proxy
./scripts/run.sh
```

The script:

1. sources `.env` if present;
2. creates `.venv` with `python3.12` (or `$PYTHON_BIN`) and installs the package
   on first run;
3. refuses to start if `CLI_PROXY_TOKEN` is unset, printing the `openssl rand
   -hex 32` command;
4. resolves `claude` from `PATH` if `CLAUDE_EXECUTABLE` is unset, and refuses to
   start if it cannot;
5. turns debug dumping on unless you have explicitly set it to `0`;
6. `exec`s `python -m cli_proxy`.

Expected output:

```
Loading configuration from .env
Claude executable: /opt/homebrew/bin/claude
Listening on:      http://127.0.0.1:8787
Bearer token:      configured (64 characters, not shown)

################################################################################
#                                                                              #
#  WARNING: DEBUG DUMPING IS ON. THIS DISABLES THE NORMAL LOGGING HYGIENE.      #
...
################################################################################
```

The token itself is never printed — only its length.

You can also run it without the script:

```bash
./.venv/bin/python -m cli_proxy
```

### Stopping it

```bash
kill $(lsof -ti tcp:8787)
```

Or `Ctrl-C` in the terminal running it.

---

## 8. Endpoint reference

| Route | Methods | Auth | Purpose |
| --- | --- | --- | --- |
| `/health` | `GET`, `HEAD` | **No** | Liveness plus live CLI probes. |
| `/v1/models`, `/v1/models/` | `GET`, `HEAD` | Yes | OpenAI model list. |
| `/v1/models/{model_id}` | `GET`, `HEAD` | Yes | One model. Unknown id → 400. |
| `/v1/chat/completions`, `/v1/chat/completions/` | `POST` | Yes | Chat Completions. |
| `/v1/responses`, `/v1/responses/` | `POST` | Yes | Responses API. |
| `/{any path}` | `OPTIONS` | No | 204 preflight with `Allow` and CORS method/header hints. |

Interactive docs (`/docs`, `/redoc`) and `/openapi.json` are disabled.

Note that the **response shape follows the payload shape, not the URL**. A
Responses-style body (`input` plus flat tool definitions) posted to
`/v1/chat/completions` gets a Responses-style reply, because some Cursor builds
do exactly that. `messages` wins when a body somehow contains both.

### `GET /health`

Unauthenticated. Returns **200** when the CLI is available and authenticated,
**503** otherwise (with the same body shape, `"status": "degraded"`). Real
output:

```json
{
  "status": "ok",
  "proxy_version": "0.1.0",
  "claude_executable": "/opt/homebrew/bin/claude",
  "claude_executable_available": true,
  "claude_version": "2.1.231 (Claude Code)",
  "claude_authenticated": true,
  "detail": "ok",
  "default_model_alias": "sonnet",
  "models": [
    "claude-cli-proxy",
    "claude-cli-sonnet",
    "claude-cli-opus"
  ],
  "max_concurrency": 1,
  "timeout_seconds": 600.0,
  "streaming": "buffered",
  "stream_opens_immediately": true,
  "debug_dump": false
}
```

`debug_dump` is a boolean only — the dump directory path is deliberately kept out
of an unauthenticated endpoint. Each call runs `claude --version` and `claude
auth status` as subprocesses with a 20 s probe timeout, so it is not free; do not
poll it aggressively.

### `GET /v1/models`

```json
{
  "object": "list",
  "data": [
    {"id": "claude-cli-proxy",  "object": "model", "created": 0, "owned_by": "cli-proxy",
     "root": "claude-cli-proxy",  "parent": null, "permission": []},
    {"id": "claude-cli-sonnet", "object": "model", "created": 0, "owned_by": "cli-proxy",
     "root": "claude-cli-sonnet", "parent": null, "permission": []},
    {"id": "claude-cli-opus",   "object": "model", "created": 0, "owned_by": "cli-proxy",
     "root": "claude-cli-opus",   "parent": null, "permission": []}
  ]
}
```

| Advertised id | Claude alias used |
| --- | --- |
| `claude-cli-proxy` | whatever `CLAUDE_DEFAULT_MODEL` says |
| `claude-cli-sonnet` | `sonnet` |
| `claude-cli-opus` | `opus` |
| *anything else* | whatever `CLAUDE_DEFAULT_MODEL` says |

### `POST /v1/chat/completions`

Standard OpenAI request. Supported fields: `model`, `messages`, `tools`,
`tool_choice`, `temperature`, `max_tokens` / `max_completion_tokens`, `stream`.
Roles: `system`, `developer`, `user`, `assistant` (with `tool_calls`), `tool`,
`function`.

Non-streaming success is a `chat.completion` object with a single choice,
`finish_reason` of `stop` or `tool_calls`, and a `usage` block populated from the
CLI's own token accounting (`input_tokens` sums the plain, cache-creation and
cache-read input counters).

Streaming success is `text/event-stream`: an opening role delta, then
`: keepalive` comments every 7 seconds, then the content chunk(s), a finish
chunk, a usage-only chunk, and `data: [DONE]`.

### `POST /v1/responses`

Accepts `input` as a string, object or array; `instructions` (mapped to a system
turn); `tools` in either the nested or flat shape; `tool_choice`; `temperature`;
`max_output_tokens` / `max_tokens`; `stream`. Item types `function_call` and
`function_call_output` / `function_call_result` carry the tool round trip.
`reasoning`, `web_search_call`, `file_search_call`, `computer_call`,
`code_interpreter_call` and `item_reference` items are ignored as instruction-free.

Streaming emits the full Responses event sequence: `response.created`,
`response.in_progress`, keepalives, then `response.output_item.added` →
`response.content_part.added` → `response.output_text.delta` →
`response.output_text.done` → `response.content_part.done` →
`response.output_item.done` per item (or the
`response.function_call_arguments.*` equivalents for tool calls), then
`response.completed` and `data: [DONE]`.

### Errors while streaming

Once the 200 has been sent, an HTTP error status is no longer available, so a
later failure is reported **in band**: for Chat Completions a `finish_reason:
"stop"` chunk followed by `data: {"error": {...}}` and `data: [DONE]`; for
Responses a `response.failed` event followed by `data: [DONE]`. Failures detected
*before* the stream opens still return an ordinary HTTP 4xx/5xx JSON error.

If the client disconnects mid-request, the Claude subprocess is cancelled and
reaped (SIGTERM, 5 s, SIGKILL).

---

## 9. Testing with curl

Export your token first so nothing below contains a literal secret:

```bash
export CLI_PROXY_TOKEN='<your-token-from-.env>'
export BASE=http://127.0.0.1:8787
```

### Health (no auth)

```bash
curl -s "$BASE/health" | jq .
```

### Model list

```bash
curl -s -H "Authorization: Bearer $CLI_PROXY_TOKEN" "$BASE/v1/models" | jq .
```

### Non-streaming chat completion

```bash
curl -s -H "Authorization: Bearer $CLI_PROXY_TOKEN" \
     -H 'Content-Type: application/json' \
     "$BASE/v1/chat/completions" \
     -d '{
       "model": "claude-cli-sonnet",
       "messages": [
         {"role": "system", "content": "Answer in exactly one short sentence."},
         {"role": "user",   "content": "What is 17 plus 25?"}
       ]
     }' | jq .
```

Response (abridged):

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1756282000,
  "model": "claude-cli-sonnet",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "17 plus 25 is 42.", "refusal": null},
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 1234, "completion_tokens": 12, "total_tokens": 1246}
}
```

### Streaming chat completion

`-N` disables curl's own buffering, which you need to see the keepalives arrive.

```bash
curl -sN -H "Authorization: Bearer $CLI_PROXY_TOKEN" \
     -H 'Content-Type: application/json' \
     "$BASE/v1/chat/completions" \
     -d '{
       "model": "claude-cli-sonnet",
       "stream": true,
       "messages": [{"role": "user", "content": "Reply with exactly: STREAM OK"}]
     }'
```

Output — note the opening chunk arrives in milliseconds, the keepalives fill the
wait, and the text arrives all at once:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...,"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}],"usage":null}

: keepalive

: keepalive

data: {"id":"chatcmpl-...","choices":[{"index":0,"delta":{"content":"STREAM OK"},"finish_reason":null}],"usage":null}

data: {"id":"chatcmpl-...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}

data: {"id":"chatcmpl-...","choices":[],"usage":{"prompt_tokens":1210,"completion_tokens":4,"total_tokens":1214}}

data: [DONE]
```

To confirm the stream really does open immediately:

```bash
curl -sN -o /dev/null -w 'time_to_first_byte=%{time_starttransfer}s total=%{time_total}s\n' \
     -H "Authorization: Bearer $CLI_PROXY_TOKEN" \
     -H 'Content-Type: application/json' \
     "$BASE/v1/chat/completions" \
     -d '{"model":"claude-cli-sonnet","stream":true,
          "messages":[{"role":"user","content":"Say hello."}]}'
```

Expect a first byte in single-digit milliseconds and a total in seconds.

### Responses API

```bash
curl -s -H "Authorization: Bearer $CLI_PROXY_TOKEN" \
     -H 'Content-Type: application/json' \
     "$BASE/v1/responses" \
     -d '{
       "model": "claude-cli-sonnet",
       "instructions": "Answer in exactly one short sentence.",
       "input": [
         {"type": "message", "role": "user",
          "content": [{"type": "input_text", "text": "Name the capital of France."}]}
       ]
     }' | jq '.status, .output_text'
```

### A tool call, and the result round trip

Offer a tool and ask for something that requires it:

```bash
curl -s -H "Authorization: Bearer $CLI_PROXY_TOKEN" \
     -H 'Content-Type: application/json' \
     "$BASE/v1/chat/completions" \
     -d '{
       "model": "claude-cli-sonnet",
       "messages": [
         {"role": "system", "content": "You are the model behind a code editor."},
         {"role": "user",   "content": "Show me the contents of src/cli_proxy/config.py"}
       ],
       "tools": [
         {"type": "function",
          "function": {
            "name": "read_file",
            "description": "Read a file from the workspace.",
            "parameters": {"type": "object",
                           "properties": {"target_file": {"type": "string"}},
                           "required": ["target_file"]}}}
       ],
       "tool_choice": "auto"
     }' | jq '.choices[0].finish_reason, .choices[0].message.tool_calls'
```

Expect `"tool_calls"` and an entry naming `read_file`. Feed the result back as a
`tool` message — this is exactly what Cursor does, and it is the round trip that
was verified working against live Cursor traffic:

```bash
curl -s -H "Authorization: Bearer $CLI_PROXY_TOKEN" \
     -H 'Content-Type: application/json' \
     "$BASE/v1/chat/completions" \
     -d '{
       "model": "claude-cli-sonnet",
       "messages": [
         {"role": "user", "content": "What does greet.py print?"},
         {"role": "assistant", "content": null,
          "tool_calls": [{"id": "call_1", "type": "function",
                          "function": {"name": "read_file",
                                       "arguments": "{\"target_file\":\"greet.py\"}"}}]},
         {"role": "tool", "tool_call_id": "call_1", "name": "read_file",
          "content": "print(\"hello world\")"}
       ],
       "tools": [
         {"type": "function",
          "function": {"name": "read_file",
                       "parameters": {"type": "object",
                                      "properties": {"target_file": {"type": "string"}}}}}
       ]
     }' | jq -r '.choices[0].message.content'
```

### Image input is refused

```bash
curl -s -o /dev/null -w '%{http_code}\n' \
     -H "Authorization: Bearer $CLI_PROXY_TOKEN" \
     -H 'Content-Type: application/json' \
     "$BASE/v1/chat/completions" \
     -d '{"model":"claude-cli-sonnet","messages":[{"role":"user","content":[
          {"type":"text","text":"what is this"},
          {"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}'
# 400
```

### The scripted smoke test

```bash
# In one shell:
./scripts/run.sh

# In another:
CLI_PROXY_TOKEN='<your-token>' ./scripts/smoke-test.sh
# or against a tunnel:
CLI_PROXY_TOKEN='<your-token>' ./scripts/smoke-test.sh https://cli-proxy.example.com
```

It runs ten checks: health, unauthenticated refusal, wrong-token refusal, model
list, non-streaming completion, streaming completion, Responses API, a tool-call
decision, a tool-result round trip, and image rejection. It reads `.env` if
`CLI_PROXY_TOKEN` is not already exported, and never echoes the token.

> **`smoke-test.sh` makes REAL Claude calls** and consumes real quota. It prints
> response bodies so you can inspect the translation, so only run it with
> harmless prompts. `pytest` is the opposite: it injects a fake runner and never
> invokes Claude at all.

---

## 10. Configuring Cursor

1. Start the proxy ([§7](#7-running-it-locally)) and expose it over HTTPS
   ([§11](#11-exposing-it-over-https-with-a-tunnel)). Cursor will not accept a
   plain-HTTP endpoint.
2. In Cursor: **Settings → Models → OpenAI API Key → Override OpenAI Base URL**.
3. **Base URL — omit the `/v1`:**

   ```
   https://cli-proxy.example.com
   ```

   Cursor appends `/v1/chat/completions` itself. Entering
   `https://cli-proxy.example.com/v1` produces requests to `/v1/v1/chat/completions`,
   which 404. This is the single most common setup mistake.
4. **API key**: paste the value of `CLI_PROXY_TOKEN`. It is sent as
   `Authorization: Bearer <token>`.
5. Verify and enable the override, then use the Agent panel as normal.

### The model picker does not select the model

Cursor sends **its own** model name in the request body — the observed value was
`"gpt-5.6-sol"`, not one of this proxy's advertised ids. Unknown ids fall back to
`CLAUDE_DEFAULT_MODEL`.

**So: `CLAUDE_DEFAULT_MODEL` in `.env` is the only real control over which Claude
model serves your Cursor sessions. Changing the model in Cursor's own picker has
no effect on this proxy.** Change the model by editing `.env` and restarting.

The three advertised ids (`claude-cli-proxy`, `claude-cli-sonnet`,
`claude-cli-opus`) are mainly useful for direct curl testing, where you control
the `model` field.

### What working traffic looks like

Two real Cursor requests were captured through a Cloudflare quick tunnel:
`POST /v1/chat/completions`, `user-agent: Cursor/1.0`, both HTTP 200. The first
returned a `Shell` tool call, which Cursor executed; the tool result came back in
the next request (the message count went from 195 to 197) and the second returned
a normal assistant message. The full tool-call round trip works.

Those same two turns reported 214,250 and 430,367 prompt tokens, at roughly
11.4 s per turn. That is the cost profile to expect — see
[§3](#3-limitations).

---

## 11. Exposing it over HTTPS with a tunnel

Cursor requires an HTTPS base URL, and the proxy binds `127.0.0.1`. Something has
to bridge the two. **Nothing in this repository creates, runs or installs a
tunnel.** `deploy/cloudflared-config.example.yml` is an annotated example you
copy and adapt yourself, outside the repo.

Keep `CLI_PROXY_HOST=127.0.0.1`. The tunnel connects to loopback; there is no
reason to bind `0.0.0.0`.

### Quick tunnel versus named tunnel

| | Quick tunnel | Named tunnel |
| --- | --- | --- |
| Command | `cloudflared tunnel --url http://127.0.0.1:8787` | `cloudflared tunnel --config ~/.cloudflared/config.yml run cli-proxy` |
| Hostname | Random `*.trycloudflare.com` | Stable, on a domain you control |
| Stability | **New hostname on every restart** — two different names were observed in a single session | Fixed |
| Access control | **None** | Can sit behind Cloudflare Access |
| Setup | None | `cloudflared tunnel login`, `create`, `route dns` |
| Good for | A five-minute experiment | Anything you use more than once |

Hostname churn is the real problem with quick tunnels: every restart means going
back into Cursor's settings and pasting a new base URL, and any Cursor session
mid-flight breaks. Use one to prove the integration works, then move to a named
tunnel.

SSE keepalives do survive a quick tunnel — verified with live Cursor traffic.

### Named tunnel setup

```bash
brew install cloudflared
cloudflared tunnel login                       # opens a browser
cloudflared tunnel create cli-proxy            # prints the tunnel UUID
cloudflared tunnel route dns cli-proxy cli-proxy.example.com

cp deploy/cloudflared-config.example.yml ~/.cloudflared/config.yml
# edit: tunnel UUID, credentials-file path, hostname
chmod 600 ~/.cloudflared/config.yml

cloudflared tunnel --config ~/.cloudflared/config.yml run cli-proxy
```

`cloudflared tunnel create` writes `~/.cloudflared/<UUID>.json`. **That file is a
credential.** Keep it 0600, never commit it, never paste it anywhere.

### Timeouts

Claude invocations can run for minutes and the proxy's own default timeout is
600 s, so the tunnel's timeouts must be generous. The example config uses
`connectTimeout: 30s`, `tcpKeepAlive: 30s` and `keepAliveTimeout: 900s`. A tunnel
that gives up at 60 s will look exactly like the proxy hanging.

### Add Cloudflare Access — strongly recommended

Without it, the bearer token is the *only* thing between the public internet and
your authenticated Claude subscription. In the Cloudflare Zero Trust dashboard:
**Access → Applications → Add a self-hosted application**, domain
`cli-proxy.example.com`, policy allowing only your identity.

Cursor sends only a bearer token and cannot complete an interactive Access login,
so use a **service token** policy (configuring the required
`CF-Access-Client-Id` / `CF-Access-Client-Secret` headers) or restrict by IP
range. If neither is workable, accept that the bearer token is your only control
and rotate it regularly.

---

## 12. Running at login with a user LaunchAgent

`deploy/com.local.cli-proxy.plist.template` is a template. **Nothing in this
repository installs it.** It only takes effect when you deliberately copy it into
`~/Library/LaunchAgents/` and run `launchctl` yourself.

Three rules matter:

1. **Absolute paths everywhere.** launchd does not expand `~`, does not run a
   login shell, and has a minimal `PATH`. Every placeholder — `__PYTHON__`,
   `__REPO__`, `__CLAUDE__`, `__TOKEN_FILE__`, `__HOME__` — must be replaced with
   a full path.
2. **A *user* LaunchAgent, never a root LaunchDaemon.** A daemon runs as root
   outside your login session and would not have your Claude authentication. The
   agent must run as you.
3. **The token goes in a 0600 file, not in the plist.** Files in
   `~/Library/LaunchAgents/` are world-readable by default and are easy to leak in
   a backup or a screen share. The template's `ProgramArguments` reads the token
   from a file at launch instead.

```bash
mkdir -p ~/.config/cli-proxy
umask 077
openssl rand -hex 32 > ~/.config/cli-proxy/token
chmod 600 ~/.config/cli-proxy/token
```

Then generate and load the agent:

```bash
mkdir -p ~/Library/LaunchAgents "$PWD/logs"

sed -e "s|__PYTHON__|$PWD/.venv/bin/python|g" \
    -e "s|__REPO__|$PWD|g" \
    -e "s|__CLAUDE__|$(which claude)|g" \
    -e "s|__TOKEN_FILE__|$HOME/.config/cli-proxy/token|g" \
    -e "s|__HOME__|$HOME|g" \
    deploy/com.local.cli-proxy.plist.template \
    > ~/Library/LaunchAgents/com.local.cli-proxy.plist

launchctl load -w ~/Library/LaunchAgents/com.local.cli-proxy.plist
launchctl list | grep com.local.cli-proxy
curl -s http://127.0.0.1:8787/health | jq .status
```

The agent sets `RunAtLoad`, restarts on unclean exit with a 30 s throttle, and
writes `logs/cli-proxy.out.log` and `logs/cli-proxy.err.log`. Those logs contain
no prompts, source code, tool arguments or Claude output, but they are still
cleartext — keep them out of any synced folder. `logs/` is gitignored.

It also means the proxy is listening whenever you are logged in. It binds
`127.0.0.1`, so it is reachable only from this machine unless you separately run
a tunnel.

---

## 13. Troubleshooting

Every client-facing error is an OpenAI-shaped body:

```json
{"error": {"message": "...", "type": "api_error", "code": "api_error", "param": null}}
```

### Symptom → cause → fix

| Symptom | Cause | Error class / HTTP | Fix |
| --- | --- | --- | --- |
| Every request returns 503 "The Claude Code CLI is not available", but startup succeeded | `.env` still has the `.env.example` placeholder `CLAUDE_EXECUTABLE=/absolute/path/to/claude`. Startup does **not** check that the path exists, so validation passes and every spawn then fails. | `ClaudeUnavailableError` / **503** | `which claude`, put that exact path in `.env`, restart. `curl -s localhost:8787/health \| jq .claude_executable` shows what the proxy is actually trying to run. |
| 503 on a machine where `claude` works fine in your shell | The proxy is running as a different user, or under launchd with a `PATH` that does not include Homebrew. | `ClaudeUnavailableError` / **503** | Set `CLAUDE_EXECUTABLE` to an absolute path. Confirm the agent is a *user* LaunchAgent. |
| 401 "Missing or invalid bearer token" | No `Authorization` header, wrong scheme, or a token mismatch. Cursor's "API key" field must hold `CLI_PROXY_TOKEN` exactly. | `AuthenticationError` / **401** | Compare against `.env`. Watch for trailing whitespace and truncated pastes. `/health` needs no auth, so if `/health` works and `/v1/models` 401s, it is the token. |
| 401 on every request even with the right token | The proxy started with no configured token. | `AuthenticationError` / **401** | Set `CLI_PROXY_TOKEN` and restart. `run.sh` refuses to start without one. |
| 404 on every Cursor request, `/health` fine | Base URL includes `/v1`; Cursor appends its own, giving `/v1/v1/chat/completions`. | FastAPI 404 | Use `https://host` with **no** `/v1` suffix. |
| Cursor shows nothing at all, no error | Requests may not be arriving. Tunnel down, hostname changed (quick tunnels change on every restart), or the base URL is wrong. | — | Turn on `CLI_PROXY_DEBUG_DUMP=1` and look in `debug-dumps/`. **A file per request means Cursor is reaching you** — read `inbound.headers` for `user-agent: Cursor/1.0` and `response.status`. **No files at all means the traffic never arrived**, so the problem is the URL or the tunnel, not the proxy. |
| Cursor spins for a long time then gives up | The turn genuinely takes that long (observed ~11.4 s, and it grows with conversation length), or a tunnel/client idle timeout fired despite the keepalives. | possibly `ClaudeTimeoutError` / **504** | Check the proxy log for `claude subprocess ... exited rc=0`. Raise the tunnel's `keepAliveTimeout`. If Cursor gives up while keepalives are still flowing, lower `_HEARTBEAT_SECONDS` in `app.py` (7 s is an untested guess at Cursor's tolerance). |
| 502 "did not return usable structured output. The proxy refuses to guess at unstructured text" | Claude answered in prose. Usually the `StructuredOutput` tool was denied and appears under `permission_denials`; also covers empty stdout, non-JSON stdout, or output that fails the contract. | `ClaudeOutputError` / **502** | Check `claude --version` against 2.1.231. A CLI upgrade may have changed `--tools` semantics or the internal tool name; a dump's `claude_result.stdout` shows the envelope. See [`LIMITATIONS.md`](LIMITATIONS.md) §1. |
| 502 "The model reported that it could not serve the request" | Claude used the contract's `error` shape — for example the conversation is contradictory, or it needs a tool Cursor did not offer. | `UpstreamModelError` / **502** | The message carries Claude's reason. Usually a prompt problem, not a proxy problem. |
| 502 "The Claude Code CLI exited unsuccessfully" | Non-zero exit, or an envelope with `is_error`. stderr is classified but never echoed, so it will not appear in the response. | `ClaudeProcessError` / **502** | Run the same prompt through `claude --print` by hand. Raise `CLI_PROXY_LOG_LEVEL` to `DEBUG`. |
| 502 "The Claude Code CLI is not authenticated" | The CLI reported an auth failure — logged out, or an expired OAuth token. Note this is **502**, not 503. | `ClaudeAuthError` / **502** | `claude auth status`, then sign in again. Confirm with `curl -s localhost:8787/health \| jq .claude_authenticated`. |
| 504 "did not finish before the configured timeout" | The invocation exceeded `CLAUDE_TIMEOUT_SECONDS` (default 600). The subprocess is SIGTERMed, given 5 s, then SIGKILLed. | `ClaudeTimeoutError` / **504** | Raise `CLAUDE_TIMEOUT_SECONDS`, and raise the tunnel's timeouts to match. Recurrent timeouts usually mean the conversation has grown very large. |
| 502 "produced more output than the configured limit" | CLI stdout exceeded `CLI_PROXY_MAX_RESPONSE_BYTES` (default 4 MiB). | `ResponseTooLargeError` / **502** | Raise the limit, or ask for a shorter answer. |
| 413 "Request body exceeds the configured maximum size" | Request over `CLI_PROXY_MAX_REQUEST_BYTES` (default 4 MiB). Long agent sessions do reach this. | `RequestTooLargeError` / **413** | Raise the limit, or start a fresh conversation in Cursor. |
| 400 "accepts text only; image, audio and file inputs are not supported" | An `image_url` / `input_image` / `input_audio` / `input_file` / `file_url` content part. | `UnsupportedContentError` / **400** | Send text. There is no workaround in this version. |
| 400 "The request payload could not be interpreted." | Malformed JSON, an empty body, no `messages` and no `input`, a bad role, a tool without a `name`, a non-numeric `temperature`. | `InvalidRequestError` / **400** | The message names the specific problem. A debug dump's `inbound.body` shows exactly what was sent. |
| Startup exits immediately with "cli-proxy configuration error: ..." | Bad environment: missing/short/placeholder token, invalid log level, out-of-range port, a non-numeric value, or `CLAUDE_WORKING_DIR` pointing at something that is not a directory. | `ConfigError`, exit code 2 | The message names the variable. |
| `/health` returns 503 with `"status": "degraded"` | `claude_executable_available` or `claude_authenticated` is false. `detail` says which. | not a `ProxyError` | Fix per the rows above. This 503 is a health verdict, not a request failure. |
| Proxy log lines read `[redacted by cli-proxy]` | The redaction filter matched something authorization-like or a long hex run in the message. | — | Working as designed. Use a debug dump if you need the detail. |

### Turning on debug dumps for an investigation

```bash
CLI_PROXY_DEBUG_DUMP=1 ./scripts/run.sh
ls -l debug-dumps/          # one 0600 JSON file per request
jq '.inbound.headers, .normalized, .response.status' debug-dumps/000001-*.json
```

Useful keys: `inbound.headers` and `inbound.body` (what Cursor sent),
`normalized` (turn count, tool names, detected flavour, resolved model alias),
`claude_argv`, `claude_stdin_prompt`, `claude_result.stdout` / `.stderr`,
`decision`, `response`, `sse_chunks`, `notes`, `errors`.

Delete them when you are done: `rm -rf debug-dumps`.

---

## 14. Uninstalling everything

Work top down. Every step is optional depending on what you actually set up.

```bash
# 1. Stop the proxy.
kill $(lsof -ti tcp:8787)

# 2. Unload and delete the LaunchAgent, if you installed one.
launchctl unload -w ~/Library/LaunchAgents/com.local.cli-proxy.plist
rm -f ~/Library/LaunchAgents/com.local.cli-proxy.plist

# 3. Stop and delete the Cloudflare tunnel, if you created one.
#    (Ctrl-C the running cloudflared first, or unload its own LaunchAgent.)
cloudflared tunnel delete cli-proxy
#    Remove the DNS record for the hostname in the Cloudflare dashboard.
rm -f ~/.cloudflared/<TUNNEL-UUID>.json   # the tunnel credential
rm -f ~/.cloudflared/cert.pem             # revokes local tunnel-creation ability
rm -f ~/.cloudflared/config.yml

# 4. Remove local secrets and artefacts.
rm -rf debug-dumps            # cleartext conversations and source code
rm -f  .env                   # your bearer token
rm -f  ~/.config/cli-proxy/token
rm -rf logs
rm -rf .venv .pytest_cache
rm -rf "${TMPDIR:-/tmp}/cli-proxy-scratch"

# 5. Remove the checkout itself.
cd .. && rm -rf cli-proxy
```

Also clear the base URL and API key override in Cursor's model settings.

**There is nothing to undo on the Claude side.** This project never modified
Claude Code's configuration, never wrote to `~/.claude`, and never read, copied or
stored your credentials. Your `claude` CLI login is exactly as it was. If you want
to sign out of it, that is an independent decision: `claude auth logout`.

---

## 15. Scope, compatibility and authorisation

Read all five. They are not boilerplate.

**1. This proxy serves only the requests Cursor points at it.** Cursor Tab
completion, codebase indexing, Auto model routing, cloud agents and subagents may
continue to use Cursor's own infrastructure regardless of this override. Setting a
custom base URL does not make Cursor stop using its own services for those
features. If your reason for using this proxy is to keep traffic off Cursor's
infrastructure entirely, this does not achieve that.

**2. Compatibility with future Cursor updates is not guaranteed.** Cursor's model
endpoint is a private integration surface. The request shape, the path it posts
to, the fields it requires, the model name it sends and its timeout behaviour can
all change in any update, without notice. Two payload shapes are handled today
because Cursor has been observed using both. A Cursor update can break this with
no change on this side. `CLI_PROXY_DEBUG_DUMP=1` is how you find out what changed.

**3. You must be authorised to do this.** Using your Claude Code subscription as a
model backend for a third-party editor may not be permitted by your employer, by
Anthropic's terms, or by Cursor's terms. **You are responsible for obtaining
authorisation from your employer and from Anthropic/Cursor before using this, and
for complying with all applicable terms of service.** Nothing in this repository
grants you permission to do anything.

**4. Do not use this to extract credentials, circumvent limits, or share access.**
Specifically, it must not be used to extract or exfiltrate credentials, to
circumvent rate limits or quotas, or to provide model access to other people. It
binds loopback by default and requires a bearer token on every `/v1` endpoint for
exactly this reason. If you expose it publicly, put Cloudflare Access or an
equivalent control in front of it, and treat the bearer token as a credential of
the same value as your Claude login.

**5. It is experimental and unsupported.** No warranty, no support, no guarantee
of correctness. Read [`LIMITATIONS.md`](LIMITATIONS.md) before relying on it for
anything that matters.
