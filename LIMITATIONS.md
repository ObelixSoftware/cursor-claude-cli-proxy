# Limitations

Known, deliberate limitations of cli-proxy. Each entry says what the limitation
is, why it exists, and what it means for you in practice.

## 1. Claude is not run with all internal tools disabled

The original intent was to launch the CLI with `--tools ""`, leaving Claude with
no built-in capability at all. That turned out to be impossible without also
breaking the proxy.

`--json-schema`, which is what forces Claude to reply with a machine-readable
decision instead of prose, is implemented inside Claude Code as a **built-in
tool named `StructuredOutput`**. Denying the built-in tool set therefore denies
structured output too: the CLI records the schema call under
`permission_denials` and answers in free-form text, which this proxy refuses to
parse. Verified against Claude Code 2.1.231.

The shipped configuration is instead:

```
--tools "StructuredOutput"
--disallowedTools "Agent,Bash,BashOutput,Edit,ExitPlanMode,Glob,Grep,KillShell,
                   MultiEdit,NotebookEdit,NotebookRead,Read,SlashCommand,Task,
                   TodoWrite,WebFetch,WebSearch,Write"
```

`StructuredOutput` only hands a JSON object back to its caller. It cannot read
files, write files, run commands or reach the network. The explicit
`--disallowedTools` list is belt and braces: even if a future CLI release
changes what `--tools` means, Claude still cannot get filesystem or shell
access. This was verified to leave the subprocess with no such access.

**What it means for you:** exactly one Claude built-in is enabled, and it is
inert. All real tool execution stays with the editor.

## 2. Streaming is buffered

The connection opens immediately; the *text* does not stream token by token.

For `"stream": true` the proxy:

1. sends HTTP 200 and the opening event(s) straight away -- the
   `{"role":"assistant","content":""}` delta for Chat Completions, or
   `response.created` and `response.in_progress` for the Responses API;
2. emits an SSE comment keepalive (`: keepalive`) every 7 seconds while the CLI
   runs, which every conforming SSE client ignores;
3. emits the entire answer in one burst when the CLI finishes, then the finish
   chunk and `data: [DONE]`.

Measured locally against Claude Code 2.1.231: time to first byte 1.2 ms, full
answer 15.8 s. Before this design, time to first byte *was* the full model
latency, which is a reliable way to trip a client's first-byte timeout and
render nothing at all.

**What it means for you:** the editor shows the request as live within
milliseconds, then the reply appears all at once rather than typing itself out.
There is no partial-token streaming, and there is no progress indication beyond
the keepalives. This is inherent to `claude --print`, which writes its JSON
envelope only on completion.

### Errors while streaming

Once the 200 has been sent, an HTTP error status is no longer available. A
failure that happens *after* the stream opened is reported in band:

* Chat Completions: a chunk with `finish_reason: "stop"`, then
  `data: {"error": {...}}`, then `data: [DONE]`.
* Responses: a `response.failed` event whose `response.error` carries the
  reason, then `data: [DONE]`.

Failures detected *before* the stream opens -- authentication, oversized body,
malformed JSON, unsupported content -- still return an ordinary HTTP 4xx/5xx
JSON error, as do all non-streaming requests.

**What it means for you:** a strict OpenAI client will raise on the in-band
error object; a lenient one may show an empty reply instead of an error. Check
the proxy's log or a debug dump if a streamed turn comes back blank.

## 3. Images are accepted; audio and files are not

Message content parts of type `image_url`, `input_image` and `image` are
accepted. Data URLs (`data:image/png;base64,...` and the other Anthropic-supported
image types) are decoded and forwarded to the Claude Code CLI as vision content
blocks. `https://` image URLs are passed through as URL sources. Local paths,
`file://`, `http://` and other schemes are rejected.

Images are inlined on stdin via `--input-format stream-json`. The CLI couples
that flag to two others: it requires `--output-format stream-json`, which under
`--print` in turn requires `--verbose`. An image request therefore switches all
three, and the reply is read from the terminal `result` event of the resulting
event stream rather than from a single JSON object. Text-only requests are
unaffected and still use `--output-format json`.

The CLI's filesystem and shell tools stay denied; nothing is written into the
user's project. Each image is capped at 5 MiB decoded, and a request may carry
at most 20 images. The existing `CLI_PROXY_MAX_REQUEST_BYTES` limit still
applies to the inbound JSON body.

`input_audio`, `audio`, `input_file`, `file` and `file_url` parts are still
rejected with HTTP 400 and a message naming the offending type. There is no
silent dropping and no attempt to describe those attachments in words.

**What it means for you:** pasting a screenshot into a chat turn works. Audio
clips and generic file attachments still fail the whole request.

## 4. Cursor compatibility is not guaranteed

This proxy is an unofficial, experimental adapter. Cursor's model endpoint is a
private integration surface: the request shape, the endpoint it posts to, the
fields it requires and its timeout behaviour can all change in any update,
without notice. Two shapes are handled today (Chat Completions and Responses,
detected from the payload rather than the URL) because Cursor has been observed
posting a Responses-style body to `/v1/chat/completions`.

**What it means for you:** a Cursor update can break this without anything in
the proxy changing. Turn on `CLI_PROXY_DEBUG_DUMP=1` (see `.env.example`) to see
exactly what Cursor sent, then adapt. End-to-end behaviour inside Cursor's UI
has not been automatically verified and needs a human to confirm.

## 5. Other operational limits

* **Up to N concurrent CLI processes (default 4).**
  `CLAUDE_MAX_CONCURRENCY` defaults to 4 so Cursor `/multitask` can run
  several agents at once. Further requests wait on the semaphore. Each
  invocation is a full process-group launch, so raising this multiplies
  memory and token cost. The console prints `cli-proxy agents running: N/M`
  on every change. When an agent finishes, is cancelled, times out, or is
  superseded, the proxy signals the whole process group (SIGTERM, 5 s,
  SIGKILL) so child processes do not leak. The reasoning behind the default
  of 4 is written out in the README, under "Why the concurrency default is
  four" in §6.
* **A prompt edit cannot be injected into a running `claude --print`.**
  That command reads stdin once. If a later request has the same
  conversation prefix and a different last user message, the in-flight
  process group is killed and a new invocation starts with the updated
  prompt. Brand-new first messages are not matched this way, so two
  first-turn `/multitask` siblings do not cancel each other. First-message
  edits depend on the editor aborting the previous HTTP stream.
* **No conversation reuse.** `--continue` and `--resume` are never used and
  session persistence is off, so every request re-sends the whole conversation
  and re-pays for the prompt tokens.
* **No token-level cost control.** `temperature` and `max_tokens` are passed
  through to the model as prose hints in the prompt, not as hard limits. The CLI
  exposes no flags for them.
* **Debug dumps are cleartext.** `CLI_PROXY_DEBUG_DUMP=1` writes full request
  and response bodies, including any source code the editor sent, to disk in
  plain text (files 0600, directory 0700). Only the authorization-style headers
  and the proxy's own bearer token are redacted. `./scripts/run.sh` turns this
  **on by default** and prints a warning. It must be off for normal use.
* **Loopback only by default.** The listener binds 127.0.0.1. Exposing it needs
  an authenticating tunnel in front; the bearer token is the only access control
  the proxy itself has.
