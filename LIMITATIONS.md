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

The *text* does not stream token by token. The proxy waits for the finished
`claude --print` envelope before opening SSE, so a session-limit or process
failure can still be an HTTP 429/502 that Cursor will render.

For `"stream": true` the proxy:

1. awaits the CLI envelope (same as a non-streaming request);
2. on a CLI / model error before any assistant token, returns the same
   OpenAI-shaped JSON error as non-stream -- HTTP **429** for a session limit,
   **502** for auth or process failure -- and does **not** open SSE;
3. on success, sends HTTP 200 and the opening event(s), then the entire answer
   in one burst, then the finish chunk and `data: [DONE]`.

A 200 stream with an in-band `data: {"error": ...}` looks like a crash or
blank turn in Cursor Agent, so the proxy refuses to open SSE when it already
knows the CLI failed. If a chunk has already gone out and something then
fails, the stream is closed in band (`data: {"error": ...}` /
`response.failed`). Rate-limit envelopes from Claude have no assistant text,
so the pre-stream 429 path covers that case. Claude's reset clock is never
forwarded.

Time-to-first-byte is therefore the full model latency. That can trip a
client's first-byte timeout on a long turn; it is the trade-off for making
errors visible in Cursor.

**What it means for you:** the reply appears all at once rather than typing
itself out. There is no partial-token streaming. This is inherent to
`claude --print`, which writes its JSON envelope only on completion. A
session-limit turn should show the fixed 429 message, not a blank reply.

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
