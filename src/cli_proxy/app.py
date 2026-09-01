"""FastAPI application: the HTTP surface of the proxy.

Endpoint behaviour:

* ``GET  /health``              -- unauthenticated liveness plus CLI probes.
* ``GET  /v1/models``           -- OpenAI model list.
* ``POST /v1/chat/completions`` -- Chat Completions, with Responses-shape rescue.
* ``POST /v1/responses``        -- Responses API.

Every ``/v1`` endpoint requires a bearer token. No route logs an authorization
header, a prompt body, source code, tool arguments or Claude output. (The
opt-in debug dump in :mod:`cli_proxy.debug_dump` deliberately does write those
to disk; it is off unless the operator switches it on.)

Streaming requests wait for the CLI envelope *before* opening SSE, so a
session-limit or process failure can still be an HTTP 429/502 that Cursor will
render. The model text itself still arrives in one burst -- see
``LIMITATIONS.md``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from importlib import resources
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .claude_runner import ClaudeDecision, ClaudeRunner
from .config import MODEL_ALIAS_MAP, Settings, load_settings
from .debug_dump import DebugDumper, Exchange
from .errors import (
    ClientDisconnectedError,
    InvalidRequestError,
    ProxyError,
    RequestTooLargeError,
    UpstreamModelError,
)
from .logging_setup import configure_logging, get_logger
from .normalize import (
    FLAVOR_CHAT,
    FLAVOR_RESPONSES,
    NormalizedRequest,
    derive_conversation_identity,
    normalize_request,
    serialize_prompt,
)
from .openai_out import (
    ChatStreamBuilder,
    ResponseStreamBuilder,
    build_chat_completion,
    build_response,
)
from .schema import KIND_ERROR
from .security import require_bearer

_LOG = get_logger()

#: How often the disconnect watcher polls while Claude is running.
_DISCONNECT_POLL_SECONDS = 0.5

_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def load_adapter_system_prompt() -> str:
    """Read the adapter system prompt shipped with the package."""
    return (
        resources.files("cli_proxy.prompts")
        .joinpath("adapter_system_prompt.md")
        .read_text(encoding="utf-8")
    )


async def _read_limited_body(request: Request, max_bytes: int) -> bytes:
    """Read the request body, refusing anything over ``max_bytes``.

    The declared ``Content-Length`` is checked first so an oversized upload is
    rejected without buffering it, then the streamed size is enforced in case
    the header was absent or dishonest.
    """
    declared = request.headers.get("content-length")
    if declared:
        try:
            if int(declared) > max_bytes:
                raise RequestTooLargeError(log_hint="content-length over limit")
        except ValueError as exc:
            raise InvalidRequestError("Malformed Content-Length header.") from exc

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_bytes:
            raise RequestTooLargeError(log_hint="streamed body over limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_json_body(raw: bytes) -> dict[str, Any]:
    if not raw.strip():
        raise InvalidRequestError("Request body is empty.")
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InvalidRequestError("Request body is not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise InvalidRequestError("Request body must be a JSON object.")
    return parsed


async def _watch_for_disconnect(request: Request) -> None:
    while True:
        try:
            if await request.is_disconnected():
                return
        except Exception:  # noqa: BLE001 - a transport quirk must not abort the request
            return
        await asyncio.sleep(_DISCONNECT_POLL_SECONDS)


async def _run_guarded(request: Request, coro: Any) -> ClaudeDecision:
    """Await ``coro``, cancelling it if the client disconnects first.

    Cancellation propagates into :meth:`ClaudeRunner.run`, which reaps the
    subprocess with SIGTERM then SIGKILL.
    """
    work = asyncio.ensure_future(coro)
    watcher = asyncio.ensure_future(_watch_for_disconnect(request))
    try:
        done, _ = await asyncio.wait(
            {work, watcher}, return_when=asyncio.FIRST_COMPLETED
        )
        if work in done:
            return work.result()

        work.cancel()
        await asyncio.gather(work, return_exceptions=True)
        _LOG.info("client disconnected; claude subprocess cancelled")
        raise ClientDisconnectedError(log_hint="client disconnected")
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        if not work.done():
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)


StreamBuilder = ChatStreamBuilder | ResponseStreamBuilder


async def _stream_decision(
    builder: StreamBuilder,
    decision: ClaudeDecision,
    exchange: Exchange,
) -> AsyncIterator[str]:
    """Emit a finished Claude decision as SSE.

    The CLI has already completed successfully. An HTTP error status is no
    longer available once this generator is returned as a ``StreamingResponse``,
    so any failure after the first chunk is reported in band.
    """

    def emit(chunks: Iterator[str]) -> Iterator[str]:
        for chunk in chunks:
            exchange.record_sse_chunk(chunk)
            yield chunk

    sent_tokens = False
    try:
        for chunk in emit(builder.opening()):
            sent_tokens = True
            yield chunk
        for chunk in emit(builder.body(decision)):
            sent_tokens = True
            yield chunk
    except Exception as exc:  # noqa: BLE001 - must not leak a traceback
        if not sent_tokens:
            raise
        failure = (
            exc.client_message
            if isinstance(exc, ProxyError)
            else "The proxy failed to handle the request."
        )
        failure_type = exc.error_type if isinstance(exc, ProxyError) else "api_error"
        exchange.record_error(exc)
        _LOG.error("stream failed after opening: %s", type(exc).__name__)
        for chunk in emit(builder.failure(failure, error_type=failure_type)):
            yield chunk
    finally:
        exchange.close()


def create_app(
    settings: Settings | None = None,
    runner: ClaudeRunner | None = None,
) -> FastAPI:
    """Build the application. Injecting ``runner`` is how tests avoid real calls."""
    resolved = settings or load_settings()
    configure_logging(resolved.log_level)

    app = FastAPI(
        title="cli-proxy",
        version=__version__,
        description=(
            "Experimental OpenAI-compatible proxy over the Claude Code CLI. "
            "The calling editor retains ownership of all tool execution."
        ),
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    app.state.settings = resolved
    app.state.runner = runner or ClaudeRunner(resolved, load_adapter_system_prompt())
    app.state.dumper = DebugDumper(resolved)

    # Print the count once at startup so the console always shows a total,
    # rather than staying silent until the first request arrives.
    app.state.runner.log_agent_count()

    if app.state.dumper.enabled:
        _LOG.warning(
            "DEBUG DUMP IS ON: full request and response bodies are being "
            "written in cleartext to %s",
            app.state.dumper.directory,
        )

    @app.exception_handler(ProxyError)
    async def _proxy_error_handler(_: Request, exc: ProxyError) -> JSONResponse:
        if exc.log_hint:
            _LOG.warning("request failed: %s (http %d)", exc.log_hint, exc.status_code)
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    def _auth(request: Request) -> None:
        require_bearer(request.headers.get("authorization"), request.app.state.settings.token)

    # -- health ------------------------------------------------------------

    # GET and HEAD together: a client verifying the endpoint exists may probe
    # with HEAD, and FastAPI does not add it implicitly.
    @app.api_route("/health", methods=["GET", "HEAD"])
    async def health(request: Request) -> JSONResponse:
        cfg: Settings = request.app.state.settings
        active: ClaudeRunner = request.app.state.runner
        probe = await active.probe_health()

        healthy = probe.executable_available and probe.authenticated
        body = {
            "status": "ok" if healthy else "degraded",
            "proxy_version": __version__,
            "claude_executable": probe.executable_path,
            "claude_executable_available": probe.executable_available,
            "claude_version": probe.version,
            "claude_authenticated": probe.authenticated,
            "detail": probe.detail,
            "default_model_alias": cfg.default_model_alias,
            "models": list(cfg.advertised_models),
            "max_concurrency": cfg.max_concurrency,
            "agents_running": active.agents_running,
            "timeout_seconds": cfg.timeout_seconds,
            # The content is still delivered in one burst. The event stream
            # opens only after the CLI envelope is known, so a 429/502 can
            # still be an HTTP status Cursor will render.
            "streaming": "buffered",
            "stream_opens_immediately": False,
            # A boolean only. /health is unauthenticated, so the dump directory
            # path stays out of it.
            "debug_dump": request.app.state.dumper.enabled,
        }
        return JSONResponse(status_code=200 if healthy else 503, content=body)

    # -- models ------------------------------------------------------------

    @app.api_route("/v1/models", methods=["GET", "HEAD"])
    @app.api_route("/v1/models/", methods=["GET", "HEAD"])
    async def list_models(request: Request) -> JSONResponse:
        _auth(request)
        cfg: Settings = request.app.state.settings
        return JSONResponse(
            content={
                "object": "list",
                "data": [
                    {
                        "id": model_id,
                        "object": "model",
                        "created": 0,
                        "owned_by": "cli-proxy",
                        "root": model_id,
                        "parent": None,
                        "permission": [],
                    }
                    for model_id in cfg.advertised_models
                ],
            }
        )

    @app.api_route("/v1/models/{model_id}", methods=["GET", "HEAD"])
    async def retrieve_model(request: Request, model_id: str) -> JSONResponse:
        _auth(request)
        if model_id not in MODEL_ALIAS_MAP:
            raise InvalidRequestError(f"Unknown model '{model_id}'.")
        return JSONResponse(
            content={
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "cli-proxy",
            }
        )

    # -- completions -------------------------------------------------------

    async def _handle(request: Request, endpoint_flavor: str) -> Any:
        cfg: Settings = request.app.state.settings
        active: ClaudeRunner = request.app.state.runner
        dumper: DebugDumper = request.app.state.dumper

        exchange = dumper.begin()
        stream_owns_exchange = False

        try:
            # Recorded before authentication so a rejected request is still
            # visible in a dump. Header values that are pure secret are removed.
            exchange.record_inbound(
                method=request.method,
                path=request.url.path,
                query=str(request.url.query or ""),
                headers=request.headers,
                body=b"",
            )
            _auth(request)

            raw = await _read_limited_body(request, cfg.max_request_bytes)
            exchange.record_inbound_body(raw)
            body = _parse_json_body(raw)

            normalized = normalize_request(
                body,
                flavor=FLAVOR_RESPONSES if endpoint_flavor == FLAVOR_RESPONSES else None,
            )
            model_alias = cfg.resolve_model_alias(normalized.requested_model)
            model_id = normalized.requested_model or "claude-cli-proxy"

            attachments = normalized.images
            exchange.record_normalized(
                flavor=normalized.api_flavor,
                model_alias=model_alias,
                model_id=model_id,
                stream=normalized.stream,
                endpoint_flavor=endpoint_flavor,
                turns=len(normalized.turns),
                tool_names=[tool.name for tool in normalized.tools],
                image_count=len(attachments),
                top_level_fields=sorted(body),
            )

            _LOG.info(
                "%s request accepted: flavor=%s turns=%d tools=%d images=%d "
                "stream=%s alias=%s",
                request.url.path,
                normalized.api_flavor,
                len(normalized.turns),
                len(normalized.tools),
                len(attachments),
                normalized.stream,
                model_alias,
            )

            prompt = serialize_prompt(normalized, model_alias)
            exchange.record_prompt(prompt)

            # Names the conversation so that a later request carrying an edited
            # prompt replaces this invocation instead of racing it.
            identity = derive_conversation_identity(body, normalized)

            # Response shape follows the *payload* shape, so a Responses-style
            # body posted to /v1/chat/completions gets a Responses-style reply.
            # This holds for the streaming path as much as the buffered one.
            wants_responses = normalized.api_flavor == FLAVOR_RESPONSES

            if normalized.stream:
                builder: StreamBuilder = (
                    ResponseStreamBuilder(model_id, normalized)
                    if wants_responses
                    else ChatStreamBuilder(model_id)
                )
                # Await the CLI before opening SSE. Cursor Agent ignores
                # in-band ``data: {"error": ...}`` on a 200 stream and only
                # renders HTTP status + JSON ``error.message``. A session
                # limit is known from the finished envelope, so fail the
                # request with 429 rather than opening a blank turn.
                decision = await _run_guarded(
                    request,
                    active.run(
                        prompt, model_alias, exchange, attachments, identity
                    ),
                )
                if decision.kind == KIND_ERROR:
                    _LOG.warning("model returned a structured error")
                    raise UpstreamModelError(
                        f"The model could not serve the request: {decision.error}",
                        log_hint="model structured error",
                    )
                exchange.record_response(
                    status=200,
                    body="text/event-stream; the emitted events are in 'sse_chunks'",
                )
                stream_owns_exchange = True
                return StreamingResponse(
                    _stream_decision(builder, decision, exchange),
                    media_type="text/event-stream",
                    headers=_SSE_HEADERS,
                )

            decision = await _run_guarded(
                request,
                active.run(prompt, model_alias, exchange, attachments, identity),
            )

            if decision.kind == KIND_ERROR:
                # Claude used the contract's error shape. Surface it as an
                # upstream failure rather than passing model prose through as an
                # answer.
                _LOG.warning("model returned a structured error")
                raise UpstreamModelError(
                    f"The model could not serve the request: {decision.error}",
                    log_hint="model structured error",
                )

            _LOG.info(
                "response ready: kind=%s tool_calls=%d",
                decision.kind,
                len(decision.tool_calls),
            )

            payload = (
                build_response(decision, model_id=model_id, request=normalized)
                if wants_responses
                else build_chat_completion(decision, model_id=model_id)
            )
            exchange.record_response(status=200, body=payload)
            return JSONResponse(content=payload)
        except ProxyError as exc:
            exchange.record_response(status=exc.status_code, body=exc.to_payload())
            exchange.record_error(exc)
            raise
        except BaseException as exc:
            exchange.record_error(exc)
            raise
        finally:
            if not stream_owns_exchange:
                exchange.close()

    @app.post("/v1/chat/completions")
    @app.post("/v1/chat/completions/")
    async def chat_completions(request: Request) -> Any:
        return await _handle(request, FLAVOR_CHAT)

    @app.post("/v1/responses")
    @app.post("/v1/responses/")
    async def responses(request: Request) -> Any:
        return await _handle(request, FLAVOR_RESPONSES)

    # A client that probes the endpoint before using it should get a clear
    # answer rather than a 405 or a redirect it might not follow.
    @app.options("/{path:path}")
    async def preflight(path: str) -> Response:
        return Response(
            status_code=204,
            headers={
                "Allow": "GET, POST, HEAD, OPTIONS",
                "Access-Control-Allow-Methods": "GET, POST, HEAD, OPTIONS",
                "Access-Control-Allow-Headers": "authorization, content-type",
            },
        )

    return app


def build_normalized_preview(body: dict[str, Any]) -> NormalizedRequest:
    """Normalise a body without running Claude. Used by tests and diagnostics."""
    return normalize_request(body)
