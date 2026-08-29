"""Error taxonomy and client-facing sanitisation.

Every message that reaches a client is a fixed string chosen from this module.
Claude's stderr, prompt bodies and model output are never echoed back, because
stderr in particular can carry environment detail and partial user content.
"""

from __future__ import annotations

from typing import Any


class ProxyError(Exception):
    """Base class for failures that map onto an HTTP status.

    ``client_message`` is the only text that may leave the process.
    """

    status_code: int = 500
    error_type: str = "proxy_error"
    client_message: str = "The proxy failed to handle the request."

    def __init__(self, client_message: str | None = None, *, log_hint: str = "") -> None:
        if client_message:
            self.client_message = client_message
        # ``log_hint`` is a short, pre-vetted classification for local logs.
        # It must never contain prompt text, source code or Claude output.
        self.log_hint = log_hint
        super().__init__(self.client_message)

    def to_payload(self) -> dict[str, Any]:
        """Render an OpenAI-shaped error body."""
        return {
            "error": {
                "message": self.client_message,
                "type": self.error_type,
                "code": self.error_type,
                "param": None,
            }
        }


class AuthenticationError(ProxyError):
    status_code = 401
    error_type = "invalid_request_error"
    client_message = "Missing or invalid bearer token."


class RequestTooLargeError(ProxyError):
    status_code = 413
    error_type = "invalid_request_error"
    client_message = "Request body exceeds the configured maximum size."


class InvalidRequestError(ProxyError):
    status_code = 400
    error_type = "invalid_request_error"
    client_message = "The request payload could not be interpreted."


class UnsupportedContentError(ProxyError):
    status_code = 400
    error_type = "invalid_request_error"
    client_message = (
        "Unsupported message content. This proxy version accepts text and "
        "images; audio and file inputs are not supported."
    )


class ClaudeUnavailableError(ProxyError):
    status_code = 503
    error_type = "api_error"
    client_message = (
        "The Claude Code CLI is not available. Check that it is installed and "
        "that CLAUDE_EXECUTABLE points at it."
    )


class ClaudeAuthError(ProxyError):
    status_code = 502
    error_type = "api_error"
    client_message = (
        "The Claude Code CLI is not authenticated. Run 'claude auth status' and "
        "sign in, then retry."
    )


class ClaudeTimeoutError(ProxyError):
    status_code = 504
    error_type = "api_error"
    client_message = "The Claude Code CLI did not finish before the configured timeout."


class ClaudeProcessError(ProxyError):
    status_code = 502
    error_type = "api_error"
    client_message = "The Claude Code CLI exited unsuccessfully."


class ClaudeOutputError(ProxyError):
    status_code = 502
    error_type = "api_error"
    client_message = (
        "The Claude Code CLI did not return usable structured output. The proxy "
        "refuses to guess at unstructured text."
    )


class ResponseTooLargeError(ProxyError):
    status_code = 502
    error_type = "api_error"
    client_message = "The Claude Code CLI produced more output than the configured limit."


class ClientDisconnectedError(ProxyError):
    status_code = 499
    error_type = "api_error"
    client_message = "The client disconnected before the response was produced."


class UpstreamModelError(ProxyError):
    """Claude itself reported a structured error via the output contract."""

    status_code = 502
    error_type = "api_error"
    client_message = "The model reported that it could not serve the request."
