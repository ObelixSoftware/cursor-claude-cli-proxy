"""Bearer-token authentication for the ``/v1`` surface.

The comparison is constant time. Nothing in this module logs the presented
credential, the expected credential or the ``Authorization`` header.
"""

from __future__ import annotations

import hmac

from .errors import AuthenticationError

_BEARER_PREFIX = "bearer "


def extract_bearer_token(authorization_header: str | None) -> str | None:
    """Pull the token out of an ``Authorization: Bearer ...`` header."""
    if not authorization_header:
        return None
    if not authorization_header.lower().startswith(_BEARER_PREFIX):
        return None
    token = authorization_header[len(_BEARER_PREFIX) :].strip()
    return token or None


def tokens_match(presented: str | None, expected: str) -> bool:
    """Constant-time token comparison.

    ``hmac.compare_digest`` is used unconditionally, including when no token was
    presented, so the code path does not branch on secret-dependent timing.
    """
    candidate = presented or ""
    return hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


def require_bearer(authorization_header: str | None, expected: str) -> None:
    """Raise :class:`AuthenticationError` unless a valid token was presented."""
    if not expected:
        raise AuthenticationError(
            "The proxy has no configured bearer token, so every request is refused. "
            "Set CLI_PROXY_TOKEN."
        )
    if not tokens_match(extract_bearer_token(authorization_header), expected):
        raise AuthenticationError()
