"""Entry point: ``python -m cli_proxy``."""

from __future__ import annotations

import sys

import uvicorn

from .config import ConfigError, load_settings
from .logging_setup import configure_logging


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"cli-proxy configuration error: {exc}", file=sys.stderr)
        return 2

    logger = configure_logging(settings.log_level)
    logger.info(
        "starting cli-proxy on http://%s:%d (claude=%s, alias=%s, concurrency=%d)",
        settings.host,
        settings.port,
        settings.claude_executable,
        settings.default_model_alias,
        settings.max_concurrency,
    )
    if settings.host not in {"127.0.0.1", "localhost", "::1"}:
        logger.warning(
            "binding to %s exposes the proxy beyond loopback; ensure an "
            "authenticating layer sits in front of it",
            settings.host,
        )

    uvicorn.run(
        "cli_proxy.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
