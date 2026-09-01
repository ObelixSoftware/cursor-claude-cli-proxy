"""cli-proxy: an OpenAI-compatible HTTP shim over the Claude Code CLI.

The proxy presents itself as a model endpoint. It never executes editor tools
and never touches the filesystem on the model's behalf; the calling editor
stays responsible for all tool execution and file changes.
"""

__version__ = "1.2.0"

__all__ = ["__version__"]
