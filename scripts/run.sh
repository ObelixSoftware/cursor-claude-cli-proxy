#!/usr/bin/env bash
#
# Start cli-proxy locally.
#
# Reads configuration from .env if present. Creates the virtual environment on
# first run. Never prints the bearer token or any Claude credential.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="$REPO_ROOT/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if [[ -f "$REPO_ROOT/.env" ]]; then
  echo "Loading configuration from .env"
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Creating virtual environment at .venv using $PYTHON_BIN"
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_BIN not found. Install Python 3.12 or set PYTHON_BIN." >&2
    exit 1
  fi
  "$PYTHON_BIN" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  "$VENV/bin/python" -m pip install --quiet -e ".[dev]"
fi

if [[ -z "${CLI_PROXY_TOKEN:-}" ]]; then
  cat >&2 <<'EOF'
ERROR: CLI_PROXY_TOKEN is not set.

Generate one and put it in .env:

    openssl rand -hex 32

EOF
  exit 1
fi

if [[ -z "${CLAUDE_EXECUTABLE:-}" ]]; then
  if ! CLAUDE_EXECUTABLE="$(command -v claude)"; then
    echo "ERROR: 'claude' not found on PATH. Set CLAUDE_EXECUTABLE." >&2
    exit 1
  fi
  export CLAUDE_EXECUTABLE
fi

# Debug dumping is ON by default for this script, because its whole purpose is
# working out what an editor actually sends. An explicit CLI_PROXY_DEBUG_DUMP=0
# (in .env or the environment) is respected and turns it off.
export CLI_PROXY_DEBUG_DUMP="${CLI_PROXY_DEBUG_DUMP:-1}"
export CLI_PROXY_DEBUG_DUMP_CONSOLE="${CLI_PROXY_DEBUG_DUMP_CONSOLE:-1}"
export CLI_PROXY_DEBUG_DUMP_DIR="${CLI_PROXY_DEBUG_DUMP_DIR:-$REPO_ROOT/debug-dumps}"

echo "Claude executable: $CLAUDE_EXECUTABLE"
echo "Listening on:      http://${CLI_PROXY_HOST:-127.0.0.1}:${CLI_PROXY_PORT:-8787}"
echo "Bearer token:      configured (${#CLI_PROXY_TOKEN} characters, not shown)"

case "$CLI_PROXY_DEBUG_DUMP" in
  1 | true | TRUE | yes | YES | on | ON)
    cat <<EOF

################################################################################
#                                                                              #
#  WARNING: DEBUG DUMPING IS ON. THIS DISABLES THE NORMAL LOGGING HYGIENE.      #
#                                                                              #
#  Every request and response is written to disk IN CLEARTEXT, in full. That    #
#  includes the entire prompt, any SOURCE CODE the editor sends, tool           #
#  arguments, tool results and the complete model reply. Anything readable by   #
#  your user account can end up in these files.                                #
#                                                                              #
#  Dump directory:  $CLI_PROXY_DEBUG_DUMP_DIR
#  Console echo:    $CLI_PROXY_DEBUG_DUMP_CONSOLE
#  File mode:       0600 (owner only), directory mode 0700                      #
#                                                                              #
#  The authorization header and this proxy's bearer token are still redacted.   #
#  Nothing else is.                                                            #
#                                                                              #
#  TURN IT OFF for normal use:                                                  #
#      CLI_PROXY_DEBUG_DUMP=0 ./scripts/run.sh                                  #
#  or set CLI_PROXY_DEBUG_DUMP=0 in .env, and delete the dumps:                 #
#      rm -rf "$CLI_PROXY_DEBUG_DUMP_DIR"
#                                                                              #
################################################################################

EOF
    ;;
  *)
    echo "Debug dumping:     off"
    echo
    ;;
esac

exec "$VENV/bin/python" -m cli_proxy
