#!/usr/bin/env bash
#
# End-to-end smoke test against a RUNNING cli-proxy.
#
# This makes REAL Claude CLI requests, so it is deliberately not part of the
# automated test suite. Start the proxy first (scripts/run.sh) in another shell.
#
# Usage:
#   CLI_PROXY_TOKEN=... ./scripts/smoke-test.sh [base_url]
#
# The token is never echoed. Response bodies are shown so you can inspect the
# translation, so run this with a harmless prompt only.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="${1:-http://127.0.0.1:${CLI_PROXY_PORT:-8787}}"

if [[ -z "${CLI_PROXY_TOKEN:-}" && -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

if [[ -z "${CLI_PROXY_TOKEN:-}" ]]; then
  echo "ERROR: CLI_PROXY_TOKEN is not set (export it or put it in .env)." >&2
  exit 1
fi

AUTH="Authorization: Bearer ${CLI_PROXY_TOKEN}"
PASS=0
FAIL=0

pretty() {
  if command -v jq >/dev/null 2>&1; then jq .; else cat; fi
}

check() {
  local label="$1" actual="$2" expected="$3"
  if [[ "$actual" == "$expected" ]]; then
    echo "  PASS  $label (HTTP $actual)"
    PASS=$((PASS + 1))
  else
    echo "  FAIL  $label (HTTP $actual, expected $expected)"
    FAIL=$((FAIL + 1))
  fi
}

hr() { printf '\n=== %s ===\n' "$1"; }

# ---------------------------------------------------------------------------
hr "1. GET /health (no auth required)"
BODY=$(mktemp)
CODE=$(curl -sS -o "$BODY" -w '%{http_code}' "$BASE_URL/health")
pretty <"$BODY"
check "health" "$CODE" 200

# ---------------------------------------------------------------------------
hr "2. GET /v1/models without a token (must be refused)"
CODE=$(curl -sS -o /dev/null -w '%{http_code}' "$BASE_URL/v1/models")
check "unauthenticated request refused" "$CODE" 401

# ---------------------------------------------------------------------------
hr "3. GET /v1/models with a wrong token (must be refused)"
CODE=$(curl -sS -o /dev/null -w '%{http_code}' \
  -H "Authorization: Bearer 0000000000000000000000000000000000000000000000000000000000000000" \
  "$BASE_URL/v1/models")
check "wrong token refused" "$CODE" 401

# ---------------------------------------------------------------------------
hr "4. GET /v1/models with the correct token"
CODE=$(curl -sS -o "$BODY" -w '%{http_code}' -H "$AUTH" "$BASE_URL/v1/models")
pretty <"$BODY"
check "model list" "$CODE" 200

# ---------------------------------------------------------------------------
hr "5. POST /v1/chat/completions (non-streaming, real Claude call)"
CODE=$(curl -sS -o "$BODY" -w '%{http_code}' -H "$AUTH" \
  -H 'Content-Type: application/json' \
  "$BASE_URL/v1/chat/completions" \
  -d '{
        "model": "claude-cli-sonnet",
        "messages": [
          {"role": "system", "content": "Answer in exactly one short sentence."},
          {"role": "user", "content": "What is 17 plus 25?"}
        ]
      }')
pretty <"$BODY"
check "chat completion" "$CODE" 200

# ---------------------------------------------------------------------------
hr "6. POST /v1/chat/completions (streaming)"
echo "--- raw event stream ---"
curl -sS -N -H "$AUTH" -H 'Content-Type: application/json' \
  "$BASE_URL/v1/chat/completions" \
  -d '{
        "model": "claude-cli-sonnet",
        "stream": true,
        "messages": [{"role": "user", "content": "Reply with exactly: STREAM OK"}]
      }' | tee "$BODY"
echo "--- end of stream ---"
if grep -q 'data: \[DONE\]' "$BODY"; then
  echo "  PASS  stream terminated with [DONE]"
  PASS=$((PASS + 1))
else
  echo "  FAIL  stream did not terminate with [DONE]"
  FAIL=$((FAIL + 1))
fi

# ---------------------------------------------------------------------------
hr "7. POST /v1/responses (Responses API shape, real Claude call)"
CODE=$(curl -sS -o "$BODY" -w '%{http_code}' -H "$AUTH" \
  -H 'Content-Type: application/json' \
  "$BASE_URL/v1/responses" \
  -d '{
        "model": "claude-cli-sonnet",
        "instructions": "Answer in exactly one short sentence.",
        "input": [
          {"type": "message", "role": "user",
           "content": [{"type": "input_text", "text": "Name the capital of France."}]}
        ]
      }')
pretty <"$BODY"
check "responses api" "$CODE" 200

# ---------------------------------------------------------------------------
hr "8. POST /v1/responses with a tool available (expect a tool call)"
CODE=$(curl -sS -o "$BODY" -w '%{http_code}' -H "$AUTH" \
  -H 'Content-Type: application/json' \
  "$BASE_URL/v1/responses" \
  -d '{
        "model": "claude-cli-sonnet",
        "instructions": "You are the model behind a code editor.",
        "input": [
          {"type": "message", "role": "user",
           "content": [{"type": "input_text",
                        "text": "Show me the contents of src/cli_proxy/config.py"}]}
        ],
        "tools": [
          {"type": "function", "name": "read_file",
           "description": "Read a file from the workspace.",
           "parameters": {"type": "object",
                          "properties": {"target_file": {"type": "string"}},
                          "required": ["target_file"]}}
        ],
        "tool_choice": "auto"
      }')
pretty <"$BODY"
check "tool call decision" "$CODE" 200

# ---------------------------------------------------------------------------
hr "9. Tool result round trip (send the result back)"
CODE=$(curl -sS -o "$BODY" -w '%{http_code}' -H "$AUTH" \
  -H 'Content-Type: application/json' \
  "$BASE_URL/v1/responses" \
  -d '{
        "model": "claude-cli-sonnet",
        "instructions": "You are the model behind a code editor. Be brief.",
        "input": [
          {"type": "message", "role": "user",
           "content": [{"type": "input_text", "text": "What does greet.py print?"}]},
          {"type": "function_call", "id": "fc_smoke1", "call_id": "call_smoke1",
           "name": "read_file", "arguments": "{\"target_file\":\"greet.py\"}"},
          {"type": "function_call_output", "call_id": "call_smoke1",
           "output": "print(\"hello world\")"}
        ],
        "tools": [
          {"type": "function", "name": "read_file",
           "parameters": {"type": "object",
                          "properties": {"target_file": {"type": "string"}}}}
        ]
      }')
pretty <"$BODY"
check "tool result round trip" "$CODE" 200

# ---------------------------------------------------------------------------
hr "10. Image input is accepted (HTTP 200, real Claude vision call)"
CODE=$(curl -sS -o "$BODY" -w '%{http_code}' -H "$AUTH" \
  -H 'Content-Type: application/json' \
  "$BASE_URL/v1/chat/completions" \
  -d '{
        "model": "claude-cli-sonnet",
        "messages": [{"role": "user", "content": [
          {"type": "text", "text": "Reply with exactly: IMAGE OK"},
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}}
        ]}]
      }')
pretty <"$BODY"
check "image input accepted" "$CODE" 200

rm -f "$BODY"

printf '\n=== SUMMARY ===\n%d passed, %d failed\n' "$PASS" "$FAIL"
[[ "$FAIL" -eq 0 ]] || exit 1
