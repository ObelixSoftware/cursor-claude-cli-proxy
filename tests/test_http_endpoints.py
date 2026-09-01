"""Endpoint tests: health, models and authentication."""

from __future__ import annotations

import httpx
import pytest

from tests.conftest import AUTH_HEADERS, TEST_TOKEN

CHAT_BODY = {
    "model": "claude-cli-sonnet",
    "messages": [{"role": "user", "content": "hi"}],
}

PROTECTED_ENDPOINTS = (
    ("get", "/v1/models", None),
    ("get", "/v1/models/claude-cli-proxy", None),
    ("post", "/v1/chat/completions", CHAT_BODY),
    ("post", "/v1/responses", {"model": "claude-cli-proxy", "input": "hi"}),
)


# -- health ----------------------------------------------------------------


async def test_health_reports_ok(client: httpx.AsyncClient, fake_mode):
    fake_mode("message")
    response = await client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["claude_executable_available"] is True
    assert body["claude_authenticated"] is True
    assert body["claude_version"] == "2.1.231 (Claude Code)"
    assert body["proxy_version"] == "1.2.0"
    assert body["streaming"] == "buffered"
    assert body["models"] == [
        "claude-cli-proxy",
        "claude-cli-sonnet",
        "claude-cli-opus",
    ]


async def test_health_needs_no_bearer_token(client: httpx.AsyncClient):
    assert (await client.get("/health")).status_code in (200, 503)


async def test_health_returns_no_account_details(client: httpx.AsyncClient):
    raw = (await client.get("/health")).text
    assert "should-never-be-exposed" not in raw
    assert "example.invalid" not in raw
    assert "orgId" not in raw
    assert "email" not in raw
    assert "subscriptionType" not in raw


async def test_health_degrades_when_not_authenticated(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("FAKE_CLAUDE_AUTH", "logged_out")
    response = await client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


# -- models ----------------------------------------------------------------


async def test_model_list_is_openai_shaped(client: httpx.AsyncClient):
    response = await client.get("/v1/models", headers=AUTH_HEADERS)
    assert response.status_code == 200

    body = response.json()
    assert body["object"] == "list"
    assert [entry["id"] for entry in body["data"]] == [
        "claude-cli-proxy",
        "claude-cli-sonnet",
        "claude-cli-opus",
    ]
    for entry in body["data"]:
        assert entry["object"] == "model"
        assert entry["owned_by"] == "cli-proxy"


async def test_model_retrieve(client: httpx.AsyncClient):
    response = await client.get("/v1/models/claude-cli-opus", headers=AUTH_HEADERS)
    assert response.status_code == 200
    assert response.json()["id"] == "claude-cli-opus"


async def test_unknown_model_retrieve_is_400(client: httpx.AsyncClient):
    response = await client.get("/v1/models/gpt-4o", headers=AUTH_HEADERS)
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_an_unknown_model_in_a_request_falls_back_rather_than_failing(
    client: httpx.AsyncClient, fake_mode
):
    """An editor sending a model id we never advertised must still get an answer."""
    fake_mode("message", text="Fell back.")
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
    }
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 200
    assert response.json()["model"] == "gpt-4o"
    assert response.json()["choices"][0]["message"]["content"] == "Fell back."


# -- client probing behaviour ---------------------------------------------


async def test_head_on_health_is_not_a_404(client: httpx.AsyncClient):
    response = await client.head("/health")
    assert response.status_code in (200, 503)


async def test_head_on_models_is_not_a_404(client: httpx.AsyncClient):
    response = await client.head("/v1/models", headers=AUTH_HEADERS)
    assert response.status_code == 200


@pytest.mark.parametrize(
    "path", ["/health", "/v1/models", "/v1/chat/completions", "/v1/responses"]
)
async def test_options_preflight_is_answered(client: httpx.AsyncClient, path):
    response = await client.request("OPTIONS", path)
    assert response.status_code == 204
    assert "POST" in response.headers["allow"]


async def test_trailing_slash_variants_work(client: httpx.AsyncClient, fake_mode):
    fake_mode("message", text="Slashed.")

    models = await client.get("/v1/models/", headers=AUTH_HEADERS)
    assert models.status_code == 200

    chat = await client.post(
        "/v1/chat/completions/",
        json={"model": "claude-cli-proxy", "messages": [{"role": "user", "content": "x"}]},
        headers=AUTH_HEADERS,
    )
    assert chat.status_code == 200
    assert chat.json()["choices"][0]["message"]["content"] == "Slashed."

    responses = await client.post(
        "/v1/responses/",
        json={"model": "claude-cli-proxy", "input": "x"},
        headers=AUTH_HEADERS,
    )
    assert responses.status_code == 200


async def test_unknown_top_level_fields_are_ignored(
    client: httpx.AsyncClient, fake_mode
):
    fake_mode("message", text="Tolerated.")
    body = {
        "model": "claude-cli-proxy",
        "messages": [{"role": "user", "content": "hi"}],
        "seed": 7,
        "top_p": 0.9,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "n": 1,
        "logprobs": False,
        "user": "someone",
        "stream_options": {"include_usage": True},
        "parallel_tool_calls": True,
        "response_format": {"type": "text"},
        "cursor_internal_flag": {"nested": ["anything"]},
        "metadata": {"trace": "abc"},
    }
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Tolerated."


# -- authentication --------------------------------------------------------


@pytest.mark.parametrize(("method", "path", "body"), PROTECTED_ENDPOINTS)
async def test_endpoints_require_a_bearer_token(
    client: httpx.AsyncClient, method, path, body
):
    response = await client.request(method.upper(), path, json=body)
    assert response.status_code == 401
    assert response.json()["error"]["message"] == "Missing or invalid bearer token."


@pytest.mark.parametrize(("method", "path", "body"), PROTECTED_ENDPOINTS)
async def test_endpoints_reject_a_wrong_bearer_token(
    client: httpx.AsyncClient, method, path, body
):
    headers = {"Authorization": "Bearer " + "b" * 64}
    response = await client.request(method.upper(), path, json=body, headers=headers)
    assert response.status_code == 401


async def test_wrong_auth_scheme_is_rejected(client: httpx.AsyncClient):
    response = await client.get(
        "/v1/models", headers={"Authorization": f"Basic {TEST_TOKEN}"}
    )
    assert response.status_code == 401


async def test_api_key_header_alone_is_not_accepted(client: httpx.AsyncClient):
    response = await client.get("/v1/models", headers={"api-key": TEST_TOKEN})
    assert response.status_code == 401


async def test_auth_error_body_never_echoes_the_token(client: httpx.AsyncClient):
    presented = "c" * 64
    response = await client.get(
        "/v1/models", headers={"Authorization": f"Bearer {presented}"}
    )
    assert presented not in response.text
    assert TEST_TOKEN not in response.text


# -- request validation ----------------------------------------------------


async def test_malformed_json_is_400(client: httpx.AsyncClient):
    response = await client.post(
        "/v1/chat/completions",
        content=b"{not json",
        headers={**AUTH_HEADERS, "Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert "not valid JSON" in response.json()["error"]["message"]


async def test_empty_body_is_400(client: httpx.AsyncClient):
    response = await client.post(
        "/v1/chat/completions",
        content=b"",
        headers={**AUTH_HEADERS, "Content-Type": "application/json"},
    )
    assert response.status_code == 400


async def test_body_without_messages_or_input_is_400(client: httpx.AsyncClient):
    response = await client.post(
        "/v1/chat/completions", json={"model": "claude-cli-proxy"}, headers=AUTH_HEADERS
    )
    assert response.status_code == 400


async def test_oversized_request_is_413(client: httpx.AsyncClient):
    body = {
        "model": "claude-cli-proxy",
        "messages": [{"role": "user", "content": "x" * 200_000}],
    }
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 413
    assert "maximum size" in response.json()["error"]["message"]


async def test_image_input_is_accepted(client: httpx.AsyncClient, fake_mode):
    fake_mode("message", text="A tiny PNG.")
    body = {
        "model": "claude-cli-proxy",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        ],
    }
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "A tiny PNG."


async def test_audio_input_is_rejected_clearly(client: httpx.AsyncClient):
    body = {
        "model": "claude-cli-proxy",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "transcribe this"},
                    {"type": "input_audio", "input_audio": {"data": "AAAA"}},
                ],
            }
        ],
    }
    response = await client.post("/v1/chat/completions", json=body, headers=AUTH_HEADERS)
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert "input_audio" in message
    assert "not supported" in message
