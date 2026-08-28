"""Unit tests for image extraction and Claude CLI stdin encoding."""

from __future__ import annotations

import json

import pytest

from cli_proxy.errors import InvalidRequestError
from cli_proxy.images import (
    MAX_IMAGE_BYTES,
    MAX_IMAGES_PER_REQUEST,
    ImageAttachment,
    encode_cli_stdin,
    parse_image_part,
)
from cli_proxy.normalize import normalize_request, serialize_prompt


PNG_DATA_URL = "data:image/png;base64,AAAA"


def test_parses_chat_completions_image_url():
    image = parse_image_part(
        {"type": "image_url", "image_url": {"url": PNG_DATA_URL}}
    )
    assert image.media_type == "image/png"
    assert image.data == "AAAA"
    assert image.url is None


def test_parses_responses_input_image_string():
    image = parse_image_part({"type": "input_image", "image_url": PNG_DATA_URL})
    assert image.media_type == "image/png"
    assert image.data == "AAAA"


def test_parses_anthropic_source_block():
    image = parse_image_part(
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "AAAA",
            },
        }
    )
    assert image.media_type == "image/jpeg"
    assert image.data == "AAAA"


def test_parses_https_url():
    image = parse_image_part(
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/shot.png"},
        }
    )
    assert image.url == "https://example.com/shot.png"
    assert image.data is None


def test_normalizes_jpg_alias():
    image = parse_image_part(
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpg", "data": "AAAA"},
        }
    )
    assert image.media_type == "image/jpeg"


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/x.png",
        "file:///tmp/secret.png",
        "/tmp/secret.png",
        "blob:https://example.com/abc",
    ],
)
def test_rejects_non_https_urls(url):
    with pytest.raises(InvalidRequestError, match="https://"):
        parse_image_part({"type": "image_url", "image_url": {"url": url}})


def test_rejects_unsupported_media_type():
    with pytest.raises(InvalidRequestError, match="media type"):
        parse_image_part(
            {
                "type": "image_url",
                "image_url": {"url": "data:image/svg+xml;base64,AAAA"},
            }
        )


def test_rejects_non_base64_data_url():
    with pytest.raises(InvalidRequestError, match="base64"):
        parse_image_part(
            {"type": "image_url", "image_url": {"url": "data:image/png,not-b64"}}
        )


def test_rejects_invalid_base64():
    with pytest.raises(InvalidRequestError, match="invalid base64"):
        parse_image_part(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "????"},
            }
        )


def test_rejects_oversized_image(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("cli_proxy.images.MAX_IMAGE_BYTES", 4)
    with pytest.raises(InvalidRequestError, match="maximum size"):
        parse_image_part(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    # 6 decoded bytes > patched 4-byte limit
                    "data": "AAAAAAAA",
                },
            }
        )


def test_rejects_too_many_images(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("cli_proxy.images.MAX_IMAGES_PER_REQUEST", 1)
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
                    {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
                ],
            }
        ]
    }
    with pytest.raises(InvalidRequestError, match="at most 1"):
        normalize_request(body)


def test_encode_cli_stdin_is_plain_text_without_images():
    assert encode_cli_stdin("hello", []) == b"hello"


def test_encode_cli_stdin_wraps_images_as_stream_json():
    image = ImageAttachment(media_type="image/png", data="AAAA")
    raw = encode_cli_stdin("describe this", [image])
    payload = json.loads(raw.decode("utf-8"))
    assert payload["type"] == "user"
    assert payload["message"]["content"][0] == {"type": "text", "text": "describe this"}
    assert payload["message"]["content"][1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
    }


def test_encode_cli_stdin_passes_https_urls_through():
    image = ImageAttachment(media_type="", url="https://example.com/a.png")
    raw = encode_cli_stdin("look", [image])
    payload = json.loads(raw.decode("utf-8"))
    assert payload["message"]["content"][1]["source"] == {
        "type": "url",
        "url": "https://example.com/a.png",
    }


def test_prompt_mentions_attached_images_without_embedding_pixels():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
                ],
            }
        ]
    }
    prompt = serialize_prompt(normalize_request(body), "sonnet")
    assert "what is this" in prompt
    assert "Attached image 1" in prompt
    assert "image/png" in prompt
    assert "AAAA" not in prompt


def test_limits_are_sensible():
    assert MAX_IMAGE_BYTES == 5 * 1024 * 1024
    assert MAX_IMAGES_PER_REQUEST == 20
