"""Image content extraction and Claude CLI stdin encoding.

Cursor sends images as OpenAI ``image_url`` / ``input_image`` / ``image``
parts. The Claude Code CLI accepts the same pixels as Anthropic content
blocks on ``--input-format stream-json`` stdin. This module is the bridge.

Filesystem and shell tools stay denied: images are inlined as base64 or
passed through as https URLs. Nothing is written into the user's project.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .errors import InvalidRequestError

#: Anthropic's documented per-image ceiling.
MAX_IMAGE_BYTES = 5 * 1024 * 1024

#: Enough for a pasted screenshot plus a handful of follow-ups.
MAX_IMAGES_PER_REQUEST = 20

ALLOWED_MEDIA_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)

_IMAGE_PART_TYPES = frozenset({"image_url", "input_image", "image"})


@dataclass(frozen=True)
class ImageAttachment:
    """One image ready to hand to the Claude CLI."""

    media_type: str
    data: str | None = None
    url: str | None = None

    def label(self) -> str:
        """Short, non-secret description for the text prompt."""
        if self.data is not None:
            nbytes = (len(self.data) * 3) // 4
            return f"{self.media_type}, {nbytes} bytes"
        return self.url or "image"

    def digest(self) -> str:
        """Stable, non-reversible identifier used to fingerprint a turn."""
        hasher = hashlib.sha256()
        hasher.update(self.media_type.encode("utf-8", errors="replace"))
        hasher.update(b"\x1e")
        hasher.update((self.data or self.url or "").encode("utf-8", errors="replace"))
        return hasher.hexdigest()[:16]

    def to_content_block(self) -> dict[str, Any]:
        if self.data is not None:
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": self.media_type,
                    "data": self.data,
                },
            }
        return {"type": "image", "source": {"type": "url", "url": self.url}}


def is_image_part_type(part_type: str) -> bool:
    return part_type in _IMAGE_PART_TYPES


def parse_image_part(part: dict[str, Any]) -> ImageAttachment:
    """Accept the common OpenAI / Responses / Anthropic image shapes."""
    source = part.get("source")
    if isinstance(source, dict):
        return _from_source(source)

    raw = part.get("image_url")
    if raw is None:
        raw = part.get("image")
    if raw is None:
        raw = part.get("url")

    if isinstance(raw, dict):
        nested = raw.get("source")
        if isinstance(nested, dict):
            return _from_source(nested)
        url = raw.get("url") or raw.get("image_url")
        if isinstance(url, str) and url.strip():
            return _from_url(url.strip())
        raise InvalidRequestError("An image part is missing a usable URL.")

    if isinstance(raw, str) and raw.strip():
        return _from_url(raw.strip())

    raise InvalidRequestError("An image part is missing a usable URL.")


def encode_cli_stdin(prompt: str, images: Sequence[ImageAttachment]) -> bytes:
    """Build the bytes written to the Claude CLI's stdin.

    Text-only requests stay as the existing prompt document. When images are
    present the document becomes the text block of a single stream-json user
    message, with image content blocks appended in conversation order.
    """
    if not images:
        return prompt.encode("utf-8")

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(image.to_content_block() for image in images)
    payload = {
        "type": "user",
        "message": {"role": "user", "content": content},
    }
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def enforce_image_count(images: Sequence[ImageAttachment]) -> None:
    if len(images) > MAX_IMAGES_PER_REQUEST:
        raise InvalidRequestError(
            f"A request may include at most {MAX_IMAGES_PER_REQUEST} images."
        )


def _from_source(source: dict[str, Any]) -> ImageAttachment:
    kind = str(source.get("type") or "").lower()
    if kind in {"base64", "image"} or (
        not kind and isinstance(source.get("data"), str)
    ):
        media = _normalize_media_type(
            source.get("media_type") or source.get("mediaType")
        )
        data = source.get("data")
        if not isinstance(data, str) or not data.strip():
            raise InvalidRequestError("A base64 image part is missing 'data'.")
        return _from_base64(media, data)

    url = source.get("url")
    if isinstance(url, str) and url.strip():
        return _from_url(url.strip())

    raise InvalidRequestError("An image source is missing base64 data or a URL.")


def _from_url(url: str) -> ImageAttachment:
    if url.startswith("data:"):
        return _from_data_url(url)

    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise InvalidRequestError(
            "Image URLs must be https:// or a data: URL. "
            "Local paths and other schemes are not accepted."
        )
    return ImageAttachment(media_type="", data=None, url=url)


def _from_data_url(url: str) -> ImageAttachment:
    try:
        header, data = url.split(",", 1)
    except ValueError as exc:
        raise InvalidRequestError("Malformed image data URL.") from exc

    header = header[5:]  # strip "data:"
    tokens = [token.strip() for token in header.split(";") if token.strip()]
    media = tokens[0].lower() if tokens else ""
    is_base64 = any(token.lower() == "base64" for token in tokens[1:])
    if not is_base64:
        raise InvalidRequestError("Image data URLs must be base64-encoded.")
    if not media.startswith("image/"):
        raise InvalidRequestError(
            "A data URL must declare an image media type "
            "(for example data:image/png;base64,...)."
        )
    return _from_base64(_normalize_media_type(media), data)


def _normalize_media_type(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidRequestError("An image part is missing a media type.")
    media = raw.strip().lower()
    if media == "image/jpg":
        media = "image/jpeg"
    if media not in ALLOWED_MEDIA_TYPES:
        raise InvalidRequestError(
            f"Unsupported image media type '{media}'. "
            "Accepted types are image/jpeg, image/png, image/gif and image/webp."
        )
    return media


def _from_base64(media_type: str, data: str) -> ImageAttachment:
    compact = "".join(data.split())
    if not compact:
        raise InvalidRequestError("An image part contains empty base64 data.")
    padded = compact + "=" * ((4 - len(compact) % 4) % 4)
    try:
        decoded = base64.b64decode(padded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise InvalidRequestError(
            "An image part contains invalid base64 data."
        ) from exc
    if not decoded:
        raise InvalidRequestError("An image part contains empty image data.")
    if len(decoded) > MAX_IMAGE_BYTES:
        raise InvalidRequestError(
            f"An image exceeds the maximum size of {MAX_IMAGE_BYTES} bytes."
        )
    return ImageAttachment(media_type=media_type, data=padded, url=None)
