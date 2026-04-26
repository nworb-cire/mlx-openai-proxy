from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any


def _path_to_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "application/octet-stream"
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _normalize_image_url(url: str, max_inline_image_bytes: int) -> str:
    if (
        url.startswith("http://")
        or url.startswith("https://")
        or url.startswith("data:")
    ):
        return url

    if url.startswith("file://"):
        path = Path(url[7:])
    else:
        path = Path(url)

    if not path.is_file():
        return url

    if path.stat().st_size > max_inline_image_bytes:
        raise ValueError(f"Image file is too large to inline: {path}")
    return _path_to_data_url(path)


def normalize_chat_images(
    body: dict[str, Any], max_inline_image_bytes: int
) -> dict[str, Any]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body

    new_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            new_messages.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            new_messages.append(message)
            continue
        new_parts: list[Any] = []
        for part in content:
            if not isinstance(part, dict):
                new_parts.append(part)
                continue
            if part.get("type") == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, dict) and isinstance(
                    image_url.get("url"), str
                ):
                    updated = dict(part)
                    updated["image_url"] = dict(image_url)
                    updated["image_url"]["url"] = _normalize_image_url(
                        image_url["url"], max_inline_image_bytes
                    )
                    new_parts.append(updated)
                    continue
            new_parts.append(part)
        new_message = dict(message)
        new_message["content"] = new_parts
        new_messages.append(new_message)

    new_body = dict(body)
    new_body["messages"] = new_messages
    return new_body


def normalize_responses_input(
    body: dict[str, Any], max_inline_image_bytes: int
) -> dict[str, Any]:
    payload = body.get("input")
    if isinstance(payload, str):
        return body

    if not isinstance(payload, list):
        return body

    items: list[Any] = []
    for item in payload:
        if not isinstance(item, dict):
            items.append(item)
            continue
        content = item.get("content")
        if not isinstance(content, list):
            items.append(item)
            continue
        new_content: list[Any] = []
        for part in content:
            if not isinstance(part, dict):
                new_content.append(part)
                continue
            if part.get("type") in {"input_image", "image_url"}:
                key = "image_url" if "image_url" in part else "file_url"
                value = part.get(key)
                if isinstance(value, str):
                    updated = dict(part)
                    updated[key] = _normalize_image_url(value, max_inline_image_bytes)
                    new_content.append(updated)
                    continue
            new_content.append(part)
        updated_item = dict(item)
        updated_item["content"] = new_content
        items.append(updated_item)
    new_body = dict(body)
    new_body["input"] = items
    return new_body
