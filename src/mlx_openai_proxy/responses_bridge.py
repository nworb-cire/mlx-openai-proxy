from __future__ import annotations

import time
import uuid
from typing import Any


def responses_input_to_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    payload = body.get("input")
    if isinstance(payload, str):
        return [{"role": "user", "content": payload}]

    if isinstance(payload, list):
        messages: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            role = item.get("role", "user")
            content = item.get("content")
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
                continue
            if isinstance(content, list):
                parts: list[dict[str, Any]] = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") in {"input_text", "text"}:
                        parts.append({"type": "text", "text": part.get("text", "")})
                    elif part.get("type") in {"input_image", "image_url"}:
                        if "image_url" in part:
                            parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": part["image_url"]},
                                }
                            )
                messages.append({"role": role, "content": parts})
        if messages:
            return messages

    instructions = body.get("instructions")
    if isinstance(instructions, str):
        return [{"role": "user", "content": instructions}]
    return []


def responses_request_to_chat(body: dict[str, Any]) -> dict[str, Any]:
    chat_body: dict[str, Any] = {
        "model": body["model"],
        "messages": responses_input_to_messages(body),
        "stream": bool(body.get("stream")),
    }
    for key in ("temperature", "top_p", "top_k", "response_format", "metadata"):
        if key in body:
            chat_body[key] = body[key]
    max_output_tokens = body.get("max_output_tokens")
    if max_output_tokens is not None:
        chat_body["max_tokens"] = max_output_tokens
    if "reasoning" in body:
        chat_body["reasoning"] = body["reasoning"]
    return chat_body


def chat_response_to_responses(chat_response: dict[str, Any]) -> dict[str, Any]:
    choice = chat_response["choices"][0]
    message = choice["message"]
    output_text = message.get("content", "")
    reasoning = message.get("reasoning_content", "")
    response_id = f"resp_{uuid.uuid4().hex}"

    output_items: list[dict[str, Any]] = []
    if reasoning:
        output_items.append(
            {
                "id": f"rs_{uuid.uuid4().hex}",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": reasoning}],
                "content": [{"type": "reasoning_text", "text": reasoning}],
            }
        )
    output_items.append(
        {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": output_text, "annotations": []}
            ],
        }
    )

    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": chat_response.get("model"),
        "output": output_items,
        "output_text": output_text,
        "usage": chat_response.get("usage", {}),
    }
