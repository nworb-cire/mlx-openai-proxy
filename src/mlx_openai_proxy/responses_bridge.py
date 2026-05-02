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
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model is required and must be a non-empty string")
    if "input" in body and not isinstance(body["input"], str | list):
        raise ValueError("input must be a string or an array")
    payload = body.get("input")
    if isinstance(payload, list):
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"input[{index}] must be an object")
            content = item.get("content")
            if content is not None and not isinstance(content, str | list):
                raise ValueError(f"input[{index}].content must be a string or an array")
    if "response_format" in body:
        raise ValueError("Responses structured output must use text.format")

    chat_body: dict[str, Any] = {
        "model": model,
        "messages": responses_input_to_messages(body),
        "stream": bool(body.get("stream")),
    }
    for key in ("temperature", "top_p", "top_k", "metadata"):
        if key in body:
            chat_body[key] = body[key]

    text = body.get("text")
    if isinstance(text, dict) and isinstance(text.get("format"), dict):
        text_format = text["format"]
        if text_format.get("type") == "json_schema":
            json_schema = {
                key: text_format[key]
                for key in ("name", "description", "schema", "strict")
                if key in text_format
            }
            chat_body["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }
        elif text_format.get("type") == "json_object":
            chat_body["response_format"] = {"type": "json_object"}
        elif text_format.get("type") == "text":
            chat_body["response_format"] = {"type": "text"}

    max_output_tokens = body.get("max_output_tokens")
    if max_output_tokens is not None:
        chat_body["max_tokens"] = max_output_tokens

    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort") is not None:
        chat_body["reasoning_effort"] = reasoning["effort"]
    return chat_body


def chat_response_to_responses(chat_response: dict[str, Any]) -> dict[str, Any]:
    choice = chat_response["choices"][0]
    message = choice["message"]
    output_text = message.get("content", "")
    response_id = f"resp_{uuid.uuid4().hex}"

    output_items: list[dict[str, Any]] = [
        {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": output_text, "annotations": []}
            ],
        }
    ]

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
