from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any


def tags_response(model_ids: list[str]) -> dict[str, Any]:
    """Return the subset of Ollama's tags response used by Home Assistant."""
    modified_at = datetime.now(UTC).isoformat()
    return {
        "models": [
            {
                "name": model_id,
                "model": model_id,
                "modified_at": modified_at,
                "size": 0,
                "digest": "local-lm-studio",
                "details": {
                    "format": "gguf",
                    "family": "gemma4",
                    "families": ["gemma4"],
                    "parameter_size": "local",
                    "quantization_level": "unknown",
                },
            }
            for model_id in model_ids
        ]
    }


def ollama_request_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    model = body.get("model")
    messages = body.get("messages")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model is required and must be a non-empty string")
    if not isinstance(messages, list):
        raise ValueError("messages is required and must be an array")

    converted_messages: list[dict[str, Any]] = []
    pending_tool_ids: list[str] = []
    next_tool_id = 0
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        converted: dict[str, Any] = {
            "role": message.get("role"),
            "content": message.get("content") or "",
        }
        if message.get("images"):
            raise ValueError("Ollama image messages are not supported by this facade")
        if tool_calls := message.get("tool_calls"):
            converted_calls: list[dict[str, Any]] = []
            for tool_call in tool_calls:
                function = tool_call.get("function") or {}
                tool_id = f"call_{next_tool_id}"
                next_tool_id += 1
                pending_tool_ids.append(tool_id)
                arguments = function.get("arguments") or {}
                converted_calls.append(
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": function.get("name", ""),
                            "arguments": json.dumps(arguments, separators=(",", ":")),
                        },
                    }
                )
            converted["tool_calls"] = converted_calls
        if converted["role"] == "tool" and pending_tool_ids:
            converted["tool_call_id"] = pending_tool_ids.pop(0)
        converted_messages.append(converted)

    request: dict[str, Any] = {
        "model": model,
        "messages": converted_messages,
        "stream": False,
    }
    if tools := body.get("tools"):
        request["tools"] = tools
        request["tool_choice"] = "auto"
    options = body.get("options") or {}
    if isinstance(options, dict):
        for ollama_key, openai_key in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("num_predict", "max_tokens"),
        ):
            if ollama_key in options:
                request[openai_key] = options[ollama_key]
    if body.get("think"):
        request["reasoning_effort"] = "medium"
    return request


def openai_response_to_ollama(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("OpenAI-compatible backend returned no choices")
    source = choices[0].get("message") or {}
    message: dict[str, Any] = {
        "role": source.get("role", "assistant"),
        "content": source.get("content") or "",
    }
    reasoning = source.get("reasoning_content") or source.get("_reasoning_content")
    if reasoning:
        message["thinking"] = reasoning
    if tool_calls := source.get("tool_calls"):
        converted_calls: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments") or "{}"
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            converted_calls.append(
                {
                    "function": {
                        "name": function.get("name", ""),
                        "arguments": arguments,
                    }
                }
            )
        message["tool_calls"] = converted_calls
    usage = response.get("usage") or {}
    return {
        "model": response.get("model"),
        "created_at": datetime.now(UTC).isoformat(),
        "message": message,
        "done": True,
        "done_reason": choices[0].get("finish_reason") or "stop",
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "eval_count": usage.get("completion_tokens", 0),
    }
