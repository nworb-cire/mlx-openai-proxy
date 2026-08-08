from __future__ import annotations

import json

from mlx_openai_proxy.ollama_compat import (
    ollama_request_to_openai,
    openai_response_to_ollama,
    tags_response,
)


def test_tags_advertises_local_models() -> None:
    response = tags_response(["gemma4:e2b"])
    assert response["models"][0]["model"] == "gemma4:e2b"


def test_converts_ollama_tool_history_to_openai() -> None:
    request = ollama_request_to_openai(
        {
            "model": "gemma4:e2b",
            "messages": [
                {"role": "user", "content": "temperature?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "GetTemperature",
                                "arguments": {"room": "kitchen"},
                            }
                        }
                    ],
                },
                {"role": "tool", "content": '{"temperature":72}'},
            ],
            "tools": [{"type": "function", "function": {"name": "GetTemperature"}}],
        }
    )
    assert request["messages"][1]["tool_calls"][0]["function"]["arguments"] == (
        '{"room":"kitchen"}'
    )
    assert request["messages"][2]["tool_call_id"] == "call_0"


def test_converts_openai_tool_call_to_ollama() -> None:
    response = openai_response_to_ollama(
        {
            "model": "gemma4:e2b",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "GetTemperature",
                                    "arguments": json.dumps({"room": "kitchen"}),
                                }
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
    )
    assert response["done"] is True
    assert response["message"]["tool_calls"][0]["function"] == {
        "name": "GetTemperature",
        "arguments": {"room": "kitchen"},
    }
