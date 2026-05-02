from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from mlx_openai_proxy.config import Settings
from mlx_openai_proxy.metrics_store import MetricsStore
from mlx_openai_proxy.request_cache import RequestCache
from mlx_openai_proxy.service import ProxyService, REQUEST_CACHE_TTL_SECONDS


class CountingStreamBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.stream_bodies: list[dict[str, Any]] = []

    async def close(self) -> None:
        return None

    async def post_stream(self, path: str, body: dict[str, Any]):
        self.calls += 1
        self.stream_bodies.append(copy.deepcopy(body))
        content = f"answer-{self.calls}"
        yield {
            "id": f"chatcmpl-{self.calls}",
            "object": "chat.completion.chunk",
            "created": self.calls,
            "model": body["model"],
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
        yield {
            "id": f"chatcmpl-{self.calls}",
            "object": "chat.completion.chunk",
            "created": self.calls,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None,
                }
            ],
        }
        yield {
            "id": f"chatcmpl-{self.calls}",
            "object": "chat.completion.chunk",
            "created": self.calls,
            "model": body["model"],
            "choices": [],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
            },
        }
        yield {
            "id": f"chatcmpl-{self.calls}",
            "object": "chat.completion.chunk",
            "created": self.calls,
            "model": body["model"],
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"},
            ],
        }


def test_request_cache_key_covers_request_content_and_options() -> None:
    cache = RequestCache(ttl_seconds=REQUEST_CACHE_TTL_SECONDS)
    body = {
        "model": "gemma4:e2b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        ],
        "reasoning_effort": "low",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "caption",
                "schema": {"type": "object"},
                "strict": True,
            },
        },
    }
    key = cache.make_key(path="/v1/chat/completions", body=body)

    for path, value in (
        (("model",), "gemma4:26b"),
        (("messages", 0, "content", 0, "text"), "Describe this image as JSON."),
        (
            ("messages", 0, "content", 1, "image_url", "url"),
            "data:image/png;base64,BBBB",
        ),
        (("reasoning_effort",), "medium"),
        (("response_format", "json_schema", "strict"), False),
    ):
        changed = copy.deepcopy(body)
        target: Any = changed
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        assert cache.make_key(path="/v1/chat/completions", body=changed) != key


@pytest.mark.asyncio
async def test_chat_nonstream_reuses_cached_response(tmp_path: Path) -> None:
    backend = CountingStreamBackend()
    settings = Settings(metrics_db_path=str(tmp_path / "metrics.db"))
    service = ProxyService(settings, backend, MetricsStore(settings.metrics_db_path))
    body = {
        "model": settings.default_model_alias,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        ],
        "reasoning_effort": "low",
        "temperature": 0,
    }

    first = await service.chat(copy.deepcopy(body))
    second = await service.chat(copy.deepcopy(body))

    assert backend.calls == 1
    assert first == second
    assert first["choices"][0]["message"]["content"] == "answer-1"
    assert backend.stream_bodies[0]["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_chat_cache_misses_when_image_changes(tmp_path: Path) -> None:
    backend = CountingStreamBackend()
    settings = Settings(metrics_db_path=str(tmp_path / "metrics.db"))
    service = ProxyService(settings, backend, MetricsStore(settings.metrics_db_path))
    body = {
        "model": settings.default_model_alias,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        ],
    }
    changed = copy.deepcopy(body)
    changed["messages"][0]["content"][1]["image_url"]["url"] = (
        "data:image/png;base64,BBBB"
    )

    first = await service.chat(body)
    second = await service.chat(changed)

    assert backend.calls == 2
    assert first["choices"][0]["message"]["content"] == "answer-1"
    assert second["choices"][0]["message"]["content"] == "answer-2"
