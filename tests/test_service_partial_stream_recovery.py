from __future__ import annotations

import httpx
import pytest

from mlx_openai_proxy.config import Settings
from mlx_openai_proxy.metrics_store import MetricsStore
from mlx_openai_proxy.service import ProxyService


class TruncatingBackend:
    async def close(self) -> None:
        return None

    async def post_stream(self, path: str, body: dict[str, object]):
        yield {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": body.get("model"),
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
        yield {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": body.get("model"),
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "thinking"},
                    "finish_reason": None,
                }
            ],
        }
        yield {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": body.get("model"),
            "choices": [
                {"index": 0, "delta": {"content": "answer"}, "finish_reason": None}
            ],
        }
        raise httpx.RemoteProtocolError("incomplete chunked read")


@pytest.mark.asyncio
async def test_buffered_chat_completion_recovers_from_partial_stream(tmp_path) -> None:
    settings = Settings(metrics_db_path=str(tmp_path / "metrics.db"))
    service = ProxyService(
        settings, TruncatingBackend(), MetricsStore(settings.metrics_db_path)
    )

    response = await service._buffered_chat_completion(
        {
            "model": settings.default_model_alias,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        }
    )

    assert response["choices"][0]["message"]["content"] == "answer"
    assert response["choices"][0]["message"]["_reasoning_content"] == "thinking"
    assert response["choices"][0]["finish_reason"] == "stop"
