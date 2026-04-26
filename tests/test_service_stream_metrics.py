from __future__ import annotations

import pytest
from fastapi.responses import StreamingResponse

from mlx_openai_proxy.config import Settings
from mlx_openai_proxy.metrics_store import MetricsStore
from mlx_openai_proxy.service import ProxyService


class FailingStreamBackend:
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
                {"index": 0, "delta": {"content": "partial"}, "finish_reason": None}
            ],
        }
        raise RuntimeError("stream failed")


@pytest.mark.asyncio
async def test_chat_stream_failure_marks_request_failed(tmp_path) -> None:
    settings = Settings(metrics_db_path=str(tmp_path / "metrics.db"))
    metrics = MetricsStore(settings.metrics_db_path)
    service = ProxyService(settings, FailingStreamBackend(), metrics)

    response = await service.chat(
        {
            "model": settings.default_model_alias,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }
    )

    assert isinstance(response, StreamingResponse)
    with pytest.raises(RuntimeError, match="stream failed"):
        async for _ in response.body_iterator:
            pass

    assert metrics.get_active_requests() == []
    history = metrics.get_history(limit=1)
    assert history[0]["status"] == "error"
    assert history[0]["error_message"] == "stream failed"
    assert history[0]["stream"] is True
