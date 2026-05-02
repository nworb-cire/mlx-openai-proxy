from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from mlx_openai_proxy.config import Settings
from mlx_openai_proxy.metrics_store import MetricsStore
from mlx_openai_proxy.model_scheduler import ModelScheduler
from mlx_openai_proxy.service import ActiveRequestTimeoutError, ProxyService
from mlx_openai_proxy.service import RequestPreemptedError


class HangingBackend:
    async def close(self) -> None:
        return None

    async def post_stream(self, path: str, body: dict[str, object]):
        await asyncio.sleep(0.05)
        if False:
            yield None


class PreemptableBackend:
    async def close(self) -> None:
        return None

    async def post_stream(self, path: str, body: dict[str, object]):
        messages = body.get("messages")
        content = ""
        if isinstance(messages, list) and messages and isinstance(messages[0], dict):
            value = messages[0].get("content")
            content = value if isinstance(value, str) else ""
        if content == "slow":
            await asyncio.sleep(60)
        yield {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": body["model"],
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
        yield {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": body["model"],
            "choices": [
                {"index": 0, "delta": {"content": "ok"}, "finish_reason": None}
            ],
        }
        yield {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": body["model"],
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        yield {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": body["model"],
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }


class FakeRuntime:
    def __init__(self) -> None:
        self.switched_to: list[str] = []

    def normalize_alias(self, value: str) -> str:
        return value

    def concurrency_for(self, alias: str) -> int:
        return 1

    async def switch_to(self, alias: str) -> None:
        self.switched_to.append(alias)


class DelayedScheduler:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.runtime = FakeRuntime()

    @asynccontextmanager
    async def slot(self, request_id: str, model: str, **kwargs):
        await asyncio.sleep(self.delay_seconds)
        yield model


def build_settings(tmp_path: Path, **overrides: object) -> Settings:
    data = {
        "metrics_db_path": str(tmp_path / "metrics.db"),
        "active_request_timeout_seconds": 0.01,
    }
    data.update(overrides)
    return Settings(**data)


@pytest.mark.asyncio
async def test_chat_times_out_only_after_service_starts(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    metrics = MetricsStore(settings.metrics_db_path)
    service = ProxyService(settings, HangingBackend(), metrics)

    body = {
        "model": settings.default_model_alias,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }

    with pytest.raises(ActiveRequestTimeoutError, match="0.01s"):
        await service.chat(body)

    history = metrics.get_history(limit=1)
    assert history[0]["status"] == "error"
    assert history[0]["queue_duration_ms"] == 0
    assert history[0]["service_duration_ms"] is not None
    assert history[0]["service_duration_ms"] >= 10
    assert "active request exceeded timeout" in history[0]["error_message"]


@pytest.mark.asyncio
async def test_queue_delay_does_not_consume_active_timeout(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    metrics = MetricsStore(settings.metrics_db_path)
    scheduler = DelayedScheduler(delay_seconds=0.05)
    service = ProxyService(settings, HangingBackend(), metrics, scheduler=scheduler)

    request_id = metrics.start_request(
        {
            "path": "/v1/chat/completions",
            "model": settings.default_model_alias,
            "stream": False,
            "execution_path": "passthrough",
            "classification_reason": "no_schema",
            "has_schema": False,
            "has_images": False,
            "asks_for_reasoning": False,
            "input_messages": 1,
            "input_chars": 5,
            "input_image_count": 0,
        }
    )

    async with service._service_slot(request_id, settings.default_model_alias):
        await asyncio.sleep(0)

    metrics.complete_request(request_id)
    history = metrics.get_history(limit=1)
    assert history[0]["status"] == "completed"
    assert history[0]["queue_duration_ms"] >= 50
    assert history[0]["service_duration_ms"] < 10


@pytest.mark.asyncio
async def test_critical_request_preempts_running_lower_priority_request(
    tmp_path: Path,
) -> None:
    settings = build_settings(tmp_path, active_request_timeout_seconds=120)
    metrics = MetricsStore(settings.metrics_db_path)
    runtime = FakeRuntime()
    scheduler = ModelScheduler(runtime, settings.default_model_alias)
    service = ProxyService(
        settings, PreemptableBackend(), metrics, scheduler=scheduler
    )

    slow = asyncio.create_task(
        service.chat(
            {
                "model": settings.default_model_alias,
                "messages": [{"role": "user", "content": "slow"}],
                "metadata": {"priority": "low"},
            }
        )
    )
    await asyncio.sleep(0.01)

    critical = await service.chat(
        {
            "model": settings.default_model_alias,
            "messages": [{"role": "user", "content": "fast"}],
            "metadata": {"priority": "critical"},
        }
    )

    with pytest.raises(RequestPreemptedError):
        await slow

    assert critical["choices"][0]["message"]["content"] == "ok"
    history = metrics.get_history(limit=2)
    assert history[0]["priority"] == "critical"
    assert history[1]["priority"] == "low"
    assert history[1]["status"] == "error"
    assert "preempted" in history[1]["error_message"]
