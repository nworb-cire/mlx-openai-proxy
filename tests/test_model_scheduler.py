from __future__ import annotations

import pytest

from mlx_openai_proxy.model_scheduler import ModelScheduler


class FakeRuntime:
    def __init__(self) -> None:
        self.switched_to: list[str] = []

    def normalize_alias(self, value: str) -> str:
        return value

    def concurrency_for(self, alias: str) -> int:
        return 1

    async def switch_to(self, alias: str) -> None:
        self.switched_to.append(alias)


@pytest.mark.asyncio
async def test_scheduler_keeps_most_recent_model_loaded_when_idle() -> None:
    runtime = FakeRuntime()
    scheduler = ModelScheduler(runtime, default_alias="gemma4:e2b")

    async with scheduler.slot("req-1", "gemma4:26b"):
        pass

    await scheduler.wait_for_idle()

    assert runtime.switched_to == ["gemma4:26b"]
    assert scheduler._active_model == "gemma4:26b"
