from __future__ import annotations

import asyncio

import pytest

from mlx_openai_proxy.model_scheduler import ModelScheduler
from mlx_openai_proxy.request_priority import RequestPriority


class FakeRuntime:
    def __init__(self, concurrency: int = 1) -> None:
        self.switched_to: list[str] = []
        self.concurrency = concurrency

    def normalize_alias(self, value: str) -> str:
        return value

    def concurrency_for(self, alias: str) -> int:
        return self.concurrency

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


@pytest.mark.asyncio
async def test_scheduler_admits_by_priority_then_fifo() -> None:
    runtime = FakeRuntime()
    scheduler = ModelScheduler(runtime, default_alias="gemma4:e2b")
    release_first = asyncio.Event()
    entered: list[str] = []

    async def run(
        request_id: str, priority: RequestPriority, release: asyncio.Event | None = None
    ) -> None:
        async with scheduler.slot(
            request_id, "gemma4:e2b", priority=priority
        ):
            entered.append(request_id)
            if release is not None:
                await release.wait()

    first = asyncio.create_task(run("first", RequestPriority.DEFAULT, release_first))
    await asyncio.sleep(0)
    tasks = [
        asyncio.create_task(run("low", RequestPriority.LOW)),
        asyncio.create_task(run("highest", RequestPriority.HIGHEST)),
        asyncio.create_task(run("high-1", RequestPriority.HIGH)),
        asyncio.create_task(run("high-2", RequestPriority.HIGH)),
        asyncio.create_task(run("lowest", RequestPriority.LOWEST)),
        asyncio.create_task(run("default", RequestPriority.DEFAULT)),
    ]
    await asyncio.sleep(0)

    release_first.set()
    await asyncio.gather(first, *tasks)

    assert entered == [
        "first",
        "highest",
        "high-1",
        "high-2",
        "default",
        "low",
        "lowest",
    ]


@pytest.mark.asyncio
async def test_critical_preempts_running_lower_priority_request() -> None:
    runtime = FakeRuntime()
    scheduler = ModelScheduler(runtime, default_alias="gemma4:e2b")
    preempted = asyncio.Event()
    entered: list[str] = []

    async def run_default() -> None:
        async with scheduler.slot(
            "default",
            "gemma4:e2b",
            priority=RequestPriority.DEFAULT,
            on_preempt=preempted.set,
        ):
            entered.append("default")
            await preempted.wait()

    async def run_critical() -> None:
        async with scheduler.slot(
            "critical", "gemma4:e2b", priority=RequestPriority.CRITICAL
        ):
            entered.append("critical")

    default_task = asyncio.create_task(run_default())
    await asyncio.sleep(0)
    critical_task = asyncio.create_task(run_critical())

    await asyncio.gather(default_task, critical_task)

    assert preempted.is_set()
    assert entered == ["default", "critical"]


@pytest.mark.asyncio
async def test_critical_does_not_preempt_running_critical_request() -> None:
    runtime = FakeRuntime()
    scheduler = ModelScheduler(runtime, default_alias="gemma4:e2b")
    release_first = asyncio.Event()
    preempted = False
    entered: list[str] = []

    def mark_preempted() -> None:
        nonlocal preempted
        preempted = True

    async def run(
        request_id: str, release: asyncio.Event | None = None
    ) -> None:
        async with scheduler.slot(
            request_id,
            "gemma4:e2b",
            priority=RequestPriority.CRITICAL,
            on_preempt=mark_preempted,
        ):
            entered.append(request_id)
            if release is not None:
                await release.wait()

    first = asyncio.create_task(run("critical-1", release_first))
    await asyncio.sleep(0)
    second = asyncio.create_task(run("critical-2"))
    await asyncio.sleep(0)

    assert entered == ["critical-1"]
    assert preempted is False

    release_first.set()
    await asyncio.gather(first, second)

    assert entered == ["critical-1", "critical-2"]
