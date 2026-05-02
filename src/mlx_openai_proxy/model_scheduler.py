from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Callable

from .model_runtime import ModelRuntimeManager
from .request_priority import RequestPriority


class QueueFullError(RuntimeError):
    pass


@dataclass
class PendingRequest:
    request_id: str
    model: str
    priority: RequestPriority
    future: asyncio.Future[None]
    sequence: int
    on_preempt: Callable[[], None] | None


@dataclass
class RunningRequest:
    request_id: str
    priority: RequestPriority
    on_preempt: Callable[[], None] | None


class ModelScheduler:
    def __init__(
        self,
        runtime: ModelRuntimeManager,
        default_alias: str,
        *,
        max_queue_size: int = 64,
    ) -> None:
        self.runtime = runtime
        self.default_alias = default_alias
        self.max_queue_size = max(0, max_queue_size)
        self._active_model = default_alias
        self._active_count = 0
        self._running_request_ids: set[str] = set()
        self._running_requests: dict[str, RunningRequest] = {}
        self._queue: deque[PendingRequest] = deque()
        self._lock = asyncio.Lock()
        self._switch_task: asyncio.Task[None] | None = None
        self._next_sequence = 0

    async def wait_for_idle(self) -> None:
        while True:
            switch_task = self._switch_task
            if switch_task is not None:
                await switch_task
                continue
            async with self._lock:
                if (
                    self._switch_task is None
                    and not self._queue
                    and self._active_count == 0
                ):
                    return
            await asyncio.sleep(0)

    async def reject_if_queue_full(self, model: str) -> None:
        normalized_model = self.runtime.normalize_alias(model)
        async with self._lock:
            if (
                not self._can_admit_immediately_locked(normalized_model)
                and len(self._queue) >= self.max_queue_size
            ):
                raise QueueFullError(
                    f"request queue is full ({self.max_queue_size} queued)"
                )

    @asynccontextmanager
    async def slot(
        self,
        request_id: str,
        model: str,
        *,
        priority: RequestPriority = RequestPriority.DEFAULT,
        on_preempt: Callable[[], None] | None = None,
    ) -> AsyncIterator[str]:
        normalized_model = self.runtime.normalize_alias(model)
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        sequence = self._next_sequence
        self._next_sequence += 1
        pending = PendingRequest(
            request_id=request_id,
            model=normalized_model,
            priority=priority,
            future=future,
            sequence=sequence,
            on_preempt=on_preempt,
        )
        async with self._lock:
            self._raise_if_queue_full_locked(normalized_model)
            self._queue.append(pending)
            self._pump_locked()
        try:
            await future
        except Exception:
            async with self._lock:
                self._remove_queued_request_locked(request_id)
            raise

        try:
            yield normalized_model
        finally:
            async with self._lock:
                self._running_request_ids.discard(request_id)
                self._running_requests.pop(request_id, None)
                self._active_count = max(0, self._active_count - 1)
                self._pump_locked()

    def _remove_queued_request_locked(self, request_id: str) -> None:
        self._queue = deque(
            item for item in self._queue if item.request_id != request_id
        )

    def _can_admit_immediately_locked(self, model: str) -> bool:
        if self._switch_task is not None:
            return False
        if self._active_model != model:
            return False
        return self._active_count < self.runtime.concurrency_for(self._active_model)

    def _raise_if_queue_full_locked(self, model: str) -> None:
        if (
            not self._can_admit_immediately_locked(model)
            and len(self._queue) >= self.max_queue_size
        ):
            raise QueueFullError(
                f"request queue is full ({self.max_queue_size} queued)"
            )

    def _pump_locked(self) -> None:
        if self._switch_task is not None:
            return

        if self._active_count > 0:
            self._preempt_for_critical_locked()
            self._admit_head_requests_locked()
            return

        if not self._queue:
            return

        head_model = self._next_pending_request_locked().model
        if self._active_model != head_model:
            self._start_switch_locked(head_model)
            return

        self._admit_head_requests_locked()

    def _admit_head_requests_locked(self) -> None:
        limit = self.runtime.concurrency_for(self._active_model)
        while self._active_count < limit:
            pending = self._pop_next_pending_for_model_locked(self._active_model)
            if pending is None:
                break
            self._active_count += 1
            self._running_request_ids.add(pending.request_id)
            self._running_requests[pending.request_id] = RunningRequest(
                request_id=pending.request_id,
                priority=pending.priority,
                on_preempt=pending.on_preempt,
            )
            if not pending.future.done():
                pending.future.set_result(None)

    def _next_pending_request_locked(self) -> PendingRequest:
        return max(self._queue, key=lambda item: (item.priority, -item.sequence))

    def _pop_next_pending_for_model_locked(self, model: str) -> PendingRequest | None:
        best_index: int | None = None
        best_item: PendingRequest | None = None
        for index, item in enumerate(self._queue):
            if item.model != model:
                continue
            if best_item is None or (item.priority, -item.sequence) > (
                best_item.priority,
                -best_item.sequence,
            ):
                best_index = index
                best_item = item
        if best_index is None:
            return None
        del self._queue[best_index]
        return best_item

    def _preempt_for_critical_locked(self) -> None:
        critical_items = [
            item for item in self._queue if item.priority == RequestPriority.CRITICAL
        ]
        if not critical_items:
            return
        limit = self.runtime.concurrency_for(self._active_model)
        needs_model_switch = any(
            item.model != self._active_model for item in critical_items
        )
        needs_capacity = (
            any(item.model == self._active_model for item in critical_items)
            and self._active_count >= limit
        )
        if not needs_model_switch and not needs_capacity:
            return
        preempted = 0
        for running in list(self._running_requests.values()):
            if running.priority >= RequestPriority.CRITICAL:
                continue
            if running.on_preempt is not None:
                running.on_preempt()
                preempted += 1
            if not needs_model_switch and preempted > 0:
                return

    def _start_switch_locked(self, target_model: str) -> None:
        self._switch_task = asyncio.create_task(self._run_switch(target_model))

    async def _run_switch(self, target_model: str) -> None:
        error: Exception | None = None
        try:
            await self.runtime.switch_to(target_model)
        except Exception as exc:
            error = exc
        async with self._lock:
            self._switch_task = None
            if error is None:
                self._active_model = target_model
            else:
                failed = [
                    item for item in self._queue if item.model == target_model
                ]
                self._queue = deque(
                    item for item in self._queue if item.model != target_model
                )
                for pending in failed:
                    if not pending.future.done():
                        pending.future.set_exception(error)
            self._pump_locked()
