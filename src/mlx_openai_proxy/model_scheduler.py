from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from .model_runtime import ModelRuntimeManager


@dataclass
class PendingRequest:
    request_id: str
    model: str
    future: asyncio.Future[None]


class ModelScheduler:
    def __init__(self, runtime: ModelRuntimeManager, default_alias: str) -> None:
        self.runtime = runtime
        self.default_alias = default_alias
        self._active_model = default_alias
        self._active_count = 0
        self._running_request_ids: set[str] = set()
        self._queue: deque[PendingRequest] = deque()
        self._lock = asyncio.Lock()
        self._switch_task: asyncio.Task[None] | None = None

    @asynccontextmanager
    async def slot(self, request_id: str, model: str) -> AsyncIterator[str]:
        normalized_model = self.runtime.normalize_alias(model)
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        pending = PendingRequest(request_id=request_id, model=normalized_model, future=future)
        async with self._lock:
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
                self._active_count = max(0, self._active_count - 1)
                self._pump_locked()

    def _remove_queued_request_locked(self, request_id: str) -> None:
        self._queue = deque(item for item in self._queue if item.request_id != request_id)

    def _pump_locked(self) -> None:
        if self._switch_task is not None:
            return

        if self._active_count > 0:
            self._admit_head_requests_locked()
            return

        if not self._queue:
            if self._active_model != self.default_alias:
                self._start_switch_locked(self.default_alias)
            return

        head_model = self._queue[0].model
        if self._active_model != head_model:
            self._start_switch_locked(head_model)
            return

        self._admit_head_requests_locked()

    def _admit_head_requests_locked(self) -> None:
        limit = self.runtime.concurrency_for(self._active_model)
        while (
            self._active_count < limit
            and self._queue
            and self._queue[0].model == self._active_model
        ):
            pending = self._queue.popleft()
            self._active_count += 1
            self._running_request_ids.add(pending.request_id)
            if not pending.future.done():
                pending.future.set_result(None)

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
                while self._queue and self._queue[0].model == target_model:
                    pending = self._queue.popleft()
                    if not pending.future.done():
                        pending.future.set_exception(error)
            self._pump_locked()
