from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class MetricsStore:
    def __init__(self, db_path: str) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._active: dict[str, dict[str, Any]] = {}
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS request_history (
                    request_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    model TEXT,
                    stream INTEGER NOT NULL,
                    execution_path TEXT,
                    classification_reason TEXT,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    started_at REAL NOT NULL,
                    completed_at REAL,
                    duration_ms INTEGER,
                    queue_duration_ms INTEGER,
                    service_duration_ms INTEGER,
                    has_schema INTEGER NOT NULL,
                    has_images INTEGER NOT NULL,
                    asks_for_reasoning INTEGER NOT NULL,
                    input_messages INTEGER NOT NULL,
                    input_chars INTEGER NOT NULL,
                    input_image_count INTEGER NOT NULL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    reasoning_tokens INTEGER,
                    output_chars INTEGER,
                    reasoning_chars INTEGER,
                    phase1_latency_ms INTEGER,
                    phase2_latency_ms INTEGER,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            self._ensure_column("queue_duration_ms", "INTEGER")
            self._ensure_column("service_duration_ms", "INTEGER")
            self._conn.commit()

    def _ensure_column(self, name: str, definition: str) -> None:
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(request_history)").fetchall()
        }
        if name not in columns:
            self._conn.execute(f"ALTER TABLE request_history ADD COLUMN {name} {definition}")

    def start_request(self, payload: dict[str, Any]) -> str:
        request_id = payload.get("request_id") or f"req_{uuid.uuid4().hex}"
        record = dict(payload)
        record["request_id"] = request_id
        record["started_at"] = record.get("started_at", time.time())
        record["status"] = record.get("status", "active")
        with self._lock:
            self._active[request_id] = record
        return request_id

    def update_request(self, request_id: str, **fields: Any) -> None:
        with self._lock:
            if request_id in self._active:
                self._active[request_id].update(fields)

    def append_progress(
        self,
        request_id: str,
        *,
        reasoning_delta: str = "",
        output_delta: str = "",
    ) -> None:
        with self._lock:
            item = self._active.get(request_id)
            if item is None:
                return
            if reasoning_delta:
                item["live_reasoning_chars"] = int(item.get("live_reasoning_chars") or 0) + len(
                    reasoning_delta
                )
            if output_delta:
                item["live_output_chars"] = int(item.get("live_output_chars") or 0) + len(
                    output_delta
                )

    def complete_request(self, request_id: str, **fields: Any) -> None:
        self._finish_request(request_id, status="completed", **fields)

    def fail_request(self, request_id: str, error_message: str, **fields: Any) -> None:
        self._finish_request(
            request_id,
            status="error",
            error_message=error_message,
            **fields,
        )

    def _finish_request(self, request_id: str, status: str, **fields: Any) -> None:
        finished_at = time.time()
        with self._lock:
            record = dict(self._active.pop(request_id, {"request_id": request_id}))
            record.update(fields)
            record["status"] = status
            record["completed_at"] = finished_at
            started_at = record.get("started_at", finished_at)
            record["duration_ms"] = int((finished_at - started_at) * 1000)
            service_duration_ms = record.get("service_duration_ms")
            service_started_at = record.get("service_started_at")
            if service_started_at is not None:
                record["service_duration_ms"] = max(
                    0, int((finished_at - float(service_started_at)) * 1000)
                )
                record["queue_duration_ms"] = max(
                    0, int((float(service_started_at) - float(started_at)) * 1000)
                )
            elif service_duration_ms is not None:
                record["queue_duration_ms"] = max(
                    0, record["duration_ms"] - int(service_duration_ms)
                )
            metadata_json = json.dumps(record, ensure_ascii=True, default=str)
            self._conn.execute(
                """
                INSERT OR REPLACE INTO request_history (
                    request_id, path, model, stream, execution_path, classification_reason,
                    status, error_message, started_at, completed_at, duration_ms,
                    queue_duration_ms, service_duration_ms,
                    has_schema, has_images, asks_for_reasoning, input_messages, input_chars,
                    input_image_count, prompt_tokens, completion_tokens, reasoning_tokens,
                    output_chars, reasoning_chars, phase1_latency_ms, phase2_latency_ms,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.get("request_id"),
                    record.get("path"),
                    record.get("model"),
                    int(bool(record.get("stream"))),
                    record.get("execution_path"),
                    record.get("classification_reason"),
                    record.get("status"),
                    record.get("error_message"),
                    record.get("started_at"),
                    record.get("completed_at"),
                    record.get("duration_ms"),
                    record.get("queue_duration_ms"),
                    record.get("service_duration_ms"),
                    int(bool(record.get("has_schema"))),
                    int(bool(record.get("has_images"))),
                    int(bool(record.get("asks_for_reasoning"))),
                    int(record.get("input_messages") or 0),
                    int(record.get("input_chars") or 0),
                    int(record.get("input_image_count") or 0),
                    record.get("prompt_tokens"),
                    record.get("completion_tokens"),
                    record.get("reasoning_tokens"),
                    record.get("output_chars"),
                    record.get("reasoning_chars"),
                    record.get("phase1_latency_ms"),
                    record.get("phase2_latency_ms"),
                    metadata_json,
                ),
            )
            self._conn.commit()

    def get_active_requests(self) -> list[dict[str, Any]]:
        with self._lock:
            items = [dict(item) for item in self._active.values()]
        items.sort(key=lambda item: item.get("started_at", 0))
        now = time.time()
        for item in items:
            item["age_ms"] = int((now - item.get("started_at", now)) * 1000)
            service_started_at = item.get("service_started_at")
            item["state"] = "running" if service_started_at else "queued"
            item["queue_ms"] = (
                max(0, int((service_started_at - item.get("started_at", now)) * 1000))
                if service_started_at
                else item["age_ms"]
            )
            item["service_ms"] = (
                max(0, int((now - service_started_at) * 1000))
                if service_started_at
                else 0
            )
            item["live_reasoning_chars"] = int(item.get("live_reasoning_chars") or 0)
            item["live_output_chars"] = int(item.get("live_output_chars") or 0)
            item["live_reasoning_tokens_est"] = self._estimate_tokens(item["live_reasoning_chars"])
            item["live_output_tokens_est"] = self._estimate_tokens(item["live_output_chars"])
        return [self._public_view(item) for item in items]

    def get_history(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT metadata_json
                FROM request_history
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._public_view(json.loads(row["metadata_json"])) for row in rows]

    def get_summary(self) -> dict[str, Any]:
        active = self.get_active_requests()
        history = self.get_history(limit=200)
        completed = [item for item in history if item.get("status") == "completed"]
        completed_with_service = [
            item for item in completed if (item.get("service_duration_ms") or 0) > 0
        ]
        completed_with_queue = [
            item
            for item in completed
            if item.get("queue_duration_ms") is not None
            and (item.get("service_duration_ms") or 0) > 0
        ]
        avg_service_duration_ms = (
            int(
                sum(item.get("service_duration_ms", 0) or 0 for item in completed_with_service)
                / len(completed_with_service)
            )
            if completed_with_service
            else 0
        )
        avg_queue_duration_ms = (
            int(
                sum(item.get("queue_duration_ms", 0) or 0 for item in completed_with_queue)
                / len(completed_with_queue)
            )
            if completed_with_queue
            else 0
        )
        return {
            "active_count": len(active),
            "queued_count": sum(1 for item in active if item.get("state") == "queued"),
            "running_count": sum(1 for item in active if item.get("state") == "running"),
            "completed_count": len(completed),
            "error_count": sum(1 for item in history if item.get("status") == "error"),
            "avg_duration_ms": avg_service_duration_ms,
            "avg_service_duration_ms": avg_service_duration_ms,
            "avg_queue_duration_ms": avg_queue_duration_ms,
        }

    @staticmethod
    def _public_view(record: dict[str, Any]) -> dict[str, Any]:
        hidden_fields: Iterable[str] = (
            "path",
            "execution_path",
            "classification_reason",
            "metadata_json",
        )
        item = dict(record)
        if (
            item.get("service_started_at") is None
            and (item.get("service_duration_ms") or 0) == 0
            and item.get("queue_duration_ms") == item.get("duration_ms")
        ):
            item["service_duration_ms"] = None
            item["queue_duration_ms"] = None
        for field in hidden_fields:
            item.pop(field, None)
        return item

    @staticmethod
    def _estimate_tokens(char_count: int) -> int:
        if char_count <= 0:
            return 0
        return max(1, round(char_count / 4))
