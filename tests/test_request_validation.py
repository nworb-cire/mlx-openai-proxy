from __future__ import annotations

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

from mlx_openai_proxy.config import Settings
from mlx_openai_proxy.main import create_app
from mlx_openai_proxy.model_runtime import ModelRuntimeError
from mlx_openai_proxy.model_scheduler import QueueFullError


class FakeAsrRuntime:
    @property
    def alias(self) -> str:
        return "parakeet:tdt-0.6b-v3"

    async def load(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def transcribe_pcm(self, pcm: bytes, sample_rate: int):
        raise AssertionError("not used")

    def create_stream(self):
        raise AssertionError("not used")


class FailingScheduler:
    class Runtime:
        def normalize_alias(self, value: str) -> str:
            return value

    runtime = Runtime()

    @asynccontextmanager
    async def slot(self, request_id: str, model: str, **kwargs):
        raise ModelRuntimeError("model service unavailable")
        yield


class FullScheduler:
    class Runtime:
        def normalize_alias(self, value: str) -> str:
            return value

    runtime = Runtime()

    async def reject_if_queue_full(self, model: str) -> None:
        raise QueueFullError("request queue is full (128 queued)")

    @asynccontextmanager
    async def slot(self, request_id: str, model: str, **kwargs):
        raise QueueFullError("request queue is full (128 queued)")
        yield


def build_client(tmp_path: Path) -> TestClient:
    settings = Settings(metrics_db_path=str(tmp_path / "metrics.db"))
    return TestClient(create_app(settings, asr_runtime=FakeAsrRuntime()))


def assert_error(response, status_code: int, message: str) -> None:
    assert response.status_code == status_code
    assert response.json() == {"error": {"message": message}}


def test_chat_rejects_invalid_json_with_400(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.post(
        "/v1/chat/completions",
        content=b'{"model":',
        headers={"content-type": "application/json"},
    )

    assert_error(response, 400, "Request body must be valid JSON.")


def test_chat_rejects_non_object_json_with_400(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.post("/v1/chat/completions", json=[])

    assert_error(response, 400, "Request body must be a JSON object.")


def test_chat_rejects_malformed_messages_with_400(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gemma4:e2b", "messages": "hello"},
    )

    assert_error(response, 400, "messages is required and must be an array")


def test_chat_rejects_malformed_message_parts_with_400(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemma4:e2b",
            "messages": [{"role": "user", "content": ["hello"]}],
        },
    )

    assert_error(response, 400, "messages[0].content[0] must be an object")


def test_chat_rejects_malformed_priority_with_400(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemma4:e2b",
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {"priority": "background"},
        },
    )

    assert_error(
        response,
        400,
        "metadata.priority must be one of: critical, highest, high, default, low, lowest",
    )


def test_chat_rejects_non_string_priority_with_400(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemma4:e2b",
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {"priority": 1},
        },
    )

    assert_error(response, 400, "metadata.priority must be a string")


def test_chat_rejects_non_object_metadata_with_400(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemma4:e2b",
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": "priority=high",
        },
    )

    assert_error(response, 400, "metadata must be an object")


def test_responses_rejects_missing_model_with_400(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.post("/v1/responses", json={"input": "hello"})

    assert_error(response, 400, "model is required and must be a non-empty string")


def test_responses_rejects_malformed_input_with_400(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.post("/v1/responses", json={"model": "gemma4:e2b", "input": 3})

    assert_error(response, 400, "input must be a string or an array")


def test_responses_rejects_malformed_priority_with_400(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.post(
        "/v1/responses",
        json={
            "model": "gemma4:e2b",
            "input": "hello",
            "metadata": {"priority": "background"},
        },
    )

    assert_error(
        response,
        400,
        "metadata.priority must be one of: critical, highest, high, default, low, lowest",
    )


def test_chat_maps_model_runtime_errors_to_503(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    client.app.state.service.scheduler = FailingScheduler()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemma4:e2b",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert_error(response, 503, "model service unavailable")


def test_chat_maps_full_queue_to_429(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    client.app.state.service.scheduler = FullScheduler()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemma4:e2b",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert_error(response, 429, "request queue is full (128 queued)")


def test_streaming_chat_maps_full_queue_to_429(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    client.app.state.service.scheduler = FullScheduler()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemma4:e2b",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )

    assert_error(response, 429, "request queue is full (128 queued)")


def test_responses_maps_model_runtime_errors_to_503(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    client.app.state.service.scheduler = FailingScheduler()

    response = client.post(
        "/v1/responses",
        json={"model": "gemma4:e2b", "input": "hello"},
    )

    assert_error(response, 503, "model service unavailable")


def test_responses_maps_full_queue_to_429(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    client.app.state.service.scheduler = FullScheduler()

    response = client.post(
        "/v1/responses",
        json={"model": "gemma4:e2b", "input": "hello"},
    )

    assert_error(response, 429, "request queue is full (128 queued)")


def test_streaming_responses_maps_full_queue_to_429(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    client.app.state.service.scheduler = FullScheduler()

    response = client.post(
        "/v1/responses",
        json={"model": "gemma4:e2b", "input": "hello", "stream": True},
    )

    assert_error(response, 429, "request queue is full (128 queued)")
