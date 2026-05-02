from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.responses import StreamingResponse

from mlx_openai_proxy.config import Settings
from mlx_openai_proxy.metrics_store import MetricsStore
from mlx_openai_proxy.responses_bridge import responses_request_to_chat
from mlx_openai_proxy.service import ProxyService


def _chat_chunk(
    *,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    model: str = "test-model",
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }


async def _stream_events(response: StreamingResponse) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    async for chunk in response.body_iterator:
        text = chunk.decode("utf-8")
        for frame in text.strip().split("\n\n"):
            if not frame.startswith("data: "):
                continue
            payload = frame.removeprefix("data: ")
            if payload == "[DONE]":
                continue
            events.append(json.loads(payload))
    return events


class ReasoningStreamBackend:
    def __init__(self) -> None:
        self.stream_bodies: list[dict[str, Any]] = []

    async def close(self) -> None:
        return None

    async def post_stream(self, path: str, body: dict[str, Any]):
        self.stream_bodies.append(body)
        yield _chat_chunk(delta={"role": "assistant"}, model=body["model"])
        yield _chat_chunk(
            delta={"reasoning_content": "private reasoning"},
            model=body["model"],
        )
        yield _chat_chunk(delta={"content": "public answer"}, model=body["model"])
        yield {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": body["model"],
            "choices": [],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 5,
                "total_tokens": 8,
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        }
        yield _chat_chunk(delta={}, finish_reason="stop", model=body["model"])


def test_responses_text_format_and_reasoning_map_to_internal_chat_shape() -> None:
    chat_body = responses_request_to_chat(
        {
            "model": "local-model",
            "input": "Return a sum.",
            "stream": True,
            "max_output_tokens": 64,
            "reasoning": {"effort": "low", "summary": "auto"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "sum",
                    "schema": {
                        "type": "object",
                        "properties": {"sum": {"type": "number"}},
                        "required": ["sum"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            },
        }
    )

    assert chat_body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "sum",
            "schema": {
                "type": "object",
                "properties": {"sum": {"type": "number"}},
                "required": ["sum"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
    assert chat_body["reasoning_effort"] == "low"
    assert "reasoning" not in chat_body
    assert chat_body["max_tokens"] == 64
    assert "text" not in chat_body


def test_responses_response_format_is_not_a_supported_structured_output_alias() -> None:
    with pytest.raises(ValueError, match="text\\.format"):
        responses_request_to_chat(
            {
                "model": "local-model",
                "input": "Return a sum.",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "legacy", "schema": {"type": "object"}},
                },
            }
        )


@pytest.mark.asyncio
async def test_chat_rejects_responses_reasoning_object(tmp_path) -> None:
    backend = ReasoningStreamBackend()
    settings = Settings(metrics_db_path=str(tmp_path / "metrics.db"))
    service = ProxyService(settings, backend, MetricsStore(settings.metrics_db_path))

    with pytest.raises(ValueError, match="reasoning_effort"):
        await service.chat(
            {
                "model": settings.default_model_alias,
                "messages": [{"role": "user", "content": "hello"}],
                "reasoning": {"effort": "low"},
            }
        )


@pytest.mark.asyncio
async def test_chat_nonstream_does_not_expose_provider_reasoning(tmp_path) -> None:
    backend = ReasoningStreamBackend()
    settings = Settings(metrics_db_path=str(tmp_path / "metrics.db"))
    service = ProxyService(settings, backend, MetricsStore(settings.metrics_db_path))

    response = await service.chat(
        {
            "model": settings.default_model_alias,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        }
    )

    message = response["choices"][0]["message"]
    assert message["content"] == "public answer"
    assert "reasoning_content" not in message
    assert "_reasoning_content" not in message
    assert response["usage"]["completion_tokens_details"]["reasoning_tokens"] == 2
    assert backend.stream_bodies[0]["reasoning_effort"] == "none"

    history = service.metrics.get_history(limit=1)
    assert history[0]["reasoning_chars"] == len("private reasoning")


@pytest.mark.asyncio
async def test_chat_stream_filters_reasoning_deltas(tmp_path) -> None:
    backend = ReasoningStreamBackend()
    settings = Settings(metrics_db_path=str(tmp_path / "metrics.db"))
    service = ProxyService(settings, backend, MetricsStore(settings.metrics_db_path))

    response = await service.chat(
        {
            "model": settings.default_model_alias,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }
    )

    assert isinstance(response, StreamingResponse)
    events = await _stream_events(response)

    serialized = json.dumps(events)
    assert "reasoning_content" not in serialized
    assert "private reasoning" not in serialized
    assert any(
        choice.get("delta", {}).get("content") == "public answer"
        for event in events
        for choice in event.get("choices", [])
    )


@pytest.mark.asyncio
async def test_responses_nonstream_has_no_reasoning_output_item(tmp_path) -> None:
    backend = ReasoningStreamBackend()
    settings = Settings(metrics_db_path=str(tmp_path / "metrics.db"))
    service = ProxyService(settings, backend, MetricsStore(settings.metrics_db_path))

    response = await service.responses(
        {
            "model": settings.default_model_alias,
            "input": "hello",
            "reasoning": {"effort": "low", "summary": "auto"},
        }
    )

    assert response["output_text"] == "public answer"
    assert [item["type"] for item in response["output"]] == ["message"]
    assert "private reasoning" not in json.dumps(response)
    assert backend.stream_bodies[0]["reasoning_effort"] == "low"
    assert "reasoning" not in backend.stream_bodies[0]


@pytest.mark.asyncio
async def test_responses_stream_has_only_output_text_deltas(tmp_path) -> None:
    backend = ReasoningStreamBackend()
    settings = Settings(metrics_db_path=str(tmp_path / "metrics.db"))
    service = ProxyService(settings, backend, MetricsStore(settings.metrics_db_path))

    response = await service.responses(
        {
            "model": settings.default_model_alias,
            "input": "hello",
            "stream": True,
        }
    )

    assert isinstance(response, StreamingResponse)
    events = await _stream_events(response)
    event_types = [event["type"] for event in events]
    assert "response.reasoning_text.delta" not in event_types
    assert "response.output_text.delta" in event_types
    assert "private reasoning" not in json.dumps(events)


class StructuredResponsesBackend:
    def __init__(self) -> None:
        self.stream_bodies: list[dict[str, Any]] = []
        self.json_bodies: list[dict[str, Any]] = []
        self.chat_content = '{"sum":3}'

    async def close(self) -> None:
        return None

    async def post_stream(self, path: str, body: dict[str, Any]):
        self.stream_bodies.append(body)
        yield _chat_chunk(delta={"role": "assistant"}, model=body["model"])
        yield _chat_chunk(delta={"content": self.chat_content}, model=body["model"])
        yield {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": body["model"],
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }
        yield _chat_chunk(delta={}, finish_reason="stop", model=body["model"])

    async def post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        self.json_bodies.append(body)
        return {
            "id": "phase2",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"sum":3}'},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }


@pytest.mark.asyncio
async def test_responses_text_format_drives_structured_output(tmp_path) -> None:
    backend = StructuredResponsesBackend()
    settings = Settings(metrics_db_path=str(tmp_path / "metrics.db"))
    service = ProxyService(settings, backend, MetricsStore(settings.metrics_db_path))

    response = await service.responses(
        {
            "model": settings.default_model_alias,
            "input": "What is 1 + 2?",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "sum",
                    "schema": {
                        "type": "object",
                        "properties": {"sum": {"type": "number"}},
                        "required": ["sum"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            },
        }
    )

    assert response["output_text"] == '{"sum":3}'
    assert backend.stream_bodies[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "sum",
            "schema": {
                "type": "object",
                "properties": {"sum": {"type": "number"}},
                "required": ["sum"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
    assert backend.stream_bodies[0]["reasoning_effort"] == "none"
    assert backend.json_bodies == []


@pytest.mark.asyncio
async def test_responses_text_format_with_reasoning_uses_formatter(tmp_path) -> None:
    backend = StructuredResponsesBackend()
    backend.chat_content = "The sum is 3."
    settings = Settings(metrics_db_path=str(tmp_path / "metrics.db"))
    service = ProxyService(settings, backend, MetricsStore(settings.metrics_db_path))

    response = await service.responses(
        {
            "model": settings.default_model_alias,
            "input": "What is 1 + 2?",
            "reasoning": {"effort": "low"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "sum",
                    "schema": {
                        "type": "object",
                        "properties": {"sum": {"type": "number"}},
                        "required": ["sum"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            },
        }
    )

    assert response["output_text"] == '{"sum":3}'
    assert "response_format" not in backend.stream_bodies[0]
    assert backend.json_bodies[0]["response_format"]["type"] == "json_schema"


class JsonObjectBackend:
    def __init__(self) -> None:
        self.stream_bodies: list[dict[str, Any]] = []
        self.json_bodies: list[dict[str, Any]] = []
        self.stream_content = "The answer is ok true."

    async def close(self) -> None:
        return None

    async def post_stream(self, path: str, body: dict[str, Any]):
        self.stream_bodies.append(body)
        yield _chat_chunk(delta={"role": "assistant"}, model=body["model"])
        yield _chat_chunk(delta={"content": self.stream_content}, model=body["model"])
        yield {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": body["model"],
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }
        yield _chat_chunk(delta={}, finish_reason="stop", model=body["model"])

    async def post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        self.json_bodies.append(body)
        return {
            "id": "phase2",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '```json\n{"ok": true}\n```',
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }


@pytest.mark.asyncio
async def test_chat_json_object_mode_uses_single_stage_when_output_is_valid(
    tmp_path,
) -> None:
    backend = JsonObjectBackend()
    backend.stream_content = '{"ok": true}'
    settings = Settings(metrics_db_path=str(tmp_path / "metrics.db"))
    service = ProxyService(settings, backend, MetricsStore(settings.metrics_db_path))

    response = await service.chat(
        {
            "model": settings.default_model_alias,
            "messages": [{"role": "user", "content": "Return JSON."}],
            "response_format": {"type": "json_object"},
            "max_tokens": 64,
            "temperature": 0,
        }
    )

    assert response["choices"][0]["message"]["content"] == '{"ok":true}'
    assert "response_format" not in backend.stream_bodies[0]
    assert backend.stream_bodies[0]["reasoning_effort"] == "none"
    assert backend.json_bodies == []


@pytest.mark.asyncio
async def test_chat_json_object_mode_repairs_invalid_single_stage_output(
    tmp_path,
) -> None:
    backend = JsonObjectBackend()
    settings = Settings(metrics_db_path=str(tmp_path / "metrics.db"))
    service = ProxyService(settings, backend, MetricsStore(settings.metrics_db_path))

    response = await service.chat(
        {
            "model": settings.default_model_alias,
            "messages": [{"role": "user", "content": "Return JSON."}],
            "response_format": {"type": "json_object"},
        }
    )

    assert response["choices"][0]["message"]["content"] == '{"ok":true}'
    assert "response_format" not in backend.stream_bodies[0]
    assert backend.stream_bodies[0]["reasoning_effort"] == "none"
    assert backend.stream_bodies[0]["messages"][-1]["role"] == "user"
    assert "valid JSON object" in backend.stream_bodies[0]["messages"][-1]["content"]
    assert "response_format" not in backend.json_bodies[0]
    assert backend.json_bodies[0]["reasoning_effort"] == "none"
    assert "Convert the canonical answer" in backend.json_bodies[0]["messages"][0][
        "content"
    ]


@pytest.mark.asyncio
async def test_responses_json_object_mode_is_emulated_for_backend(tmp_path) -> None:
    backend = JsonObjectBackend()
    settings = Settings(metrics_db_path=str(tmp_path / "metrics.db"))
    service = ProxyService(settings, backend, MetricsStore(settings.metrics_db_path))

    response = await service.responses(
        {
            "model": settings.default_model_alias,
            "input": "Return JSON.",
            "text": {"format": {"type": "json_object"}},
        }
    )

    assert response["output_text"] == '{"ok":true}'
    assert "response_format" not in backend.stream_bodies[0]
    assert "response_format" not in backend.json_bodies[0]


@pytest.mark.asyncio
async def test_image_json_object_mode_preserves_image_only_in_phase1(tmp_path) -> None:
    backend = JsonObjectBackend()
    settings = Settings(metrics_db_path=str(tmp_path / "metrics.db"))
    service = ProxyService(settings, backend, MetricsStore(settings.metrics_db_path))
    image_url = "data:image/png;base64,AAAA"

    response = await service.responses(
        {
            "model": settings.default_model_alias,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this as JSON."},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
            "text": {"format": {"type": "json_object"}},
        }
    )

    assert response["output_text"] == '{"ok":true}'
    phase1_messages = backend.stream_bodies[0]["messages"]
    assert phase1_messages[0]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": image_url},
    }
    assert phase1_messages[-1]["role"] == "user"
    assert "valid JSON object" in phase1_messages[-1]["content"]
    formatter_prompt = backend.json_bodies[0]["messages"][0]["content"]
    assert image_url not in formatter_prompt
    assert '"image":"<omitted>"' in formatter_prompt
