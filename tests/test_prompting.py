from __future__ import annotations

from pathlib import Path

import pytest

from mlx_openai_proxy.config import Settings
from mlx_openai_proxy.metrics_store import MetricsStore
from mlx_openai_proxy.prompting import build_phase2_messages
from mlx_openai_proxy.service import ProxyService


def test_build_phase2_messages_omits_inline_image_payloads() -> None:
    data_url = "data:image/png;base64," + ("A" * 8192)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image as JSON."},
                {
                    "type": "image_url",
                    "image_url": {"url": data_url, "detail": "high"},
                },
            ],
        }
    ]

    phase2_messages = build_phase2_messages(
        messages,
        schema={"type": "object", "properties": {"caption": {"type": "string"}}},
        canonical_answer="A gray cat sitting on a chair.",
    )

    serialized_prompt = phase2_messages[1]["content"]
    assert isinstance(serialized_prompt, str)
    assert "Describe this image as JSON." in serialized_prompt
    assert data_url not in serialized_prompt
    assert '"image":"<omitted>"' in serialized_prompt
    assert '"detail":"high"' in serialized_prompt


class CaptureBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def close(self) -> None:
        return None

    async def post_json(self, path: str, body: dict[str, object]) -> dict[str, object]:
        self.calls.append(body)
        if len(self.calls) == 1:
            return {
                "id": "phase1",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "A gray cat sitting on a chair."},
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }
        return {
            "id": "phase2",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"caption":"A gray cat"}'},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }


@pytest.mark.asyncio
async def test_structured_flow_does_not_forward_data_url_to_formatter(tmp_path: Path) -> None:
    backend = CaptureBackend()
    settings = Settings(metrics_db_path=str(tmp_path / "metrics.db"))
    metrics = MetricsStore(settings.metrics_db_path)
    service = ProxyService(settings, backend, metrics)

    data_url = "data:image/png;base64," + ("B" * 16384)
    body = {
        "model": settings.default_model_alias,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image as JSON."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "caption",
                "schema": {
                    "type": "object",
                    "properties": {"caption": {"type": "string"}},
                    "required": ["caption"],
                    "additionalProperties": False,
                },
            },
        },
    }

    response = await service._chat_structured_nonstream_without_metrics(body)

    assert response["choices"][0]["message"]["content"] == '{"caption":"A gray cat"}'
    assert len(backend.calls) == 2
    formatter_prompt = backend.calls[1]["messages"][1]["content"]
    assert isinstance(formatter_prompt, str)
    assert data_url not in formatter_prompt
    assert '"image":"<omitted>"' in formatter_prompt
