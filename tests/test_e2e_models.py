from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest


pytestmark = pytest.mark.e2e


def _e2e_enabled() -> bool:
    return os.environ.get("MLX_PROXY_RUN_E2E_MODELS") == "1"


def _base_url() -> str:
    return os.environ.get("MLX_PROXY_E2E_BASE_URL", "http://127.0.0.1:8080").rstrip(
        "/"
    )


def _model_config_path() -> Path:
    return Path(
        os.environ.get("MLX_PROXY_MODEL_CONFIG_PATH", "config/models.json")
    ).resolve()


def _configured_chat_models() -> list[str]:
    with _model_config_path().open(encoding="utf-8") as handle:
        data = json.load(handle)
    assert isinstance(data, list), "model config must be a JSON list"
    aliases = [item["alias"] for item in data]
    assert aliases, "model config must contain at least one model"
    return aliases


def _chat_payload(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Reply with exactly this word and no punctuation: ready"
                ),
            }
        ],
        "temperature": 0,
        "max_tokens": 8,
    }


@pytest.fixture(scope="module")
def live_client() -> httpx.Client:
    if not _e2e_enabled():
        pytest.skip("set MLX_PROXY_RUN_E2E_MODELS=1 to run live model e2e tests")
    with httpx.Client(base_url=_base_url(), timeout=900.0) as client:
        try:
            response = client.get("/healthz")
        except httpx.TransportError as exc:
            pytest.fail(f"proxy is not reachable at {_base_url()}: {exc}")
        assert response.status_code == 200, response.text
        assert response.json().get("ok") is True
        yield client


def test_live_api_advertises_every_configured_chat_model(
    live_client: httpx.Client,
) -> None:
    response = live_client.get("/v1/models")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["object"] == "list"
    advertised = {item["id"] for item in payload["data"]}
    for model in _configured_chat_models():
        assert model in advertised


@pytest.mark.parametrize("model", _configured_chat_models())
def test_live_chat_completion_serves_configured_model(
    live_client: httpx.Client, model: str
) -> None:
    response = live_client.post("/v1/chat/completions", json=_chat_payload(model))

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["model"] == model
    assert payload["choices"][0]["finish_reason"] in {"stop", "length"}

    message = payload["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert isinstance(message.get("content"), str)
    assert message["content"].strip()

    usage = payload.get("usage") or {}
    assert usage.get("prompt_tokens", 0) > 0
    assert usage.get("completion_tokens", 0) > 0


@pytest.mark.parametrize("model", _configured_chat_models())
def test_live_responses_api_serves_configured_model(
    live_client: httpx.Client, model: str
) -> None:
    response = live_client.post(
        "/v1/responses",
        json={
            "model": model,
            "input": "Reply with exactly this word and no punctuation: ready",
            "temperature": 0,
            "max_output_tokens": 8,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["object"] == "response"
    assert payload["model"] == model
    assert payload["status"] == "completed"
    assert payload["output_text"].strip()
