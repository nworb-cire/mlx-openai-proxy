from __future__ import annotations

from fastapi.testclient import TestClient

from mlx_openai_proxy.main import create_app


def test_admin_summary_includes_loaded_model() -> None:
    app = create_app()

    def fake_current_active_alias() -> str:
        return "gemma4:26b"

    async def fake_current_loaded_aliases() -> list[str]:
        return ["gemma4:26b"]

    async def fake_normalize_residency(preferred_alias: str | None = None) -> str:
        return "gemma4:26b"

    app.state.runtime.current_active_alias = fake_current_active_alias  # type: ignore[method-assign]
    app.state.runtime.current_loaded_aliases = fake_current_loaded_aliases  # type: ignore[method-assign]
    app.state.runtime.normalize_residency = fake_normalize_residency  # type: ignore[method-assign]

    with TestClient(app) as client:
        response = client.get("/admin/api/summary")

    assert response.status_code == 200
    assert response.json()["loaded_model"] == "gemma4:26b"
    assert response.json()["active_model"] == "gemma4:26b"
    assert response.json()["loaded_models"] == ["gemma4:26b"]


def test_admin_summary_does_not_probe_loaded_models_during_active_request() -> None:
    app = create_app()

    def fake_current_active_alias() -> str:
        return "gemma4:e2b"

    async def fake_current_loaded_aliases() -> list[str]:
        raise AssertionError("dashboard summary should not shell out while active")

    async def fake_normalize_residency(preferred_alias: str | None = None) -> str:
        return "gemma4:e2b"

    app.state.runtime.current_active_alias = fake_current_active_alias  # type: ignore[method-assign]
    app.state.runtime.current_loaded_aliases = fake_current_loaded_aliases  # type: ignore[method-assign]
    app.state.runtime.normalize_residency = fake_normalize_residency  # type: ignore[method-assign]

    app.state.metrics.start_request(
        {
            "path": "/v1/chat/completions",
            "model": "gemma4:e2b",
            "stream": False,
            "has_schema": True,
            "has_images": True,
            "asks_for_reasoning": True,
            "input_messages": 1,
            "input_chars": 1646,
            "input_image_count": 2,
        }
    )

    with TestClient(app) as client:
        response = client.get("/admin/api/summary")

    assert response.status_code == 200
    assert response.json()["active_count"] == 1
    assert response.json()["loaded_model"] == "gemma4:e2b"
    assert response.json()["loaded_models"] == ["gemma4:e2b"]
