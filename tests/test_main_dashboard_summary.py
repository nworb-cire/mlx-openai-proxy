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
