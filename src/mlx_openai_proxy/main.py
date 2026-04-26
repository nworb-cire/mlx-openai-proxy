from __future__ import annotations

import argparse
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .backend import BackendClient, BackendError
from .config import Settings
from .dashboard import dashboard_html
from .logging_utils import configure_logging
from .metrics_store import MetricsStore
from .model_runtime import ModelRuntimeManager
from .model_scheduler import ModelScheduler
from .service import ActiveRequestTimeoutError, ProxyService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)

    backend = BackendClient(
        base_url=settings.backend_base_url,
        timeout_seconds=settings.backend_timeout_seconds,
    )
    metrics = MetricsStore(settings.metrics_db_path)
    runtime = ModelRuntimeManager(settings)
    scheduler = ModelScheduler(runtime, settings.default_model_alias)
    service = ProxyService(settings, backend, metrics, scheduler=scheduler)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            await runtime.normalize_residency()
            yield
        finally:
            await backend.close()

    app = FastAPI(title="MLX OpenAI Proxy", lifespan=lifespan)
    app.state.settings = settings
    app.state.service = service
    app.state.metrics = metrics
    app.state.runtime = runtime
    app.state.scheduler = scheduler

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return await service.health()

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        return await service.models()

    @app.get("/admin/dashboard", response_class=HTMLResponse)
    async def admin_dashboard() -> str:
        return dashboard_html()

    @app.get("/admin/api/summary")
    async def admin_summary() -> dict[str, object]:
        summary = metrics.get_summary()
        try:
            summary["active_model"] = runtime.current_active_alias()
            summary["loaded_models"] = await runtime.current_loaded_aliases()
            summary["loaded_model"] = summary["active_model"]
        except Exception:
            summary["active_model"] = None
            summary["loaded_models"] = []
            summary["loaded_model"] = None
        return summary

    @app.get("/admin/api/active")
    async def admin_active() -> dict[str, object]:
        return {"items": metrics.get_active_requests()}

    @app.get("/admin/api/history")
    async def admin_history(limit: int = 100) -> dict[str, object]:
        return {"items": metrics.get_history(limit=max(1, min(limit, 1000)))}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        try:
            return await service.chat(body)
        except ActiveRequestTimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except BackendError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/responses")
    async def responses(request: Request):
        body = await request.json()
        try:
            return await service.responses(body)
        except ActiveRequestTimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except BackendError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code, content={"error": {"message": exc.detail}}
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MLX OpenAI proxy")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    settings = Settings()
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
