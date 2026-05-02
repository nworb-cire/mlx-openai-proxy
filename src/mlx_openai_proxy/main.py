from __future__ import annotations

import argparse
import json
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from .asr import AsrError, AsrRuntime, ParakeetAsrRuntime, ResidentAsrService
from .backend import BackendClient, BackendError
from .config import Settings
from .dashboard import dashboard_html
from .logging_utils import configure_logging
from .metrics_store import MetricsStore
from .model_runtime import ModelRuntimeManager
from .model_runtime import ModelRuntimeError
from .model_scheduler import ModelScheduler
from .service import ActiveRequestTimeoutError, ProxyService


async def _json_object_body(request: Request) -> dict[str, object]:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Request body must be valid JSON.",
        ) from exc
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400,
            detail="Request body must be a JSON object.",
        )
    return body


def create_app(
    settings: Settings | None = None, *, asr_runtime: AsrRuntime | None = None
) -> FastAPI:
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
    asr = ResidentAsrService(
        settings.asr,
        asr_runtime or ParakeetAsrRuntime(settings.asr),
        metrics,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            await runtime.normalize_residency()
            await asr.load()
            yield
        finally:
            await asr.close()
            await backend.close()

    app = FastAPI(title="MLX OpenAI Proxy", lifespan=lifespan)
    app.state.settings = settings
    app.state.service = service
    app.state.metrics = metrics
    app.state.runtime = runtime
    app.state.scheduler = scheduler
    app.state.asr = asr

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        health = await service.health()
        health["asr"] = asr.health()
        return health

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        payload = await service.models()
        data = list(payload.get("data", []))
        data.append(asr.advertised_model())
        return {"object": "list", "data": data}

    @app.get("/admin/dashboard", response_class=HTMLResponse)
    async def admin_dashboard() -> str:
        return dashboard_html()

    @app.get("/admin/api/summary")
    async def admin_summary() -> dict[str, object]:
        summary = metrics.get_summary()
        try:
            summary["active_model"] = runtime.current_active_alias()
            if summary["active_count"]:
                summary["loaded_models"] = (
                    [summary["active_model"]] if summary["active_model"] else []
                )
            else:
                summary["loaded_models"] = await runtime.current_loaded_aliases()
            summary["loaded_model"] = summary["active_model"]
            summary["asr"] = asr.health()
        except Exception:
            summary["active_model"] = None
            summary["loaded_models"] = []
            summary["loaded_model"] = None
            summary["asr"] = {"loaded": False, "model": settings.asr.alias}
        return summary

    @app.get("/admin/api/active")
    async def admin_active() -> dict[str, object]:
        return {"items": metrics.get_active_requests()}

    @app.get("/admin/api/history")
    async def admin_history(limit: int = 100) -> dict[str, object]:
        return {"items": metrics.get_history(limit=max(1, min(limit, 1000)))}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await _json_object_body(request)
        try:
            return await service.chat(body)
        except ActiveRequestTimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except BackendError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ModelRuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/responses")
    async def responses(request: Request):
        body = await _json_object_body(request)
        try:
            return await service.responses(body)
        except ActiveRequestTimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except BackendError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ModelRuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/audio/transcriptions")
    async def audio_transcriptions(
        file: UploadFile = File(...),
        model: str | None = Form(default=None),
        language: str | None = Form(default=None),
        prompt: str | None = Form(default=None),
        response_format: str | None = Form(default=None),
    ):
        del language, prompt
        try:
            result = await asr.transcribe_upload(
                upload=file,
                model=model,
                response_format=response_format,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except AsrError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if isinstance(result, str):
            return PlainTextResponse(result)
        return result

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket):
        model = websocket.query_params.get("model")
        if model and model not in {settings.asr.alias, settings.asr.model_id}:
            await websocket.accept()
            await websocket.send_json(
                {"type": "error", "error": {"message": f"unsupported ASR model '{model}'"}}
            )
            await websocket.close(code=1008)
            return
        await asr.realtime(websocket)

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
