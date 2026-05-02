from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx


class BackendError(RuntimeError):
    pass


class BackendClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def get_json(self, path: str) -> dict[str, Any]:
        try:
            response = await self._client.get(path)
        except httpx.TransportError as exc:
            raise BackendError(str(exc)) from exc
        self._raise_for_status(response)
        return response.json()

    async def post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(path, json=body)
        except httpx.TransportError as exc:
            raise BackendError(str(exc)) from exc
        self._raise_for_status(response)
        return response.json()

    async def post_stream(
        self, path: str, body: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any] | None]:
        try:
            async with self._client.stream("POST", path, json=body) as response:
                await self._raise_for_status_async(response)
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        yield None
                        break
                    yield json.loads(payload)
        except httpx.TransportError as exc:
            raise BackendError(str(exc)) from exc

    async def proxy_stream(
        self, path: str, body: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        try:
            async with self._client.stream("POST", path, json=body) as response:
                await self._raise_for_status_async(response)
                async for chunk in response.aiter_raw():
                    yield chunk
        except httpx.TransportError as exc:
            raise BackendError(str(exc)) from exc

    async def post_raw(self, path: str, body: dict[str, Any]) -> httpx.Response:
        try:
            response = await self._client.post(path, json=body)
        except httpx.TransportError as exc:
            raise BackendError(str(exc)) from exc
        self._raise_for_status(response)
        return response

    @staticmethod
    async def _raise_for_status_async(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.text
            except httpx.ResponseNotRead:
                detail = (await exc.response.aread()).decode("utf-8", errors="replace")
            raise BackendError(detail) from exc

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.text
            except httpx.ResponseNotRead:
                detail = ""
            raise BackendError(detail) from exc
