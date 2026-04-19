from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi.responses import StreamingResponse

from .backend import BackendClient, BackendError
from .classifier import ExecutionPath, RequestClassification, classify_chat_request
from .config import ReasoningVisibility, Settings
from .images import normalize_chat_images, normalize_responses_input
from .logging_utils import log_event
from .metrics_store import MetricsStore
from .model_scheduler import ModelScheduler
from .prompting import (
    build_phase1_messages,
    build_phase2_messages,
    build_repair_messages,
)
from .responses_bridge import chat_response_to_responses, responses_request_to_chat
from .schema_utils import (
    chunk_text,
    compact_json,
    extract_json_schema,
    validate_incoming_schema,
    validate_json_text,
    validation_error_message,
)

GEMMA_THOUGHT_OPEN = "<|channel>thought"
GEMMA_THOUGHT_CLOSE = "<channel|>"


class GemmaThoughtStreamParser:
    def __init__(self) -> None:
        self._buffer = ""
        self._leading_whitespace = ""
        self._state = "prefix"
        self._passthrough = False

    def feed(self, text: str) -> tuple[list[str], list[str]]:
        reasoning: list[str] = []
        content: list[str] = []
        self._buffer += text
        while self._buffer:
            if self._passthrough:
                content.append(self._buffer)
                self._buffer = ""
                break
            if self._state == "prefix":
                while self._buffer and self._buffer[0].isspace():
                    self._leading_whitespace += self._buffer[0]
                    self._buffer = self._buffer[1:]
                if not self._buffer:
                    break
                if self._buffer.startswith(GEMMA_THOUGHT_OPEN):
                    if self._leading_whitespace:
                        reasoning.append(self._leading_whitespace)
                        self._leading_whitespace = ""
                    self._buffer = self._buffer[len(GEMMA_THOUGHT_OPEN) :]
                    if self._buffer.startswith("\n"):
                        self._buffer = self._buffer[1:]
                    self._state = "thought"
                    continue
                if GEMMA_THOUGHT_OPEN.startswith(self._buffer):
                    break
                if self._leading_whitespace:
                    self._buffer = f"{self._leading_whitespace}{self._buffer}"
                    self._leading_whitespace = ""
                self._passthrough = True
                continue
            if self._state == "thought":
                close_index = self._buffer.find(GEMMA_THOUGHT_CLOSE)
                if close_index != -1:
                    thought_text = self._buffer[:close_index]
                    if thought_text:
                        reasoning.append(thought_text)
                    self._buffer = self._buffer[close_index + len(GEMMA_THOUGHT_CLOSE) :]
                    if self._buffer.startswith("\n"):
                        self._buffer = self._buffer[1:]
                    self._state = "content"
                    continue
                safe_len = max(0, len(self._buffer) - len(GEMMA_THOUGHT_CLOSE) + 1)
                if safe_len > 0:
                    reasoning.append(self._buffer[:safe_len])
                    self._buffer = self._buffer[safe_len:]
                break
            content.append(self._buffer)
            self._buffer = ""
            break
        return reasoning, content

    def finish(self) -> tuple[list[str], list[str]]:
        if not self._buffer and not self._leading_whitespace:
            return [], []
        if self._passthrough or self._state in {"prefix", "content"}:
            content = [f"{self._leading_whitespace}{self._buffer}"]
            self._leading_whitespace = ""
            self._buffer = ""
            return [], content
        reasoning = [f"{self._leading_whitespace}{self._buffer}"]
        self._leading_whitespace = ""
        self._buffer = ""
        return reasoning, []


class ProxyService:
    def __init__(
        self,
        settings: Settings,
        backend: BackendClient,
        metrics: MetricsStore,
        *,
        scheduler: ModelScheduler | None = None,
    ) -> None:
        self.settings = settings
        self.backend = backend
        self.metrics = metrics
        self.scheduler = scheduler
        self.logger = logging.getLogger("mlx_openai_proxy")
        self._upstream_slots = asyncio.Semaphore(max(1, settings.max_upstream_concurrency))

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "backend_base_url": self.settings.backend_base_url}

    async def models(self) -> dict[str, Any]:
        if self.scheduler is not None:
            return self.scheduler.runtime.advertised_models()
        return {
            "object": "list",
            "data": [
                {
                    "id": self.settings.default_model_alias,
                    "object": "model",
                    "owned_by": "organization_owner",
                },
                {
                    "id": self.settings.burst_model_alias,
                    "object": "model",
                    "owned_by": "organization_owner",
                },
            ],
        }

    async def chat(self, body: dict[str, Any]) -> dict[str, Any] | StreamingResponse:
        body = normalize_chat_images(body, self.settings.max_inline_image_bytes)
        body = self._prepare_reasoning_request(body)
        classification = classify_chat_request(body, self.settings.schema_mode)
        request_id = self.metrics.start_request(
            self._build_request_record(
                path="/v1/chat/completions",
                body=body,
                classification=classification,
            )
        )
        log_event(
            self.logger,
            "request_classified",
            path="/v1/chat/completions",
            execution_path=classification.execution_path,
            reason=classification.reason,
            model=body.get("model"),
            stream=bool(body.get("stream")),
        )
        try:
            if classification.execution_path == ExecutionPath.PASSTHROUGH:
                if body.get("stream"):
                    return StreamingResponse(
                        self._passthrough_chat_stream(body, request_id),
                        media_type="text/event-stream",
                    )
                async with self._service_slot(request_id, body.get("model")):
                    response = await self._buffered_chat_completion(body, request_id=request_id)
                self._complete_chat_metrics(request_id, response)
                return response

            if classification.execution_path == ExecutionPath.STRICT_STRUCTURED_FAST_PATH:
                if body.get("stream"):
                    return StreamingResponse(
                        self._passthrough_chat_stream(body, request_id),
                        media_type="text/event-stream",
                    )
                async with self._service_slot(request_id, body.get("model")):
                    response = await self._buffered_chat_completion(body, request_id=request_id)
                self._complete_chat_metrics(request_id, response)
                return response

            if body.get("stream"):
                return StreamingResponse(
                    self._chat_structured_stream(body, classification, request_id),
                    media_type="text/event-stream",
                )
            return await self._chat_structured_nonstream(body, classification, request_id)
        except Exception as exc:
            self.metrics.fail_request(request_id, error_message=str(exc))
            raise

    async def responses(self, body: dict[str, Any]) -> dict[str, Any] | StreamingResponse:
        body = normalize_responses_input(body, self.settings.max_inline_image_bytes)
        chat_body = responses_request_to_chat(body)
        chat_body = self._prepare_reasoning_request(chat_body)
        classification = classify_chat_request(chat_body, self.settings.schema_mode)
        request_id = self.metrics.start_request(
            self._build_request_record(
                path="/v1/responses",
                body=chat_body,
                classification=classification,
            )
        )
        if body.get("stream"):
            return StreamingResponse(
                self._responses_stream(chat_body, request_id),
                media_type="text/event-stream",
            )
        try:
            chat_response = await self._chat_no_metrics(chat_body, request_id=request_id)
            response = chat_response_to_responses(chat_response)
            self._complete_responses_metrics(request_id, response, chat_response)
            return response
        except Exception as exc:
            self.metrics.fail_request(request_id, error_message=str(exc))
            raise

    async def _chat_structured_nonstream(
        self,
        original_body: dict[str, Any],
        classification: RequestClassification,
        request_id: str,
    ) -> dict[str, Any]:
        schema = extract_json_schema(original_body)
        if schema is None:
            raise ValueError("Structured execution requires a JSON schema")
        validate_incoming_schema(schema)

        async with self._service_slot(request_id, original_body.get("model")):
            phase1_started = time.perf_counter()
            phase1_response = await self._buffered_chat_completion(
                self._make_phase1_body(original_body, stream=False),
                request_id=request_id,
            )
            phase1_elapsed = time.perf_counter() - phase1_started
            phase1_message = phase1_response["choices"][0]["message"]
            canonical_answer = phase1_message.get("content", "")
            reasoning_content = phase1_message.get("reasoning_content", "")

            phase2_started = time.perf_counter()
            phase2_response, final_json_text = await self._run_phase2(
                original_body=original_body,
                schema=schema,
                canonical_answer=canonical_answer,
            )
            phase2_elapsed = time.perf_counter() - phase2_started

        merged = copy.deepcopy(phase2_response)
        merged_choice = merged["choices"][0]["message"]
        merged_choice["content"] = final_json_text
        if self.settings.reasoning_visibility != ReasoningVisibility.OFF:
            merged_choice["reasoning_content"] = reasoning_content
        merged["usage"] = self._merge_usage(
            phase1_response.get("usage", {}),
            phase2_response.get("usage", {}),
        )

        log_event(
            self.logger,
            "structured_nonstream_completed",
            model=original_body.get("model"),
            phase1_latency_s=round(phase1_elapsed, 3),
            phase2_latency_s=round(phase2_elapsed, 3),
            execution_path=classification.execution_path,
        )
        self._complete_chat_metrics(
            request_id,
            merged,
            phase1_latency_ms=int(phase1_elapsed * 1000),
            phase2_latency_ms=int(phase2_elapsed * 1000),
        )
        return merged

    async def _chat_structured_stream(
        self,
        original_body: dict[str, Any],
        classification: RequestClassification,
        request_id: str,
    ) -> AsyncIterator[bytes]:
        schema = extract_json_schema(original_body)
        if schema is None:
            raise ValueError("Structured execution requires a JSON schema")
        validate_incoming_schema(schema)

        response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        model = original_body.get("model")
        stream_include_usage = bool(
            original_body.get("stream_options", {}).get("include_usage")
        )

        reasoning_visible: list[str] = []
        content_accumulator: list[str] = []
        phase1_usage: dict[str, Any] = {}
        phase1_started = time.perf_counter()

        yield self._sse(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )

        async with self._service_slot(request_id, original_body.get("model")):
            async for event in self.backend.post_stream(
                "/chat/completions",
                self._make_phase1_body(original_body, stream=True),
            ):
                if event is None:
                    break
                if "usage" in event:
                    phase1_usage = event.get("usage", {})
                choice = (event.get("choices") or [{}])[0]
                delta = choice.get("delta", {})
                reasoning_delta = delta.get("reasoning_content")
                if isinstance(reasoning_delta, str):
                    reasoning_visible.append(reasoning_delta)
                    self.metrics.append_progress(
                        request_id,
                        reasoning_delta=reasoning_delta,
                    )
                    if self.settings.reasoning_visibility != ReasoningVisibility.OFF:
                        yield self._sse(
                            {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"reasoning_content": reasoning_delta},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                content_delta = delta.get("content")
                if isinstance(content_delta, str):
                    content_accumulator.append(content_delta)
                    self.metrics.append_progress(
                        request_id,
                        output_delta=content_delta,
                    )

            canonical_answer = "".join(content_accumulator).strip()
            phase1_elapsed = time.perf_counter() - phase1_started
            phase2_started = time.perf_counter()
            phase2_response, final_json_text = await self._run_phase2(
                original_body=original_body,
                schema=schema,
                canonical_answer=canonical_answer,
            )
            phase2_elapsed = time.perf_counter() - phase2_started
            phase2_usage = phase2_response.get("usage", {})

        for chunk in chunk_text(final_json_text, self.settings.final_chunk_size):
            yield self._sse(
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
            )

        if stream_include_usage:
            yield self._sse(
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": self._merge_usage(phase1_usage, phase2_usage),
                }
            )

        yield self._sse(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        yield b"data: [DONE]\n\n"

        log_event(
            self.logger,
            "structured_stream_completed",
            model=model,
            execution_path=classification.execution_path,
            phase1_completion_tokens=phase1_usage.get("completion_tokens"),
            phase2_completion_tokens=phase2_usage.get("completion_tokens"),
        )
        self.metrics.complete_request(
            request_id,
            prompt_tokens=(phase1_usage.get("prompt_tokens") or 0)
            + (phase2_usage.get("prompt_tokens") or 0),
            completion_tokens=(phase1_usage.get("completion_tokens") or 0)
            + (phase2_usage.get("completion_tokens") or 0),
            reasoning_tokens=((phase1_usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
            + ((phase2_usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
            output_chars=len(final_json_text),
            reasoning_chars=len("".join(reasoning_visible)),
            phase1_latency_ms=int(phase1_elapsed * 1000),
            phase2_latency_ms=int(phase2_elapsed * 1000),
        )

    async def _responses_stream(self, chat_body: dict[str, Any], request_id: str) -> AsyncIterator[bytes]:
        response_id = f"resp_{uuid.uuid4().hex}"
        created = int(time.time())

        yield self._sse({"type": "response.created", "response": {"id": response_id, "object": "response"}})

        try:
            chat_response = await self._chat_no_metrics(chat_body, request_id=request_id)
            response_obj = chat_response_to_responses(chat_response)
        except Exception as exc:
            self.metrics.fail_request(request_id, error_message=str(exc))
            raise

        reasoning = chat_response["choices"][0]["message"].get("reasoning_content", "")
        if reasoning:
            for chunk in chunk_text(reasoning, self.settings.final_chunk_size):
                yield self._sse(
                    {
                        "type": "response.reasoning_text.delta",
                        "response_id": response_id,
                        "delta": chunk,
                    }
                )

        for chunk in chunk_text(response_obj["output_text"], self.settings.final_chunk_size):
            yield self._sse(
                {
                    "type": "response.output_text.delta",
                    "response_id": response_id,
                    "delta": chunk,
                }
            )

        response_obj["id"] = response_id
        response_obj["created_at"] = created
        yield self._sse({"type": "response.completed", "response": response_obj})
        yield b"data: [DONE]\n\n"
        self._complete_responses_metrics(request_id, response_obj, chat_response)

    async def _chat_no_metrics(
        self,
        body: dict[str, Any],
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        body = normalize_chat_images(body, self.settings.max_inline_image_bytes)
        body = self._prepare_reasoning_request(body)
        classification = classify_chat_request(body, self.settings.schema_mode)
        if classification.execution_path == ExecutionPath.PASSTHROUGH:
            if request_id is None:
                return await self.backend.post_json("/chat/completions", body)
            async with self._service_slot(request_id, body.get("model")):
                return await self._buffered_chat_completion(body, request_id=request_id)
        if classification.execution_path == ExecutionPath.STRICT_STRUCTURED_FAST_PATH:
            if request_id is None:
                return await self.backend.post_json("/chat/completions", body)
            async with self._service_slot(request_id, body.get("model")):
                return await self._buffered_chat_completion(body, request_id=request_id)
        return await self._chat_structured_nonstream_without_metrics(body, request_id=request_id)

    async def _chat_structured_nonstream_without_metrics(
        self,
        original_body: dict[str, Any],
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        schema = extract_json_schema(original_body)
        if schema is None:
            raise ValueError("Structured execution requires a JSON schema")
        validate_incoming_schema(schema)

        if request_id is None:
            phase1_response = await self.backend.post_json(
                "/chat/completions",
                self._make_phase1_body(original_body, stream=False),
            )
            phase1_message = phase1_response["choices"][0]["message"]
            canonical_answer = phase1_message.get("content", "")
            reasoning_content = phase1_message.get("reasoning_content", "")
            phase2_response, final_json_text = await self._run_phase2(
                original_body=original_body,
                schema=schema,
                canonical_answer=canonical_answer,
            )
        else:
            async with self._service_slot(request_id, original_body.get("model")):
                phase1_response = await self._buffered_chat_completion(
                    self._make_phase1_body(original_body, stream=False),
                    request_id=request_id,
                )
                phase1_message = phase1_response["choices"][0]["message"]
                canonical_answer = phase1_message.get("content", "")
                reasoning_content = phase1_message.get("reasoning_content", "")
                phase2_response, final_json_text = await self._run_phase2(
                    original_body=original_body,
                    schema=schema,
                    canonical_answer=canonical_answer,
                )
        merged = copy.deepcopy(phase2_response)
        merged_choice = merged["choices"][0]["message"]
        merged_choice["content"] = final_json_text
        if self.settings.reasoning_visibility != ReasoningVisibility.OFF:
            merged_choice["reasoning_content"] = reasoning_content
        merged["usage"] = self._merge_usage(
            phase1_response.get("usage", {}),
            phase2_response.get("usage", {}),
        )
        return merged

    async def _passthrough_chat_stream(
        self,
        body: dict[str, Any],
        request_id: str,
    ) -> AsyncIterator[bytes]:
        usage: dict[str, Any] = {}
        reasoning_parts: list[str] = []
        output_parts: list[str] = []
        gemma_parser = self._make_gemma_stream_parser(body.get("model"))
        async with self._service_slot(request_id, body.get("model")):
            async for event in self.backend.post_stream("/chat/completions", body):
                if event is None:
                    break
                if "usage" in event:
                    usage = event.get("usage", {})
                    yield self._sse(event)
                    continue
                if gemma_parser is None:
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        reasoning = delta.get("reasoning_content")
                        if isinstance(reasoning, str):
                            reasoning_parts.append(reasoning)
                            self.metrics.append_progress(
                                request_id,
                                reasoning_delta=reasoning,
                            )
                        content = delta.get("content")
                        if isinstance(content, str):
                            output_parts.append(content)
                            self.metrics.append_progress(
                                request_id,
                                output_delta=content,
                            )
                    yield self._sse(event)
                    continue
                for choice in event.get("choices", []):
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta", {})
                    if not isinstance(delta, dict):
                        delta = {}
                    index = int(choice.get("index") or 0)
                    if isinstance(delta.get("role"), str):
                        yield self._sse(
                            self._stream_event_from_choice(
                                event,
                                index=index,
                                delta={"role": delta["role"]},
                                finish_reason=None,
                            )
                        )
                    reasoning = delta.get("reasoning_content")
                    if isinstance(reasoning, str):
                        reasoning_parts.append(reasoning)
                        self.metrics.append_progress(request_id, reasoning_delta=reasoning)
                        yield self._sse(
                            self._stream_event_from_choice(
                                event,
                                index=index,
                                delta={"reasoning_content": reasoning},
                                finish_reason=None,
                            )
                        )
                    content_delta = delta.get("content")
                    if isinstance(content_delta, str):
                        parsed_reasoning, parsed_content = gemma_parser.feed(content_delta)
                        for piece in parsed_reasoning:
                            reasoning_parts.append(piece)
                            self.metrics.append_progress(request_id, reasoning_delta=piece)
                            yield self._sse(
                                self._stream_event_from_choice(
                                    event,
                                    index=index,
                                    delta={"reasoning_content": piece},
                                    finish_reason=None,
                                )
                            )
                        for piece in parsed_content:
                            output_parts.append(piece)
                            self.metrics.append_progress(request_id, output_delta=piece)
                            yield self._sse(
                                self._stream_event_from_choice(
                                    event,
                                    index=index,
                                    delta={"content": piece},
                                    finish_reason=None,
                                )
                            )
                    finish_reason = choice.get("finish_reason")
                    if finish_reason is not None:
                        yield self._sse(
                            self._stream_event_from_choice(
                                event,
                                index=index,
                                delta={},
                                finish_reason=finish_reason,
                            )
                        )
        if gemma_parser is not None:
            final_reasoning, final_content = gemma_parser.finish()
            for piece in final_reasoning:
                reasoning_parts.append(piece)
                self.metrics.append_progress(request_id, reasoning_delta=piece)
            for piece in final_content:
                output_parts.append(piece)
                self.metrics.append_progress(request_id, output_delta=piece)
        yield b"data: [DONE]\n\n"
        self.metrics.complete_request(
            request_id,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            output_chars=len("".join(output_parts)),
            reasoning_chars=len("".join(reasoning_parts)),
        )

    async def _buffered_chat_completion(
        self,
        body: dict[str, Any],
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        stream_body = copy.deepcopy(body)
        stream_body["stream"] = True
        stream_options = dict(stream_body.get("stream_options") or {})
        stream_options["include_usage"] = True
        stream_body["stream_options"] = stream_options

        response_id: str | None = None
        created: int | None = None
        model = stream_body.get("model")
        system_fingerprint: str | None = None
        finish_reason: str | None = None
        role = "assistant"
        usage: dict[str, Any] = {}
        reasoning_parts: list[str] = []
        output_parts: list[str] = []
        gemma_parser = self._make_gemma_stream_parser(stream_body.get("model"))

        async for event in self.backend.post_stream("/chat/completions", stream_body):
            if event is None:
                break
            if response_id is None and isinstance(event.get("id"), str):
                response_id = event["id"]
            if created is None and isinstance(event.get("created"), int):
                created = event["created"]
            if isinstance(event.get("model"), str):
                model = event["model"]
            if isinstance(event.get("system_fingerprint"), str):
                system_fingerprint = event["system_fingerprint"]
            if "usage" in event and isinstance(event.get("usage"), dict):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    delta = {}
                if isinstance(delta.get("role"), str):
                    role = delta["role"]
                reasoning_delta = delta.get("reasoning_content")
                if isinstance(reasoning_delta, str):
                    reasoning_parts.append(reasoning_delta)
                    if request_id is not None:
                        self.metrics.append_progress(request_id, reasoning_delta=reasoning_delta)
                content_delta = delta.get("content")
                if isinstance(content_delta, str):
                    if gemma_parser is None:
                        output_parts.append(content_delta)
                        if request_id is not None:
                            self.metrics.append_progress(request_id, output_delta=content_delta)
                    else:
                        parsed_reasoning, parsed_content = gemma_parser.feed(content_delta)
                        for piece in parsed_reasoning:
                            reasoning_parts.append(piece)
                            if request_id is not None:
                                self.metrics.append_progress(request_id, reasoning_delta=piece)
                        for piece in parsed_content:
                            output_parts.append(piece)
                            if request_id is not None:
                                self.metrics.append_progress(request_id, output_delta=piece)
                if isinstance(choice.get("finish_reason"), str):
                    finish_reason = choice["finish_reason"]

        if gemma_parser is not None:
            final_reasoning, final_content = gemma_parser.finish()
            reasoning_parts.extend(final_reasoning)
            output_parts.extend(final_content)
            if request_id is not None:
                for piece in final_reasoning:
                    self.metrics.append_progress(request_id, reasoning_delta=piece)
                for piece in final_content:
                    self.metrics.append_progress(request_id, output_delta=piece)

        message: dict[str, Any] = {
            "role": role,
            "content": "".join(output_parts),
            "tool_calls": [],
        }
        reasoning_text = "".join(reasoning_parts)
        if reasoning_text:
            message["reasoning_content"] = reasoning_text

        response: dict[str, Any] = {
            "id": response_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": created or int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "logprobs": None,
                    "finish_reason": finish_reason or "stop",
                }
            ],
            "usage": usage,
        }
        if system_fingerprint is not None:
            response["system_fingerprint"] = system_fingerprint
        return response

    def _make_phase1_body(self, original_body: dict[str, Any], stream: bool) -> dict[str, Any]:
        body = copy.deepcopy(original_body)
        body["messages"] = build_phase1_messages(body.get("messages", []))
        body.pop("response_format", None)
        body["stream"] = stream
        return body

    def _prepare_reasoning_request(self, body: dict[str, Any]) -> dict[str, Any]:
        return body

    def _make_gemma_stream_parser(self, model: Any) -> GemmaThoughtStreamParser | None:
        if self._is_gemma4_model(model):
            return GemmaThoughtStreamParser()
        return None

    @staticmethod
    def _is_gemma4_model(model: Any) -> bool:
        if not isinstance(model, str):
            return False
        normalized = model.lower().replace(" ", "")
        return "gemma4" in normalized or "gemma-4" in normalized

    @staticmethod
    def _stream_event_from_choice(
        event: dict[str, Any],
        *,
        index: int,
        delta: dict[str, Any],
        finish_reason: Any,
    ) -> dict[str, Any]:
        patched = {
            "id": event.get("id"),
            "object": event.get("object"),
            "created": event.get("created"),
            "model": event.get("model"),
            "choices": [
                {
                    "index": index,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        if "system_fingerprint" in event:
            patched["system_fingerprint"] = event["system_fingerprint"]
        return patched

    async def _run_phase2(
        self,
        original_body: dict[str, Any],
        schema: dict[str, Any],
        canonical_answer: str,
    ) -> tuple[dict[str, Any], str]:
        phase2_body: dict[str, Any] = {
            "model": original_body["model"],
            "messages": build_phase2_messages(
                original_body.get("messages", []), schema, canonical_answer
            ),
            "stream": False,
            "temperature": self.settings.phase2_temperature,
            "top_p": self.settings.phase2_top_p,
            "top_k": self.settings.phase2_top_k,
            "max_tokens": self.settings.phase2_max_tokens,
            "response_format": original_body["response_format"],
        }

        phase2_response = await self.backend.post_json("/chat/completions", phase2_body)
        raw_text = phase2_response["choices"][0]["message"].get("content", "")

        try:
            _, normalized = validate_json_text(raw_text, schema)
            return phase2_response, normalized
        except Exception as exc:
            last_error = exc

        for _ in range(self.settings.max_repair_attempts):
            repair_body: dict[str, Any] = {
                "model": original_body["model"],
                "messages": build_repair_messages(
                    schema=schema,
                    canonical_answer=canonical_answer,
                    invalid_output=raw_text,
                    validation_error=validation_error_message(last_error),
                ),
                "stream": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "max_tokens": self.settings.phase2_max_tokens,
                "response_format": original_body["response_format"],
            }
            phase2_response = await self.backend.post_json("/chat/completions", repair_body)
            raw_text = phase2_response["choices"][0]["message"].get("content", "")
            try:
                _, normalized = validate_json_text(raw_text, schema)
                return phase2_response, normalized
            except Exception as exc:
                last_error = exc

        raise BackendError(f"Schema repair failed: {validation_error_message(last_error)}")

    @staticmethod
    def _merge_usage(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        merged = {
            "prompt_tokens": (left.get("prompt_tokens") or 0) + (right.get("prompt_tokens") or 0),
            "completion_tokens": (left.get("completion_tokens") or 0)
            + (right.get("completion_tokens") or 0),
            "total_tokens": (left.get("total_tokens") or 0) + (right.get("total_tokens") or 0),
        }

        left_details = left.get("completion_tokens_details") or {}
        right_details = right.get("completion_tokens_details") or {}
        reasoning_tokens = (left_details.get("reasoning_tokens") or 0) + (
            right_details.get("reasoning_tokens") or 0
        )
        if reasoning_tokens:
            merged["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
        return merged

    @staticmethod
    def _sse(payload: dict[str, Any]) -> bytes:
        return f"data: {compact_json(payload)}\n\n".encode("utf-8")

    def _build_request_record(
        self,
        path: str,
        body: dict[str, Any],
        classification: RequestClassification,
    ) -> dict[str, Any]:
        messages = body.get("messages")
        messages = messages if isinstance(messages, list) else []
        input_messages = len(messages)
        input_chars = 0
        input_image_count = 0
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                input_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") in {"text", "input_text"} and isinstance(part.get("text"), str):
                        input_chars += len(part["text"])
                    if part.get("type") in {"image_url", "input_image"}:
                        input_image_count += 1
        return {
            "path": path,
            "model": body.get("model"),
            "stream": bool(body.get("stream")),
            "execution_path": classification.execution_path,
            "classification_reason": classification.reason,
            "has_schema": classification.has_schema,
            "has_images": classification.has_images,
            "asks_for_reasoning": classification.asks_for_reasoning,
            "input_messages": input_messages,
            "input_chars": input_chars,
            "input_image_count": input_image_count,
        }

    def _complete_chat_metrics(
        self,
        request_id: str,
        response: dict[str, Any],
        *,
        phase1_latency_ms: int | None = None,
        phase2_latency_ms: int | None = None,
        service_duration_ms: int | None = None,
    ) -> None:
        message = response["choices"][0]["message"]
        usage = response.get("usage", {})
        self.metrics.complete_request(
            request_id,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            output_chars=len(message.get("content", "") or ""),
            reasoning_chars=len(message.get("reasoning_content", "") or ""),
            phase1_latency_ms=phase1_latency_ms,
            phase2_latency_ms=phase2_latency_ms,
            service_duration_ms=service_duration_ms,
        )

    def _complete_responses_metrics(
        self,
        request_id: str,
        response_obj: dict[str, Any],
        chat_response: dict[str, Any],
    ) -> None:
        usage = response_obj.get("usage", {})
        reasoning = chat_response["choices"][0]["message"].get("reasoning_content", "") or ""
        self.metrics.complete_request(
            request_id,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            output_chars=len(response_obj.get("output_text", "") or ""),
            reasoning_chars=len(reasoning),
        )

    @asynccontextmanager
    async def _service_slot(self, request_id: str, model: Any | None = None) -> AsyncIterator[None]:
        if self.scheduler is not None:
            requested_model = model or self.settings.default_model_alias
            async with self.scheduler.slot(request_id, str(requested_model)):
                service_started_at = time.time()
                self.metrics.update_request(
                    request_id,
                    service_started_at=service_started_at,
                    model=self.scheduler.runtime.normalize_alias(requested_model),
                )
                yield
            return

        await self._upstream_slots.acquire()
        service_started_at = time.time()
        self.metrics.update_request(request_id, service_started_at=service_started_at)
        try:
            yield
        finally:
            self._upstream_slots.release()
