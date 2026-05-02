from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .config import StructuredMode
from .schema_utils import extract_json_schema


class ExecutionPath(StrEnum):
    PASSTHROUGH = "passthrough"
    STRICT_STRUCTURED_FAST_PATH = "strict_structured_fast_path"
    REASON_THEN_STRUCTURE = "reason_then_structure"


@dataclass(slots=True)
class RequestClassification:
    execution_path: ExecutionPath
    has_schema: bool
    has_images: bool
    asks_for_reasoning: bool
    reason: str


def _body_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in {
                    "text",
                    "input_text",
                }:
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
    return "\n".join(parts).lower()


def _has_images(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in {
                "image_url",
                "input_image",
            }:
                return True
    return False


def _explicit_reasoning_requested(body: dict[str, Any]) -> bool:
    effort = body.get("reasoning_effort")
    if isinstance(effort, str):
        return effort.lower() != "none"
    return False


def classify_chat_request(
    body: dict[str, Any], mode: StructuredMode
) -> RequestClassification:
    messages = body.get("messages")
    messages = messages if isinstance(messages, list) else []
    has_schema = extract_json_schema(body) is not None
    has_images = _has_images(messages)
    asks_for_reasoning = _explicit_reasoning_requested(body)

    if not has_schema:
        return RequestClassification(
            execution_path=ExecutionPath.PASSTHROUGH,
            has_schema=False,
            has_images=has_images,
            asks_for_reasoning=asks_for_reasoning,
            reason="no_schema",
        )

    if mode == StructuredMode.STRICT_FAST_PATH:
        return RequestClassification(
            execution_path=ExecutionPath.STRICT_STRUCTURED_FAST_PATH,
            has_schema=True,
            has_images=has_images,
            asks_for_reasoning=asks_for_reasoning,
            reason="forced_fast_path",
        )

    if mode == StructuredMode.REASON_THEN_STRUCTURE:
        return RequestClassification(
            execution_path=ExecutionPath.REASON_THEN_STRUCTURE,
            has_schema=True,
            has_images=has_images,
            asks_for_reasoning=asks_for_reasoning,
            reason="forced_two_phase",
        )

    if not asks_for_reasoning:
        return RequestClassification(
            execution_path=ExecutionPath.STRICT_STRUCTURED_FAST_PATH,
            has_schema=True,
            has_images=has_images,
            asks_for_reasoning=asks_for_reasoning,
            reason="no_reasoning_requested",
        )

    return RequestClassification(
        execution_path=ExecutionPath.REASON_THEN_STRUCTURE,
        has_schema=True,
        has_images=has_images,
        asks_for_reasoning=asks_for_reasoning,
        reason="reasoning_requested",
    )
