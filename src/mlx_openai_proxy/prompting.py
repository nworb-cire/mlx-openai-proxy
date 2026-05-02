from __future__ import annotations

from typing import Any

from .schema_utils import compact_json


PHASE1_SYSTEM_PROMPT = """You are phase 1 of a two-phase response pipeline.
Solve the user's request carefully.
Reason normally and explicitly if the model supports visible reasoning.
Your final assistant answer must be the concise canonical answer payload needed by a downstream formatter.
Do not use markdown fences in the final assistant answer."""


PHASE2_SYSTEM_PROMPT = """You are a formatter.
You will receive an already computed canonical answer and a JSON schema.
Emit only valid JSON matching the schema.
Do not add markdown.
Do not solve the task again unless the canonical answer is ambiguous."""


REPAIR_SYSTEM_PROMPT = """You are repairing JSON output.
Return only valid JSON that matches the provided schema.
Do not add markdown or commentary."""


JSON_OBJECT_FORMATTER_SYSTEM_PROMPT = """Output JSON only.
Convert the canonical answer to one JSON object."""


JSON_OBJECT_REPAIR_SYSTEM_PROMPT = """You are repairing JSON output.
Output JSON only."""


def build_phase1_messages(
    original_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [{"role": "system", "content": PHASE1_SYSTEM_PROMPT}, *original_messages]


def _sanitize_formatter_part(part: Any) -> Any:
    if not isinstance(part, dict):
        return part

    part_type = part.get("type")
    if part_type in {"text", "input_text"}:
        text = part.get("text")
        if isinstance(text, str):
            return {"type": part_type, "text": text}
        return {"type": part_type}

    if part_type in {"image_url", "input_image"}:
        detail: Any = None
        if part_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                detail = image_url.get("detail")
        else:
            detail = part.get("detail")

        sanitized = {"type": part_type, "image": "<omitted>"}
        if isinstance(detail, str) and detail:
            sanitized["detail"] = detail
        return sanitized

    sanitized: dict[str, Any] = {}
    for key, value in part.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
    return sanitized or {"type": part_type or "unknown"}


def sanitize_messages_for_formatter(
    original_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sanitized_messages: list[dict[str, Any]] = []
    for message in original_messages:
        if not isinstance(message, dict):
            continue

        sanitized_message: dict[str, Any] = {}
        for key in ("role", "name"):
            value = message.get(key)
            if isinstance(value, str):
                sanitized_message[key] = value

        content = message.get("content")
        if isinstance(content, str):
            sanitized_message["content"] = content
        elif isinstance(content, list):
            sanitized_message["content"] = [
                _sanitize_formatter_part(part) for part in content
            ]

        sanitized_messages.append(sanitized_message)
    return sanitized_messages


def build_phase2_messages(
    original_messages: list[dict[str, Any]],
    schema: dict[str, Any],
    canonical_answer: str,
) -> list[dict[str, Any]]:
    summarized_messages = sanitize_messages_for_formatter(original_messages)
    return [
        {"role": "system", "content": PHASE2_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Original user request:\n"
                f"{compact_json(summarized_messages)}\n\n"
                "JSON schema:\n"
                f"{compact_json(schema)}\n\n"
                "Canonical answer:\n"
                f"{canonical_answer}\n"
            ),
        },
    ]


def build_repair_messages(
    schema: dict[str, Any],
    canonical_answer: str,
    invalid_output: str,
    validation_error: str,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Schema:\n"
                f"{compact_json(schema)}\n\n"
                "Canonical answer:\n"
                f"{canonical_answer}\n\n"
                "Invalid output:\n"
                f"{invalid_output}\n\n"
                "Validation error:\n"
                f"{validation_error}\n"
            ),
        },
    ]


def build_json_object_formatter_messages(
    original_messages: list[dict[str, Any]],
    canonical_answer: str,
) -> list[dict[str, Any]]:
    summarized_messages = sanitize_messages_for_formatter(original_messages)
    return [
        {
            "role": "user",
            "content": (
                f"{JSON_OBJECT_FORMATTER_SYSTEM_PROMPT}\n\n"
                "Original user request:\n"
                f"{compact_json(summarized_messages)}\n\n"
                "Canonical answer:\n"
                f"{canonical_answer}\n"
            ),
        },
    ]


def build_json_object_repair_messages(
    canonical_answer: str,
    invalid_output: str,
    validation_error: str,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": (
                f"{JSON_OBJECT_REPAIR_SYSTEM_PROMPT}\n\n"
                "Canonical answer:\n"
                f"{canonical_answer}\n\n"
                "Invalid output:\n"
                f"{invalid_output}\n\n"
                "Validation error:\n"
                f"{validation_error}\n"
            ),
        },
    ]
