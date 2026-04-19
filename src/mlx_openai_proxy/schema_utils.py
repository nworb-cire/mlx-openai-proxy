from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


def extract_json_schema(body: dict[str, Any]) -> dict[str, Any] | None:
    response_format = body.get("response_format")
    if not isinstance(response_format, dict):
        return None
    if response_format.get("type") != "json_schema":
        return None
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        return None
    schema = json_schema.get("schema")
    return schema if isinstance(schema, dict) else None


def validate_incoming_schema(schema: dict[str, Any]) -> None:
    Draft202012Validator.check_schema(schema)


def normalize_json_text(text: str) -> str:
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    parsed = json.loads(text)
    return json.dumps(parsed, separators=(",", ":"), ensure_ascii=True)


def validate_json_text(text: str, schema: dict[str, Any]) -> tuple[dict[str, Any], str]:
    normalized = normalize_json_text(text)
    parsed = json.loads(normalized)
    Draft202012Validator(schema).validate(parsed)
    return parsed, normalized


def validation_error_message(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return exc.message
    return str(exc)


def compact_json(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=True)


def chunk_text(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]
