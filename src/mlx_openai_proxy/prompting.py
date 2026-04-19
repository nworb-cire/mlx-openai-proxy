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


def build_phase1_messages(original_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"role": "system", "content": PHASE1_SYSTEM_PROMPT}, *original_messages]


def build_phase2_messages(
    original_messages: list[dict[str, Any]],
    schema: dict[str, Any],
    canonical_answer: str,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": PHASE2_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Original user request:\n"
                f"{compact_json(original_messages)}\n\n"
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
