from __future__ import annotations

from enum import IntEnum
from typing import Any


class RequestPriority(IntEnum):
    LOWEST = 0
    LOW = 1
    DEFAULT = 2
    HIGH = 3
    HIGHEST = 4
    CRITICAL = 5

    @property
    def label(self) -> str:
        return self.name.lower()


_PRIORITIES_BY_LABEL = {priority.label: priority for priority in RequestPriority}


def parse_request_priority(body: dict[str, Any]) -> RequestPriority:
    metadata = body.get("metadata")
    if metadata is None:
        return RequestPriority.DEFAULT
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    value = metadata.get("priority")
    if value is None:
        return RequestPriority.DEFAULT
    if not isinstance(value, str):
        raise ValueError("metadata.priority must be a string")
    priority = _PRIORITIES_BY_LABEL.get(value)
    if priority is None:
        accepted = ", ".join(priority.label for priority in reversed(RequestPriority))
        raise ValueError(f"metadata.priority must be one of: {accepted}")
    return priority


def strip_local_priority_metadata(body: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(body)
    metadata = prepared.get("metadata")
    if not isinstance(metadata, dict) or "priority" not in metadata:
        return prepared
    next_metadata = dict(metadata)
    next_metadata.pop("priority", None)
    if next_metadata:
        prepared["metadata"] = next_metadata
    else:
        prepared.pop("metadata", None)
    return prepared
