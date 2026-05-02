from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class RequestCacheEntry:
    kind: str
    payload: Any
    expires_at: float


class RequestCache:
    def __init__(self, ttl_seconds: float) -> None:
        self.ttl_seconds = ttl_seconds
        self._items: dict[str, RequestCacheEntry] = {}

    def make_key(self, *, path: str, body: dict[str, Any]) -> str:
        payload = {
            "version": 1,
            "path": path,
            "body": body,
        }
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, key: str) -> RequestCacheEntry | None:
        now = time.monotonic()
        entry = self._items.get(key)
        if entry is None:
            return None
        if entry.expires_at <= now:
            self._items.pop(key, None)
            return None
        return RequestCacheEntry(
            kind=entry.kind,
            payload=copy.deepcopy(entry.payload),
            expires_at=entry.expires_at,
        )

    def set(self, key: str, *, kind: str, payload: Any) -> None:
        self._items[key] = RequestCacheEntry(
            kind=kind,
            payload=copy.deepcopy(payload),
            expires_at=time.monotonic() + self.ttl_seconds,
        )
        self._prune_expired()

    def _prune_expired(self) -> None:
        now = time.monotonic()
        expired = [key for key, entry in self._items.items() if entry.expires_at <= now]
        for key in expired:
            self._items.pop(key, None)
