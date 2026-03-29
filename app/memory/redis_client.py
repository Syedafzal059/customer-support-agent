"""In-process storage with Redis-like string/list usage (no Redis server required for MVP)."""

from __future__ import annotations

import threading


class MemoryStore:
    """Thread-safe dict-backed store for string keys and append-only list values."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._strings: dict[str, str] = {}
        self._lists: dict[str, list[str]] = {}

    def list_append(self, key: str, value: str, *, max_length: int | None = None) -> None:
        with self._lock:
            bucket = self._lists.setdefault(key, [])
            bucket.append(value)
            if max_length is not None and len(bucket) > max_length:
                self._lists[key] = bucket[-max_length:]

    def list_get(self, key: str) -> list[str]:
        with self._lock:
            return list(self._lists.get(key, []))

    def string_get(self, key: str) -> str | None:
        with self._lock:
            return self._strings.get(key)

    def string_set(self, key: str, value: str) -> None:
        with self._lock:
            self._strings[key] = value


_store_singleton: MemoryStore | None = None
_store_lock = threading.Lock()


def get_memory_store() -> MemoryStore:
    """Process-wide singleton for the in-memory backend."""
    global _store_singleton
    with _store_lock:
        if _store_singleton is None:
            _store_singleton = MemoryStore()
        return _store_singleton
