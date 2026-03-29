"""Chat history and query cache using MemoryStore and Redis-style key names."""

from __future__ import annotations

import hashlib
import json

from app.memory.redis_client import MemoryStore

CHAT_HISTORY_LIMIT = 5


def _chat_key(user_id: str) -> str:
    return f"chat:{user_id}"


def _cache_key_for_query(query: str) -> str:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return f"cache:{digest}"


def get_chat_history(user_id: str, store: MemoryStore) -> list[dict[str, str]]:
    key = _chat_key(user_id)
    return [json.loads(raw) for raw in store.list_get(key)]


def append_message(user_id: str, role: str, message: str, store: MemoryStore) -> None:
    payload = json.dumps({"role": role, "message": message}, ensure_ascii=False)
    store.list_append(_chat_key(user_id), payload, max_length=CHAT_HISTORY_LIMIT)


def get_cache(query: str, store: MemoryStore) -> str | None:
    return store.string_get(_cache_key_for_query(query))


def set_cache(query: str, response: str, store: MemoryStore) -> None:
    store.string_set(_cache_key_for_query(query), response)
