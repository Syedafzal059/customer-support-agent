"""Append-only feedback events (MemoryStore) plus durable JSONL under data/feedback/."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from app.memory.redis_client import MemoryStore

FEEDBACK_LIST_KEY = "feedback:events"
REVIEW_QUEUE_KEY = "feedback:review_queue"

_file_lock = threading.Lock()


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _feedback_dir() -> Path:
    d = _project_root() / "data" / "feedback"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _append_jsonl(filename: str, line: str) -> None:
    path = _feedback_dir() / filename
    with _file_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def turn_key(request_id: str) -> str:
    return f"chat:turn:{request_id}"


def store_turn_snapshot(
    store: MemoryStore,
    *,
    request_id: str,
    user_id: str,
    user_message: str,
    assistant_response: str,
    source: str,
    intent: str | None,
) -> None:
    preview = (assistant_response or "")[:800]
    payload = {
        "user_id": user_id,
        "user_message": user_message,
        "assistant_preview": preview,
        "source": source,
        "intent": intent,
    }
    store.string_set(turn_key(request_id), json.dumps(payload, ensure_ascii=False))


def get_turn_snapshot(store: MemoryStore, request_id: str) -> dict[str, Any] | None:
    raw = store.string_get(turn_key(request_id))
    if not raw:
        return None
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None


def append_feedback(store: MemoryStore, payload: dict[str, Any]) -> bool:
    """Persist one feedback row; return True if also queued for human review."""
    line = json.dumps(payload, ensure_ascii=False)
    store.list_append(FEEDBACK_LIST_KEY, line, max_length=50_000)
    _append_jsonl("events.jsonl", line)

    thumbs = payload.get("thumbs")
    rating = payload.get("rating")
    needs_review = thumbs == "down" or (rating is not None and int(rating) <= 2)
    if needs_review:
        store.list_append(REVIEW_QUEUE_KEY, line, max_length=10_000)
        _append_jsonl("review_queue.jsonl", line)
    return needs_review
