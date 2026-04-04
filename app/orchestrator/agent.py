"""Orchestrates cache → intent → route (RAG + ticket paths with LLM generation)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Literal

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree, set_run_metadata
from openai import OpenAIError

from app.core.config import AppSettings
from app.core.logger import get_logger
from app.integrations.jira_mock import get_ticket
from app.llm.generation import generate_rag_answer, generate_ticket_narrative
from app.llm.prompts import RAG_UNKNOWN_REPLY
from app.llm.schemas import IntentClassification
from app.memory import chat_memory
from app.memory.redis_client import MemoryStore
from app.orchestrator.intent_classifier import classify_intent
from app.retrieval.faiss_store import FaissKnowledgeIndex, get_knowledge_index

logger = get_logger(__name__)

# Bump when cached reply semantics change (invalidates Phase 2 query cache).
_RESPONSE_CACHE_VERSION = "v3"

Source = Literal["question", "ticket"]


def _user_id_trace_hash(user_id: str) -> str:
    """Stable pseudonymous id for LangSmith metadata (not reversible to raw user_id)."""
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]


def _merge_chat_turn_trace_metadata(**fields: Any) -> None:
    """Attach metadata to the current `chat_turn` span when a LangSmith run is active."""
    if get_current_run_tree() is None:
        return
    set_run_metadata(**{k: v for k, v in fields.items() if v is not None})


class MissingOpenAIKeyError(Exception):
    """Raised when intent classification is needed but no API key is configured."""


@dataclass(frozen=True)
class ChatTurnOutcome:
    response: str
    source: Source
    from_cache: bool
    intent: str | None


def _cache_lookup_key(user_id: str, message: str) -> str:
    return f"{_RESPONSE_CACHE_VERSION}\n{user_id}\n{message}"


def _serialize_cache_entry(response: str, source: Source) -> str:
    return json.dumps({"response": response, "source": source}, ensure_ascii=False)


def _deserialize_cache_entry(raw: str) -> tuple[str, Source]:
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "response" in data:
            src = data.get("source", "question")
            if src not in ("question", "ticket"):
                src = "question"
            return str(data["response"]), src  # type: ignore[return-value]
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    return raw, "question"


def _normalize_ticket_id_from_classifier(raw: str | None) -> str | None:
    """Validate LLM-supplied issue key (no regex extraction from user text)."""
    if raw is None:
        return None
    s = raw.strip().upper()
    if not s or len(s) > 40:
        return None
    if not all(c.isalnum() or c == "-" for c in s):
        return None
    return s


def _answer_ticket_path(
    intent_result: IntentClassification,
    user_message: str,
    settings: AppSettings,
) -> str:
    tid = _normalize_ticket_id_from_classifier(intent_result.ticket_id)
    if tid is None:
        return (
            "I can look that up in the ticket system, but I need your issue key "
            "(for example PROJECT-123). Reply with the key from your confirmation or portal."
        )
    data = get_ticket(tid)
    return generate_ticket_narrative(
        ticket_fields=data,
        user_message=user_message,
        settings=settings,
    )


@traceable(name="rag_retrieval", run_type="retriever")
def retrieve_rag_chunks(
    *,
    query: str,
    settings: AppSettings,
    kb: FaissKnowledgeIndex,
) -> list[str]:
    """FAISS top-k retrieval; appears as a retriever span in LangSmith with chunk texts as output."""
    chunks = kb.search(query, settings)
    if get_current_run_tree() is not None:
        set_run_metadata(
            rag_top_k=settings.rag_top_k,
            retrieved_chunk_count=len(chunks),
        )
    return chunks


def _answer_question_path(
    message: str,
    history: list[dict[str, str]],
    settings: AppSettings,
) -> str:
    kb = get_knowledge_index()
    if kb is None:
        return "[RAG] Knowledge index is not initialized."
    hits = retrieve_rag_chunks(query=message, settings=settings, kb=kb)
    if not hits:
        return RAG_UNKNOWN_REPLY
    return generate_rag_answer(
        context_chunks=hits,
        current_message=message,
        history=history,
        settings=settings,
    )


@traceable(name="chat_turn", run_type="chain")
def run_chat_turn(
    *,
    user_id: str,
    message: str,
    store: MemoryStore,
    settings: AppSettings,
    request_id: str | None = None,
) -> ChatTurnOutcome:
    _merge_chat_turn_trace_metadata(
        request_id=request_id,
        user_id_hash=_user_id_trace_hash(user_id),
        message_length=len(message),
    )

    prior = chat_memory.get_chat_history(user_id, store)
    cache_key = _cache_lookup_key(user_id, message)
    cached_raw = chat_memory.get_cache(cache_key, store)
    if cached_raw is not None:
        reply, source = _deserialize_cache_entry(cached_raw)
        _merge_chat_turn_trace_metadata(
            cache_hit=True,
            source=source,
        )
        logger.info(
            "orchestrator_cache_hit",
            extra={
                "structured": {
                    "user_id": user_id,
                    "source": source,
                    "response_length": len(reply),
                }
            },
        )
        return ChatTurnOutcome(
            response=reply,
            source=source,
            from_cache=True,
            intent=None,
        )

    logger.info(
        "orchestrator_cache_miss",
        extra={"structured": {"user_id": user_id}},
    )

    _merge_chat_turn_trace_metadata(cache_hit=False)

    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise MissingOpenAIKeyError("OPENAI_API_KEY is not configured")

    try:
        intent_result = classify_intent(
            current_message=message,
            history=prior,
            settings=settings,
        )
        branch: Source = intent_result.intent
        logger.info(
            "orchestrator_intent",
            extra={
                "structured": {
                    "user_id": user_id,
                    "intent": branch,
                    "ticket_id": intent_result.ticket_id,
                    "intent_confidence": intent_result.confidence,
                    "intent_rationale": intent_result.rationale,
                }
            },
        )

        kb_index = get_knowledge_index()
        if branch == "question":
            reply = _answer_question_path(message, prior, settings)
        else:
            reply = _answer_ticket_path(intent_result, message, settings)
        logger.info(
            "orchestrator_route",
            extra={
                "structured": {
                    "user_id": user_id,
                    "branch": branch,
                    "kb_chunk_count": kb_index.chunk_count if kb_index else 0,
                }
            },
        )
        _merge_chat_turn_trace_metadata(
            source=branch,
            intent=branch,
            kb_index_chunk_count=kb_index.chunk_count if kb_index else 0,
        )
    except (OpenAIError, RuntimeError):
        logger.exception(
            "orchestrator_llm_failed",
            extra={"structured": {"user_id": user_id}},
        )
        raise

    chat_memory.set_cache(cache_key, _serialize_cache_entry(reply, branch), store)

    return ChatTurnOutcome(
        response=reply,
        source=branch,
        from_cache=False,
        intent=branch,
    )
