"""API route handlers."""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from openai import OpenAIError

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
)
from app.core.config import AppSettings, get_settings
from app.core.logger import get_logger
from app.memory import chat_memory
from app.memory.feedback_store import append_feedback, get_turn_snapshot, store_turn_snapshot
from app.memory.redis_client import MemoryStore, get_memory_store
from app.orchestrator.agent import ChatTurnOutcome, MissingOpenAIKeyError, run_chat_turn

router = APIRouter()
logger = get_logger(__name__)


def get_memory_store_dep(settings: AppSettings = Depends(get_settings)) -> MemoryStore:
    if settings.redis_backend != "memory":
        raise HTTPException(
            status_code=501,
            detail="Only redis.backend=memory is implemented; install Redis and wire redis client to switch.",
        )
    return get_memory_store()


@router.get("/health", response_model=HealthResponse)
def health(settings: AppSettings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(status="healthy", app_name=settings.app_name)


def _log_chat_completed(
    body: ChatRequest,
    outcome: ChatTurnOutcome,
    settings: AppSettings,
    prior_count: int,
) -> None:
    structured: dict[str, str | int | float | bool] = {
        "user_id": body.user_id,
        "message_length": len(body.message),
        "chat_history_prior_count": prior_count,
        "source": outcome.source,
        "cached": outcome.from_cache,
    }
    if outcome.intent is not None:
        structured["intent"] = outcome.intent
    if settings.log_chat_message_body:
        structured["message"] = body.message
    logger.info("chat_completed", extra={"structured": structured})


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: Request,
    body: ChatRequest,
    settings: AppSettings = Depends(get_settings),
    store: MemoryStore = Depends(get_memory_store_dep),
) -> ChatResponse:
    prior_count = len(chat_memory.get_chat_history(body.user_id, store))
    request_id = getattr(request.state, "request_id", None)
    try:
        outcome = run_chat_turn(
            user_id=body.user_id,
            message=body.message,
            store=store,
            settings=settings,
            request_id=request_id,
        )
    except MissingOpenAIKeyError as exc:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured; intent classification requires it when cache misses.",
        ) from exc
    except OpenAIError as exc:
        raise HTTPException(
            status_code=503,
            detail="OpenAI request failed (intent classification or answer generation).",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="LLM returned an invalid or empty response.",
        ) from exc

    _log_chat_completed(body, outcome, settings, prior_count)

    chat_memory.append_message(body.user_id, "user", body.message, store)
    chat_memory.append_message(body.user_id, "assistant", outcome.response, store)

    if request_id:
        store_turn_snapshot(
            store,
            request_id=request_id,
            user_id=body.user_id,
            user_message=body.message,
            assistant_response=outcome.response,
            source=outcome.source,
            intent=outcome.intent,
        )

    return ChatResponse(
        response=outcome.response,
        source=outcome.source,
        cached=outcome.from_cache,
        intent=outcome.intent,
        request_id=request_id,
    )


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    body: FeedbackRequest,
    store: MemoryStore = Depends(get_memory_store_dep),
) -> FeedbackResponse:
    snapshot = get_turn_snapshot(store, body.request_id)
    if snapshot is not None and snapshot.get("user_id") != body.user_id:
        raise HTTPException(
            status_code=403,
            detail="request_id does not match this user_id.",
        )

    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_id": body.request_id,
        "user_id": body.user_id,
        "rating": body.rating,
        "thumbs": body.thumbs,
        "comment": body.comment,
    }
    if snapshot:
        payload["user_message"] = snapshot.get("user_message")
        payload["assistant_preview"] = snapshot.get("assistant_preview")
        payload["source"] = snapshot.get("source")
        payload["intent"] = snapshot.get("intent")

    queued = append_feedback(store, payload)
    logger.info(
        "feedback_received",
        extra={
            "structured": {
                "request_id": body.request_id,
                "user_id": body.user_id,
                "rating": body.rating,
                "thumbs": body.thumbs,
                "queued_for_review": queued,
                "has_turn_context": snapshot is not None,
            }
        },
    )
    return FeedbackResponse(queued_for_review=queued)
