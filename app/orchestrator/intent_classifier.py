"""LLM-based binary intent: question (RAG) vs ticket (API)."""

from __future__ import annotations

from app.core.config import AppSettings
from app.llm.client import complete_parsed, get_openai_client
from app.llm.prompts import build_intent_classifier_messages
from app.llm.router import model_for_task
from app.llm.schemas import IntentClassification


def classify_intent(
    *,
    current_message: str,
    history: list[dict[str, str]],
    settings: AppSettings,
) -> IntentClassification:
    """Uses OpenAI structured output; `history` is prior turns only (excludes current message)."""
    client = get_openai_client(settings)
    messages = build_intent_classifier_messages(history, current_message)
    return complete_parsed(
        client,
        model_for_task(settings, "intent_classification"),
        messages,
        IntentClassification,
    )
