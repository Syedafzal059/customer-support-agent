"""LLM-based binary intent: question (RAG) vs ticket (API)."""

from __future__ import annotations

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree, set_run_metadata

from app.core.config import AppSettings
from app.llm.client import complete_parsed, get_openai_client, helicone_extra_headers
from app.llm.prompts import build_intent_classifier_messages
from app.llm.router import model_for_task
from app.llm.schemas import IntentClassification


@traceable(name="intent_classification", run_type="chain")
def classify_intent(
    *,
    current_message: str,
    history: list[dict[str, str]],
    settings: AppSettings,
) -> IntentClassification:
    """Uses OpenAI structured output; `history` is prior turns only (excludes current message)."""
    model = model_for_task(settings, "intent_classification")
    if get_current_run_tree() is not None:
        set_run_metadata(
            intent_model=model,
            history_turns=len(history),
        )
    client = get_openai_client(settings)
    messages = build_intent_classifier_messages(history, current_message)
    return complete_parsed(
        client,
        model,
        messages,
        IntentClassification,
        extra_headers=helicone_extra_headers(settings, step="intent_classification"),
    )
