"""Resolve OpenAI model id per task (Phase 7)."""

from __future__ import annotations

from typing import Literal

from app.core.config import AppSettings

LlmTask = Literal["intent_classification", "rag_qa", "ticket_summary"]


def model_for_task(settings: AppSettings, task: LlmTask) -> str:
    if task == "intent_classification":
        return settings.intent_classifier_model
    if task == "rag_qa":
        return settings.rag_qa_model
    return settings.ticket_summary_model
