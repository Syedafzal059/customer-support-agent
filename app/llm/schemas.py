"""Pydantic models for LLM structured outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IntentClassification(BaseModel):
    """Binary routing: RAG (question) vs ticket system (API)."""

    intent: Literal["question", "ticket"] = Field(
        description='"question" = answer from knowledge/RAG; "ticket" = fetch/update via ticket API',
    )
    ticket_id: str | None = Field(
        default=None,
        description=(
            "When intent is ticket: the issue key to fetch (e.g. ABC-123), taken only from the "
            "current message or recent history—never invented. Normalize to uppercase PROJECT-NUMBER. "
            "When intent is question: must be null."
        ),
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence in this label")
    rationale: str = Field(
        max_length=600,
        description="One short sentence explaining the choice (for logs and debugging)",
    )
