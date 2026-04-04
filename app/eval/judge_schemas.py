"""Structured outputs for LLM eval judges."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ScoreWithReason(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="0 = bad, 1 = meets criteria")
    reason: str = Field(max_length=600, description="One short sentence")
