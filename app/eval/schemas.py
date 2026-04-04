"""Pydantic models for offline evaluation datasets (Phase 1.1)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

RouteExpected = Literal["question", "ticket"]


class EvalCase(BaseModel):
    """One labeled example for regression / LangSmith import.

    `history` matches chat memory turns: {"role": "user"|"assistant", "message": "..."}.
    """

    id: str = Field(min_length=1, max_length=256, description="Stable id for baselines and diffs")
    message: str = Field(min_length=1, max_length=50_000)
    route_expected: RouteExpected = Field(
        description="Expected orchestrator branch after intent classification",
    )
    ticket_id_expected: str | None = Field(
        default=None,
        description="Expected normalized issue key when route is ticket; null if none extracted",
    )
    reference_answer: str | None = Field(
        default=None,
        description="Optional gold or near-gold text for RAG correctness checks",
    )
    expected_behavior: str | None = Field(
        default=None,
        description="Rubric for judges when answers vary (tickets, off-KB, ambiguous)",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Edge-case labels, e.g. off_kb, pii_like",
    )
    user_id: str | None = Field(
        default=None,
        max_length=256,
        description="Optional eval user id; runner may default to eval-{case_id}",
    )
    history: list[dict[str, str]] | None = Field(
        default=None,
        description="Prior turns for multi-turn routing tests",
    )

    @field_validator("id", "message", mode="before")
    @classmethod
    def strip_strings(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(t).strip().lower() for t in value if str(t).strip()]
        return value

    @field_validator("ticket_id_expected", mode="before")
    @classmethod
    def normalize_ticket_id(cls, value: object) -> object:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            s = value.strip().upper()
            return s or None
        return value

    @field_validator("history", mode="before")
    @classmethod
    def validate_history(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, list):
            raise TypeError("history must be a list of dicts with role and message")
        cleaned: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                raise TypeError("each history turn must be a dict")
            role = str(item.get("role", "")).strip()
            msg = str(item.get("message", "")).strip()
            if not role or not msg:
                raise ValueError("history entries need non-empty role and message")
            cleaned.append({"role": role, "message": msg})
        return cleaned
