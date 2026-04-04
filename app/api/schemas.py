"""Pydantic request/response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class HealthResponse(BaseModel):
    status: str = Field(description="Service health indicator")
    app_name: str = Field(description="Application name from config")


class ChatRequest(BaseModel):
    user_id: str = Field(
        min_length=1,
        max_length=256,
        description="Client or session identifier",
    )
    message: str = Field(
        min_length=1,
        max_length=50_000,
        description="User message text",
    )

    @field_validator("user_id", "message", mode="before")
    @classmethod
    def strip_outer_whitespace(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class ChatResponse(BaseModel):
    response: str = Field(description="Assistant reply text")
    source: Literal["question", "ticket"] = Field(
        description="Routing branch: RAG path vs ticket API path",
    )
    cached: bool = Field(description="True if reply was served from Phase 2 cache")
    intent: str | None = Field(
        default=None,
        description="LLM intent when computed this turn; null on cache hit",
    )
    request_id: str | None = Field(
        default=None,
        description="Same as response header X-Request-ID for correlating POST /feedback",
    )


class FeedbackRequest(BaseModel):
    request_id: str = Field(
        min_length=8,
        max_length=128,
        description="From X-Request-ID (header or ChatResponse) after /chat",
    )
    user_id: str = Field(min_length=1, max_length=256)
    rating: int | None = Field(default=None, ge=1, le=5)
    thumbs: Literal["up", "down"] | None = None
    comment: str | None = Field(default=None, max_length=2000)

    @field_validator("user_id", "request_id", "comment", mode="before")
    @classmethod
    def strip_strings(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def at_least_one_signal(self) -> FeedbackRequest:
        has_comment = bool(self.comment and self.comment.strip())
        if self.rating is None and self.thumbs is None and not has_comment:
            raise ValueError("Provide at least one of: rating, thumbs, or non-empty comment")
        return self


class FeedbackResponse(BaseModel):
    accepted: bool = True
    queued_for_review: bool = False
