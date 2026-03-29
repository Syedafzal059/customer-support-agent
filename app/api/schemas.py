"""Pydantic request/response models."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


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
