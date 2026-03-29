"""OpenAI client wrapper for chat completions with structured parsing."""

from __future__ import annotations

from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from app.core.config import AppSettings

T = TypeVar("T", bound=BaseModel)


def get_openai_client(settings: AppSettings) -> OpenAI:
    kwargs: dict = {}
    if settings.openai_base_url.strip():
        kwargs["base_url"] = settings.openai_base_url.strip()
    return OpenAI(**kwargs)


def complete_parsed(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    response_model: type[T],
) -> T:
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=response_model,
    )
    choice = completion.choices[0].message
    if choice.refusal:
        raise RuntimeError(f"Model refused: {choice.refusal}")
    if choice.parsed is None:
        raise RuntimeError("Structured parse returned no content")
    return choice.parsed


def complete_text(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
) -> str:
    """Plain chat completion for RAG answers and ticket narrative (no structured parse)."""
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    choice = completion.choices[0].message
    content = (choice.content or "").strip()
    if not content:
        raise RuntimeError("Chat completion returned empty content")
    return content
