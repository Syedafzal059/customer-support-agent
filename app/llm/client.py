"""OpenAI client wrapper for chat completions with structured parsing."""

from __future__ import annotations
import os
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from app.core.config import AppSettings

T = TypeVar("T", bound=BaseModel)


def _langsmith_api_key() -> str:
    return (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()


def apply_langsmith_env_from_settings(settings: AppSettings) -> None:
    """Set tracing env before any @traceable runs so LangSmith uses the configured project.

    Without this, the first spans (e.g. chat_turn) start before get_openai_client runs and
    often land in LangSmith's default project.
    """
    if not settings.langsmith_enabled or not _langsmith_api_key():
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    proj = settings.langsmith_project.strip()
    if proj:
        os.environ["LANGSMITH_PROJECT"] = proj
        os.environ["LANGCHAIN_PROJECT"] = proj


def get_openai_client(settings: AppSettings) -> OpenAI:
    kwargs: dict = {}
    if settings.openai_base_url.strip():
        kwargs["base_url"] = settings.openai_base_url.strip()
    client = OpenAI(**kwargs)

    if not settings.langsmith_enabled or not _langsmith_api_key():
        return client

    apply_langsmith_env_from_settings(settings)
    
    from langsmith.wrappers import wrap_openai
    
    return wrap_openai(client)




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
