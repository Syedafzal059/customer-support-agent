"""OpenAI client wrapper for chat completions with structured parsing."""

from __future__ import annotations
import os
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from app.core.config import AppSettings
from app.core.logger import get_logger

T = TypeVar("T", bound=BaseModel)

logger = get_logger(__name__)


def _helicone_api_key() -> str:
    return (os.getenv("HELICONE_API_KEY") or "").strip()


def helicone_extra_headers(
    settings: AppSettings,
    *,
    branch: str | None = None,
    step: str | None = None,
) -> dict[str, str] | None:
    """Per-request Helicone custom properties (only when proxy is enabled)."""
    if not settings.helicone_enabled:
        return None
    out: dict[str, str] = {}
    if branch:
        out["Helicone-Property-Branch"] = branch
    if step:
        out["Helicone-Property-Step"] = step
    return out or None

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
    use_helicone = settings.helicone_enabled and bool(_helicone_api_key())
    if settings.helicone_enabled and not _helicone_api_key():
        logger.warning(
            "helicone_enabled_but_missing_api_key",
            extra={"structured": {"hint": "Set HELICONE_API_KEY or set helicone.enable false"}},
        )
    if use_helicone:
        kwargs["base_url"] = settings.helicone_openai_proxy_base_url.strip()
        kwargs["default_headers"] = {
            "Helicone-Auth": f"Bearer {_helicone_api_key()}",
        }
    elif settings.openai_base_url.strip():
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
    *,
    extra_headers: dict[str, str] | None = None,
) -> T:
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=response_model,
        extra_headers=extra_headers,
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
    extra_headers: dict[str, str] | None = None,
) -> str:
    """Plain chat completion for RAG answers and ticket narrative (no structured parse)."""
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        extra_headers=extra_headers,
    )
    choice = completion.choices[0].message
    content = (choice.content or "").strip()
    if not content:
        raise RuntimeError("Chat completion returned empty content")
    return content
