"""OpenAI text generation for RAG answers and ticket summaries (Phases 7–8)."""

from __future__ import annotations

from app.core.config import AppSettings
from app.llm.client import complete_text, get_openai_client
from app.llm.prompts import build_rag_qa_messages, build_ticket_summary_messages
from app.llm.router import model_for_task


def generate_rag_answer(
    *,
    context_chunks: list[str],
    current_message: str,
    history: list[dict[str, str]],
    settings: AppSettings,
) -> str:
    client = get_openai_client(settings)
    messages = build_rag_qa_messages(context_chunks, history, current_message)
    model = model_for_task(settings, "rag_qa")
    return complete_text(client, model, messages)


def generate_ticket_narrative(
    *,
    ticket_fields: dict[str, str],
    user_message: str,
    settings: AppSettings,
) -> str:
    client = get_openai_client(settings)
    messages = build_ticket_summary_messages(ticket_fields, user_message)
    model = model_for_task(settings, "ticket_summary")
    return complete_text(client, model, messages)
