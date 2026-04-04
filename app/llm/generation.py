"""OpenAI text generation for RAG answers and ticket summaries (Phases 7–8)."""

from __future__ import annotations

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree, set_run_metadata

from app.core.config import AppSettings
from app.llm.client import complete_text, get_openai_client
from app.llm.prompts import build_rag_qa_messages, build_ticket_summary_messages
from app.llm.router import model_for_task


@traceable(name="generate_rag_answer", run_type="chain")
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
    if get_current_run_tree() is not None:
        set_run_metadata(
            rag_model=model,
            chunk_count=len(context_chunks),
        )
    return complete_text(client, model, messages)

@traceable(name="generate_ticket_narrative", run_type="chain")
def generate_ticket_narrative(
    *,
    ticket_fields: dict[str, str],
    user_message: str,
    settings: AppSettings,
) -> str:
    client = get_openai_client(settings)
    messages = build_ticket_summary_messages(ticket_fields, user_message)
    model = model_for_task(settings, "ticket_summary")
    if get_current_run_tree() is not None:
        set_run_metadata(ticket_summary_model=model)
    return complete_text(client, model, messages)
