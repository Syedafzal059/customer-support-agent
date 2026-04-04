"""LLM judges and rule-based retrieval checks for offline eval."""

from __future__ import annotations

import os

from app.core.config import AppSettings
from app.eval.judge_schemas import ScoreWithReason
from app.eval.schemas import EvalCase
from app.llm.client import complete_parsed, get_openai_client, helicone_extra_headers


def _judge_model(settings: AppSettings) -> str:
    return (os.getenv("OPENAI_EVAL_JUDGE_MODEL") or "").strip() or settings.rag_qa_model


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def score_correctness_judge(
    *,
    case: EvalCase,
    assistant_response: str,
    settings: AppSettings,
) -> ScoreWithReason | None:
    if case.reference_answer:
        spec = "REFERENCE (gold):\n" + case.reference_answer.strip()
    elif case.expected_behavior:
        spec = "RUBRIC (criteria):\n" + case.expected_behavior.strip()
    else:
        return None

    client = get_openai_client(settings)
    model = _judge_model(settings)
    system = (
        "You are a strict evaluation judge. Compare the ASSISTANT response to the "
        "reference or rubric. Score 1.0 if facts and intent align; lower for omissions, "
        "wrong facts, or violations of the rubric. Output structured JSON only via the schema."
    )
    user = (
        f"USER_MESSAGE:\n{case.message}\n\n"
        f"ASSISTANT_RESPONSE:\n{assistant_response.strip()}\n\n"
        f"{spec}\n"
    )
    return complete_parsed(
        client,
        model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        ScoreWithReason,
        extra_headers=helicone_extra_headers(settings, step="eval_correctness_judge"),
    )


def score_faithfulness_judge(
    *,
    assistant_response: str,
    context_chunks: list[str],
    settings: AppSettings,
) -> ScoreWithReason | None:
    if not context_chunks:
        return None

    context_blob = _truncate("\n\n---\n\n".join(context_chunks), 14_000)
    client = get_openai_client(settings)
    model = _judge_model(settings)
    system = (
        "You judge grounding. CONTEXT is the only evidence. If the assistant adds "
        "important facts not supported by CONTEXT, or contradicts CONTEXT, score low. "
        "Score 1.0 if fully supported. Output structured JSON only."
    )
    user = f"CONTEXT:\n{context_blob}\n\nASSISTANT_RESPONSE:\n{assistant_response.strip()}\n"
    return complete_parsed(
        client,
        model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        ScoreWithReason,
        extra_headers=helicone_extra_headers(settings, step="eval_faithfulness_judge"),
    )


def score_retrieval_sources(case: EvalCase, context_chunks: list[str]) -> bool | None:
    """Each expected substring must appear somewhere in retrieved chunks."""
    expected = case.expected_kb_sources
    if not expected:
        return None
    blob = "\n".join(context_chunks)
    for fragment in expected:
        if fragment not in blob:
            return False
    return True
