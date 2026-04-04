"""Run eval cases through run_chat_turn and print scores (Phase 1.4 + 1.3 judges)."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from app.core.config import get_settings
from app.llm.client import apply_langsmith_env_from_settings
from app.eval.judges import (
    score_correctness_judge,
    score_faithfulness_judge,
    score_retrieval_sources,
)
from app.eval.load_dataset import default_dataset_path, load_eval_cases
from app.eval.metrics import score_routing
from app.memory.redis_client import MemoryStore
from app.orchestrator.agent import MissingOpenAIKeyError, retrieve_rag_chunks, run_chat_turn
from app.retrieval.faiss_store import get_knowledge_index, rebuild_knowledge_index


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _env_run_judges() -> bool:
    raw = os.getenv("EVAL_RUN_JUDGES", "true").strip().lower()
    return raw in ("1", "true", "yes", "on")


def main() -> int:
    load_dotenv(_project_root() / ".env")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        print("ERROR: OPENAI_API_KEY is not set (needed on cache miss).", file=sys.stderr)
        return 1

    settings = get_settings()
    apply_langsmith_env_from_settings(settings)
    rebuild_knowledge_index(settings)
    cases = load_eval_cases(default_dataset_path())
    store = MemoryStore()
    run_judges = _env_run_judges()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    reports_dir = _project_root() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"eval_run_{stamp}.jsonl"

    ok = 0
    with report_path.open("w", encoding="utf-8") as rep:
        for case in cases:
            user_id = case.user_id or f"eval-{case.id}"
            try:
                outcome = run_chat_turn(
                    user_id=user_id,
                    message=case.message,
                    store=store,
                    settings=settings,
                    request_id=f"eval-{stamp}-{case.id}",
                )
            except MissingOpenAIKeyError as e:
                print(f"{case.id} ERROR: {e}", file=sys.stderr)
                row = {
                    "eval_run_stamp": stamp,
                    "case_id": case.id,
                    "route_expected": case.route_expected,
                    "route_ok": False,
                    "error": str(e),
                    "source": None,
                    "cached": None,
                    "intent": None,
                    "response_preview": None,
                }
                rep.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    f"{case.id}\t"
                    "route_ok=False\t"
                    "source=-\t"
                    "cached=-\t"
                    "intent=-\t"
                    f"error={type(e).__name__}",
                )
                print(f"\nRouting pass: {ok}/{len(cases)}")
                print(f"Wrote {report_path}")
                return 1
            except Exception as e:
                print(f"{case.id} ERROR: {e}", file=sys.stderr)
                row = {
                    "eval_run_stamp": stamp,
                    "case_id": case.id,
                    "route_expected": case.route_expected,
                    "route_ok": False,
                    "error": str(e),
                    "source": None,
                    "cached": None,
                    "intent": None,
                    "response_preview": None,
                }
                rep.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    f"{case.id}\t"
                    "route_ok=False\t"
                    f"source=-\t"
                    f"cached=-\t"
                    f"intent=-\t"
                    f"error={type(e).__name__}",
                )
                continue

            m = score_routing(case, outcome)
            row: dict = {
                "eval_run_stamp": stamp,
                "case_id": case.id,
                "route_expected": case.route_expected,
                "route_ok": m.route_matches_expected,
                "source": outcome.source,
                "cached": outcome.from_cache,
                "intent": outcome.intent,
                "response_preview": outcome.response[:400]
                + ("…" if len(outcome.response) > 400 else ""),
            }

            chunks: list[str] = []
            if outcome.source == "question":
                kb = get_knowledge_index()
                if kb is not None:
                    chunks = retrieve_rag_chunks(
                        query=case.message,
                        settings=settings,
                        kb=kb,
                    )

            if run_judges:
                try:
                    corr = score_correctness_judge(
                        case=case,
                        assistant_response=outcome.response,
                        settings=settings,
                    )
                    if corr is not None:
                        row["correctness_score"] = corr.score
                        row["correctness_reason"] = corr.reason
                except Exception as je:
                    row["correctness_error"] = str(je)

                if outcome.source == "question":
                    try:
                        faith = score_faithfulness_judge(
                            assistant_response=outcome.response,
                            context_chunks=chunks,
                            settings=settings,
                        )
                        if faith is not None:
                            row["faithfulness_score"] = faith.score
                            row["faithfulness_reason"] = faith.reason
                    except Exception as je:
                        row["faithfulness_error"] = str(je)

                if outcome.source == "question" and chunks:
                    r_ok = score_retrieval_sources(case, chunks)
                    if r_ok is not None:
                        row["retrieval_sources_ok"] = r_ok

            rep.write(json.dumps(row, ensure_ascii=False) + "\n")
            if m.route_matches_expected:
                ok += 1

            extras: list[str] = []
            if "correctness_score" in row:
                extras.append(f"cor={row['correctness_score']:.2f}")
            if "faithfulness_score" in row:
                extras.append(f"faith={row['faithfulness_score']:.2f}")
            if "retrieval_sources_ok" in row:
                extras.append(f"ret_src={row['retrieval_sources_ok']}")
            extra_s = ("\t" + "\t".join(extras)) if extras else ""
            print(
                f"{case.id}\t"
                f"route_ok={m.route_matches_expected}\t"
                f"source={outcome.source}\t"
                f"cached={outcome.from_cache}\t"
                f"intent={outcome.intent!r}"
                f"{extra_s}",
            )

    print(f"\nRouting pass: {ok}/{len(cases)}")
    print(f"Wrote {report_path}")
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
