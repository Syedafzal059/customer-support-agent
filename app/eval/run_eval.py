"""Run eval cases through run_chat_turn and print routing scores (Phase 1.4)."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from app.core.config import get_settings
from app.eval.load_dataset import default_dataset_path, load_eval_cases
from app.eval.metrics import score_routing
from app.memory.redis_client import MemoryStore
from app.orchestrator.agent import MissingOpenAIKeyError, run_chat_turn
from app.retrieval.faiss_store import rebuild_knowledge_index


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def main() -> int:
    load_dotenv(_project_root() / ".env")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        print("ERROR: OPENAI_API_KEY is not set (needed on cache miss).", file=sys.stderr)
        return 1

    settings = get_settings()
    rebuild_knowledge_index(settings)
    cases = load_eval_cases(default_dataset_path())
    store = MemoryStore()

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
            row = {
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
            rep.write(json.dumps(row, ensure_ascii=False) + "\n")
            if m.route_matches_expected:
                ok += 1
            print(
                f"{case.id}\t"
                f"route_ok={m.route_matches_expected}\t"
                f"source={outcome.source}\t"
                f"cached={outcome.from_cache}\t"
                f"intent={outcome.intent!r}",
            )

    print(f"\nRouting pass: {ok}/{len(cases)}")
    print(f"Wrote {report_path}")
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
