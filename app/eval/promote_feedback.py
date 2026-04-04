"""Build draft eval JSONL lines from the feedback review queue (Phase 5.3).

Reads ``data/feedback/review_queue.jsonl`` (append-only, not committed). Human review
still required: verify ``route_expected``, add ``reference_answer`` / ``ticket_id_expected``,
then append to a versioned dataset and update ``regression_baseline.json`` if intended.

Usage (from repo root)::

    python -m app.eval.promote_feedback
    python -m app.eval.promote_feedback --input data/feedback/review_queue.jsonl --out /tmp/draft.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _stable_case_id(request_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", request_id.strip())[:80].strip("-")
    return f"fb-{safe}" if safe else "fb-unknown"


def main() -> int:
    p = argparse.ArgumentParser(description="Draft eval cases from feedback review_queue.jsonl")
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Default: <repo>/data/feedback/review_queue.jsonl",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSONL here; default: stdout",
    )
    args = p.parse_args()
    root = _project_root()
    in_path = (
        args.input if args.input is not None else root / "data" / "feedback" / "review_queue.jsonl"
    )
    if not in_path.is_file():
        print(f"No file at {in_path}", file=sys.stderr)
        return 1

    lines_out: list[str] = []
    for line in in_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        msg = row.get("user_message")
        if not isinstance(msg, str) or not msg.strip():
            continue
        rid = str(row.get("request_id") or "")
        src = row.get("source")
        route = src if src in ("question", "ticket") else "question"
        case = {
            "id": _stable_case_id(rid),
            "message": msg.strip(),
            "route_expected": route,
            "tags": ["from_feedback"],
            "expected_behavior": "Imported from feedback queue; set reference_answer or ticket expectations after review.",
        }
        lines_out.append(json.dumps(case, ensure_ascii=False))

    if not lines_out:
        print("No promotable rows (need user_message on each JSON line).", file=sys.stderr)
        return 1

    text = "\n".join(lines_out) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {len(lines_out)} rows to {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
