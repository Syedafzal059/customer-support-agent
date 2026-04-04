"""Load evaluation datasets (JSONL) into EvalCase rows."""

from __future__ import annotations
import json 
from collections.abc import Iterator
from pathlib import Path
from app.eval.schemas import EvalCase


def iter_eval_cases_jsonl(path: Path | str) -> Iterator[EvalCase]:
    """Yield one EvalCase per non-empty line. Raises ValueError with file:line on failure."""
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}: {line_no}: Invalid JSON: {e}") from e

            try:
                yield EvalCase.model_validate(raw)
            except Exception as e:
                raise ValueError(f"{p}:{line_no}: invalid EvalCase: {e}") from e



def load_eval_cases(path: Path | str) -> list[EvalCase]:
    """Load the whole JSONL into memory (fine for small regression sets)."""
    return list(iter_eval_cases_jsonl(path))



def default_dataset_path() -> Path:
    """Default: app/eval/datasets/support_eval_v1.jsonl next to this package."""
    return Path(__file__).resolve().parent / "datasets" / "support_eval_v1.jsonl"