"""
Build faiss_index.bin + chunk_store.db for CI offline eval (no Drive API).

Text and source_name values align with app/eval/datasets/support_eval_v1.jsonl so
routing eval can load the same RAG path used in production.

Usage (from repo root):
    python scripts/build_ci_kb_fixtures.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from sync_pipeline import sync_to_faiss  # noqa: E402

_RENT_FILE_ID = "ci_fixture_rent_agreement"
_RESUME_FILE_ID = "ci_fixture_resume"


def main() -> int:
    os.environ["FAISS_INDEX_PATH"] = str(_REPO_ROOT / "faiss_index.bin")
    os.environ["CHUNK_STORE_DB"] = str(_REPO_ROOT / "chunk_store.db")

    results = [
        {
            "file_id": _RENT_FILE_ID,
            "name": "Current_Address_Rent_Agreement.pdf",
            "modified_time": "2026-07-07T00:00:00.000Z",
            "char_count": 800,
            "hash": "ci-rent-fixture",
            "status": "NEW",
            "chunk_count": 0,
        },
        {
            "file_id": _RESUME_FILE_ID,
            "name": "S_M_AFZAL_HASHMI.pdf",
            "modified_time": "2026-07-07T00:00:00.000Z",
            "char_count": 400,
            "hash": "ci-resume-fixture",
            "status": "NEW",
            "chunk_count": 0,
        },
    ]
    text_lookup = {
        _RENT_FILE_ID: (
            "LEASE AGREEMENT made at Bangalore.\n\n"
            "The lessor (OWNER) is Mrs. SREE RAMADEVI M, residing at No 72, 3rd Main, "
            "Karnataka Layout, 1st Stage, WCR Bangalore North, Bangalore - 560086.\n\n"
            "The lessee shall pay a monthly rent of INR 23,000 (Rupees Twenty Three "
            "Thousand Only) by the first of every month in advance.\n\n"
            "The lessee has paid INR 46,000 as an interest-free security deposit; "
            "INR 23,000 of that is non-refundable for cleaning, painting, and "
            "depreciation of furnishings."
        ),
        _RESUME_FILE_ID: (
            "S M Afzal Hashmi\n"
            "Experience\n"
            "Apexon | Senior Data Scientist — Team Lead May 2024 – Present\n"
            "Senior AI Engineer · Agentic AI & LLM Systems"
        ),
    }

    sync_to_faiss(results, [], text_lookup)
    print("CI KB fixtures written to faiss_index.bin and chunk_store.db")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
