"""
Controlled regression demo: "stale chunk after document edit" RAG bug.

Proves why delete-then-add sync is necessary by:
  1. Baseline — correct sync (skip_deletion=False)
  2. Manual PDF edit + eval case setup (user-driven pause)
  3. Broken sync — skip_deletion=True (naive add-only)
  4. Fixed sync — skip_deletion=False (proper purge + re-add)
  5. Summary table with chunk counts and eval scores

Usage (from repo root):
    python scripts/regression_stale_chunk_demo.py

Requires: GDRIVE_FOLDER_ID, service account, chunk_store.db, faiss_index.bin
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gdrive_kb  # noqa: E402
from chunk_store import get_chunk_count, init_db  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from embedder import get_embedding_dim  # noqa: E402
from faiss_store import load_index  # noqa: E402
from sync_pipeline import sync_to_faiss  # noqa: E402

RENT_PDF_NAME = "Current_Address_Rent_Agreement.pdf"
EVAL_CASE_ID = "pdf-rent-updated-001"
EVAL_DATASET = "app/eval/datasets/support_eval_v1.jsonl"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _banner(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n  {title}\n{line}\n")


def _subheading(title: str) -> None:
    print(f"\n--- {title} ---")


def _faiss_index_path() -> str:
    return os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin").strip() or "faiss_index.bin"


def _get_store_faiss_counts() -> tuple[int, int]:
    init_db()
    store_count = get_chunk_count()
    index = load_index(_faiss_index_path(), dim=get_embedding_dim())
    faiss_count = int(index.ntotal)
    return store_count, faiss_count


def _print_counts(label: str) -> tuple[int, int]:
    store_count, faiss_count = _get_store_faiss_counts()
    match = "MATCH" if store_count == faiss_count else "MISMATCH"
    print(f"{label}")
    print(f"  chunk_store rows: {store_count}")
    print(f"  FAISS ntotal:     {faiss_count}")
    print(f"  store vs FAISS:   {match}")
    return store_count, faiss_count


def _find_rent_entry(results: list[dict]) -> dict | None:
    for entry in results:
        if entry.get("name") == RENT_PDF_NAME:
            return entry
    return None


def _print_eval_case_instructions() -> None:
    _subheading("Manual step: add eval case")
    print(
        f"Add a new line to {_project_root() / EVAL_DATASET} with id "
        f'"{EVAL_CASE_ID}":\n'
    )
    example = {
        "id": EVAL_CASE_ID,
        "message": "What is the monthly rent in the lease agreement?",
        "route_expected": "question",
        "ticket_id_expected": None,
        "reference_answer": (
            "UPDATE THIS after editing the PDF with the new rent amount in step C"
        ),
        "expected_behavior": None,
        "tags": ["regression_stale_chunk", "pdf_rent"],
        "expected_kb_sources": [RENT_PDF_NAME],
    }
    print(json.dumps(example, ensure_ascii=False))
    print(
        f"\nThen edit {RENT_PDF_NAME} in Google Drive:\n"
        "  1. Open the file in Drive → right-click → Manage versions → Upload new version\n"
        "     (keep the same file_id — do NOT delete and re-upload as a new file)\n"
        "  2. Change the monthly rent amount in the PDF to a clearly different value\n"
        f"  3. Update reference_answer in the eval case to match the NEW rent amount\n"
    )


def _wait_for_manual_setup() -> None:
    print(
        "Press Enter when you have:\n"
        f"  (1) uploaded a new version of {RENT_PDF_NAME} via Manage versions, AND\n"
        f"  (2) updated reference_answer for {EVAL_CASE_ID} in {EVAL_DATASET}\n"
    )
    input(">>> Ready to continue? [Enter] ")


def _print_eval_run_instructions(phase_label: str, expectation: str) -> None:
    _subheading(f"Manual step: run eval ({phase_label})")
    print("From the repo root, run:\n")
    print("    python -m app.eval.run_eval\n")
    print(f"Expected for {EVAL_CASE_ID}: {expectation}\n")
    print(
        "Note the report path printed at the end (reports/eval_run_<timestamp>.jsonl).\n"
    )


def _prompt_eval_report(phase_label: str) -> tuple[str | None, float | None]:
    raw_path = input(
        f">>> Paste the {phase_label} eval report path "
        f"(e.g. reports/eval_run_20260707_120000.jsonl), or Enter to skip: "
    ).strip()
    if not raw_path:
        print(f"  (skipped — no {phase_label} report path recorded)")
        return None, None

    report_path = Path(raw_path)
    if not report_path.is_absolute():
        report_path = _project_root() / report_path
    if not report_path.is_file():
        print(f"  WARN: file not found: {report_path}")
        return str(report_path), None

    score = _read_eval_score(report_path, EVAL_CASE_ID)
    if score is not None:
        print(f"  {EVAL_CASE_ID} correctness_score = {score:.2f}")
    else:
        print(f"  WARN: could not find {EVAL_CASE_ID} score in {report_path}")
    return str(report_path.relative_to(_project_root())), score


def _read_eval_score(report_path: Path, case_id: str) -> float | None:
    for line in report_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("case_id") == case_id:
            raw = row.get("correctness_score")
            return float(raw) if raw is not None else None
    return None


def _print_summary_table(
    baseline: tuple[int, int],
    broken: tuple[int, int],
    fixed: tuple[int, int],
    broken_report: str | None,
    fixed_report: str | None,
    broken_score: float | None,
    fixed_score: float | None,
) -> None:
    _banner("STEP E — Final summary")

    def _fmt_counts(counts: tuple[int, int]) -> str:
        store, faiss = counts
        return f"store={store}, FAISS={faiss}"

    rows = [
        ("Baseline (correct sync)", _fmt_counts(baseline), "—"),
        ("Broken (skip_deletion=True)", _fmt_counts(broken), _fmt_score(broken_score)),
        ("Fixed (skip_deletion=False)", _fmt_counts(fixed), _fmt_score(fixed_score)),
    ]

    col_w = [28, 24, 18]
    header = f"{'Phase':<{col_w[0]}} {'chunk_store / FAISS':<{col_w[1]}} {'eval score':<{col_w[2]}}"
    print(header)
    print("-" * len(header))
    for phase, counts, score in rows:
        print(f"{phase:<{col_w[0]}} {counts:<{col_w[1]}} {score:<{col_w[2]}}")

    print("\nEval report artifacts:")
    print(f"  BEFORE (broken): {broken_report or '(not recorded)'}")
    print(f"  AFTER  (fixed):  {fixed_report or '(not recorded)'}")

    if broken[0] > baseline[0] or broken[1] > baseline[1]:
        print(
            "\nEvidence: broken sync INCREASED vector counts — stale chunks "
            "coexisted with new ones."
        )
    if fixed[0] <= baseline[0] and fixed[1] <= baseline[1]:
        print(
            "Evidence: fixed sync restored counts to baseline — duplicates "
            "and stale chunks were purged."
        )
    if broken_score is not None and fixed_score is not None:
        delta = fixed_score - broken_score
        print(
            f"\nEval delta for {EVAL_CASE_ID}: {broken_score:.2f} → {fixed_score:.2f} "
            f"({delta:+.2f})"
        )


def _fmt_score(score: float | None) -> str:
    return f"{score:.2f}" if score is not None else "(not recorded)"


def main() -> int:
    load_dotenv(_project_root() / ".env")

    _banner("Stale-chunk regression demo")
    print(
        "This script deliberately reproduces the classic RAG bug where edited "
        "documents leave stale chunks in FAISS, then proves the delete-then-add "
        "fix resolves it.\n"
        "Production sync always uses skip_deletion=False (the default)."
    )

    # ── STEP A: baseline ──────────────────────────────────────────────────
    _banner("STEP A — Capture baseline (correct sync)")
    print(
        "Running gdrive_kb.run() + sync_to_faiss(skip_deletion=False) against "
        "your real Drive folder to establish a known-good index.\n"
    )

    results_a, removals_a, text_lookup_a = gdrive_kb.run()
    sync_to_faiss(results_a, removals_a, text_lookup_a, skip_deletion=False)
    baseline_counts = _print_counts("Baseline counts after correct sync:")

    # ── STEP B: manual setup ────────────────────────────────────────────
    _banner("STEP B — Manual PDF edit + eval case")
    _print_eval_case_instructions()
    _wait_for_manual_setup()

    # ── STEP C: broken sync ─────────────────────────────────────────────
    _banner("STEP C — Reproduce the failure (naive / broken sync)")
    print(
        "WHY: A naive pipeline only ever adds embeddings on file change — it never "
        "removes the old vectors. We simulate that with skip_deletion=True.\n"
        "WHAT TO EXPECT: chunk_store and FAISS counts will RISE above baseline "
        "because old rent chunks stay indexed alongside the new ones.\n"
    )

    results_c, removals_c, text_lookup_c = gdrive_kb.run()
    rent_entry = _find_rent_entry(results_c)
    if rent_entry is None:
        print(
            f"WARN: {RENT_PDF_NAME} not in gdrive_kb results — "
            "Drive may not have detected the edit yet. Continuing anyway.\n"
        )
    else:
        print(
            f"Detected {RENT_PDF_NAME}: status={rent_entry['status']!r}, "
            f"file_id={rent_entry['file_id']!r}, "
            f"chunks={rent_entry['chunk_count']}\n"
        )

    sync_to_faiss(results_c, removals_c, text_lookup_c, skip_deletion=True)
    broken_counts = _print_counts("Counts after BROKEN sync (skip_deletion=True):")

    if broken_counts[0] > baseline_counts[0] or broken_counts[1] > baseline_counts[1]:
        print(
            "\n  ✓ Counts rose above baseline — stale + new chunks are coexisting "
            "(classic bug reproduced)."
        )
    else:
        print(
            "\n  NOTE: counts did not rise — the edit may not have changed chunk "
            "boundaries, or FAISS replaced same-ID vectors in-place. The eval "
            "score comparison in steps C/D is still the primary evidence."
        )

    _print_eval_run_instructions(
        "BROKEN state",
        "stale or mixed answer citing the OLD rent amount → low correctness score",
    )
    broken_report, broken_score = _prompt_eval_report("BROKEN")
    if broken_report:
        print(f"\nBROKEN-STATE REPORT: {broken_report} — keep this file, it's your 'before' evidence.")

    # ── STEP D: fixed sync ──────────────────────────────────────────────
    _banner("STEP D — Apply the real fix (correct sync)")
    print(
        "WHY: Production sync deletes ALL existing chunks for a changed file_id "
        "before re-embedding — including duplicates from the broken run above.\n"
        "Reusing the Step C gdrive payload (Drive already recorded the new hash; "
        "a fresh gdrive_kb.run() would skip the file as unchanged).\n"
    )

    sync_to_faiss(results_c, removals_c, text_lookup_c, skip_deletion=False)
    fixed_counts = _print_counts("Counts after FIXED sync (skip_deletion=False):")

    if fixed_counts[0] <= baseline_counts[0] and fixed_counts[1] <= baseline_counts[1]:
        print(
            "\n  ✓ Counts back at or below baseline — stale duplicates purged, "
            "only current chunks remain."
        )

    _print_eval_run_instructions(
        "FIXED state",
        "answer cites the NEW rent amount → high / passing correctness score",
    )
    fixed_report, fixed_score = _prompt_eval_report("FIXED")
    if fixed_report:
        print(f"\nFIXED-STATE REPORT: {fixed_report} — keep this file, it's your 'after' evidence.")

    # ── STEP E: summary ─────────────────────────────────────────────────
    _print_summary_table(
        baseline=baseline_counts,
        broken=broken_counts,
        fixed=fixed_counts,
        broken_report=broken_report,
        fixed_report=fixed_report,
        broken_score=broken_score,
        fixed_score=fixed_score,
    )

    print(
        "\n" + "=" * 72 + "\n"
        "REMINDER: This script used skip_deletion=True to simulate a naive sync. "
        "Production sync_pipeline.py calls always use skip_deletion=False "
        "(the default) — this flag exists only for this demonstration and must "
        "never be used in the real Drive sync job.\n"
        + "=" * 72
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
