"""
End-to-end wiring: gdrive_kb results/removals + text -> chunk -> embed -> FAISS.

Consumes output from gdrive_kb.run(); does not call the Drive API.

Usage (from repo root):
    python scripts/sync_pipeline.py   # self-check with fake data
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunk_store import (  # noqa: E402
    delete_chunks_for_file,
    get_chunk_count,
    get_chunk_ids_for_file,
    init_db,
    save_chunks,
)
from chunker import chunk_text  # noqa: E402
from embedder import embed_texts, get_embedding_dim  # noqa: E402
from faiss_store import add_chunks, load_index, remove_chunks, save_index  # noqa: E402

DEFAULT_FAISS_INDEX_PATH = "faiss_index.bin"


def _faiss_index_path() -> str:
    return os.environ.get("FAISS_INDEX_PATH", DEFAULT_FAISS_INDEX_PATH).strip() or (
        DEFAULT_FAISS_INDEX_PATH
    )


def _purge_file_chunks(index, file_id: str) -> None:
    """Remove a file's vectors from FAISS, then delete its chunk_store rows."""
    old_ids = get_chunk_ids_for_file(file_id)
    if not old_ids:
        return

    removed_count = remove_chunks(index, old_ids)
    if removed_count != len(old_ids):
        print(
            f"WARN: FAISS removed {removed_count} of {len(old_ids)} chunk(s) "
            f"for file_id {file_id!r} — investigate index/store drift"
        )
    delete_chunks_for_file(file_id)


def sync_to_faiss(
    results: list[dict],
    removals: list[str],
    text_lookup: dict[str, str],
) -> None:
    """
    Sync chunk metadata and FAISS vectors from gdrive_kb.run() output.

    results: files needing re-embedding (NEW / CHANGED / REPLACED)
    removals: file_ids whose old chunks must be deleted first
    text_lookup: file_id -> full extracted text for chunking
    """
    init_db()
    faiss_path = _faiss_index_path()
    index = load_index(faiss_path, dim=get_embedding_dim())

    for file_id in removals:
        _purge_file_chunks(index, file_id)

    files_added_or_updated = 0
    for file_entry in results:
        file_id = file_entry["file_id"]
        status = file_entry.get("status", "")

        if status in ("CHANGED", "REPLACED") and get_chunk_ids_for_file(file_id):
            _purge_file_chunks(index, file_id)

        if file_id not in text_lookup:
            print(f"WARN: no text for file_id {file_id!r} ({file_entry.get('name')}) — skipping")
            continue

        text = text_lookup[file_id]
        new_chunks = chunk_text(text, file_id, source_name=file_entry["name"])
        if not new_chunks:
            print(
                f"WARN: chunk_text produced no chunks for {file_entry.get('name')!r} "
                f"(file_id {file_id!r}) — skipping"
            )
            continue

        vectors = embed_texts([chunk["text"] for chunk in new_chunks])
        chunk_ids = [chunk["chunk_id"] for chunk in new_chunks]
        add_chunks(index, chunk_ids, vectors)
        save_chunks(new_chunks)
        files_added_or_updated += 1

    save_index(index, faiss_path)

    store_count = get_chunk_count()
    faiss_count = int(index.ntotal)
    counts_match = store_count == faiss_count

    print("\n--- sync summary ---")
    print(f"Files added/updated: {files_added_or_updated}")
    print(f"Files removed:       {len(removals)}")
    print(f"chunk_store rows:  {store_count}")
    print(f"FAISS vectors:     {faiss_count}")
    if counts_match:
        print("Sanity check:      PASS — chunk_store count matches FAISS ntotal")
    else:
        print(
            "Sanity check:      *** FAIL *** — chunk_store and FAISS counts diverge! "
            f"store={store_count}, faiss={faiss_count}"
        )


if __name__ == "__main__":
    import gc
    import time
    from tempfile import TemporaryDirectory

    from chunker import make_chunk_id  # noqa: E402

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        os.environ["CHUNK_STORE_DB"] = str(tmp_path / "test_sync_chunk_store.db")
        os.environ["FAISS_INDEX_PATH"] = str(tmp_path / "test_sync_faiss.bin")

        file_new = "fake_file_new"
        file_changed = "fake_file_changed"
        file_removed = "fake_file_removed"

        init_db()
        stale_chunks = [
            {
                "chunk_id": make_chunk_id(file_changed, 0),
                "file_id": file_changed,
                "source_name": "old_policy.pdf",
                "chunk_index": 0,
                "text": "Old refund policy text that will be replaced.",
            },
            {
                "chunk_id": make_chunk_id(file_removed, 0),
                "file_id": file_removed,
                "source_name": "deleted_doc.pdf",
                "chunk_index": 0,
                "text": "This document was deleted from Drive.",
            },
        ]
        save_chunks(stale_chunks)

        index = load_index(_faiss_index_path(), dim=get_embedding_dim())
        stale_vectors = embed_texts([chunk["text"] for chunk in stale_chunks])
        add_chunks(index, [chunk["chunk_id"] for chunk in stale_chunks], stale_vectors)
        save_index(index, _faiss_index_path())

        results = [
            {
                "file_id": file_new,
                "name": "welcome.pdf",
                "modified_time": "2026-07-07T00:00:00.000Z",
                "char_count": 120,
                "hash": "abc123",
                "status": "NEW",
                "chunk_count": 0,
            },
            {
                "file_id": file_changed,
                "name": "refund_policy.pdf",
                "modified_time": "2026-07-07T01:00:00.000Z",
                "char_count": 200,
                "hash": "def456",
                "status": "CHANGED",
                "chunk_count": 0,
            },
        ]
        removals = [file_removed, file_changed]
        text_lookup = {
            file_new: (
                "Welcome to our support knowledge base.\n\n"
                "This guide explains how to open a ticket and what to expect."
            ),
            file_changed: (
                "Refunds are available within thirty days of purchase.\n\n"
                "After thirty days, manager approval is required before processing."
            ),
        }

        print("Running sync_to_faiss self-check with fake Drive output...\n")
        sync_to_faiss(results, removals, text_lookup)

        remaining_ids = get_chunk_ids_for_file(file_removed)
        changed_ids = get_chunk_ids_for_file(file_changed)
        new_ids = get_chunk_ids_for_file(file_new)
        demo_ok = (
            remaining_ids == []
            and len(changed_ids) >= 1
            and len(new_ids) >= 1
            and get_chunk_count() == load_index(_faiss_index_path(), dim=get_embedding_dim()).ntotal
        )
        print(f"\nSelf-check wiring: {'PASS' if demo_ok else 'FAIL'}")

        gc.collect()
        faiss_file = Path(os.environ["FAISS_INDEX_PATH"])
        db_file = Path(os.environ["CHUNK_STORE_DB"])
        for _ in range(10):
            try:
                faiss_file.unlink(missing_ok=True)
                db_file.unlink(missing_ok=True)
                break
            except PermissionError:
                time.sleep(0.05)
