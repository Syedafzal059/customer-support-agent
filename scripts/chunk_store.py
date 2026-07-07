"""
SQLite-backed metadata store for text chunks destined for FAISS IndexIDMap.

FAISS only stores vectors and integer IDs — this side-store tracks file_id,
source_name, chunk_index, and text for each chunk_id.

Usage (from repo root):
    python scripts/chunk_store.py   # runs the self-check demo at the bottom
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any

DEFAULT_DB_PATH = "chunk_store.db"

_REQUIRED_CHUNK_KEYS = frozenset({"chunk_id", "file_id", "source_name", "chunk_index", "text"})


class ChunkStoreError(Exception):
    """Base exception for chunk store operations."""


class ChunkStoreConnectionError(ChunkStoreError):
    """Raised when opening a connection to the SQLite database fails."""


class ChunkStoreOperationError(ChunkStoreError):
    """Raised when a read/write operation against the chunk store fails."""


def _db_path() -> str:
    return os.environ.get("CHUNK_STORE_DB", DEFAULT_DB_PATH).strip() or DEFAULT_DB_PATH


def _connect() -> sqlite3.Connection:
    path = _db_path()
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as exc:
        raise ChunkStoreConnectionError(
            f"Failed to connect to chunk store at {path!r}: {exc}"
        ) from exc


def init_db() -> None:
    """Create the chunks table and file_id index if they do not already exist."""
    try:
        with _connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id INTEGER PRIMARY KEY,
                    file_id TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_file_id ON chunks(file_id);
                """
            )
    except ChunkStoreConnectionError:
        raise
    except sqlite3.Error as exc:
        raise ChunkStoreOperationError(
            f"Failed to initialize chunk store schema: {exc}"
        ) from exc


def save_chunks(chunks: list[dict[str, Any]]) -> None:
    """Insert or replace rows for a list of chunk dicts keyed on chunk_id."""
    if not chunks:
        return

    rows: list[tuple[int, str, str, int, str]] = []
    for chunk in chunks:
        missing = _REQUIRED_CHUNK_KEYS - chunk.keys()
        if missing:
            raise ChunkStoreOperationError(
                f"Chunk dict missing required keys {sorted(missing)}: {chunk!r}"
            )
        rows.append(
            (
                int(chunk["chunk_id"]),
                str(chunk["file_id"]),
                str(chunk["source_name"]),
                int(chunk["chunk_index"]),
                str(chunk["text"]),
            )
        )

    try:
        with _connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks
                    (chunk_id, file_id, source_name, chunk_index, text)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
    except ChunkStoreConnectionError:
        raise
    except sqlite3.Error as exc:
        raise ChunkStoreOperationError(f"Failed to save {len(rows)} chunk(s): {exc}") from exc


def get_chunk_ids_for_file(file_id: str) -> list[int]:
    """Return all chunk_ids stored for file_id, or [] if none exist."""
    try:
        with _connect() as conn:
            cursor = conn.execute(
                "SELECT chunk_id FROM chunks WHERE file_id = ? ORDER BY chunk_index",
                (file_id,),
            )
            return [int(row["chunk_id"]) for row in cursor.fetchall()]
    except ChunkStoreConnectionError:
        raise
    except sqlite3.Error as exc:
        raise ChunkStoreOperationError(
            f"Failed to look up chunk_ids for file_id {file_id!r}: {exc}"
        ) from exc


def delete_chunks_for_file(file_id: str) -> int:
    """Delete all rows for file_id. Call only after FAISS vectors are removed."""
    try:
        with _connect() as conn:
            cursor = conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
            return int(cursor.rowcount)
    except ChunkStoreConnectionError:
        raise
    except sqlite3.Error as exc:
        raise ChunkStoreOperationError(
            f"Failed to delete chunks for file_id {file_id!r}: {exc}"
        ) from exc


def get_chunk_text(chunk_id: int) -> dict[str, str | int] | None:
    """Return metadata and text for chunk_id, or None if it does not exist."""
    try:
        with _connect() as conn:
            row = conn.execute(
                """
                SELECT file_id, source_name, chunk_index, text
                FROM chunks
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()
    except ChunkStoreConnectionError:
        raise
    except sqlite3.Error as exc:
        raise ChunkStoreOperationError(
            f"Failed to look up chunk_id {chunk_id}: {exc}"
        ) from exc

    if row is None:
        return None

    return {
        "file_id": str(row["file_id"]),
        "source_name": str(row["source_name"]),
        "chunk_index": int(row["chunk_index"]),
        "text": str(row["text"]),
    }


def get_all_file_ids() -> set[str]:
    """Return the distinct set of file_ids currently represented in the store."""
    try:
        with _connect() as conn:
            cursor = conn.execute("SELECT DISTINCT file_id FROM chunks")
            return {str(row["file_id"]) for row in cursor.fetchall()}
    except ChunkStoreConnectionError:
        raise
    except sqlite3.Error as exc:
        raise ChunkStoreOperationError(f"Failed to list file_ids: {exc}") from exc


def get_chunk_count() -> int:
    """Return the total number of chunk rows in the store."""
    try:
        with _connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
            return int(row["n"]) if row is not None else 0
    except ChunkStoreConnectionError:
        raise
    except sqlite3.Error as exc:
        raise ChunkStoreOperationError(f"Failed to count chunks: {exc}") from exc


def _check(label: str, condition: bool, detail: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"{status}: {label} {detail}")


if __name__ == "__main__":
    import gc
    import time
    from pathlib import Path

    test_db = Path("test_chunk_store.db")
    if test_db.exists():
        test_db.unlink()

    os.environ["CHUNK_STORE_DB"] = str(test_db)

    file_a = "gdrive_file_aaa"
    file_b = "gdrive_file_bbb"
    chunk_a0 = {
        "chunk_id": 100001,
        "file_id": file_a,
        "source_name": "refund_policy.pdf",
        "chunk_index": 0,
        "text": "Refunds are available within thirty days.",
    }
    chunk_a1 = {
        "chunk_id": 100002,
        "file_id": file_a,
        "source_name": "refund_policy.pdf",
        "chunk_index": 1,
        "text": "After thirty days, manager approval is required.",
    }
    chunk_b0 = {
        "chunk_id": 200001,
        "file_id": file_b,
        "source_name": "shipping_faq.pdf",
        "chunk_index": 0,
        "text": "Standard shipping takes three to five business days.",
    }

    init_db()
    save_chunks([chunk_a0, chunk_a1, chunk_b0])

    ids_for_a = get_chunk_ids_for_file(file_a)
    _check(
        "get_chunk_ids_for_file",
        ids_for_a == [100001, 100002],
        f"returned {len(ids_for_a)} ids as expected"
        if ids_for_a == [100001, 100002]
        else f"expected [100001, 100002], got {ids_for_a}",
    )

    unknown_ids = get_chunk_ids_for_file("nonexistent_file")
    _check(
        "get_chunk_ids_for_file (unknown file)",
        unknown_ids == [],
        "returned empty list for unknown file_id"
        if unknown_ids == []
        else f"expected [], got {unknown_ids}",
    )

    row = get_chunk_text(100001)
    expected_row = {
        "file_id": file_a,
        "source_name": "refund_policy.pdf",
        "chunk_index": 0,
        "text": chunk_a0["text"],
    }
    _check(
        "get_chunk_text",
        row == expected_row,
        "returned the expected row for chunk_id 100001"
        if row == expected_row
        else f"expected {expected_row!r}, got {row!r}",
    )

    missing_row = get_chunk_text(999999)
    _check(
        "get_chunk_text (unknown chunk)",
        missing_row is None,
        "returned None for unknown chunk_id"
        if missing_row is None
        else f"expected None, got {missing_row!r}",
    )

    file_ids = get_all_file_ids()
    _check(
        "get_all_file_ids",
        file_ids == {file_a, file_b},
        f"returned {len(file_ids)} distinct file_ids as expected"
        if file_ids == {file_a, file_b}
        else f"expected {{{file_a!r}, {file_b!r}}}, got {file_ids!r}",
    )

    deleted = delete_chunks_for_file(file_a)
    _check(
        "delete_chunks_for_file",
        deleted == 2,
        f"deleted {deleted} row(s) as expected"
        if deleted == 2
        else f"expected 2 deleted rows, got {deleted}",
    )

    ids_after_delete = get_chunk_ids_for_file(file_a)
    _check(
        "get_chunk_ids_for_file after deletion",
        ids_after_delete == [],
        "returned empty list after deletion"
        if ids_after_delete == []
        else f"expected 0 ids after deletion, got {len(ids_after_delete)}",
    )

    remaining_file_ids = get_all_file_ids()
    _check(
        "get_all_file_ids after deletion",
        remaining_file_ids == {file_b},
        "only the untouched file_id remains"
        if remaining_file_ids == {file_b}
        else f"expected {{{file_b!r}}}, got {remaining_file_ids!r}",
    )

    # Re-save with same chunk_id to verify INSERT OR REPLACE overwrites cleanly.
    updated_chunk = dict(chunk_a0)
    updated_chunk["text"] = "Updated refund policy text."
    save_chunks([updated_chunk])
    updated_row = get_chunk_text(100001)
    _check(
        "save_chunks INSERT OR REPLACE",
        updated_row is not None and updated_row["text"] == updated_chunk["text"],
        "overwrote existing chunk_id without duplication"
        if updated_row and updated_row["text"] == updated_chunk["text"]
        else "INSERT OR REPLACE did not update the row as expected",
    )

    gc.collect()
    for _ in range(10):
        try:
            test_db.unlink(missing_ok=True)
            print(f"Cleaned up {test_db}")
            break
        except PermissionError:
            time.sleep(0.05)
    else:
        print(f"WARN: could not remove {test_db} (file still in use)")
