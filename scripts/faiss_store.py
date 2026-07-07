"""
FAISS IndexIDMap wrapper for add-by-id, remove-by-id, search, and persistence.

A plain FAISS index only supports positional add — no delete-by-chunk_id.
IndexIDMap wraps IndexFlatL2 so we can use the same deterministic chunk_id
integers produced by chunker.py and stored in chunk_store.py.

This module uses dummy random vectors for testing so FAISS/ID-mapping bugs can
be isolated from embedding-model bugs. No real embedding calls here.

Usage (from repo root):
    python scripts/faiss_store.py   # runs the self-check demo at the bottom
"""

from __future__ import annotations

import os
import sys

import faiss
import numpy as np

VECTOR_DIM = 128
DEFAULT_INDEX_PATH = "faiss_index.bin"

FAISS_NO_RESULT_ID = -1


class FaissStoreError(Exception):
    """Raised when index persistence fails."""


def _index_path() -> str:
    return os.environ.get("FAISS_INDEX_PATH", DEFAULT_INDEX_PATH).strip() or DEFAULT_INDEX_PATH


def build_index(dim: int = VECTOR_DIM) -> faiss.IndexIDMap:
    """Create a fresh IndexIDMap over IndexFlatL2(dim).

    IndexIDMap is required because our pipeline assigns its own chunk_id per
    vector — we need add_with_ids / remove_ids, not positional add only.
    """
    base = faiss.IndexFlatL2(dim)
    return faiss.IndexIDMap(base)


def load_index(path: str | None = None, dim: int = VECTOR_DIM) -> faiss.IndexIDMap:
    """Load a persisted index from disk, or return a fresh one if missing."""
    index_path = path if path is not None else _index_path()
    if os.path.isfile(index_path):
        try:
            return faiss.read_index(index_path)
        except Exception as exc:
            raise FaissStoreError(f"Failed to read FAISS index from {index_path!r}: {exc}") from exc
    return build_index(dim)


def save_index(index: faiss.IndexIDMap, path: str | None = None) -> None:
    """Atomically persist index to disk via a temp file + os.replace.

    Writing directly to the target path risks a corrupted index if the process
    crashes mid-write. The tmp-then-replace pattern keeps the on-disk index
    always valid or absent, never half-written.
    """
    index_path = path if path is not None else _index_path()
    tmp_path = f"{index_path}.tmp"

    try:
        faiss.write_index(index, tmp_path)
        os.replace(tmp_path, index_path)
    except Exception as exc:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise FaissStoreError(f"Failed to save FAISS index to {index_path!r}: {exc}") from exc


def add_chunks(
    index: faiss.IndexIDMap,
    chunk_ids: list[int],
    vectors: np.ndarray,
) -> None:
    """Add vectors keyed by chunk_id so later removal targets exact chunks."""
    if not chunk_ids:
        return

    expected_rows = len(chunk_ids)
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2-D (n, dim); got shape {vectors.shape!r}")
    if vectors.shape[0] != expected_rows:
        raise ValueError(
            f"len(chunk_ids)={expected_rows} but vectors has "
            f"{vectors.shape[0]} rows (shape {vectors.shape!r})"
        )
    if vectors.shape[1] != index.d:
        raise ValueError(f"vectors dim {vectors.shape[1]} does not match index dim {index.d}")

    id_array = np.array(chunk_ids, dtype="int64")
    index.add_with_ids(vectors.astype("float32"), id_array)


def remove_chunks(index: faiss.IndexIDMap, chunk_ids: list[int]) -> int:
    """Remove vectors by chunk_id — the reason we use IndexIDMap at all.

    When a Drive file changes, chunk_store tells us which chunk_ids belonged
    to the old version; we must drop those exact vectors before re-adding the
    re-chunked set. A plain index has no remove-by-id — only IndexIDMap does.
    The returned count is surfaced so callers can detect store/index drift.
    """
    if not chunk_ids:
        return 0

    id_array = np.array(chunk_ids, dtype="int64")
    removed = index.remove_ids(id_array)
    return int(removed)


def search(
    index: faiss.IndexIDMap,
    query_vector: np.ndarray,
    k: int = 5,
) -> list[tuple[int, float]]:
    """Return up to k (chunk_id, L2 distance) pairs, nearest first.

    FAISS pads short result lists with id -1 when fewer than k vectors exist;
    those sentinels must be filtered out or RAG would try to look up nonsense IDs.
    """
    if k <= 0:
        return []

    query = np.asarray(query_vector, dtype="float32")
    if query.ndim == 1:
        query = query.reshape(1, -1)
    elif query.ndim != 2 or query.shape[0] != 1:
        raise ValueError(
            f"query_vector must be 1-D (dim,) or 2-D (1, dim); got shape {query.shape!r}"
        )
    if query.shape[1] != index.d:
        raise ValueError(f"query dim {query.shape[1]} does not match index dim {index.d}")

    distances, ids = index.search(query, k)
    results: list[tuple[int, float]] = []
    for chunk_id, distance in zip(ids[0], distances[0]):
        if int(chunk_id) == FAISS_NO_RESULT_ID:
            continue
        results.append((int(chunk_id), float(distance)))
    return results


def _check(label: str, condition: bool, detail: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"{status}: {label} {detail}")
    return condition


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    checks_passed = 0
    checks_total = 0

    index = build_index()

    file_a_ids = [101, 102, 103, 104, 105]
    file_a_vectors = rng.standard_normal((len(file_a_ids), VECTOR_DIM)).astype("float32")
    vector_103 = file_a_vectors[2].copy()
    query_near_103 = (vector_103 + rng.standard_normal(VECTOR_DIM).astype("float32") * 0.01).astype(
        "float32"
    )

    add_chunks(index, file_a_ids, file_a_vectors)

    results_before_remove = search(index, query_near_103, k=3)
    top_id = results_before_remove[0][0] if results_before_remove else None
    checks_total += 1
    if _check(
        "search before remove",
        top_id == 103,
        f"chunk_id 103 is nearest (got top={top_id}, results={results_before_remove})"
        if top_id == 103
        else f"expected chunk_id 103 as top result, got top={top_id}, results={results_before_remove}",
    ):
        checks_passed += 1

    removed_count = remove_chunks(index, file_a_ids)
    checks_total += 1
    if _check(
        "remove_chunks count",
        removed_count == 5,
        f"removed {removed_count} vectors as expected"
        if removed_count == 5
        else f"expected 5 removed, got {removed_count}",
    ):
        checks_passed += 1

    results_after_remove = search(index, query_near_103, k=3)
    ids_after_remove = [chunk_id for chunk_id, _ in results_after_remove]
    checks_total += 1
    if _check(
        "search after remove",
        103 not in ids_after_remove,
        "chunk_id 103 no longer appears in results"
        if 103 not in ids_after_remove
        else f"chunk_id 103 still present after removal: {results_after_remove}",
    ):
        checks_passed += 1

    new_file_a_ids = [201, 202, 203]
    new_vectors = rng.standard_normal((len(new_file_a_ids), VECTOR_DIM)).astype("float32")
    vector_202 = new_vectors[1].copy()
    query_near_202 = (vector_202 + rng.standard_normal(VECTOR_DIM).astype("float32") * 0.01).astype(
        "float32"
    )

    add_chunks(index, new_file_a_ids, new_vectors)

    results_after_readd = search(index, query_near_202, k=3)
    top_id_after_readd = results_after_readd[0][0] if results_after_readd else None
    checks_total += 1
    if _check(
        "search after re-add",
        top_id_after_readd == 202,
        f"chunk_id 202 is nearest (got top={top_id_after_readd}, results={results_after_readd})"
        if top_id_after_readd == 202
        else f"expected chunk_id 202 as top result, got top={top_id_after_readd}, results={results_after_readd}",
    ):
        checks_passed += 1

    print(f"\n{checks_passed}/{checks_total} checks passed")

    if checks_passed != checks_total:
        sys.exit(1)
