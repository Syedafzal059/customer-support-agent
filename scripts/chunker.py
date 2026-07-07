"""
Paragraph-aware text chunker with deterministic chunk IDs.

Standalone and testable without touching Drive, FAISS, or embeddings —
feed it any extracted text and a file_id, get back a list of chunk dicts.

Usage (from repo root):
    python scripts/chunker.py   # runs the demo at the bottom on sample text
"""

from __future__ import annotations

import hashlib

TARGET_CHUNK_SIZE = 400  # target characters per chunk
OVERLAP_SIZE = 50  # characters of overlap carried into the next chunk


def make_chunk_id(file_id: str, chunk_index: int) -> int:
    """Deterministic integer ID: same file_id + position always yields the
    same ID, so Phase 3 can look up and remove exact chunk sets on file changes.

    Masked to signed int64 range for SQLite INTEGER and FAISS int64 IDs."""
    raw = f"{file_id}:{chunk_index}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) & 0x7FFFFFFFFFFFFFFF


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; fall back to single-newline splits if the PDF
    extraction produced no blank-line paragraph breaks at all."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return paragraphs


def chunk_text(
    text: str,
    file_id: str,
    source_name: str,
    target_size: int = TARGET_CHUNK_SIZE,
    overlap: int = OVERLAP_SIZE,
) -> list[dict[str, str | int]]:
    """
    Groups paragraphs into chunks close to target_size, carrying a small
    overlap from the end of one chunk into the start of the next so a
    question whose answer straddles a boundary can still retrieve well.

    Returns a list of dicts: {chunk_id, file_id, source_name, chunk_index, text}
    """
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if current and len(current) + len(para) + 1 > target_size:
            chunks.append(current.strip())
            # carry the tail of this chunk forward as overlap context
            current = current[-overlap:] + "\n" + para
        else:
            current = (current + "\n" + para) if current else para

    if current.strip():
        chunks.append(current.strip())

    return [
        {
            "chunk_id": make_chunk_id(file_id, i),
            "file_id": file_id,
            "source_name": source_name,
            "chunk_index": i,
            "text": chunk,
        }
        for i, chunk in enumerate(chunks)
    ]


if __name__ == "__main__":
    sample_text = (
        "This is the first paragraph of a sample document. It explains the "
        "general policy for handling refunds within thirty days of purchase.\n\n"
        "This is the second paragraph. It covers what happens if a customer "
        "requests a refund after the thirty day window has already passed, "
        "which requires manager approval before it can be processed.\n\n"
        "This is the third paragraph, describing the escalation path for "
        "disputed charges and how the support team should route them to "
        "billing specialists for further review."
    )

    result = chunk_text(sample_text, file_id="demo_file_123", source_name="refund_policy.pdf")

    print(f"Produced {len(result)} chunk(s):\n")
    for c in result:
        print(
            f"  chunk_id={c['chunk_id']}  index={c['chunk_index']}  "
            f"len={len(c['text'])}  source={c['source_name']}"
        )
        print(f"    text: {c['text'][:80]}...\n")
