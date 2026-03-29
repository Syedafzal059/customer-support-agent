"""Token-based text chunking for RAG (Phase 5)."""

from __future__ import annotations

import tiktoken


def chunk_text_by_tokens(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    """
    Split text into chunks of at most `max_tokens` tokens with `overlap_tokens` overlap
    between consecutive chunks. Uses cl100k as a language-agnostic proxy for English KB docs.
    """
    if max_tokens < 1:
        return []
    overlap = min(max(0, overlap_tokens), max_tokens - 1)
    stride = max_tokens - overlap

    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text.strip())
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start += stride
    return chunks
