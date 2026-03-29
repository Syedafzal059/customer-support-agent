"""Hugging Face sentence embeddings for RAG (Phase 5).

Default model **BAAI/bge-base-en-v1.5** is a common pick on the public **MTEB** embedding
leaderboard (https://huggingface.co/spaces/mteb/leaderboard, https://mteb.info/): strong
English retrieval vs. latency for its size (~109M params, 768-d). Use **BAAI/bge-small-en-v1.5**
when you need faster CPU inference or smaller memory (~33M params, 384-d) at some quality cost.

Models load lazily so the API can start before the first encode.
"""

from __future__ import annotations

from threading import Lock
from typing import Any

import numpy as np

_model_cache: dict[str, Any] = {}
_model_lock = Lock()


def get_embedding_model(model_id: str) -> Any:
    """Load and cache a Sentence-Transformers model by Hugging Face id."""
    from sentence_transformers import SentenceTransformer

    with _model_lock:
        if model_id not in _model_cache:
            _model_cache[model_id] = SentenceTransformer(model_id)
        return _model_cache[model_id]


def embed_texts(
    texts: list[str],
    model_id: str,
    *,
    normalize: bool = True,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Return a float32 matrix of shape (len(texts), dim).

    For FAISS inner product on L2-normalized vectors, cosine similarity matches IP scores.
    """
    if not texts:
        return np.array([], dtype=np.float32).reshape(0, 0)

    model = get_embedding_model(model_id)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)
