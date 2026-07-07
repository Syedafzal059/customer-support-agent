"""
Text chunk embedder for the Drive -> FAISS sync pipeline.

Embedding model: **BAAI/bge-base-en-v1.5** via sentence-transformers — the same
library and default model as app/retrieval/embeddings.py, so vectors produced here
stay compatible with the FastAPI RAG path. Override with EMBEDDING_MODEL_ID.

Usage (from repo root):
    python scripts/embedder.py
"""

from __future__ import annotations

import os

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "BAAI/bge-base-en-v1.5").strip() or (
    "BAAI/bge-base-en-v1.5"
)
_model = SentenceTransformer(_MODEL_ID)
_EMBEDDING_DIM = int(_model.get_sentence_embedding_dimension())


def get_embedding_dim() -> int:
    """Return the model's output vector dimension (768 for bge-base-en-v1.5)."""
    return _EMBEDDING_DIM


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a batch of strings.

    Returns a float32 array of shape (len(texts), get_embedding_dim()).
    Empty input yields shape (0, dim), not an error.
    """
    dim = get_embedding_dim()
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)

    vectors = _model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


if __name__ == "__main__":
    samples = [
        "Refunds are available within thirty days of purchase.",
        "Standard shipping takes three to five business days.",
        "Contact billing for disputed charges after manager review.",
    ]

    vectors = embed_texts(samples)
    dim = get_embedding_dim()
    expected_shape = (3, dim)
    actual_shape = vectors.shape

    print(f"Model: {_MODEL_ID}")
    print(f"Embedding dim: {dim}")
    print(f"Output shape: {actual_shape}")
    print(f"Expected shape: {expected_shape}")

    empty = embed_texts([])
    empty_ok = empty.shape == (0, dim)
    print(f"Empty input shape: {empty.shape} — {'PASS' if empty_ok else 'FAIL'}")

    shape_ok = actual_shape == expected_shape
    print(f"Sample batch: {'PASS' if shape_ok else 'FAIL'}")
