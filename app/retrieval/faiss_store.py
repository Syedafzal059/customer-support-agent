"""Load Drive-synced FAISS index + chunk store and search for RAG retrieval."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import faiss

from app.core.config import AppSettings
from app.core.logger import get_logger

logger = get_logger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _ensure_scripts_importable() -> None:
    root = str(_project_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _resolve_pipeline_path(env_var: str, default_name: str) -> str:
    raw = os.environ.get(env_var, default_name).strip() or default_name
    path = Path(raw)
    if not path.is_absolute():
        path = _project_root() / path
    return str(path.resolve())


def _configure_pipeline_env() -> tuple[str, str]:
    """Ensure FAISS/chunk paths resolve to repo root regardless of process cwd."""
    index_path = _resolve_pipeline_path("FAISS_INDEX_PATH", "faiss_index.bin")
    db_path = _resolve_pipeline_path("CHUNK_STORE_DB", "chunk_store.db")
    os.environ["FAISS_INDEX_PATH"] = index_path
    os.environ["CHUNK_STORE_DB"] = db_path
    return index_path, db_path


@dataclass(frozen=True)
class IndexedChunk:
    text: str
    source_relpath: str


class FaissKnowledgeIndex:
    """FAISS IndexIDMap (Drive sync pipeline) over chunk_store metadata."""

    def __init__(self) -> None:
        self._index: faiss.IndexIDMap | None = None
        self._index_path: str = ""

    @property
    def chunk_count(self) -> int:
        if self._index is None:
            return 0
        return int(self._index.ntotal)

    def build(self, settings: AppSettings) -> None:
        """Load persisted Drive-sync index at startup (not per-query)."""
        _ensure_scripts_importable()
        from scripts import chunk_store as pipeline_chunk_store
        from scripts import embedder as pipeline_embedder
        from scripts import faiss_store as pipeline_faiss_store

        index_path, db_path = _configure_pipeline_env()
        self._index_path = index_path

        if not os.path.isfile(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_path!r}. "
                "Run the Google Drive sync pipeline first "
                "(e.g. python scripts/sync_pipeline.py after gdrive_kb sync) "
                "to build faiss_index.bin and chunk_store.db."
            )
        if not os.path.isfile(db_path):
            raise FileNotFoundError(
                f"Chunk store not found at {db_path!r}. "
                "Run the Google Drive sync pipeline first to build chunk_store.db."
            )

        pipeline_model_id = (
            os.environ.get("EMBEDDING_MODEL_ID", "BAAI/bge-base-en-v1.5").strip()
            or "BAAI/bge-base-en-v1.5"
        )
        app_model_id = settings.embedding_model_id.strip() or pipeline_model_id
        if app_model_id != pipeline_model_id:
            raise RuntimeError(
                "Embedding model mismatch: AppSettings.embedding_model_id="
                f"{app_model_id!r} but the Drive sync pipeline uses "
                f"scripts/embedder.py with EMBEDDING_MODEL_ID={pipeline_model_id!r}. "
                "Query and index vectors must use the same model or similarity search "
                "will silently return garbage results."
            )

        dim = pipeline_embedder.get_embedding_dim()

        self._index = pipeline_faiss_store.load_index(index_path, dim=dim)
        if self._index.ntotal == 0:
            raise RuntimeError(
                f"FAISS index at {index_path!r} is empty. Re-run the Drive sync pipeline."
            )

        stored_chunks = pipeline_chunk_store.get_chunk_count()
        logger.info(
            "knowledge_index_loaded",
            extra={
                "structured": {
                    "faiss_index_path": index_path,
                    "chunk_store_path": db_path,
                    "faiss_vector_count": int(self._index.ntotal),
                    "chunk_store_row_count": stored_chunks,
                    "embedding_model": os.environ.get(
                        "EMBEDDING_MODEL_ID", "BAAI/bge-base-en-v1.5"
                    ),
                    "embedding_dim": dim,
                }
            },
        )

    def search(self, query: str, settings: AppSettings) -> list[str]:
        if self._index is None or self._index.ntotal == 0:
            return []

        _ensure_scripts_importable()
        from scripts import chunk_store as pipeline_chunk_store
        from scripts import embedder as pipeline_embedder
        from scripts import faiss_store as pipeline_faiss_store

        top_k = min(settings.rag_top_k, self._index.ntotal)
        query_vector = pipeline_embedder.embed_texts([query])[0]
        hits = pipeline_faiss_store.search(self._index, query_vector, k=top_k)

        out: list[str] = []
        for chunk_id, _distance in hits:
            meta = pipeline_chunk_store.get_chunk_text(chunk_id)
            if meta is None:
                logger.warning(
                    "knowledge_index_dangling_chunk_id",
                    extra={
                        "structured": {
                            "chunk_id": chunk_id,
                            "faiss_index_path": self._index_path,
                        }
                    },
                )
                continue
            source_name = str(meta["source_name"])
            text = str(meta["text"])
            header = f"(source: {source_name})"
            out.append(f"{header}\n{text}")
        return out


# ---------------------------------------------------------------------------
# TODO: old local-KB path preserved for rollback, remove once new pipeline is
# confirmed stable in eval.
#
# from app.retrieval.chunking import chunk_text_by_tokens
# from app.retrieval.embeddings import embed_texts
#
# def _load_and_chunk_documents(kb_root: Path, settings: AppSettings) -> tuple[list[str], list[str]]:
#     ...
#
# Old FaissKnowledgeIndex used IndexFlatIP + in-memory IndexedChunk list built from
# data/knowledge_base/*.md at startup via embed_texts(..., settings.embedding_model_id).
# ---------------------------------------------------------------------------


_kb_singleton: FaissKnowledgeIndex | None = None


def rebuild_knowledge_index(settings: AppSettings) -> FaissKnowledgeIndex:
    """Load the global Drive-sync FAISS index from disk (startup only)."""
    global _kb_singleton
    index = FaissKnowledgeIndex()
    index.build(settings)
    _kb_singleton = index
    return index


def get_knowledge_index() -> FaissKnowledgeIndex | None:
    return _kb_singleton


def format_rag_reply(chunks: list[str]) -> str:
    """Debug helper: join retrieved chunks as markdown (orchestrator uses LLM synthesis instead)."""
    if not chunks:
        return (
            "[RAG] No passages matched your question in the knowledge base. "
            "Run the Drive sync pipeline to populate faiss_index.bin and chunk_store.db."
        )
    parts = [f"### Match {i + 1}\n{c}" for i, c in enumerate(chunks)]
    return "Retrieved passages (raw):\n\n" + "\n\n".join(parts)
