"""Load knowledge-base files, embed chunks, and search with FAISS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss

from app.core.config import AppSettings
from app.core.logger import get_logger
from app.retrieval.chunking import chunk_text_by_tokens
from app.retrieval.embeddings import embed_texts

logger = get_logger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class IndexedChunk:
    text: str
    source_relpath: str


class FaissKnowledgeIndex:
    """In-memory FAISS index (IndexFlatIP) over L2-normalized chunk embeddings."""

    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[IndexedChunk] = []
        self._vector_dim: int = 0

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def _set_empty_index(self, settings: AppSettings) -> None:
        probe = embed_texts([" "], settings.embedding_model_id)
        self._vector_dim = int(probe.shape[1])
        self._index = faiss.IndexFlatIP(self._vector_dim)
        self._chunks = []

    def build(self, settings: AppSettings) -> None:
        kb_root = (_project_root() / settings.knowledge_base_dir).resolve()
        texts, sources = _load_and_chunk_documents(kb_root, settings)
        if not texts:
            logger.warning(
                "knowledge_index_empty",
                extra={"structured": {"kb_dir": str(kb_root)}},
            )
            self._set_empty_index(settings)
            return

        vectors = embed_texts(texts, settings.embedding_model_id)
        self._vector_dim = int(vectors.shape[1])
        self._index = faiss.IndexFlatIP(self._vector_dim)
        self._index.add(vectors)
        self._chunks = [IndexedChunk(text=t, source_relpath=s) for t, s in zip(texts, sources)]
        logger.info(
            "knowledge_index_built",
            extra={
                "structured": {
                    "kb_dir": str(kb_root),
                    "chunk_count": len(self._chunks),
                    "embedding_model": settings.embedding_model_id,
                }
            },
        )

    def search(self, query: str, settings: AppSettings) -> list[str]:
        if self._index is None or self._index.ntotal == 0:
            return []
        top_k = min(settings.rag_top_k, self._index.ntotal)
        q = embed_texts([query], settings.embedding_model_id)
        _scores, indices = self._index.search(q, top_k)
        out: list[str] = []
        for idx in indices[0]:
            i = int(idx)
            if i < 0 or i >= len(self._chunks):
                continue
            ch = self._chunks[i]
            header = f"(source: {ch.source_relpath})"
            out.append(f"{header}\n{ch.text}")
        return out


_kb_singleton: FaissKnowledgeIndex | None = None


def _load_and_chunk_documents(kb_root: Path, settings: AppSettings) -> tuple[list[str], list[str]]:
    if not kb_root.is_dir():
        logger.warning(
            "knowledge_base_missing",
            extra={"structured": {"path": str(kb_root)}},
        )
        return [], []

    max_t = settings.rag_chunk_size_tokens
    ov = settings.rag_chunk_overlap_tokens
    if ov >= max_t:
        ov = max(0, max_t - 1)

    texts_out: list[str] = []
    sources_out: list[str] = []

    patterns = ("*.md", "*.txt", "*.MD", "*.TXT")
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in patterns:
        for path in kb_root.rglob(pattern):
            if path.is_file() and path.name != ".gitkeep" and path not in seen:
                seen.add(path)
                files.append(path)
    files.sort(key=lambda p: str(p).lower())

    for file_path in files:
        try:
            raw = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning(
                "knowledge_file_read_failed",
                extra={"structured": {"path": str(file_path), "error": str(exc)}},
            )
            continue
        rel = str(file_path.relative_to(kb_root)).replace("\\", "/")
        for chunk in chunk_text_by_tokens(
            raw,
            max_tokens=max_t,
            overlap_tokens=ov,
        ):
            texts_out.append(chunk)
            sources_out.append(rel)
    return texts_out, sources_out


def rebuild_knowledge_index(settings: AppSettings) -> FaissKnowledgeIndex:
    """Build (or rebuild) the global index from disk + config."""
    global _kb_singleton
    index = FaissKnowledgeIndex()
    try:
        index.build(settings)
    except Exception:
        logger.exception(
            "knowledge_index_build_failed",
            extra={"structured": {"model": settings.embedding_model_id}},
        )
        index = FaissKnowledgeIndex()
        try:
            index._set_empty_index(settings)
        except Exception:
            logger.exception("knowledge_index_fallback_empty_failed")
            index._index = None
            index._chunks = []
    _kb_singleton = index
    return index


def get_knowledge_index() -> FaissKnowledgeIndex | None:
    return _kb_singleton


def format_rag_reply(chunks: list[str]) -> str:
    """Debug helper: join retrieved chunks as markdown (orchestrator uses LLM synthesis instead)."""
    if not chunks:
        return (
            "[RAG] No passages matched your question in the knowledge base. "
            "Add .md or .txt files under data/knowledge_base/."
        )
    parts = [f"### Match {i + 1}\n{c}" for i, c in enumerate(chunks)]
    return "Retrieved passages (raw):\n\n" + "\n\n".join(parts)
