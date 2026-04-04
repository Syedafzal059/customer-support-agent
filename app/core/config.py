"""Load YAML config from configs/config.yaml and secrets from .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _config_yaml_path() -> Path:
    return _project_root() / "configs" / "config.yaml"


def _env_bool(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class AppSettings:
    app_name: str
    debug: bool
    server_host: str
    server_port: int
    cors_origins: tuple[str, ...]
    log_level: str
    log_chat_message_body: bool
    redis_backend: str
    redis_url: str
    intent_classifier_model: str
    rag_qa_model: str
    ticket_summary_model: str
    openai_base_url: str
    embedding_model_id: str
    knowledge_base_dir: str
    rag_chunk_size_tokens: int
    rag_chunk_overlap_tokens: int
    rag_top_k: int
    langsmith_enabled: bool
    langsmith_project: str


def _parse_cors_origins(raw: dict) -> tuple[str, ...]:
    env = os.getenv("CORS_ORIGINS", "").strip()
    if env:
        return tuple(part.strip() for part in env.split(",") if part.strip())
    server = raw.get("server") or {}
    origins = server.get("cors_origins")
    if isinstance(origins, list):
        return tuple(str(x).strip() for x in origins if str(x).strip())
    return (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    )


def _read_yaml() -> dict:
    path = _config_yaml_path()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("config.yaml must parse to a mapping at the root")
    return data


@lru_cache
def get_settings() -> AppSettings:
    load_dotenv(_project_root() / ".env")
    raw = _read_yaml()
    app = raw.get("app") or {}
    server = raw.get("server") or {}
    logging_cfg = raw.get("logging") or {}
    log_level = os.getenv("LOG_LEVEL") or logging_cfg.get("level", "INFO")
    log_body = _env_bool("LOG_CHAT_MESSAGE_BODY")
    if log_body is None:
        log_body = bool(logging_cfg.get("log_chat_message_body", False))
    redis_cfg = raw.get("redis") or {}
    redis_backend = os.getenv("REDIS_BACKEND") or str(redis_cfg.get("backend", "memory")).lower()
    redis_url = os.getenv("REDIS_URL") or str(redis_cfg.get("url", "redis://localhost:6379/0"))
    llm_cfg = raw.get("llm") or {}
    intent_model = os.getenv("OPENAI_INTENT_MODEL") or str(
        llm_cfg.get("intent_classifier_model", "gpt-4o-mini")
    )
    rag_qa_model = os.getenv("OPENAI_RAG_QA_MODEL") or str(
        llm_cfg.get("rag_qa_model", intent_model)
    )
    ticket_summary_model = os.getenv("OPENAI_TICKET_SUMMARY_MODEL") or str(
        llm_cfg.get("ticket_summary_model", intent_model)
    )
    openai_base = os.getenv("OPENAI_BASE_URL") or str(llm_cfg.get("openai_base_url", "") or "")
    emb_cfg = raw.get("embeddings") or {}
    embedding_model = os.getenv("EMBEDDING_MODEL_ID") or str(
        emb_cfg.get("model_id", "BAAI/bge-base-en-v1.5")
    )
    rag_cfg = raw.get("rag") or {}
    kb_dir = os.getenv("KNOWLEDGE_BASE_DIR") or str(
        rag_cfg.get("knowledge_base_dir", "data/knowledge_base")
    )
    chunk_size = int(os.getenv("RAG_CHUNK_SIZE_TOKENS") or rag_cfg.get("chunk_size_tokens", 400))
    chunk_overlap = int(
        os.getenv("RAG_CHUNK_OVERLAP_TOKENS") or rag_cfg.get("chunk_overlap_tokens", 50)
    )
    rag_top_k = int(os.getenv("RAG_TOP_K") or rag_cfg.get("top_k", 3))
    langsmith_cfg = raw.get("langsmith") or {}
    yaml_ls_enable = bool(langsmith_cfg.get("enable", False))
    ls_tracing_env = _env_bool("LANGSMITH_TRACING")
    if ls_tracing_env is None:
        ls_tracing_env = _env_bool("LANGCHAIN_TRACING_V2")
    langsmith_enabled = ls_tracing_env if ls_tracing_env is not None else yaml_ls_enable
    langsmith_project = (
        (os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "").strip()
        or str(langsmith_cfg.get("project", "") or "").strip()
    )
    cors_origins = _parse_cors_origins(raw)
    return AppSettings(
        app_name=str(app.get("name", "Xactly AI Support")),
        debug=bool(app.get("debug", False)),
        server_host=str(server.get("host", "0.0.0.0")),
        server_port=int(server.get("port", 8000)),
        cors_origins=cors_origins,
        log_level=str(log_level).upper(),
        log_chat_message_body=log_body,
        redis_backend=redis_backend,
        redis_url=redis_url,
        intent_classifier_model=intent_model,
        rag_qa_model=rag_qa_model,
        ticket_summary_model=ticket_summary_model,
        openai_base_url=openai_base,
        embedding_model_id=embedding_model,
        knowledge_base_dir=kb_dir,
        rag_chunk_size_tokens=chunk_size,
        rag_chunk_overlap_tokens=chunk_overlap,
        rag_top_k=rag_top_k,
        langsmith_enabled=langsmith_enabled,
        langsmith_project=langsmith_project,
    )
