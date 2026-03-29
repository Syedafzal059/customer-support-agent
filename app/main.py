"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.logger import get_logger, setup_logging
from app.retrieval.faiss_store import rebuild_knowledge_index

logger = get_logger(__name__)

# Any loopback port (Vite, preview, IDE embedded browser) — avoids OPTIONS → 405 when Origin is not in the static allow_origins list.
_LOCALHOST_CORS_ORIGIN_REGEX = r"https?://(localhost|127\.0\.0\.1)(:\d+)?$"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info(
        "application_start",
        extra={"structured": {"app_name": settings.app_name, "debug": settings.debug}},
    )
    rebuild_knowledge_index(settings)
    yield
    logger.info("application_stop", extra={"structured": {}})


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_origin_regex=_LOCALHOST_CORS_ORIGIN_REGEX,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(router)
    return application


app = create_app()
