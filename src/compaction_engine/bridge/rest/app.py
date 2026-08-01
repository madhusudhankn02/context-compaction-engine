"""
FastAPI application factory.

Using the lifespan context manager (not deprecated @app.on_event) so the
container is built before the first request is served and torn down cleanly
on shutdown. The factory pattern (create_app) makes it trivial to spin up
the app with a custom container in tests (no monkey-patching of globals).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from compaction_engine.bridge.container import BridgeContainer, build_container
from compaction_engine.bridge.rest.routes import router
from compaction_engine.config import EngineSettings
from compaction_engine.utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def create_app(
    settings: EngineSettings | None = None,
    container: BridgeContainer | None = None,
) -> FastAPI:
    """
    Factory function. Accepts an optional pre-built container so tests can
    inject a FakeStructuredExtractor without going through `build_container`.
    If neither is supplied, reads settings from the environment.
    """
    resolved_settings = settings or EngineSettings()
    configure_logging(resolved_settings.log_level, resolved_settings.enable_json_logging)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        # Startup
        _app.state.container = container or build_container(resolved_settings)
        logger.info("REST bridge online")
        yield
        # Shutdown (add cleanup here in Phase 4: close DB pool, OTel flush, etc.)
        logger.info("REST bridge shutting down")

    app = FastAPI(
        title="Compaction Engine — Multi-Protocol Bridge",
        description=(
            "Context-Aware State Compaction Engine REST API. "
            "Compress multi-agent dialogue into structured, schema-validated state "
            "and validate compressed trajectories against gold-standard references."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],    # tighten in production via config
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


# Module-level app instance used by uvicorn/gunicorn entry points:
#   uvicorn compaction_engine.bridge.rest.app:app --reload
# Tests should use `create_app(container=test_container)` instead.
app = create_app()
