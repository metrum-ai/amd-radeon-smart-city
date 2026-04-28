# Created by Metrum AI for AMD

"""FastAPI application factory for the Smart City platform."""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)


def _build_lifespan(config: Optional[Any]):
    """Return an asynccontextmanager lifespan bound to config.

    Args:
        config: Loaded AppSettings or None.

    Returns:
        Async context manager for FastAPI lifespan.
    """

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        if config is not None:
            from smart_city.api.startup import (
                init_app_state,
                shutdown_app_state,
            )

            await init_app_state(app, config)
        else:
            # Minimal state for dev / testing without real services
            from smart_city.api.websocket import WebSocketHub

            app.state.ws_hub = WebSocketHub()
            app.state.db_pool = None
            app.state.report_generator = None
            app.state.density_cache = {}
            app.state.heatmap_cache = {}
            app.state.report_jobs = {}
            from smart_city.api.startup import StreamRegistry
            app.state.pipeline = StreamRegistry(app)

        logger.info("Smart City platform ready.")
        yield
        if config is not None:
            from smart_city.api.startup import shutdown_app_state

            await shutdown_app_state(app)
        logger.info("Smart City platform stopped.")

    return _lifespan


def create_app(config: Optional[Any] = None) -> FastAPI:
    """Build and configure the FastAPI application.

    Args:
        config: AppSettings instance; uses defaults/stubs if None.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="Smart City Public Safety API",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=_build_lifespan(config),
    )

    # ---------------------------------------------------------------
    # Prometheus HTTP instrumentation middleware
    # ---------------------------------------------------------------
    from smart_city.observability.metrics import HTTP_REQUEST_DURATION

    @app.middleware("http")
    async def _record_http_duration(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            path=request.url.path,
            status_code=str(response.status_code),
        ).observe(duration)
        return response

    # ---------------------------------------------------------------
    # CORS
    # ---------------------------------------------------------------
    cors_origins = (
        config.api.cors_origins
        if config
        else ["http://localhost:5173", "http://localhost:3000"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    )

    # ---------------------------------------------------------------
    # Exception handlers
    # ---------------------------------------------------------------
    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "Validation error", "detail": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def _unhandled(
        _request: Request, exc: Exception
    ) -> JSONResponse:
        import uuid

        error_id = str(uuid.uuid4())
        logger.error(
            "Unhandled exception %s: %s", error_id, exc, exc_info=True
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "error_id": error_id,
            },
        )

    # ---------------------------------------------------------------
    # Prometheus metrics endpoint
    # ---------------------------------------------------------------
    @app.get("/metrics", include_in_schema=False)
    async def _metrics() -> PlainTextResponse:
        try:
            from prometheus_client import (
                CONTENT_TYPE_LATEST,
                generate_latest,
            )

            return PlainTextResponse(
                generate_latest().decode("utf-8"),
                media_type=CONTENT_TYPE_LATEST,
            )
        except (ImportError, OSError, ValueError, UnicodeDecodeError):
            logger.exception("Prometheus metrics generation failed")
            return PlainTextResponse("# metrics unavailable\n")

    # ---------------------------------------------------------------
    # Health check (no auth required)
    # ---------------------------------------------------------------
    _start_time = time.time()

    @app.get("/api/v1/health", tags=["system"])
    async def _health() -> dict:
        return {
            "status": "ok",
            "version": "1.0.0",
            "uptime_s": round(time.time() - _start_time, 1),
        }

    # ---------------------------------------------------------------
    # Static assets (recorded replay videos)
    # ---------------------------------------------------------------
    videos_dir = Path(__file__).resolve().parents[2] / "videos"
    app.mount(
        "/videos",
        StaticFiles(directory=str(videos_dir), check_dir=False),
        name="videos",
    )

    # ---------------------------------------------------------------
    # Mount routers
    # ---------------------------------------------------------------
    from smart_city.api.routes.alerts import router as alerts_router
    from smart_city.api.routes.analytics import (
        router as analytics_router,
    )
    from smart_city.api.routes.dashboard import (
        router as dashboard_router,
    )
    from smart_city.api.routes.reports import router as reports_router
    from smart_city.api.routes.streams import router as streams_router
    from smart_city.api.routes.simulator import (
        router as simulator_router,
    )
    from smart_city.api.routes.ws import router as ws_router

    prefix = "/api/v1"
    app.include_router(streams_router, prefix=prefix)
    app.include_router(alerts_router, prefix=prefix)
    app.include_router(analytics_router, prefix=prefix)
    app.include_router(dashboard_router, prefix=prefix)
    app.include_router(reports_router, prefix=prefix)
    app.include_router(simulator_router, prefix=prefix)
    # WebSocket routes carry their full paths internally
    app.include_router(ws_router)

    return app
