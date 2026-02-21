from contextlib import asynccontextmanager
from pathlib import Path

import sentry_sdk
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.configs import settings
from app.core.celery.app import celery_app
from app.core.vector_store import initialize_qdrant
from app.logging import logger
from app.middlewares import RequestResponseLoggingMiddleware
from app.utils.sentry import init_sentry
from app.v2.api import api_router_v2
from app.v3.api import api_router_v3
from app.v3.endpoints.agent_chat.utils.memory import (
    cleanup_connection_pool,
    setup_checkpointer,
)
from app.v3.endpoints.report_generator.utils.memory import (
    cleanup_connection_pool as cleanup_report_pool,
)
from app.v3.endpoints.report_generator.utils.memory import (
    setup_checkpointer as setup_report_checkpointer,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    init_sentry()
    logger.info("Sentry initialized")
    initialize_qdrant()
    logger.info("Qdrant initialized")

    Path(settings.PDF_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.SAVE_DIR).mkdir(parents=True, exist_ok=True)
    logger.info("Cache directories created")

    await setup_checkpointer()
    logger.info("Agent chat checkpointer initialized")

    await setup_report_checkpointer()
    logger.info("Report generator checkpointer initialized")

    logger.info("Lifespan initialized")
    yield

    logger.info("Shutting down...")
    await cleanup_connection_pool()
    await cleanup_report_pool()
    logger.info("Cleanup completed")


app = FastAPI(lifespan=lifespan)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler that ensures all unhandled exceptions
    are properly captured by Sentry with request context.
    """
    if sentry_sdk.Hub.current.client:
        with sentry_sdk.configure_scope() as scope:
            scope.set_context(
                "request",
                {
                    "url": str(request.url),
                    "method": request.method,
                    "client_host": request.client.host if request.client else "unknown",
                    "headers": dict(request.headers),
                },
            )
        sentry_sdk.capture_exception(exc)

    raise exc


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log 422 validation errors with request details."""
    from app.logging import logger

    logger.error(f"Validation error on {request.method} {request.url}: {exc.errors()}")

    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Custom handler for HTTP exceptions that adds Sentry context
    for non-5xx errors that might not be captured otherwise.
    """
    if 400 <= exc.status_code < 500 and sentry_sdk.Hub.current.client:
        with sentry_sdk.push_scope() as scope:
            scope.set_level("info")
            scope.set_context(
                "request",
                {
                    "url": str(request.url),
                    "method": request.method,
                },
            )
            scope.set_tag("status_code", exc.status_code)
            sentry_sdk.capture_message(f"HTTP {exc.status_code}: {exc.detail}")

    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


origins = [
    "https://api.server.delineate-data.com/",
    "https://api.cosmos.staging.delineate.pro/",
    "http://localhost:3000/",
    "http://localhost:3000",
    "localhost:3000",
    "null",
]

app.add_middleware(RequestResponseLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(api_router_v2)
app.include_router(api_router_v3)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/sentry-test")
async def sentry_test():
    """
    Test endpoint to verify Sentry integration
    """
    if not sentry_sdk.Hub.current.client:
        return {
            "status": "error",
            "message": "Sentry is not properly configured",
            "details": "Check SENTRY_DSN environment variable",
        }

    try:
        sentry_sdk.capture_message("Sentry test from API", level="info")

        _ = 1 / 0
    except Exception as e:
        sentry_sdk.capture_exception(e)

        return {
            "status": "success",
            "message": "Test exception captured and sent to Sentry",
            "error_type": type(e).__name__,
            "details": "Check your Sentry dashboard to verify the error was reported",
        }


@app.get("/sentry-test/celery")
async def sentry_test_celery():
    celery_app.send_task("sentry-test-celery")
