"""FastAPI application factory.

Cross-cutting concerns only: request identity, structured access logs, and turning application errors into
RFC 9457 problem documents. Domain logic lives in ``aletheia.ml`` and ``aletheia.services``, never here.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from aletheia import __version__
from aletheia.api.routers import audit, explanations, labels, models, predictions, system
from aletheia.core.config import Settings, get_settings
from aletheia.core.errors import AletheiaError, ValidationFailed
from aletheia.core.logging import configure_logging, get_logger
from aletheia.db.connection import get_engine, init_db

REQUEST_ID_HEADER = "X-Request-Id"
PROBLEM_MEDIA_TYPE = "application/problem+json"


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings if settings is not None else get_settings()
    configure_logging(resolved.log_level, json=resolved.log_json)
    log = get_logger("aletheia.api")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        if resolved.database_url:
            try:
                engine = get_engine(resolved.database_url)
                init_db(engine)
                log.info("database_tables_initialized")
            except Exception as exc:
                log.warning("database_initialization_failed", error=str(exc))
        yield

    app = FastAPI(
        title="Aletheia",
        description=(
            "Explainable AI and model governance. "
            "Explanations describe model behaviour, not causation."
        ),
        version=__version__,
        docs_url=None if resolved.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if resolved.is_production else "/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = resolved

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(
            request_id=request_id, method=request.method, path=request.url.path
        )
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception(
                "request_failed", duration_ms=round((time.perf_counter() - started) * 1000, 2)
            )
            structlog.contextvars.clear_contextvars()
            raise
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers[REQUEST_ID_HEADER] = request_id
        log.info("request", status=response.status_code, duration_ms=duration_ms)
        structlog.contextvars.clear_contextvars()
        return response

    @app.exception_handler(AletheiaError)
    async def application_error(request: Request, exc: AletheiaError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status,
            content=exc.to_problem(
                request_id=getattr(request.state, "request_id", None),
                instance=request.url.path,
            ),
            media_type=PROBLEM_MEDIA_TYPE,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        problem = ValidationFailed("one or more fields are invalid").to_problem(
            request_id=getattr(request.state, "request_id", None),
            instance=request.url.path,
        )
        problem["errors"] = [
            {
                "location": ".".join(str(part) for part in error.get("loc", ())),
                "message": error.get("msg", "invalid value"),
                "type": error.get("type", "value_error"),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422, content=_jsonable(problem), media_type=PROBLEM_MEDIA_TYPE
        )

    app.include_router(system.router)
    app.include_router(system.api_router, prefix="/api/v1")
    app.include_router(models.router)
    app.include_router(predictions.router)
    app.include_router(explanations.router)
    app.include_router(labels.router)
    app.include_router(audit.router)

    return app


def _jsonable(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce a problem document into JSON-safe primitives."""
    return {key: value for key, value in payload.items()}


app = create_app()
