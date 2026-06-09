"""FastAPI application entry point.

Wires up logging, loads the Detoxify model once via the lifespan context, and
registers the analyze + meta routers.

Bearer-token authentication is optional and disabled by default (see app.security);
set ``OPENSPECTIVE_API_TOKENS`` to require it on the analyze endpoint.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.config import get_settings
from app.routers import analyze, chart, meta
from app.security import AuthError, RateLimitError, rate_limit, require_auth
from app.services import classifier, redis_client
from app.services.metrics import REQUEST_COUNT, REQUEST_LATENCY


def _configure_logging(level: str) -> None:
    """Configure root logging at the given level (idempotent for reloads)."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup and release resources at shutdown."""
    settings = get_settings()
    _configure_logging(settings.log_level)
    logging.getLogger("openspective").info(
        "Starting openspective v%s (model=%s)", __version__, settings.model_variant
    )
    # Load the model once, before serving any request.
    classifier.load_model(settings.model_variant)
    try:
        yield
    finally:
        classifier.shutdown()
        await redis_client.close()


app = FastAPI(
    title="openspective",
    version=__version__,
    description="Self-hosted, open-source drop-in replacement for the Perspective API.",
    lifespan=lifespan,
)

# Auth + rate limiting guard the scoring endpoints (analyze + chart, which both run
# inference); operational routes stay open for probes and scrapers.
_scoring_guards = [Depends(require_auth), Depends(rate_limit)]
app.include_router(analyze.router, dependencies=_scoring_guards)
app.include_router(chart.router, dependencies=_scoring_guards)
app.include_router(meta.router)


@app.exception_handler(AuthError)
async def _auth_error_handler(request: Request, exc: AuthError) -> JSONResponse:
    """Render auth failures in the same structured shape as other errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "unauthorized", "detail": exc.detail},
        headers=exc.headers,
    )


@app.exception_handler(RateLimitError)
async def _rate_limit_error_handler(request: Request, exc: RateLimitError) -> JSONResponse:
    """Render rate-limit rejections as a structured 429."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "rate_limited", "detail": exc.detail},
        headers=exc.headers,
    )


def _route_label(request: Request) -> str:
    """Return the matched route template (low cardinality), or the raw path.

    Using the template (e.g. ``/v1alpha1/comments:analyze``) instead of the raw URL
    keeps Prometheus label cardinality bounded.
    """
    route = request.scope.get("route")
    return getattr(route, "path", request.url.path)


@app.middleware("http")
async def _instrument_requests(request: Request, call_next):
    """Record request count and latency for every HTTP request."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    path = _route_label(request)
    REQUEST_LATENCY.labels(method=request.method, path=path).observe(elapsed)
    REQUEST_COUNT.labels(
        method=request.method, path=path, status=str(response.status_code)
    ).inc()
    return response
