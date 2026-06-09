"""FastAPI application entry point.

Wires up logging, loads the Detoxify model once via the lifespan context, and
registers the analyze + meta routers.

TODO (v0.2): add optional Bearer-token authentication. v0.1 intentionally ships
without auth so it is a frictionless drop-in for the (also unauthenticated for
self-hosters) Perspective replacement use case.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app import __version__
from app.config import get_settings
from app.routers import analyze, meta
from app.services import cache, classifier
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
        await cache.close()


app = FastAPI(
    title="openspective",
    version=__version__,
    description="Self-hosted, open-source drop-in replacement for the Perspective API.",
    lifespan=lifespan,
)

app.include_router(analyze.router)
app.include_router(meta.router)


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
