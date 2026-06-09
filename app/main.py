"""FastAPI application entry point.

Wires up logging, loads the Detoxify model once via the lifespan context, and
registers the analyze + meta routers.

TODO (v0.2): add optional Bearer-token authentication. v0.1 intentionally ships
without auth so it is a frictionless drop-in for the (also unauthenticated for
self-hosters) Perspective replacement use case.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.config import get_settings
from app.routers import analyze, meta
from app.services import cache, classifier


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
