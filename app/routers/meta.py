"""Operational endpoints: health, readiness, model listing, and metrics."""

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.services import classifier, metrics

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict:
    """Liveness/readiness probe.

    Reports ``ok`` together with the configured model variant. The loaded model
    name is included separately so callers can distinguish "configured" from
    "actually loaded".
    """
    settings = get_settings()
    return {
        "status": "ok",
        "model": classifier.model_name() or settings.model_variant,
    }


@router.get("/readyz")
async def readyz():
    """Readiness probe.

    Distinct from ``/healthz``: returns 200 only once the model has actually
    finished loading, so orchestrators don't route traffic before the first
    request would succeed. Returns 503 while the model is still loading.
    """
    if classifier.is_loaded():
        return {"status": "ready", "model": classifier.model_name()}
    return JSONResponse(status_code=503, content={"status": "not_ready"})


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    """Expose Prometheus metrics in the standard text exposition format."""
    payload, content_type = metrics.render()
    return Response(content=payload, media_type=content_type)


@router.get("/v1/models")
async def list_models() -> dict:
    """List the selectable Detoxify variants and the currently loaded model."""
    return {
        "models": list(classifier.available_variants()),
        "loaded": classifier.model_name(),
    }
