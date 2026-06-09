"""Operational endpoints: health check and model listing."""

from fastapi import APIRouter

from app.config import get_settings
from app.services import classifier

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


@router.get("/v1/models")
async def list_models() -> dict:
    """List the selectable Detoxify variants and the currently loaded model."""
    return {
        "models": list(classifier.available_variants()),
        "loaded": classifier.model_name(),
    }
