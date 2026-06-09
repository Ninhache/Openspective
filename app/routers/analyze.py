"""``POST /v1alpha1/comments:analyze`` — the Perspective-compatible scoring endpoint."""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.models import (
    ALL_ATTRIBUTES,
    DETOXIFY_TO_PERSPECTIVE,
    PERSPECTIVE_TO_DETOXIFY,
    AnalyzeRequest,
    AnalyzeResponse,
    AttributeScore,
    ErrorResponse,
    SummaryScore,
)
from app.services import cache, classifier
from app.services.detector import detect_language
from app.services.normalizer import normalize

logger = logging.getLogger("openspective.analyze")

router = APIRouter()


def _resolve_requested(request: AnalyzeRequest) -> list[str]:
    """Return the Perspective attribute names to include in the response.

    Defaults to all available attributes when ``requestedAttributes`` is omitted or
    empty. Unknown attribute names are ignored (Perspective behaviour).
    """
    requested = request.requestedAttributes
    if not requested:
        return list(ALL_ATTRIBUTES)
    return [name for name in requested if name in PERSPECTIVE_TO_DETOXIFY]


def _to_perspective_scores(
    detoxify_scores: dict[str, float], requested: list[str]
) -> dict[str, AttributeScore]:
    """Map Detoxify scores to Perspective attribute scores, filtered to ``requested``."""
    attribute_scores: dict[str, AttributeScore] = {}
    for detox_key, value in detoxify_scores.items():
        perspective_name = DETOXIFY_TO_PERSPECTIVE.get(detox_key)
        # Filtering happens *after* inference: drop attributes not requested.
        if perspective_name is None or perspective_name not in requested:
            continue
        attribute_scores[perspective_name] = AttributeScore(
            summaryScore=SummaryScore(value=value)
        )
    return attribute_scores


# Route registered with an explicit path; FastAPI handles the literal ``:analyze``.
@router.post("/v1alpha1/comments:analyze")
async def analyze(request: AnalyzeRequest):
    """Score a comment and return Perspective-compatible attribute scores."""
    normalized = normalize(request.comment.text)
    requested = _resolve_requested(request)

    # Cache is keyed on normalised text + the requested attribute set.
    cache_key = cache.make_key(normalized, requested)
    detoxify_scores = await cache.get_scores(cache_key)

    if detoxify_scores is None:
        try:
            detoxify_scores = await classifier.predict(normalized)
        except Exception as exc:  # noqa: BLE001 — surface as structured HTTP 500
            # Do not silently swallow: log and return a structured error body.
            logger.exception("Inference failed")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(error="inference_failed", detail=str(exc)).model_dump(),
            )
        await cache.set_scores(cache_key, detoxify_scores)

    attribute_scores = _to_perspective_scores(detoxify_scores, requested)

    # Language handling: auto-detect from the original text, but let a caller-supplied
    # ``languages`` value override what we report as the effective languages.
    detected = detect_language(request.comment.text)
    languages = request.languages if request.languages else detected

    return AnalyzeResponse(
        attributeScores=attribute_scores,
        languages=languages,
        detectedLanguages=detected,
        clientToken=request.clientToken,
    )
