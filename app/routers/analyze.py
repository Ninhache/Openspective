"""``POST /v1alpha1/comments:analyze`` — the Perspective-compatible scoring endpoint."""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.models import (
    ALL_ATTRIBUTES,
    DETOXIFY_TO_PERSPECTIVE,
    PERSPECTIVE_TO_DETOXIFY,
    AnalyzeRequest,
    AnalyzeResponse,
    AttributeScore,
    ErrorResponse,
    SpanScore,
    SummaryScore,
)
from app.services import cache, classifier
from app.services.detector import detect_language
from app.services.metrics import CACHE_HITS, CACHE_MISSES
from app.services.normalizer import normalize
from app.services.spans import split_spans

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


def _thresholds(request: AnalyzeRequest, requested: list[str]) -> dict[str, float]:
    """Resolve the span-score threshold per requested attribute.

    A per-attribute ``scoreThreshold`` in ``requestedAttributes`` overrides the
    ``OPENSPECTIVE_SCORE_THRESHOLD`` default.
    """
    default = get_settings().score_threshold
    requested_attrs = request.requestedAttributes or {}
    thresholds: dict[str, float] = {}
    for name in requested:
        config = requested_attrs.get(name) or {}
        thresholds[name] = float(config.get("scoreThreshold", default))
    return thresholds


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


async def _attach_span_scores(
    text: str,
    attribute_scores: dict[str, AttributeScore],
    requested: list[str],
    thresholds: dict[str, float],
) -> None:
    """Compute per-span scores for each requested attribute and attach them in place.

    Each sentence-like span is scored independently; spans scoring below the
    attribute's threshold are omitted. Span scores are intentionally not cached
    (they multiply inference cost by the number of spans).
    """
    spans = split_spans(text)
    per_attribute: dict[str, list[SpanScore]] = {name: [] for name in requested}
    for segment, begin, end in spans:
        segment_scores = await classifier.predict(normalize(segment))
        for name in requested:
            value = segment_scores[PERSPECTIVE_TO_DETOXIFY[name]]
            if value >= thresholds[name]:
                per_attribute[name].append(
                    SpanScore(begin=begin, end=end, score=SummaryScore(value=value))
                )
    for name in requested:
        if name in attribute_scores:
            attribute_scores[name].spanScores = per_attribute[name]


@router.post(
    "/v1alpha1/comments:analyze",
    response_model=AnalyzeResponse,
    response_model_exclude_none=True,
)
async def analyze(request: AnalyzeRequest):
    """Score a comment and return Perspective-compatible attribute scores."""
    normalized = normalize(request.comment.text)
    requested = _resolve_requested(request)

    # Cache is keyed on normalised text + the requested attribute set (summary only).
    cache_key = cache.make_key(normalized, requested)
    detoxify_scores = await cache.get_scores(cache_key)

    if detoxify_scores is None:
        CACHE_MISSES.inc()
        try:
            detoxify_scores = await classifier.predict(normalized)
        except Exception as exc:  # noqa: BLE001 — surface as structured HTTP 500
            logger.exception("Inference failed")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(error="inference_failed", detail=str(exc)).model_dump(),
            )
        await cache.set_scores(cache_key, detoxify_scores)
    else:
        CACHE_HITS.inc()

    attribute_scores = _to_perspective_scores(detoxify_scores, requested)

    if request.spanAnnotations:
        try:
            await _attach_span_scores(
                request.comment.text, attribute_scores, requested, _thresholds(request, requested)
            )
        except Exception as exc:  # noqa: BLE001 — span inference failure is still a 500
            logger.exception("Span inference failed")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(error="inference_failed", detail=str(exc)).model_dump(),
            )

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
