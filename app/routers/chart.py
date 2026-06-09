"""``GET /chart`` — render a comment's attribute scores as an SVG of circular gauges.

Embeddable in an ``<img>`` tag: ``<img src="/chart?text=you%20are%20an%20idiot">``.
Runs the same scoring pipeline as the JSON analyze endpoint (so it is guarded by the
same auth + rate limiting when those are enabled).
"""

import logging

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, Response

from app.models import ALL_ATTRIBUTES, PERSPECTIVE_TO_DETOXIFY, ErrorResponse
from app.routers.validation import oversize_response
from app.services import chart, scoring
from app.services.detector import detect_language

logger = logging.getLogger("openspective.chart")

router = APIRouter()


def _resolve_attributes(attributes: str | None) -> list[str]:
    """Parse the comma-separated ``attributes`` query param (defaults to all)."""
    if not attributes:
        return list(ALL_ATTRIBUTES)
    names = [name.strip().upper() for name in attributes.split(",")]
    resolved = [name for name in names if name in PERSPECTIVE_TO_DETOXIFY]
    return resolved or list(ALL_ATTRIBUTES)


@router.get("/chart")
async def chart_svg(
    text: str = Query(..., description="Comment text to score and visualise."),
    attributes: str | None = Query(
        None, description="Comma-separated Perspective attribute names (default: all)."
    ),
    style: str = Query(
        "gauges", description="Chart style: gauges | radar | polar."
    ),
):
    """Return an SVG chart of the comment's attribute scores."""
    if (oversize := oversize_response(text)) is not None:
        return oversize
    requested = _resolve_attributes(attributes)
    chart_style = style if style in chart.STYLES else "gauges"
    try:
        scores = await scoring.summary_scores(text, requested)
    except Exception as exc:  # noqa: BLE001 — surface as structured HTTP 500
        logger.exception("Inference failed")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error="inference_failed", detail=str(exc)).model_dump(),
        )

    language = detect_language(text)[0]
    svg = chart.render_svg(text, scores, language=language, style=chart_style)
    # Short cache so embeds don't hammer inference, but stay reasonably fresh.
    return Response(
        content=svg,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=60"},
    )
