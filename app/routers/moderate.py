"""``POST /v1/moderate`` — one allow/flag/block verdict for a piece of text.

Composes the two detectors so callers don't have to: the deterministic lexicon is
the primary signal (and the only one for short fields like usernames), and the
Detoxify model is a fallback for longer, notes-like text where it actually performs
well. The lexicon always wins when it fires; ML only runs on a clean, long string.
"""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.models import ErrorResponse, ModerateRequest, ModerateResponse
from app.routers.validation import oversize_response
from app.services import lexicon, scoring

logger = logging.getLogger("openspective.moderate")

router = APIRouter()

# Attributes the ML fallback considers (the "is this nasty?" signals).
_ML_ATTRIBUTES = ["TOXICITY", "INSULT", "IDENTITY_ATTACK", "THREAT", "OBSCENE"]


@router.post("/v1/moderate", response_model=ModerateResponse, response_model_exclude_none=True)
async def moderate(request: ModerateRequest):
    """Return a moderation verdict for ``request.text``."""
    if (oversize := oversize_response(request.text)) is not None:
        return oversize

    # 1) Lexicon (primary, deterministic). A hit settles it immediately.
    verdict = lexicon.evaluate(request.text)
    if verdict.decision != "allow":
        return ModerateResponse(
            decision=verdict.decision, reasons=verdict.reasons, tier=verdict.tier
        )

    # 2) ML fallback — only on clean, long-enough text (keeps short fields fast).
    settings = get_settings()
    if settings.moderate_ml_min_chars and len(request.text) >= settings.moderate_ml_min_chars:
        try:
            scores = await scoring.summary_scores(request.text, _ML_ATTRIBUTES)
        except Exception as exc:  # noqa: BLE001 — surface as structured HTTP 500
            logger.exception("Moderation ML fallback failed")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(error="inference_failed", detail=str(exc)).model_dump(),
            )
        top = max(scores.values(), default=0.0)
        if top >= settings.moderate_ml_threshold:
            worst = max(scores, key=scores.get)
            return ModerateResponse(
                decision="flag", reasons=["ml", worst], tier="ml", mlScore=round(top, 4)
            )

    return ModerateResponse(decision="allow")
