"""``POST /v1/moderate`` — one allow/flag/block verdict for a piece of text.

Folds the two detectors into a single severity ``score`` in ``[0, 1]``, then maps it
to a decision via two thresholds:

* the deterministic **lexicon** scores clear hits (slur = 1.0, soft term = 0.6) — the
  primary signal, and the only one for short fields like usernames;
* the **Detoxify model** scores *contextual* toxicity (e.g. a neutral word used as an
  insult) that no word list can catch — run on longer / lexicon-clean text.

``score = max(lexicon, ml)`` → ``block`` if ``>= block_threshold``, ``flag`` if
``>= flag_threshold`` (review by a moderator), else ``allow``.
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

# Identity-field contexts where generic swears are blocked outright (a username can't be
# "fuckankama"). Other contexts (build names, bios, notes) leave swears to the ML, which
# tells colloquial use from targeted abuse.
_STRICT_CONTEXTS = {"username", "slug", "displayname", "display_name", "name", "handle"}

# Attributes the ML layer considers — only **target-directed** abuse signals. We exclude
# both ``TOXICITY`` (fires on game combat jargon: "nuke the boss") and ``OBSCENE`` (fires
# on gibberish and innocent build names, and double-counts profanity the lexicon already
# handles better). The remaining signals separate real abuse from game text cleanly:
# a full-dataset sweep flagged only insult/threat/identity hits, no false positives.
_ML_ATTRIBUTES = ["INSULT", "IDENTITY_ATTACK", "THREAT", "SEVERE_TOXICITY"]


def _decide(score: float, settings) -> str:
    """Map a unified severity score to a decision via the configured thresholds."""
    if score >= settings.moderate_block_threshold:
        return "block"
    if score >= settings.moderate_flag_threshold:
        return "flag"
    return "allow"


@router.post("/v1/moderate", response_model=ModerateResponse, response_model_exclude_none=True)
async def moderate(request: ModerateRequest):
    """Return a moderation verdict (decision + unified score) for ``request.text``."""
    if (oversize := oversize_response(request.text)) is not None:
        return oversize

    settings = get_settings()

    # 1) Lexicon (deterministic). A block hit is a certainty — settle immediately.
    # Identity fields (username/slug) block generic swears too; free text leaves them to ML.
    strict = (request.context or "").strip().lower() in _STRICT_CONTEXTS
    verdict = lexicon.evaluate(request.text, strict=strict)
    if verdict.decision == "block":
        return ModerateResponse(
            decision="block", score=verdict.score, reasons=verdict.reasons, tier="block"
        )

    # 2) ML (contextual). Run on long-enough text and take the strongest signal.
    score = verdict.score  # 0.0, or the flag-band score from a soft lexicon hit
    reasons = list(verdict.reasons)
    tier = verdict.tier
    ml_score: float | None = None
    if settings.moderate_ml_min_chars and len(request.text) >= settings.moderate_ml_min_chars:
        try:
            scores = await scoring.summary_scores(request.text, _ML_ATTRIBUTES)
        except Exception as exc:  # noqa: BLE001 — surface as structured HTTP 500
            logger.exception("Moderation ML scoring failed")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(error="inference_failed", detail=str(exc)).model_dump(),
            )
        if scores:
            ml_score = round(max(scores.values()), 4)
            if ml_score > score:
                score, tier, reasons = ml_score, "ml", ["ml", max(scores, key=scores.get)]

    decision = _decide(score, settings)
    if decision == "allow":
        return ModerateResponse(decision="allow", score=round(score, 4), mlScore=ml_score)
    return ModerateResponse(
        decision=decision, score=round(score, 4), reasons=reasons, tier=tier, mlScore=ml_score
    )
