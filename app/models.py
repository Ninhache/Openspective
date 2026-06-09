"""Pydantic request/response schemas, mirroring the Google Perspective API.

Field names use Perspective's exact camelCase keys (``attributeScores``,
``summaryScore``, ``requestedAttributes``, ``detectedLanguages``, ``clientToken``)
so existing Perspective integrations can swap the endpoint URL with minimal changes.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Mapping from Detoxify's output keys to Perspective attribute names.
# Detoxify always scores every attribute; ``requestedAttributes`` filtering happens
# after inference (see routers/analyze.py).
DETOXIFY_TO_PERSPECTIVE: dict[str, str] = {
    "toxicity": "TOXICITY",
    "severe_toxicity": "SEVERE_TOXICITY",
    "obscene": "OBSCENE",
    "threat": "THREAT",
    "insult": "INSULT",
    "identity_attack": "IDENTITY_ATTACK",
}

# Reverse map (Perspective name -> Detoxify key) for convenience.
PERSPECTIVE_TO_DETOXIFY: dict[str, str] = {v: k for k, v in DETOXIFY_TO_PERSPECTIVE.items()}

# All Perspective attribute names openspective can return.
ALL_ATTRIBUTES: tuple[str, ...] = tuple(DETOXIFY_TO_PERSPECTIVE.values())


class Comment(BaseModel):
    """The text to be analysed."""

    text: str


class AnalyzeRequest(BaseModel):
    """A Perspective-compatible ``comments:analyze`` request body."""

    comment: Comment
    # Optional: if omitted/empty, all available attributes are returned. Each value
    # may carry a ``scoreThreshold`` (float) used to filter returned span scores.
    requestedAttributes: dict[str, dict] | None = None  # noqa: N815 (Perspective schema)
    # Optional: if omitted, language is auto-detected.
    languages: list[str] | None = None
    # Optional: when true, include per-span (sentence) scores per attribute.
    spanAnnotations: bool = False  # noqa: N815 (Perspective schema)
    clientToken: str | None = None  # noqa: N815 (Perspective schema)


class SummaryScore(BaseModel):
    """A single attribute's summary score."""

    value: float
    type: Literal["PROBABILITY"] = "PROBABILITY"


class SpanScore(BaseModel):
    """A score for a single span (sentence) of the comment.

    ``begin``/``end`` are character offsets into the original comment text.
    """

    begin: int
    end: int
    score: SummaryScore


class AttributeScore(BaseModel):
    """Wrapper holding the summary (and optional span) scores for one attribute."""

    summaryScore: SummaryScore  # noqa: N815 (Perspective schema)
    # Present only when ``spanAnnotations`` was requested; omitted otherwise.
    spanScores: list[SpanScore] | None = None  # noqa: N815 (Perspective schema)


class AnalyzeResponse(BaseModel):
    """A Perspective-compatible ``comments:analyze`` response body."""

    attributeScores: dict[str, AttributeScore]  # noqa: N815 (Perspective schema)
    languages: list[str] = Field(default_factory=list)
    detectedLanguages: list[str] = Field(default_factory=list)  # noqa: N815
    clientToken: str | None = None  # noqa: N815 (Perspective schema)


class ErrorResponse(BaseModel):
    """Structured error body returned on inference failure (HTTP 500)."""

    error: str
    detail: str
