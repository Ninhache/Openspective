"""Language detection for populating ``detectedLanguages``.

Perspective auto-detects the comment language when the caller does not supply one.
We use ``langdetect`` (pure-Python, no model download, offline) for parity. Detection
is best-effort: on very short / empty / undetectable input we return ``["unknown"]``
rather than failing the request.
"""

import logging

from langdetect import DetectorFactory, LangDetectException, detect

logger = logging.getLogger("openspective.detector")

# langdetect samples randomly; fixing the seed makes results deterministic so the
# same text always yields the same code (and the same cache behaviour upstream).
DetectorFactory.seed = 0

# Returned when language cannot be determined.
UNKNOWN = "unknown"


def detect_language(text: str) -> list[str]:
    """Best-effort language detection.

    :param text: The comment text (raw or normalised both work).
    :returns: A single-element list with an ISO 639-1 code, or ``["unknown"]``.
    """
    stripped = text.strip()
    if not stripped:
        return [UNKNOWN]
    try:
        return [detect(stripped)]
    except LangDetectException:
        # Too short, no detectable features, etc. — degrade gracefully.
        logger.debug("Language detection failed for input of length %d", len(stripped))
        return [UNKNOWN]
