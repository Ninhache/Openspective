"""Shared scoring orchestration.

Both the JSON analyze endpoint and the SVG chart endpoint need the same thing:
summary scores for a comment, with caching and metrics. That logic lives here so
the two routers stay thin and consistent (normalise -> cache lookup -> infer ->
cache store -> map to Perspective attribute names).
"""

from app.models import DETOXIFY_TO_PERSPECTIVE
from app.services import cache, classifier
from app.services.metrics import CACHE_HITS, CACHE_MISSES
from app.services.normalizer import normalize


async def summary_scores(text: str, requested: list[str]) -> dict[str, float]:
    """Return summary scores for ``text``, keyed by Perspective attribute name.

    Filtering to ``requested`` happens after inference (Detoxify always scores all
    attributes). Raises whatever ``classifier.predict`` raises on inference failure;
    callers map that to an HTTP 500.

    :param text: Raw comment text (normalisation happens here).
    :param requested: Perspective attribute names to keep in the result.
    :returns: Mapping of Perspective attribute name -> probability in ``[0, 1]``.
    """
    normalized = normalize(text)
    cache_key = cache.make_key(normalized, requested)
    detoxify_scores = await cache.get_scores(cache_key)

    if detoxify_scores is None:
        CACHE_MISSES.inc()
        detoxify_scores = await classifier.predict(normalized)
        await cache.set_scores(cache_key, detoxify_scores)
    else:
        CACHE_HITS.inc()

    return {
        DETOXIFY_TO_PERSPECTIVE[key]: value
        for key, value in detoxify_scores.items()
        if DETOXIFY_TO_PERSPECTIVE.get(key) in requested
    }
