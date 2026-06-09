"""Redis-backed score cache with graceful degradation.

All Redis access is isolated behind this module (and the shared ``redis_client``) —
routers never import redis directly. If Redis is unavailable the service logs a
warning and continues without caching, so a Redis outage degrades performance but
never causes request failures.

Cache key:  ``sha256(normalized_text + "|" + ",".join(sorted(requested_attributes)))``
Cache value: JSON-serialised ``attributeScores`` dict.
"""

import hashlib
import json
import logging
from collections.abc import Iterable

from app.config import get_settings
from app.services import redis_client
from app.services.redis_client import REDIS_ERRORS

logger = logging.getLogger("openspective.cache")


def make_key(normalized_text: str, requested_attributes: Iterable[str]) -> str:
    """Build the deterministic cache key for a request.

    :param normalized_text: Text after normalisation.
    :param requested_attributes: The Perspective attribute names requested.
    :returns: Hex sha256 digest used as the Redis key.
    """
    payload = normalized_text + "|" + ",".join(sorted(requested_attributes))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def get_scores(key: str) -> dict[str, float] | None:
    """Return cached Detoxify scores for ``key``, or ``None`` on miss/unavailable.

    A Redis error is logged once and treated as a cache miss.
    """
    client = redis_client.get_client()
    if client is None:
        return None
    try:
        raw = await client.get(key)
    except REDIS_ERRORS as exc:  # connection refused, timeout, etc.
        redis_client.mark_unavailable(exc)
        return None
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Discarding corrupt cache entry for key=%s", key)
        return None


async def set_scores(key: str, scores: dict[str, float]) -> None:
    """Cache ``scores`` under ``key`` with the configured TTL.

    Failures are swallowed (logged once) — caching is best-effort.
    """
    client = redis_client.get_client()
    if client is None:
        return
    settings = get_settings()
    try:
        await client.set(key, json.dumps(scores), ex=settings.cache_ttl)
    except REDIS_ERRORS as exc:
        redis_client.mark_unavailable(exc)
