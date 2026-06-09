"""Redis-backed score cache with graceful degradation.

All Redis access is isolated in this module — routers never import redis directly.
If Redis is unavailable the service logs a warning and continues without caching,
so a Redis outage degrades performance but never causes request failures.

Cache key:  ``sha256(normalized_text + "|" + ",".join(sorted(requested_attributes)))``
Cache value: JSON-serialised ``attributeScores`` dict.
"""

import hashlib
import json
import logging
from collections.abc import Iterable

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from app.config import get_settings

logger = logging.getLogger("openspective.cache")

# Lazily-created shared client. ``None`` until first use; never recreated on error.
_client: aioredis.Redis | None = None
# Once a connection has hard-failed we stop retrying for the process lifetime to
# avoid logging a warning on every request.
_disabled = False


def make_key(normalized_text: str, requested_attributes: Iterable[str]) -> str:
    """Build the deterministic cache key for a request.

    :param normalized_text: Text after normalisation.
    :param requested_attributes: The Perspective attribute names requested.
    :returns: Hex sha256 digest used as the Redis key.
    """
    payload = normalized_text + "|" + ",".join(sorted(requested_attributes))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _get_client() -> aioredis.Redis | None:
    """Return the shared async Redis client, or ``None`` if caching is disabled."""
    global _client
    if _disabled:
        return None
    if _client is None:
        settings = get_settings()
        _client = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _client


async def get_scores(key: str) -> dict[str, float] | None:
    """Return cached Detoxify scores for ``key``, or ``None`` on miss/unavailable.

    A Redis error is logged once and treated as a cache miss.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        raw = await client.get(key)
    except (RedisError, OSError) as exc:  # connection refused, timeout, etc.
        _warn_unavailable(exc)
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
    client = _get_client()
    if client is None:
        return
    settings = get_settings()
    try:
        await client.set(key, json.dumps(scores), ex=settings.cache_ttl)
    except (RedisError, OSError) as exc:
        _warn_unavailable(exc)


async def close() -> None:
    """Close the Redis connection pool (called from the lifespan shutdown)."""
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except (RedisError, OSError):
            pass
        _client = None


def _warn_unavailable(exc: Exception) -> None:
    """Log the first Redis failure and disable caching for the process."""
    global _disabled
    if not _disabled:
        logger.warning("Redis unavailable, continuing without cache: %s", exc)
        _disabled = True
