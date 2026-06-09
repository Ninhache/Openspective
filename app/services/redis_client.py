"""Shared async Redis client with graceful degradation.

Both the score cache and the rate limiter use Redis. They share a single
connection pool and a single "Redis is down" latch here so that a Redis outage
degrades every Redis-backed feature consistently (cache misses, rate limiting
fails open) without each module reimplementing the connect/disable dance.

Routers never import this directly — they go through ``cache`` / ``ratelimit``.
"""

import logging

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from app.config import get_settings

logger = logging.getLogger("openspective.redis")

# Lazily-created shared client. ``None`` until first use.
_client: aioredis.Redis | None = None
# Once a connection has hard-failed we stop retrying for the process lifetime to
# avoid logging a warning on every request.
_disabled = False

# Exceptions that mean "Redis is unreachable" and should trigger fail-open.
REDIS_ERRORS = (RedisError, OSError)


def get_client() -> aioredis.Redis | None:
    """Return the shared async Redis client, or ``None`` if Redis is disabled."""
    global _client
    if _disabled:
        return None
    if _client is None:
        settings = get_settings()
        _client = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _client


def mark_unavailable(exc: Exception) -> None:
    """Latch Redis as unavailable and log the first failure only."""
    global _disabled
    if not _disabled:
        logger.warning("Redis unavailable, continuing in degraded mode: %s", exc)
        _disabled = True


async def close() -> None:
    """Close the Redis connection pool (called from the lifespan shutdown)."""
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except REDIS_ERRORS:
            pass
        _client = None
