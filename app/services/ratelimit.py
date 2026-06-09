"""Redis-backed fixed-window rate limiter with fail-open semantics.

Disabled by default (``OPENSPECTIVE_RATE_LIMIT <= 0``). When enabled, each client is
allowed at most ``rate_limit`` requests per ``rate_limit_window`` seconds. The window
is a simple fixed window: a Redis counter keyed by ``(client, window_bucket)`` that is
``INCR``'d per request and expires at the end of the bucket.

If Redis is unavailable the limiter **fails open** (allows the request) — consistent
with the cache's graceful degradation — rather than rejecting traffic.
"""

import time
from dataclasses import dataclass

from app.services import redis_client
from app.services.redis_client import REDIS_ERRORS

# Redis key namespace for rate-limit counters.
_KEY_PREFIX = "openspective:rl"


@dataclass(frozen=True)
class RateLimitStatus:
    """Outcome of a rate-limit check for a single request."""

    allowed: bool
    limit: int
    remaining: int
    reset_seconds: int  # seconds until the current window resets


async def check(client_id: str, limit: int, window: int) -> RateLimitStatus:
    """Record a request for ``client_id`` and report whether it is within the limit.

    :param client_id: Identifier to rate-limit on (token or client IP).
    :param limit: Max requests allowed per window.
    :param window: Window length in seconds.
    :returns: A :class:`RateLimitStatus`. Fails open (``allowed=True``) if Redis is down.
    """
    client = redis_client.get_client()
    now = int(time.time())
    reset_seconds = window - (now % window)

    # Redis disabled/unavailable -> fail open.
    if client is None:
        return RateLimitStatus(
            allowed=True, limit=limit, remaining=limit, reset_seconds=reset_seconds
        )

    bucket = now // window
    key = f"{_KEY_PREFIX}:{client_id}:{bucket}"
    try:
        count = await client.incr(key)
        if count == 1:
            # First hit in this window — set the bucket to expire when the window ends.
            await client.expire(key, window)
    except REDIS_ERRORS as exc:
        redis_client.mark_unavailable(exc)
        return RateLimitStatus(
            allowed=True, limit=limit, remaining=limit, reset_seconds=reset_seconds
        )

    remaining = max(0, limit - count)
    return RateLimitStatus(
        allowed=count <= limit,
        limit=limit,
        remaining=remaining,
        reset_seconds=reset_seconds,
    )
