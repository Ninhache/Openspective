"""Tests for Redis-backed rate limiting (HTTP behaviour + window logic)."""

import asyncio

from app.services import ratelimit
from app.services.ratelimit import RateLimitStatus

ANALYZE_URL = "/v1alpha1/comments:analyze"
PAYLOAD = {"comment": {"text": "hello"}}


# --- HTTP-level behaviour (rate-limit service mocked) ------------------------


def test_rate_limit_disabled_by_default(client):
    """With rate_limit=0, no limiting occurs and no X-RateLimit headers are set."""
    resp = client.post(ANALYZE_URL, json=PAYLOAD)
    assert resp.status_code == 200
    assert "X-RateLimit-Limit" not in resp.headers


def test_allowed_request_sets_headers(rl_client, monkeypatch):
    """An allowed request returns 200 and exposes X-RateLimit-* headers."""
    async def _allowed(client_id, limit, window):
        return RateLimitStatus(allowed=True, limit=2, remaining=1, reset_seconds=42)

    monkeypatch.setattr(ratelimit, "check", _allowed)
    resp = rl_client.post(ANALYZE_URL, json=PAYLOAD)
    assert resp.status_code == 200
    assert resp.headers["X-RateLimit-Limit"] == "2"
    assert resp.headers["X-RateLimit-Remaining"] == "1"
    assert resp.headers["X-RateLimit-Reset"] == "42"


def test_exceeding_limit_returns_429(rl_client, monkeypatch):
    """A blocked request returns a structured 429 with a Retry-After header."""
    async def _blocked(client_id, limit, window):
        return RateLimitStatus(allowed=False, limit=2, remaining=0, reset_seconds=30)

    monkeypatch.setattr(ratelimit, "check", _blocked)
    resp = rl_client.post(ANALYZE_URL, json=PAYLOAD)
    assert resp.status_code == 429
    assert resp.json()["error"] == "rate_limited"
    assert resp.headers["Retry-After"] == "30"


# --- Window logic (Redis client faked) ---------------------------------------


class _FakeRedis:
    """Minimal async stand-in supporting the INCR/EXPIRE the limiter uses."""

    def __init__(self):
        self.counts = {}

    async def incr(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    async def expire(self, key, ttl):
        return True


def test_check_counts_requests_within_window(monkeypatch):
    """check() allows up to the limit then blocks subsequent requests in the window."""
    from app.services import redis_client

    # Bind ONE fake instance so counts persist across calls (a fresh one per call
    # would reset the window counter).
    fake = _FakeRedis()
    monkeypatch.setattr(redis_client, "get_client", lambda: fake)

    first = asyncio.run(ratelimit.check("ip:1.2.3.4", limit=2, window=60))
    second = asyncio.run(ratelimit.check("ip:1.2.3.4", limit=2, window=60))
    third = asyncio.run(ratelimit.check("ip:1.2.3.4", limit=2, window=60))

    assert first.allowed and first.remaining == 1
    assert second.allowed and second.remaining == 0
    assert not third.allowed


def test_check_fails_open_when_redis_unavailable(monkeypatch):
    """When Redis is unavailable, check() fails open (allows the request)."""
    from app.services import redis_client

    monkeypatch.setattr(redis_client, "get_client", lambda: None)
    status = asyncio.run(ratelimit.check("ip:1.2.3.4", limit=2, window=60))
    assert status.allowed
    assert status.remaining == 2
