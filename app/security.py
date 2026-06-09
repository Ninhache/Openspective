"""Optional Bearer-token authentication.

Authentication is **disabled by default** (no tokens configured) so openspective
stays a drop-in replacement out of the box. Set ``OPENSPECTIVE_API_TOKENS`` to a
comma-separated list of tokens to require ``Authorization: Bearer <token>`` on the
analyze endpoint. Operational endpoints (/healthz, /readyz, /metrics, /v1/models)
remain unauthenticated so probes and scrapers can always reach them.
"""

import secrets

from fastapi import Depends, Header, HTTPException, Request, Response

from app.config import Settings, get_settings
from app.services import ratelimit


class AuthError(HTTPException):
    """Raised on a missing/invalid Bearer token; rendered as a structured 401."""

    def __init__(self, detail: str) -> None:
        # WWW-Authenticate signals the scheme to clients per RFC 6750.
        super().__init__(status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"})


class RateLimitError(HTTPException):
    """Raised when a client exceeds its rate limit; rendered as a structured 429."""

    def __init__(self, detail: str, retry_after: int) -> None:
        super().__init__(
            status_code=429, detail=detail, headers={"Retry-After": str(retry_after)}
        )


def _token_matches(presented: str, configured: set[str]) -> bool:
    """Constant-time membership check to avoid leaking token bytes via timing."""
    return any(secrets.compare_digest(presented, token) for token in configured)


async def require_auth(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """FastAPI dependency enforcing Bearer auth when tokens are configured.

    :raises AuthError: If auth is enabled and the token is missing or invalid.
    """
    tokens = settings.api_token_set
    if not tokens:
        # Auth disabled — allow through.
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise AuthError("Missing or malformed Authorization header; expected 'Bearer <token>'")
    presented = authorization[len("Bearer "):].strip()
    if not presented or not _token_matches(presented, tokens):
        raise AuthError("Invalid API token")


def _client_id(request: Request, authorization: str | None) -> str:
    """Derive the rate-limit identity: the Bearer token if present, else client IP."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization[len("Bearer "):].strip()
        if token:
            return f"token:{token}"
    client = request.client
    return f"ip:{client.host}" if client else "ip:unknown"


async def rate_limit(
    request: Request,
    response: Response,
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """FastAPI dependency enforcing a per-client rate limit when configured.

    Disabled when ``OPENSPECTIVE_RATE_LIMIT <= 0``. Keys on the Bearer token if one is
    present, otherwise the client IP. Sets ``X-RateLimit-*`` headers on the response.

    :raises RateLimitError: If the client has exceeded its allowance this window.
    """
    if settings.rate_limit <= 0:
        return
    status = await ratelimit.check(
        _client_id(request, authorization), settings.rate_limit, settings.rate_limit_window
    )
    response.headers["X-RateLimit-Limit"] = str(status.limit)
    response.headers["X-RateLimit-Remaining"] = str(status.remaining)
    response.headers["X-RateLimit-Reset"] = str(status.reset_seconds)
    if not status.allowed:
        raise RateLimitError(
            f"Rate limit exceeded: {status.limit} requests per "
            f"{settings.rate_limit_window}s",
            retry_after=status.reset_seconds,
        )
