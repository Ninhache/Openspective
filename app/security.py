"""Optional Bearer-token authentication.

Authentication is **disabled by default** (no tokens configured) so openspective
stays a drop-in replacement out of the box. Set ``OPENSPECTIVE_API_TOKENS`` to a
comma-separated list of tokens to require ``Authorization: Bearer <token>`` on the
analyze endpoint. Operational endpoints (/healthz, /readyz, /metrics, /v1/models)
remain unauthenticated so probes and scrapers can always reach them.
"""

import secrets

from fastapi import Depends, Header, HTTPException

from app.config import Settings, get_settings


class AuthError(HTTPException):
    """Raised on a missing/invalid Bearer token; rendered as a structured 401."""

    def __init__(self, detail: str) -> None:
        # WWW-Authenticate signals the scheme to clients per RFC 6750.
        super().__init__(status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"})


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
