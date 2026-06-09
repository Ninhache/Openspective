"""Shared request validation helpers for the scoring routers."""

from fastapi.responses import JSONResponse

from app.config import get_settings
from app.models import ErrorResponse


def oversize_response(text: str) -> JSONResponse | None:
    """Return a 400 response if ``text`` exceeds the configured size limit, else ``None``.

    Enforced at the router boundary so oversized comments are rejected explicitly
    rather than silently truncated or scored over an unbounded number of chunks.
    The limit is ``OPENSPECTIVE_MAX_TEXT_CHARS`` (default mirrors Perspective's 20480).
    """
    limit = get_settings().max_text_chars
    if len(text) > limit:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="text_too_large",
                detail=f"Comment text exceeds the {limit}-character limit ({len(text)} chars).",
            ).model_dump(),
        )
    return None
