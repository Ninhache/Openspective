"""Application configuration loaded from environment variables.

All settings are read from ``OPENSPECTIVE_*`` environment variables via
pydantic-settings. ``get_settings()`` is cached so the same instance is reused
across the process (and can be overridden in tests via ``get_settings.cache_clear()``).
"""

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Valid Detoxify model variants. The default ``multilingual`` is XLM-RoBERTa and
# supports EN/FR/ES/IT/PT/TR/RU. See README for the trade-offs of each variant.
VALID_MODEL_VARIANTS = ("original", "unbiased", "multilingual")


class Settings(BaseSettings):
    """Runtime settings, populated from ``OPENSPECTIVE_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="OPENSPECTIVE_",
        env_file=".env",
        extra="ignore",
        # ``model_`` is a protected namespace in pydantic v2; we deliberately use
        # ``model_variant`` and opt out of the resulting warning.
        protected_namespaces=(),
    )

    # The spec documents this env var as ``OPENSPECTIVE_MODEL`` (not
    # ``OPENSPECTIVE_MODEL_VARIANT``); accept both via an alias. The value is either
    # a built-in variant name or a path to a fine-tuned Detoxify checkpoint file.
    model_variant: str = Field(
        default="multilingual",
        validation_alias=AliasChoices("OPENSPECTIVE_MODEL", "OPENSPECTIVE_MODEL_VARIANT"),
    )
    redis_url: str = "redis://redis:6379"  # OPENSPECTIVE_REDIS_URL
    cache_ttl: int = 3600  # OPENSPECTIVE_CACHE_TTL (seconds)
    log_level: str = "INFO"  # OPENSPECTIVE_LOG_LEVEL
    workers: int = 1  # OPENSPECTIVE_WORKERS
    # Developer conveniences: DEBUG logging + permissive CORS (for a local frontend
    # on another port). Off by default. Hot-reload is enabled via the launch command
    # (scripts/dev.sh or docker-compose.dev.yml), not here.
    dev_mode: bool = False  # OPENSPECTIVE_DEV_MODE

    @property
    def effective_log_level(self) -> str:
        """DEBUG when dev mode is on, otherwise the configured level."""
        return "DEBUG" if self.dev_mode else self.log_level
    # Comma-separated Bearer tokens. Empty (default) disables auth entirely, which
    # keeps openspective a frictionless drop-in. Set OPENSPECTIVE_API_TOKENS to enable.
    api_tokens: str = ""  # OPENSPECTIVE_API_TOKENS

    # Rate limiting (Redis-backed fixed window). ``rate_limit <= 0`` disables it
    # (the default), keeping drop-in parity. When enabled, a client may make at most
    # ``rate_limit`` requests per ``rate_limit_window`` seconds.
    rate_limit: int = 0  # OPENSPECTIVE_RATE_LIMIT (max requests per window)
    rate_limit_window: int = 60  # OPENSPECTIVE_RATE_LIMIT_WINDOW (seconds)

    # Default span-score threshold: when span annotations are requested, spans scoring
    # below this value are omitted. A per-attribute ``scoreThreshold`` in the request
    # overrides this. 0.0 (default) returns all spans.
    score_threshold: float = 0.0  # OPENSPECTIVE_SCORE_THRESHOLD

    # Maximum comment length (characters). Longer text is rejected with HTTP 400 rather
    # than silently truncated or scored over many chunks. Default mirrors Google
    # Perspective's comment size limit (20480 bytes) for drop-in parity.
    max_text_chars: int = 20480  # OPENSPECTIVE_MAX_TEXT_CHARS

    @property
    def api_token_set(self) -> set[str]:
        """Return the configured Bearer tokens as a set (empty == auth disabled)."""
        return {token.strip() for token in self.api_tokens.split(",") if token.strip()}


@lru_cache
def get_settings() -> Settings:
    """Return the cached :class:`Settings` instance for dependency injection."""
    return Settings()
