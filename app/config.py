"""Application configuration loaded from environment variables.

All settings are read from ``OPENSPECTIVE_*`` environment variables via
pydantic-settings. ``get_settings()`` is cached so the same instance is reused
across the process (and so it can be overridden in tests).
"""

from functools import lru_cache

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
        # ``model_variant`` and opt out of the warning.
        protected_namespaces=(),
    )

    model_variant: str = "multilingual"  # OPENSPECTIVE_MODEL -> see alias below
    redis_url: str = "redis://redis:6379"  # OPENSPECTIVE_REDIS_URL
    cache_ttl: int = 3600  # OPENSPECTIVE_CACHE_TTL (seconds)
    log_level: str = "INFO"  # OPENSPECTIVE_LOG_LEVEL
    workers: int = 1  # OPENSPECTIVE_WORKERS

    # The spec exposes the model variant as ``OPENSPECTIVE_MODEL`` (not
    # ``OPENSPECTIVE_MODEL_VARIANT``); map that env var onto ``model_variant``.
    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):  # noqa: D401, E501
        return (init_settings, _ModelEnvAlias(env_settings), dotenv_settings, file_secret_settings)


class _ModelEnvAlias:
    """Wrap the env source so ``OPENSPECTIVE_MODEL`` feeds ``model_variant``."""

    def __init__(self, inner):
        self._inner = inner

    def __call__(self):
        values = dict(self._inner())
        # Accept the documented OPENSPECTIVE_MODEL env var as an alias.
        import os

        raw = os.environ.get("OPENSPECTIVE_MODEL")
        if raw is not None:
            values["model_variant"] = raw
        return values


@lru_cache
def get_settings() -> Settings:
    """Return the cached :class:`Settings` instance for dependency injection."""
    return Settings()
