"""Shared test fixtures.

The real Detoxify model (torch + ~1GB of weights) is never loaded in tests. We
patch the classifier's ``load_model``/``predict`` and the Redis cache so the suite
runs fast and offline, exercising the HTTP layer and business logic in isolation.
"""

import pytest
from fastapi.testclient import TestClient

# Fixed, deterministic Detoxify-shaped output used by the fake classifier.
FAKE_SCORES = {
    "toxicity": 0.92,
    "severe_toxicity": 0.10,
    "obscene": 0.30,
    "threat": 0.04,
    "insult": 0.71,
    "identity_attack": 0.05,
}

# Token used by the authenticated-client fixture.
TEST_TOKEN = "test-secret-token"


def _apply_common_mocks(monkeypatch):
    """Stub the model and Redis cache so tests run offline and fast."""
    from app.services import cache, classifier

    monkeypatch.setattr(classifier, "load_model", lambda variant: None)
    monkeypatch.setattr(classifier, "model_name", lambda: "multilingual")

    async def _fake_predict(text: str) -> dict[str, float]:
        return dict(FAKE_SCORES)

    monkeypatch.setattr(classifier, "predict", _fake_predict)

    async def _miss(key):
        return None

    async def _noop(key, scores):
        return None

    monkeypatch.setattr(cache, "get_scores", _miss)
    monkeypatch.setattr(cache, "set_scores", _noop)


@pytest.fixture
def client(monkeypatch):
    """Yield a TestClient with the model and Redis cache stubbed out (no auth)."""
    _apply_common_mocks(monkeypatch)
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def rl_client(monkeypatch):
    """Yield a TestClient with rate limiting enabled (limit 2 / 60s window)."""
    from app.config import get_settings

    monkeypatch.setenv("OPENSPECTIVE_RATE_LIMIT", "2")
    monkeypatch.setenv("OPENSPECTIVE_RATE_LIMIT_WINDOW", "60")
    get_settings.cache_clear()
    _apply_common_mocks(monkeypatch)
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


@pytest.fixture
def auth_client(monkeypatch):
    """Yield a TestClient with Bearer auth enabled (token == TEST_TOKEN)."""
    from app.config import get_settings

    monkeypatch.setenv("OPENSPECTIVE_API_TOKENS", TEST_TOKEN)
    # Settings are cached; clear so the env var is picked up, and clear again on teardown.
    get_settings.cache_clear()
    _apply_common_mocks(monkeypatch)
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()
