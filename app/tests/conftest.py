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


@pytest.fixture
def client(monkeypatch):
    """Yield a TestClient with the model and Redis cache stubbed out."""
    from app.services import cache, classifier

    # Avoid importing torch / downloading weights during the lifespan startup.
    monkeypatch.setattr(classifier, "load_model", lambda variant: None)
    monkeypatch.setattr(classifier, "model_name", lambda: "multilingual")

    async def _fake_predict(text: str) -> dict[str, float]:
        return dict(FAKE_SCORES)

    monkeypatch.setattr(classifier, "predict", _fake_predict)

    # Disable the cache entirely: always a miss, set is a no-op.
    async def _miss(key):
        return None

    async def _noop(key, scores):
        return None

    monkeypatch.setattr(cache, "get_scores", _miss)
    monkeypatch.setattr(cache, "set_scores", _noop)

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
