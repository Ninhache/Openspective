"""Integration tests for POST /v1/moderate.

Uses the shared ``client`` fixture (model + Redis stubbed). Synthetic strings only.
"""

import pytest

from app.services import lexicon

MODERATE_URL = "/v1/moderate"


@pytest.fixture(autouse=True)
def _fresh_lexicon():
    lexicon.get_lexicon.cache_clear()
    yield
    lexicon.get_lexicon.cache_clear()


def test_block_decision(client):
    resp = client.post(MODERATE_URL, json={"text": "fuckankama", "context": "username"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "block"
    assert body["tier"] == "block"
    assert "fuck" in body["reasons"]


def test_allow_decision_for_domain_vocab(client):
    resp = client.post(MODERATE_URL, json={"text": "zob heal", "context": "build_name"})
    assert resp.json()["decision"] == "allow"


def test_flag_decision_soft(client):
    resp = client.post(MODERATE_URL, json={"text": "what a pile of shit"})
    assert resp.json()["decision"] == "flag"


def test_short_clean_text_skips_ml(client):
    """A short, clean username stays lexicon-only (no ML call) and is allowed."""
    resp = client.post(MODERATE_URL, json={"text": "Mathius"})
    body = resp.json()
    assert body["decision"] == "allow"
    assert "mlScore" not in body  # ML never ran


def test_long_clean_text_triggers_ml_fallback(client):
    """Long, lexicon-clean text falls back to the model; the fake scorer flags it."""
    text = "this is a perfectly ordinary sentence with no banned words at all here"
    resp = client.post(MODERATE_URL, json={"text": text})
    body = resp.json()
    # conftest's fake classifier returns toxicity 0.92 (>= 0.8 threshold) -> flag via ML
    assert body["decision"] == "flag"
    assert body["tier"] == "ml"
    assert body["mlScore"] >= 0.8


def test_ml_fallback_below_threshold_allows(client, monkeypatch):
    """When the ML score is low, clean long text is allowed."""
    from app.routers import moderate

    async def _low(text, attrs):
        return {a: 0.01 for a in attrs}

    monkeypatch.setattr(moderate.scoring, "summary_scores", _low)
    text = "another long and entirely harmless sentence about gardening and tea"
    resp = client.post(MODERATE_URL, json={"text": text})
    assert resp.json()["decision"] == "allow"


def test_oversize_text_rejected(client):
    from app.config import get_settings

    limit = get_settings().max_text_chars
    resp = client.post(MODERATE_URL, json={"text": "a" * (limit + 1)})
    assert resp.status_code == 400
    assert resp.json()["error"] == "text_too_large"
