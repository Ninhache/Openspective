"""Integration tests for POST /v1/moderate.

Uses the shared ``client`` fixture (model + Redis stubbed). Synthetic strings only.
The fake classifier returns a high toxicity (0.92) for any text, so ML-branch tests
either keep text under the ML min-length or monkeypatch ``summary_scores`` explicitly.
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
    assert body["score"] == 1.0
    assert "fuck" in body["reasons"]


def test_allow_decision_for_domain_vocab(client):
    # short (< ml_min_chars) so it stays lexicon-only
    resp = client.post(MODERATE_URL, json={"text": "zob heal", "context": "build_name"})
    body = resp.json()
    assert body["decision"] == "allow"
    assert body["score"] == 0.0


def test_flag_decision_soft_lexicon(client):
    # 'shit' is a flag-tier token; keep it short so ML doesn't run and escalate it
    resp = client.post(MODERATE_URL, json={"text": "shit"})
    body = resp.json()
    assert body["decision"] == "flag"
    assert body["tier"] == "flag"
    assert body["score"] == 0.6


def test_short_clean_text_skips_ml(client):
    resp = client.post(MODERATE_URL, json={"text": "Mathius"})
    body = resp.json()
    assert body["decision"] == "allow"
    assert "mlScore" not in body  # ML never ran


def test_ml_high_score_blocks(client, monkeypatch):
    """Long, lexicon-clean text scored high by the model is blocked (>= block threshold)."""
    from app.routers import moderate

    async def _high(text, attrs):
        return {a: 0.95 for a in attrs}

    monkeypatch.setattr(moderate.scoring, "summary_scores", _high)
    resp = client.post(MODERATE_URL, json={"text": "a clean long sentence with no bad words"})
    body = resp.json()
    assert body["decision"] == "block"
    assert body["tier"] == "ml"
    assert body["mlScore"] >= 0.9


def test_ml_midrange_flags(client, monkeypatch):
    """A mid-range ML score lands in the flag (review) band."""
    from app.routers import moderate

    async def _mid(text, attrs):
        return {a: 0.6 for a in attrs}

    monkeypatch.setattr(moderate.scoring, "summary_scores", _mid)
    resp = client.post(MODERATE_URL, json={"text": "a perfectly clean longer sentence here"})
    body = resp.json()
    assert body["decision"] == "flag"
    assert body["tier"] == "ml"


def test_ml_low_score_allows(client, monkeypatch):
    from app.routers import moderate

    async def _low(text, attrs):
        return {a: 0.01 for a in attrs}

    monkeypatch.setattr(moderate.scoring, "summary_scores", _low)
    resp = client.post(MODERATE_URL, json={"text": "another harmless longer sentence here"})
    assert resp.json()["decision"] == "allow"


def test_oversize_text_rejected(client):
    from app.config import get_settings

    limit = get_settings().max_text_chars
    resp = client.post(MODERATE_URL, json={"text": "a" * (limit + 1)})
    assert resp.status_code == 400
    assert resp.json()["error"] == "text_too_large"
