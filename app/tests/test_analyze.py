"""Integration tests for the analyze endpoint and operational routes.

These exercise the HTTP layer with the Detoxify model and Redis cache stubbed out
(see conftest.py). The fake classifier returns the fixed ``FAKE_SCORES`` mapping.
"""

from app.models import ALL_ATTRIBUTES
from app.tests.conftest import FAKE_SCORES

ANALYZE_URL = "/v1alpha1/comments:analyze"


def test_returns_all_attributes_by_default(client):
    """When requestedAttributes is omitted, all six attributes are returned."""
    resp = client.post(ANALYZE_URL, json={"comment": {"text": "you are an idiot"}})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["attributeScores"].keys()) == set(ALL_ATTRIBUTES)


def test_response_matches_perspective_schema(client):
    """Each attribute carries a summaryScore with a value and PROBABILITY type."""
    resp = client.post(ANALYZE_URL, json={"comment": {"text": "hello"}})
    body = resp.json()
    toxicity = body["attributeScores"]["TOXICITY"]["summaryScore"]
    assert toxicity["type"] == "PROBABILITY"
    # Maps Detoxify 'toxicity' (0.92) -> Perspective TOXICITY.
    assert toxicity["value"] == FAKE_SCORES["toxicity"]


def test_requested_attributes_filtering(client):
    """Only requested attributes appear in the response; others are dropped."""
    resp = client.post(
        ANALYZE_URL,
        json={
            "comment": {"text": "you are an idiot"},
            "requestedAttributes": {"TOXICITY": {}, "INSULT": {}},
        },
    )
    body = resp.json()
    assert set(body["attributeScores"].keys()) == {"TOXICITY", "INSULT"}


def test_unknown_requested_attribute_ignored(client):
    """Unknown attribute names are ignored rather than erroring."""
    resp = client.post(
        ANALYZE_URL,
        json={
            "comment": {"text": "hi"},
            "requestedAttributes": {"TOXICITY": {}, "NOT_A_REAL_ATTR": {}},
        },
    )
    assert resp.status_code == 200
    assert set(resp.json()["attributeScores"].keys()) == {"TOXICITY"}


def test_languages_passthrough(client):
    """Provided languages are echoed into languages and detectedLanguages."""
    resp = client.post(
        ANALYZE_URL,
        json={"comment": {"text": "hola"}, "languages": ["es"]},
    )
    body = resp.json()
    assert body["languages"] == ["es"]
    assert body["detectedLanguages"] == ["es"]


def test_detected_languages_unknown_when_absent(client):
    """When languages is omitted, detectedLanguages falls back to ['unknown']."""
    resp = client.post(ANALYZE_URL, json={"comment": {"text": "hi"}})
    body = resp.json()
    assert body["languages"] == []
    assert body["detectedLanguages"] == ["unknown"]


def test_client_token_passthrough(client):
    """clientToken is echoed back unchanged."""
    resp = client.post(
        ANALYZE_URL,
        json={"comment": {"text": "hi"}, "clientToken": "abc-123"},
    )
    assert resp.json()["clientToken"] == "abc-123"


def test_inference_failure_returns_structured_500(client, monkeypatch):
    """A classifier error surfaces as HTTP 500 with a structured error body."""
    from app.services import classifier

    async def _boom(text: str):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(classifier, "predict", _boom)

    resp = client.post(ANALYZE_URL, json={"comment": {"text": "boom"}})
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "inference_failed"
    assert "model exploded" in body["detail"]


def test_healthz(client):
    """Health endpoint reports ok and the model name."""
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": "multilingual"}


def test_list_models(client):
    """Models endpoint lists variants and the loaded model."""
    resp = client.get("/v1/models")
    body = resp.json()
    assert "multilingual" in body["models"]
    assert body["loaded"] == "multilingual"
