"""Tests for optional Bearer-token authentication."""

from app.tests.conftest import TEST_TOKEN

ANALYZE_URL = "/v1alpha1/comments:analyze"
PAYLOAD = {"comment": {"text": "hello"}}


def test_auth_disabled_by_default(client):
    """With no tokens configured, the analyze endpoint needs no Authorization header."""
    resp = client.post(ANALYZE_URL, json=PAYLOAD)
    assert resp.status_code == 200


def test_valid_token_allowed(auth_client):
    """A correct Bearer token is accepted when auth is enabled."""
    resp = auth_client.post(
        ANALYZE_URL, json=PAYLOAD, headers={"Authorization": f"Bearer {TEST_TOKEN}"}
    )
    assert resp.status_code == 200


def test_missing_header_rejected(auth_client):
    """A missing Authorization header is rejected with a structured 401."""
    resp = auth_client.post(ANALYZE_URL, json=PAYLOAD)
    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthorized"
    assert resp.headers["WWW-Authenticate"] == "Bearer"


def test_wrong_token_rejected(auth_client):
    """An incorrect token is rejected with 401."""
    resp = auth_client.post(
        ANALYZE_URL, json=PAYLOAD, headers={"Authorization": "Bearer not-the-token"}
    )
    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthorized"


def test_malformed_header_rejected(auth_client):
    """A non-Bearer Authorization header is rejected with 401."""
    resp = auth_client.post(
        ANALYZE_URL, json=PAYLOAD, headers={"Authorization": TEST_TOKEN}
    )
    assert resp.status_code == 401


def test_operational_endpoints_stay_open(auth_client):
    """Health/readiness/metrics remain unauthenticated even with auth enabled."""
    assert auth_client.get("/healthz").status_code == 200
    assert auth_client.get("/metrics").status_code == 200
    # /readyz returns 503 (model not loaded in tests) but is reachable without a token.
    assert auth_client.get("/readyz").status_code in (200, 503)
