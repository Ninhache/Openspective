"""Tests for dev mode (permissive CORS + always-on docs)."""


def test_no_cors_headers_when_dev_mode_off(client):
    """Without dev mode, responses carry no permissive CORS header."""
    resp = client.get("/healthz", headers={"Origin": "http://localhost:5173"})
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers


def test_cors_headers_present_in_dev_mode(dev_client):
    """In dev mode, responses include permissive CORS headers."""
    resp = dev_client.get("/healthz", headers={"Origin": "http://localhost:5173"})
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "*"


def test_cors_preflight_short_circuited_in_dev_mode(dev_client):
    """An OPTIONS preflight is answered with 204 + CORS headers in dev mode."""
    resp = dev_client.options(
        "/v1alpha1/comments:analyze",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.status_code == 204
    assert resp.headers["access-control-allow-origin"] == "*"
    assert resp.headers["access-control-allow-methods"] == "*"


def test_docs_available(client):
    """The OpenAPI schema (and thus Swagger docs) are reachable."""
    assert client.get("/openapi.json").status_code == 200
