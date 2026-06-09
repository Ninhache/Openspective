"""Tests for the observability endpoints (/metrics, /readyz)."""

ANALYZE_URL = "/v1alpha1/comments:analyze"


def _metric_value(metrics_text: str, name: str) -> float:
    """Extract a single (unlabelled) counter value from Prometheus exposition text."""
    for line in metrics_text.splitlines():
        if line.startswith(name + " "):
            return float(line.split(" ", 1)[1])
    return 0.0


def test_metrics_endpoint_exposes_known_series(client):
    """The /metrics endpoint returns Prometheus exposition with our series names."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "openspective_requests_total" in body
    assert "openspective_cache_misses_total" in body
    assert "openspective_inference_duration_seconds" in body


def test_cache_miss_counter_increments_on_analyze(client):
    """A cache-miss request increments the cache_misses counter."""
    before = _metric_value(client.get("/metrics").text, "openspective_cache_misses_total")
    client.post(ANALYZE_URL, json={"comment": {"text": "increment me"}})
    after = _metric_value(client.get("/metrics").text, "openspective_cache_misses_total")
    assert after == before + 1


def test_readyz_returns_503_when_model_not_loaded(client):
    """Readiness is not OK until the model has loaded (model stays unloaded in tests)."""
    resp = client.get("/readyz")
    assert resp.status_code == 503
    assert resp.json()["status"] == "not_ready"


def test_readyz_returns_200_when_model_loaded(client, monkeypatch):
    """Readiness reports ready once the model is loaded."""
    from app.services import classifier

    monkeypatch.setattr(classifier, "is_loaded", lambda: True)
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
