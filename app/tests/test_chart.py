"""Tests for the SVG chart endpoint and renderer."""

from app.services.chart import render_svg
from app.tests.conftest import TEST_TOKEN


def test_render_svg_contains_gauges_and_percentages():
    """The renderer emits an SVG with a gauge label and percentage per attribute."""
    svg = render_svg("hello", {"TOXICITY": 0.92, "THREAT": 0.04}, language="en")
    assert svg.startswith("<svg")
    assert "TOXICITY" in svg and "THREAT" in svg
    assert "92%" in svg and "4%" in svg
    assert "detected language: en" in svg


def test_chart_endpoint_returns_svg(client):
    """GET /chart returns an SVG image with the requested attribute gauges."""
    resp = client.get("/chart", params={"text": "you are an idiot"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/svg+xml")
    assert resp.text.startswith("<svg")
    # FAKE toxicity 0.92 -> 92% gauge present.
    assert "92%" in resp.text


def test_chart_endpoint_respects_attributes_filter(client):
    """The attributes query param limits which gauges are drawn."""
    resp = client.get("/chart", params={"text": "hi", "attributes": "TOXICITY,INSULT"})
    body = resp.text
    assert "TOXICITY" in body and "INSULT" in body
    assert "THREAT" not in body and "OBSCENE" not in body


def test_render_radar_is_single_svg_with_all_labels():
    """The radar style emits one polygon over all attributes with their labels."""
    svg = render_svg("hi", {"TOXICITY": 0.9, "INSULT": 0.5, "THREAT": 0.1}, style="radar")
    assert svg.startswith("<svg")
    assert "<polygon" in svg  # the data polygon
    for name in ("TOXICITY", "INSULT", "THREAT"):
        assert name in svg


def test_render_polar_uses_wedges():
    """The polar style emits one wedge (path) per attribute."""
    svg = render_svg("hi", {"TOXICITY": 0.9, "INSULT": 0.5}, style="polar")
    assert svg.count("<path") == 2


def test_chart_endpoint_radar_style(client):
    """GET /chart?style=radar returns a radar SVG."""
    resp = client.get("/chart", params={"text": "you are an idiot", "style": "radar"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/svg+xml")
    assert "<polygon" in resp.text


def test_chart_endpoint_unknown_style_falls_back_to_gauges(client):
    """An unknown style falls back to gauges rather than erroring."""
    resp = client.get("/chart", params={"text": "hi", "style": "bogus"})
    assert resp.status_code == 200
    assert "92%" in resp.text  # gauge percentage text


def test_chart_endpoint_requires_auth_when_enabled(auth_client):
    """When auth is enabled, /chart needs a valid Bearer token (it runs inference)."""
    unauth = auth_client.get("/chart", params={"text": "hi"})
    assert unauth.status_code == 401

    ok = auth_client.get(
        "/chart", params={"text": "hi"}, headers={"Authorization": f"Bearer {TEST_TOKEN}"}
    )
    assert ok.status_code == 200
    assert ok.text.startswith("<svg")
