"""Tests for span segmentation and span/threshold scoring."""

from app.services.spans import split_spans
from app.tests.conftest import FAKE_SCORES

ANALYZE_URL = "/v1alpha1/comments:analyze"


# --- split_spans unit tests --------------------------------------------------


def test_split_spans_multiple_sentences():
    text = "Hello world. You idiot!"
    spans = split_spans(text)
    assert spans == [("Hello world.", 0, 12), ("You idiot!", 13, 23)]
    # Offsets index back into the original text.
    for segment, begin, end in spans:
        assert text[begin:end] == segment


def test_split_spans_no_terminator():
    assert split_spans("no punctuation here") == [("no punctuation here", 0, 19)]


def test_split_spans_empty():
    assert split_spans("") == []
    assert split_spans("   ") == []


# --- HTTP span scoring -------------------------------------------------------


def test_no_span_scores_by_default(client):
    """Without spanAnnotations, attribute scores carry no spanScores key."""
    resp = client.post(ANALYZE_URL, json={"comment": {"text": "you are an idiot. really."}})
    body = resp.json()
    assert "spanScores" not in body["attributeScores"]["TOXICITY"]


def test_span_annotations_returns_span_scores(client):
    """With spanAnnotations, each requested attribute gets one span per sentence."""
    resp = client.post(
        ANALYZE_URL,
        json={
            "comment": {"text": "you are an idiot. really."},
            "requestedAttributes": {"TOXICITY": {}},
            "spanAnnotations": True,
        },
    )
    body = resp.json()
    spans = body["attributeScores"]["TOXICITY"]["spanScores"]
    assert len(spans) == 2  # two sentences
    first = spans[0]
    assert first["begin"] == 0
    assert first["score"]["value"] == FAKE_SCORES["toxicity"]
    assert first["score"]["type"] == "PROBABILITY"


def test_score_threshold_filters_spans(client):
    """A scoreThreshold above the span score drops the span; below keeps it."""
    base = {"comment": {"text": "one. two. three."}, "spanAnnotations": True}

    # FAKE toxicity is 0.92; a 0.95 threshold filters everything out.
    high = client.post(
        ANALYZE_URL, json={**base, "requestedAttributes": {"TOXICITY": {"scoreThreshold": 0.95}}}
    )
    assert high.json()["attributeScores"]["TOXICITY"]["spanScores"] == []

    # A 0.5 threshold keeps all three spans.
    low = client.post(
        ANALYZE_URL, json={**base, "requestedAttributes": {"TOXICITY": {"scoreThreshold": 0.5}}}
    )
    assert len(low.json()["attributeScores"]["TOXICITY"]["spanScores"]) == 3
