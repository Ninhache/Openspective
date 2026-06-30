"""Unit tests for the language detector."""

import pytest

from app.services import detector
from app.services.detector import detect_language


@pytest.fixture(autouse=True)
def _force_langdetect(monkeypatch):
    """Pin the langdetect fallback so these tests don't depend on a cached fastText model."""
    monkeypatch.setattr(detector, "_attempted", True)
    monkeypatch.setattr(detector, "_model", None)


def test_detects_english():
    assert detect_language("this is a clearly english sentence") == ["en"]


def test_detects_french():
    assert detect_language("ceci est clairement une phrase en français") == ["fr"]


def test_empty_string_is_unknown():
    assert detect_language("") == ["unknown"]


def test_whitespace_only_is_unknown():
    assert detect_language("   \n\t ") == ["unknown"]


def test_no_alphabetic_content_is_unknown():
    assert detect_language("12345 !!!") == ["unknown"]


def test_fasttext_detects_spanish_when_available():
    """fastText tags Spanish correctly (langdetect mislabels it Italian). Skipped if absent."""
    fasttext = pytest.importorskip("fasttext")
    if not detector._model_path().exists():
        pytest.skip("fastText lid.176 model not available")
    model = fasttext.load_model(str(detector._model_path()))
    assert detector._detect_fasttext(model, "eres un puto gilipollas") == ["es"]


def test_fasttext_low_confidence_is_unknown(monkeypatch):
    """A bare name has no real language → below the confidence floor → 'unknown'."""
    fasttext = pytest.importorskip("fasttext")
    if not detector._model_path().exists():
        pytest.skip("fastText lid.176 model not available")
    model = fasttext.load_model(str(detector._model_path()))
    monkeypatch.setenv("OPENSPECTIVE_LANGID_MIN_CONFIDENCE", "0.9")
    from app.config import get_settings
    get_settings.cache_clear()
    assert detector._detect_fasttext(model, "Mathius") == ["unknown"]
    get_settings.cache_clear()
