"""Unit tests for the language detector."""

from app.services.detector import detect_language


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
