"""Unit tests for token-aware chunking.

The chunker is model-agnostic: a fake ``count_tokens`` (word count) stands in for
the real tokenizer so these run without torch.
"""

from app.services.chunking import chunk_text


def _word_count(text: str) -> int:
    """Stand-in tokenizer: one 'token' per whitespace-separated word."""
    return len(text.split())


def test_short_text_returns_single_chunk_unchanged():
    """Text within the window is returned as-is (fast path, no splitting)."""
    text = "you are an idiot."
    chunks = chunk_text(text, _word_count, window=10)
    assert chunks == [text]


def test_long_text_splits_into_window_sized_chunks():
    """Text over the window is split into multiple chunks, each within the window."""
    text = "a b c d. e f g h. i j k l. m n o p."  # four 4-word sentences
    chunks = chunk_text(text, _word_count, window=6)
    assert len(chunks) > 1
    assert all(_word_count(c) <= 6 for c in chunks)


def test_long_text_split_per_sentence_not_packed():
    """Over-window text is split one sentence per chunk (no greedy packing).

    Packing re-dilutes a lone toxic sentence, so each sentence is scored on its own.
    """
    text = "a b. c d. e f. g h."  # four 2-word sentences, total 8 > window
    chunks = chunk_text(text, _word_count, window=4)
    assert len(chunks) == 4
    assert all(_word_count(c) <= 2 for c in chunks)


def test_oversize_single_sentence_is_hard_split():
    """A lone run-on sentence larger than the window is split on word boundaries."""
    text = "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10"  # no sentence punctuation, 10 words
    chunks = chunk_text(text, _word_count, window=4)
    assert len(chunks) >= 3
    assert all(_word_count(c) <= 4 for c in chunks)
    # Nothing is dropped: every word survives across the chunks.
    assert " ".join(chunks).split() == text.split()


def test_empty_text_returns_single_chunk():
    """Empty/whitespace text degrades to a single (empty) chunk rather than []."""
    assert chunk_text("", _word_count, window=4) == [""]
