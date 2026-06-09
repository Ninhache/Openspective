"""Token-aware chunking for long comments.

The toxicity models have a fixed input window (512 tokens for XLM-RoBERTa).
Detoxify truncates anything beyond it *silently*, so a toxic passage past the
window is never scored. To stay correct on long text we split it into chunks that
each fit the window, score every chunk, and pool the results (max) upstream.

Chunks are sentence-granular (reusing the span splitter) because a single toxic
sentence diluted inside a large token window scores far lower than that sentence
on its own — see ``scripts/experiment_chunking.py``. A lone run-on sentence that
itself exceeds the window is hard-split on word boundaries as a last resort so
nothing is ever silently dropped.

This module is intentionally model-agnostic: the caller injects a ``count_tokens``
function, which keeps it unit-testable without loading torch.
"""

from collections.abc import Callable

from app.services.spans import split_spans

# Usable window in tokens, leaving room for the 2 special tokens (<s>/</s>) the
# tokenizer adds. Detoxify's models use a 512-token maximum.
MODEL_WINDOW = 510


def _hard_split(sentence: str, count_tokens: Callable[[str], int], window: int) -> list[str]:
    """Split a single over-window sentence into <=``window``-token pieces by words.

    Only triggered for pathological run-on text with no sentence punctuation; the
    input-size guard bounds how large this can get.
    """
    pieces: list[str] = []
    current: list[str] = []
    for word in sentence.split():
        current.append(word)
        if count_tokens(" ".join(current)) > window:
            current.pop()
            if current:
                pieces.append(" ".join(current))
            current = [word]
    if current:
        pieces.append(" ".join(current))
    return pieces or [sentence]


def chunk_text(
    text: str,
    count_tokens: Callable[[str], int],
    window: int = MODEL_WINDOW,
) -> list[str]:
    """Split ``text`` into per-sentence chunks for pooled scoring.

    Fast path: when the whole text already fits the window, returns ``[text]``
    unchanged so short comments incur no extra work (a single inference pass).

    When the text exceeds the window it is split into individual *sentences* — not
    packed into larger window-sized groups. Packing was measured to re-dilute a lone
    toxic sentence among many benign ones (see ``scripts/experiment_chunking.py``:
    sentence-granular ``max`` recovers ~0.99 where window-packed ``max`` only reaches
    ~0.18). One run-on sentence that itself exceeds the window is hard-split on words.

    :param text: The text to chunk (already normalised by the caller).
    :param count_tokens: Returns the token count of a string (model tokenizer).
    :param window: Maximum tokens per chunk.
    :returns: Non-empty list of chunk strings, each <= ``window`` tokens.
    """
    if count_tokens(text) <= window:
        return [text]

    chunks: list[str] = []
    for segment, _begin, _end in split_spans(text):
        if count_tokens(segment) > window:
            chunks.extend(_hard_split(segment, count_tokens, window))
        else:
            chunks.append(segment)
    return chunks or [text]
