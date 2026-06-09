"""Sentence/span segmentation for per-span scoring.

Detoxify scores a whole string; Perspective can additionally return per-span
scores. We approximate spans by splitting the comment into sentence-like chunks
and reporting each chunk's character offsets into the *original* text.

This is a deliberately dependency-free splitter (no nltk/spaCy): it breaks on
``.``/``!``/``?`` runs. It is good enough for highlighting toxic sentences; it is
not a linguistically perfect sentence tokenizer (abbreviations, decimals, etc.).
"""

import re

# Matches a run of text up to and including a terminating .!? group, or the final
# trailing chunk with no terminator. DOTALL so sentences can span newlines.
_CHUNK_RE = re.compile(r".*?(?:[.!?]+|\Z)", flags=re.DOTALL)


def split_spans(text: str) -> list[tuple[str, int, int]]:
    """Split ``text`` into sentence-like spans with character offsets.

    :param text: The original (un-normalised) comment text.
    :returns: List of ``(segment, begin, end)`` where ``begin``/``end`` are offsets
        into ``text`` and ``segment`` is the whitespace-trimmed span text. Empty /
        whitespace-only spans are skipped.
    """
    spans: list[tuple[str, int, int]] = []
    for match in _CHUNK_RE.finditer(text):
        raw = match.group()
        if not raw.strip():
            continue
        # Trim surrounding whitespace while keeping begin/end aligned to the trim.
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        begin = match.start() + leading
        end = match.end() - trailing
        spans.append((raw.strip(), begin, end))
    return spans
