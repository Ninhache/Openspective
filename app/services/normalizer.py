"""Text pre-processing applied before inference.

The normaliser defends against common evasion tricks (homoglyphs, leet speak,
character flooding) so that, e.g., ``"іd!07"``-style obfuscations still reach the
classifier as recognisable text. Steps are applied in a fixed order:

1. Unicode homoglyph normalisation (lookalike chars -> ASCII).
2. Leet-speak expansion (``3->e``, ``4->a`` ...).
3. Strip excessive repetition (max 2 consecutive identical chars).
4. Lowercase.

The mappings are intentionally hardcoded (no external dependency) so behaviour is
deterministic and auditable.
"""

import re

# 1. Homoglyph map: visually similar non-ASCII characters -> ASCII equivalents.
# Covers the most common Cyrillic and Greek lookalikes plus a couple of symbols.
# Note: applied before lowercasing, so include both cases where they differ.
HOMOGLYPHS: dict[str, str] = {
    # Cyrillic lowercase
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ѕ": "s", "ј": "j", "к": "k", "н": "h", "т": "t", "в": "b", "м": "m",
    # Cyrillic uppercase
    "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "У": "Y", "Х": "X",
    "І": "I", "К": "K", "Н": "H", "Т": "T", "В": "B", "М": "M",
    # Greek
    "α": "a", "ο": "o", "ι": "i", "ε": "e", "ρ": "p", "υ": "u", "τ": "t",
    "Α": "A", "Ο": "O", "Ι": "I", "Ε": "E", "Ρ": "P", "Τ": "T",
    # Symbol substitutions commonly used to spell words
    "@": "a", "$": "s",
}

# 2. Leet-speak substitutions. ``0->o`` and ``1->i`` are the conventional choices
# for spelling out words; they intentionally win over the homoglyph table above.
LEET: dict[str, str] = {
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "!": "i",
    "0": "o",
    "1": "i",
}

# 3. Collapse any run of 3+ identical characters down to exactly 2.
_REPEAT_RE = re.compile(r"(.)\1{2,}", flags=re.DOTALL)

# Translation tables built once at import time for speed.
_HOMOGLYPH_TABLE = str.maketrans(HOMOGLYPHS)
_LEET_TABLE = str.maketrans(LEET)


def normalize(text: str) -> str:
    """Normalise ``text`` for classification.

    :param text: Raw user-supplied comment text.
    :returns: The normalised text (homoglyph-folded, de-leeted, de-flooded, lowercased).
    """
    # 1. Homoglyphs -> ASCII.
    text = text.translate(_HOMOGLYPH_TABLE)
    # 2. Leet expansion.
    text = text.translate(_LEET_TABLE)
    # 3. Strip excessive repetition: "aaaaa" -> "aa".
    text = _REPEAT_RE.sub(r"\1\1", text)
    # 4. Lowercase.
    return text.lower()
