"""Deterministic, tiered profanity lexicon for short-field moderation.

The Detoxify model is weak on short identifiers (usernames, slugs, build names),
especially in FR/ES/PT. This module is the *primary* detector for those: it
normalizes (reusing the service normalizer), segments into tokens, and matches a
curated multilingual lexicon split into ``block`` / ``flag`` tiers, with an
``allow`` list that suppresses domain vocab and known false positives.

The lexicon lives in editable text files (``app/data/lexicon/`` by default, or
``OPENSPECTIVE_LEXICON_DIR``) so it can grow via the human review loop without code
changes — "static" here means deterministic, not frozen.

Matching is whole-token by default (precise: ``reputation`` does not match ``pute``).
A trailing ``*`` on a term opts into substring matching to catch concatenations
(``fuck*`` matches ``fuckankama``); use it only for stems that never appear inside
innocent words.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.services.normalizer import normalize

logger = logging.getLogger("openspective.lexicon")

# Score assigned to a lexicon hit per tier, so /moderate can fold lexicon + ML into a
# single number. A clear slur is a certainty (1.0); a soft/borderline term sits in the
# review band.
_BLOCK_SCORE = 1.0
_FLAG_SCORE = 0.6


def _fold_accents(text: str) -> str:
    """Strip diacritics so accented profanity matches its de-accented lexicon entry.

    ``pédé`` -> ``pede``, ``enculé`` -> ``encule``. Applied to both the lexicon terms
    and the input so the two always meet. NFKD splits a letter from its combining
    accent; we drop the combining marks.
    """
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))

# Default lexicon directory (this file's sibling ``../data/lexicon``).
_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "lexicon"

# Split normalized text into tokens: break on non-alphanumerics and on letter/digit
# boundaries so ``zob200`` -> {zob, 200} and ``x_killer`` -> {x, killer}.
_SPLIT_RE = re.compile(r"[^a-z0-9]+|(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])")

# Terms shorter than this are always matched as whole tokens, never substrings,
# regardless of a ``*`` suffix — short substrings are false-positive magnets.
_MIN_SUBSTRING_LEN = 4


@dataclass(frozen=True)
class Lexicon:
    """A loaded lexicon: token/substring terms per tier plus an allowlist.

    The ``swear`` tier holds generic profanity (``fuck``, ``shit``, ``putain``) that
    blocks only in *strict* (identity) contexts like usernames; in free text it is left
    to the ML, which tells colloquial use (``it's fucked up`` ~0.1) from targeted abuse
    (``fuck you`` ~0.95). Slurs stay in ``block`` and fire everywhere.
    """

    block_tokens: frozenset[str]
    block_substrings: frozenset[str]
    swear_tokens: frozenset[str]
    swear_substrings: frozenset[str]
    flag_tokens: frozenset[str]
    flag_substrings: frozenset[str]
    allow: frozenset[str]


@dataclass
class Verdict:
    """Result of evaluating a single piece of text against the lexicon."""

    decision: str = "allow"  # block | flag | allow
    tier: str | None = None  # which tier fired (block/flag), or None
    reasons: list[str] = field(default_factory=list)  # matched terms
    score: float = 0.0  # severity in [0, 1] (block tier = 1.0, flag tier = 0.6)


def tokenize(normalized: str) -> set[str]:
    """Split already-normalized text into a set of alphanumeric tokens."""
    return {t for t in _SPLIT_RE.split(normalized) if t}


def _parse_file(path: Path) -> tuple[set[str], set[str]]:
    """Parse a tier file into (token_terms, substring_terms).

    Lines are lowercased; blank lines and ``#`` comments are ignored. A trailing
    ``*`` marks a substring term (unless the stem is too short to be safe).
    """
    tokens: set[str] = set()
    substrings: set[str] = set()
    if not path.exists():
        return tokens, substrings
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = _fold_accents(raw.strip().lower())
        if not line or line.startswith("#"):
            continue
        if line.endswith("*"):
            stem = line[:-1]
            if len(stem) >= _MIN_SUBSTRING_LEN:
                substrings.add(stem)
            else:
                tokens.add(stem)  # too short to substring-match safely
        else:
            tokens.add(line)
    return tokens, substrings


@lru_cache(maxsize=1)
def get_lexicon() -> Lexicon:
    """Load and cache the lexicon from the configured directory."""
    settings = get_settings()
    directory = Path(settings.lexicon_dir) if settings.lexicon_dir else _DEFAULT_DIR
    block_t, block_s = _parse_file(directory / "block.txt")
    swear_t, swear_s = _parse_file(directory / "swear.txt")
    flag_t, flag_s = _parse_file(directory / "flag.txt")
    allow_t, allow_s = _parse_file(directory / "allow.txt")
    lex = Lexicon(
        block_tokens=frozenset(block_t),
        block_substrings=frozenset(block_s),
        swear_tokens=frozenset(swear_t),
        swear_substrings=frozenset(swear_s),
        flag_tokens=frozenset(flag_t),
        flag_substrings=frozenset(flag_s),
        allow=frozenset(allow_t | allow_s),
    )
    logger.info(
        "Loaded lexicon from %s: block=%d flag=%d allow=%d",
        directory,
        len(lex.block_tokens) + len(lex.block_substrings),
        len(lex.flag_tokens) + len(lex.flag_substrings),
        len(lex.allow),
    )
    return lex


def _suppressed(stem: str, tokens: set[str], allow: frozenset[str]) -> bool:
    """True if ``stem`` only appears inside allowlisted tokens (so it's a false hit).

    A substring stem found solely within allowlisted words (e.g. a stem inside a
    Wakfu class name) is suppressed; if it also appears in a non-allowlisted token,
    it stands.
    """
    containing = [t for t in tokens if stem in t]
    if not containing:
        return False  # spans a token boundary; can't attribute to an allow word
    return all(t in allow for t in containing)


def _matches(tokens: set[str], norm: str, term_tokens: frozenset[str],
             term_substrings: frozenset[str], allow: frozenset[str]) -> list[str]:
    """Return the terms from one tier that match, after allowlist suppression."""
    hits: list[str] = []
    for term in term_tokens:
        if term in tokens and term not in allow:
            hits.append(term)
    for term in term_substrings:
        if term in norm and not _suppressed(term, tokens, allow):
            hits.append(term)
    return hits


def evaluate(text: str, strict: bool = False) -> Verdict:
    """Evaluate ``text`` against the lexicon and return a :class:`Verdict`.

    Block matches take precedence over flag matches. Returns an ``allow`` verdict
    when nothing matches.

    :param text: Raw text (normalization happens here).
    :param strict: When ``True`` (identity fields like usernames), generic swears block
        too; when ``False`` (free text) swears are left to the ML. Slurs block either way.
    :returns: A :class:`Verdict` with decision, tier, and matched terms.
    """
    lex = get_lexicon()
    # Fold accents so 'pédé' matches the de-accented 'pede' entry (and vice versa).
    norm = _fold_accents(normalize(text))
    tokens = tokenize(norm)

    block_hits = _matches(tokens, norm, lex.block_tokens, lex.block_substrings, lex.allow)
    if strict:
        block_hits += _matches(tokens, norm, lex.swear_tokens, lex.swear_substrings, lex.allow)
    if block_hits:
        return Verdict(
            decision="block", tier="block", reasons=sorted(set(block_hits)), score=_BLOCK_SCORE
        )

    flag_hits = _matches(tokens, norm, lex.flag_tokens, lex.flag_substrings, lex.allow)
    if flag_hits:
        return Verdict(
            decision="flag", tier="flag", reasons=sorted(set(flag_hits)), score=_FLAG_SCORE
        )

    return Verdict(decision="allow", score=0.0)
