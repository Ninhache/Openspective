"""Unit tests for the tiered moderation lexicon.

All test strings are synthetic — no real user data is committed. Tests run against
the bundled default lexicon (app/data/lexicon).
"""

import pytest

from app.services import lexicon


@pytest.fixture(autouse=True)
def _fresh_lexicon():
    """Clear the cached lexicon so each test loads the bundled default cleanly."""
    lexicon.get_lexicon.cache_clear()
    yield
    lexicon.get_lexicon.cache_clear()


def test_clean_text_is_allowed():
    assert lexicon.evaluate("Mathius the brave").decision == "allow"


def test_block_substring_catches_concatenation():
    """A '*' term (fuck) matches even when glued to other text."""
    v = lexicon.evaluate("fuckankama")
    assert v.decision == "block"
    assert "fuck" in v.reasons


def test_block_token_match():
    """A whole-token block term (salope) fires as its own word."""
    assert lexicon.evaluate("espece de salope").decision == "block"


def test_block_spanish_and_portuguese():
    assert lexicon.evaluate("puta madre").decision == "block"
    assert lexicon.evaluate("vai tomar no caralho").decision == "block"


def test_flag_tier_is_soft():
    """'shit' is a flag (soft), not a block."""
    v = lexicon.evaluate("this is shit")
    assert v.decision == "flag"
    assert v.tier == "flag"


def test_bullshit_not_over_flagged():
    """'shit' is token-only, so legit 'bullshit' build names are not flagged."""
    assert lexicon.evaluate("Ozzy bullshit build").decision == "allow"


def test_allowlist_domain_vocab_not_flagged():
    """'zob' is the Zobal class in Wakfu, not profanity — allowlisted."""
    assert lexicon.evaluate("zob heal 200").decision == "allow"


def test_token_matching_avoids_substring_false_positives():
    """Whole-token default means 'pute' does not fire inside 'reputation'/'dispute'."""
    assert lexicon.evaluate("ma reputation est bonne").decision == "allow"
    assert lexicon.evaluate("une longue dispute").decision == "allow"


def test_spanish_con_is_not_profanity():
    """'con' (Spanish 'with') was a false positive in review — not in the lexicon."""
    assert lexicon.evaluate("latita con botas").decision == "allow"


def test_digit_and_separator_segmentation():
    """zob200 / x_killer split into tokens so allowlist + tokens still apply."""
    assert lexicon.evaluate("zob200").decision == "allow"
    assert lexicon.evaluate("x_killer_pro").decision == "allow"


def test_suppressed_helper_only_inside_allow_tokens():
    allow = frozenset({"class"})
    # 'ass' appears only inside the allowlisted 'class' -> suppressed
    assert lexicon._suppressed("ass", {"class"}, allow) is True
    # also present in a standalone bad token -> not suppressed
    assert lexicon._suppressed("ass", {"class", "ass"}, allow) is False
