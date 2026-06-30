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


def test_slur_blocks_in_any_context():
    """Slurs / person-directed insults block regardless of strict."""
    assert lexicon.evaluate("espece de salope").decision == "block"
    assert lexicon.evaluate("puta madre").decision == "block"
    assert lexicon.evaluate("seu viado").decision == "block"


def test_swear_blocks_only_in_strict_context():
    """Generic swears (fuck) block in strict (username) but not in free text."""
    assert lexicon.evaluate("fuckankama", strict=True).decision == "block"
    assert lexicon.evaluate("fuckankama").decision == "allow"  # lenient -> left to the ML


def test_swear_substring_catches_concatenation_in_strict():
    v = lexicon.evaluate("fuckankama", strict=True)
    assert v.decision == "block"
    assert "fuck" in v.reasons


def test_flag_tier_is_soft():
    """A borderline flag-tier term yields 'flag', not block."""
    v = lexicon.evaluate("sale tamere")
    assert v.decision == "flag"
    assert v.tier == "flag"


def test_colloquial_swear_allowed_in_free_text():
    """'shit'/'bullshit' in free text are left to the ML (lenient), not lexicon-blocked."""
    assert lexicon.evaluate("Ozzy bullshit build").decision == "allow"
    assert lexicon.evaluate("this build is shit").decision == "allow"


def test_allowlist_domain_vocab_not_flagged():
    """'zob' is the Zobal class in Wakfu, not profanity — allowlisted."""
    assert lexicon.evaluate("zob heal 200").decision == "allow"


def test_accent_folding_matches_de_accented_entry():
    """Accented profanity matches its de-accented lexicon entry (salopé -> salope)."""
    assert lexicon.evaluate("salopé").decision == "block"
    assert lexicon.evaluate("sale salopé").decision == "block"


def test_score_is_set_per_tier():
    """Block hits score 1.0, flag hits 0.6, clean text 0.0."""
    assert lexicon.evaluate("salope").score == 1.0
    assert lexicon.evaluate("fuck", strict=True).score == 1.0
    assert lexicon.evaluate("tamere").score == 0.6
    assert lexicon.evaluate("Mathius").score == 0.0


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
