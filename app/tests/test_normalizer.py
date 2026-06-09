"""Unit tests for the text normaliser."""

from app.services.normalizer import normalize


def test_clean_text_only_lowercased():
    """Clean ASCII text should pass through unchanged except for casing."""
    assert normalize("Hello World") == "hello world"


def test_leet_speak_expanded():
    """Leet substitutions are expanded to their letters."""
    # l33t -> leet, 5pe4k -> speak (5->s, 4->a), ! -> i
    assert normalize("l33t 5pe4k") == "leet speak"
    assert normalize("h!") == "hi"


def test_cyrillic_homoglyphs_folded_to_ascii():
    """Cyrillic lookalikes are folded to ASCII before classification."""
    # "іdіоt" mixes Cyrillic і/о with ASCII; should become "idiot".
    assert normalize("іdіоt") == "idiot"


def test_excessive_repetition_collapsed():
    """Runs of 3+ identical characters collapse to exactly two."""
    assert normalize("aaaaa") == "aa"
    assert normalize("noooo waaay") == "noo waay"


def test_mixed_scripts_and_substitutions():
    """A combination of homoglyphs, leet and symbols resolves to plain text."""
    # @ -> a, $ -> s, 0 -> o, 1 -> i
    assert normalize("@$$h0le") == "asshole"
    assert normalize("1d10t") == "idiot"


def test_punctuation_heavy_string_preserved():
    """Punctuation that is not a known substitution is left intact (lowercased)."""
    # ',' '?' '.' are not substitutions and there are no 3+ runs, so only casing changes.
    assert normalize("Hey, really? Yes.") == "hey, really? yes."


def test_leet_bang_and_repetition_interact():
    """'!' is a leet substitution, so '!!!' -> 'iii' -> collapsed to 'ii'."""
    # leet: '!' -> 'i' gives "wowiii", then repetition collapses "iii" -> "ii".
    assert normalize("Wow!!!") == "wowii"


def test_empty_string():
    """An empty string normalises to an empty string."""
    assert normalize("") == ""
