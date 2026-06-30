"""Build a tiered openspective lexicon from a Wakfu censor dump + LDNOOBW.

This is generic code (MIT). It reads a game censorship JSON (Ankama's
``NN.named.json`` format) plus the LDNOOBW multilingual profanity lists and emits
``block.txt`` / ``flag.txt`` / ``allow.txt`` into an output directory that you then
point ``OPENSPECTIVE_LEXICON_DIR`` at. The game data stays local — only this
transform script lives in the repo.

Curation rules (learned empirically against real Wakfuli data):
  * ``m_censorType`` 0/1 = chat-forbidden profanity -> block.
  * ``m_censorType`` 2  = forbidden-as-name. It mixes game vocab (``support``,
    ``astrub``, class names) with slurs (``bamboula``). Metadata can't separate them,
    so we cross-reference against LDNOOBW (+ a hate/extremism set): a type-2 word that
    is also in LDNOOBW is a real slur -> block; the rest is game vocab -> not blocked.
  * ``m_language`` 7 is Ankama's spam/URL/RMT bucket (TLDs, "test", numbers) -> drop.
  * ``m_deepSearch`` true + len>=5 -> substring ('*'); else whole-token.
  * Accents are folded (pédé == pede); a STOP list removes cross-language false
    positives; block always wins over allow.

Usage (from repo root, with the venv):
  python scripts/build_lexicon.py --dump path/to/13.named.json --out ~/wakfu-lexicon
LDNOOBW lists are fetched to a cache dir unless --ldnoobw-dir already has them.
"""

import argparse
import json
import urllib.request
from pathlib import Path

from app.services.lexicon import _fold_accents
from app.services.normalizer import normalize

EXCLUDE_LANG = {7}  # spam / URL / RMT bucket
LDNOOBW_LANGS = ("en", "fr", "es", "pt")
LDNOOBW_URL = (
    "https://raw.githubusercontent.com/LDNOOBW/"
    "List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/{lang}"
)
# Slurs/extremism terms not always in LDNOOBW but unequivocally block-worthy.
HATE = {"daesh", "isis", "nazi", "hitler", "kkk", "bamboula", "bougnoul", "negro", "bicot"}
# Cross-language false positives (innocent in some language) + leet artifacts.
STOP = {"con", "cu", "tit", "titi", "tae", "tai", "ass", "bb", "kk", "vv", "ww", "tg",
        "lul", "zebi", "bedo", "gringo", "pok", "pokpok", "hie", "erat", "erad", "weed",
        "bullshit", "isad"}
# Mild / context-dependent words: NOT hard-blocked. They're fine colloquially
# ("I'm stupid", "this sucks") and the ML scores targeted use ("you're stupid") instead.
MILD = {"stupid", "suck", "sucks", "jerk", "damn", "hell", "idiot", "idiots", "dumb",
        "moron", "loser", "noob", "trash", "dummy", "crap"}
STOP |= MILD
# Known short offensive terms that bypass the len>=3/4 guards.
SHORT_BLOCK = {"cum", "fdp", "ntm", "pute", "pd"}
# Generic swears (not slurs / not person-directed): routed to the swear tier, which
# blocks only in strict (identity) contexts and is left to the ML in free text.
SWEAR = {"fuck", "shit", "bullshit", "putain", "merde", "mierda", "joder", "cono",
         "caralho", "foder", "porra", "merda"}


def _norm(text: str) -> str:
    return _fold_accents(normalize(text)).replace(" ", "").strip()


def load_ldnoobw(cache_dir: Path) -> set[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    words: set[str] = set()
    for lang in LDNOOBW_LANGS:
        f = cache_dir / f"ldnoobw_{lang}.txt"
        if not f.exists():
            urllib.request.urlretrieve(LDNOOBW_URL.format(lang=lang), f)  # noqa: S310
        for line in f.read_text(encoding="utf-8").splitlines():
            w = _norm(line)
            if len(w) >= 3:
                words.add(w)
    return words


def _is_slur(word: str, ldnoobw: set[str]) -> bool:
    return (
        word in ldnoobw
        or word in HATE
        or any(s in word for s in ldnoobw if len(s) >= 4)
        or any(h in word for h in HATE)
    )


def _read_seed(path: Path) -> tuple[set[str], set[str]]:
    """Parse a seed file into (substring_terms, token_terms), folding accents."""
    sub: set[str] = set()
    tok: set[str] = set()
    if not path.exists():
        return sub, tok
    for line in path.read_text(encoding="utf-8").splitlines():
        line = _fold_accents(line.strip().lower())
        if not line or line.startswith("#"):
            continue
        (sub if line.endswith("*") and len(line[:-1]) >= 4 else tok).add(line.rstrip("*"))
    return sub, tok


def build(dump: Path, ldnoobw: set[str], seed_dir: Path) -> dict[str, set[str]]:
    """Return a dict of tiers: block_sub/block_tok, swear_sub/swear_tok, flag."""
    data = json.loads(dump.read_text(encoding="utf-8"))
    block_sub: set[str] = set()
    block_tok: set[str] = set()
    swear_sub: set[str] = set()
    swear_tok: set[str] = set()

    def add(word: str, deep: bool) -> None:
        sub, tok = (swear_sub, swear_tok) if word in SWEAR else (block_sub, block_tok)
        if deep and len(word) >= 5:
            sub.add(word)
        elif len(word) >= 4:
            tok.add(word)

    for d in data:
        ct, lang = d["m_censorType"], d["m_language"]
        if lang in EXCLUDE_LANG:
            continue
        w = _norm(d["m_text"])
        if not w or w[0] == "." or any(c.isdigit() for c in w) or w in STOP:
            continue
        if w in SHORT_BLOCK:
            block_tok.add(w)
            continue
        if ct in (0, 1):  # chat-forbidden profanity
            add(w, d["m_deepSearch"])
        elif ct == 2 and _is_slur(w, ldnoobw):  # name-forbidden, but a real slur
            add(w, d["m_deepSearch"])

    # Merge the bundled seed tiers (block / swear / flag), preserving '*' markers.
    bs, bt = _read_seed(seed_dir / "block.txt")
    block_sub |= bs
    block_tok |= bt
    ss, st = _read_seed(seed_dir / "swear.txt")
    swear_sub |= ss
    swear_tok |= st
    flag_sub, flag_tok = _read_seed(seed_dir / "flag.txt")

    def clean(s: set[str]) -> set[str]:
        return {t for t in s if t not in STOP and len(t) >= 3}

    return {
        "block_sub": clean(block_sub), "block_tok": clean(block_tok),
        "swear_sub": clean(swear_sub), "swear_tok": clean(swear_tok),
        "flag": clean(flag_sub | flag_tok),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True, type=Path, help="Path to the NN.named.json dump")
    ap.add_argument("--out", required=True, type=Path, help="Output lexicon directory")
    ap.add_argument("--ldnoobw-dir", type=Path, default=Path("/tmp/ldnoobw"),
                    help="Cache dir for LDNOOBW lists (fetched if absent)")
    ap.add_argument("--seed-dir", type=Path, default=Path("app/data/lexicon"),
                    help="Bundled seed lexicon to merge in")
    args = ap.parse_args()

    ldnoobw = load_ldnoobw(args.ldnoobw_dir)
    tiers = build(args.dump, ldnoobw, args.seed_dir)
    args.out.mkdir(parents=True, exist_ok=True)

    def write_tier(name: str, sub: set[str], tok: set[str]) -> None:
        (args.out / name).write_text("\n".join(sorted(s + "*" for s in sub) + sorted(tok)) + "\n")

    write_tier("block.txt", tiers["block_sub"], tiers["block_tok"])
    write_tier("swear.txt", tiers["swear_sub"], tiers["swear_tok"])
    (args.out / "flag.txt").write_text("\n".join(sorted(tiers["flag"])) + "\n")
    # Allowlist: explicit Wakfu vocab only (NOT derived from censorType 2).
    blocked = tiers["block_sub"] | tiers["block_tok"] | tiers["swear_sub"] | tiers["swear_tok"]
    allow = {"zob", "zobal", "iop", "cra", "sram", "eniripsa", "ecaflip", "enutrof", "sadida",
             "osamodas", "feca", "xelor", "sacrieur", "pandawa", "roublard", "ouginak",
             "huppermage", "eliotrope", "steamer", "ogrest", "support", "astrub"} - blocked
    (args.out / "allow.txt").write_text("\n".join(sorted(allow)) + "\n")
    nblock = len(tiers["block_sub"]) + len(tiers["block_tok"])
    nswear = len(tiers["swear_sub"]) + len(tiers["swear_tok"])
    print(f"wrote {args.out}: block={nblock} swear={nswear} flag={len(tiers['flag'])} "
          f"allow={len(allow)}")


if __name__ == "__main__":
    main()
