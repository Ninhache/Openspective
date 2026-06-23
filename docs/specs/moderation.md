# Moderation layer (`/v1/moderate`)

## Why

openspective's Detoxify model is a *conversational-toxicity* classifier. Empirically
(on real Wakfuli data) it misses ~95% of bad **short** strings — usernames, slugs,
build names — especially in FR/ES/PT, because bare profanity in a short handle is
out-of-distribution. The right primary detector for short fields is a **lexicon**
(deterministic, you control it, instant); the ML is the *secondary* net for novel /
contextual toxicity and the *primary* for long free-text (build notes).

This layer composes both behind one endpoint so callers get a single verdict.

## Design

```
short text  → normalize → segment → tiered lexicon → block / flag / allow
long text   → (clean lexicon) → Detoxify fallback → flag on high score
```

### Tiers (data-driven, editable files — "static" = deterministic, not frozen)

- `app/data/lexicon/block.txt` — strong profanity / slurs → **block**
- `app/data/lexicon/flag.txt`  — soft / borderline → **flag** (send to a human)
- `app/data/lexicon/allow.txt` — allowlist: domain vocab (`zob`=Zobal class, Wakfu
  class names) + known false-positive carriers → never contributes a match

Override the directory with `OPENSPECTIVE_LEXICON_DIR` to ship a custom list without
touching code. Grow the lists via the human-in-the-loop review cycle (no redeploy).

### Matching

Default is **whole-token** matching (precise: avoids `réputation`→`pute`,
`drapeau`→`rape`). A trailing `*` on a term opts it into **substring** matching to
catch concatenations (`fuck*` → `fuckankama`) — used only for unambiguous stems where
the false-positive risk is ~0. Allowlisted tokens suppress matches inside them.

Trade-off: token-default misses some concatenated FR profanity (`talkapute`); the
feedback loop and a future fuzzy/embedding layer (v2) close that gap. Precision over
recall on FR short strings is the right v1 call (a flagged `réputation` is worse than
a missed rare pun).

### Decision

`block` if any block term hits; else `flag` if any flag term hits (or ML fallback
scores high on long text); else `allow`. ML fallback runs only when the lexicon is
clean **and** the text is long enough to be notes-like (keeps usernames fast).

## API

`POST /v1/moderate` → `{ "text": "...", "context": "username|slug|build_name|notes" }`
→ `{ "decision": "block|flag|allow", "reasons": [...], "tier": "...", "mlScore": 0.97 }`

## Out of scope (v1)

- Fuzzy / edit-distance / embedding expansion (v2).
- Fine-tuned Wakfuli model (needs accumulated labels; the loop bootstraps it).
- Email-as-username detection — that's input validation / PII, not moderation.
