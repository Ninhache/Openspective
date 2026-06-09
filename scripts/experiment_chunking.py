"""Experiment: compare summary-score aggregation strategies for long text.

The model has a 512-token window; Detoxify silently truncates beyond it. This
script chunks each input into <=510-token windows, scores every chunk, and shows
what each candidate aggregation would report:

  - direct : current behaviour (one predict() call, truncated at 512 tokens)
  - max    : highest-scoring chunk
  - mean   : plain mean across chunks
  - wmean  : length-weighted mean (weighted by chunk token count)

Run:  .venv/bin/python scripts/experiment_chunking.py
"""

import re

from detoxify import Detoxify

ATTR = "toxicity"
WINDOW = 510  # leave room for the 2 special tokens (<s> ... </s>)

# Same sentence splitter the span scorer uses (.!? runs).
_SENT_RE = re.compile(r".*?(?:[.!?]+|\Z)", flags=re.DOTALL)

TOXIC = "you are a disgusting idiot and everyone here hates your guts"
BENIGN = "I really enjoyed the weather today and the food was lovely. "


def token_chunks(model, text):
    """Split text into <=WINDOW-token windows, returned as decoded strings."""
    tok = model.tokenizer
    ids = tok(text, add_special_tokens=False)["input_ids"]
    chunks = []
    for i in range(0, len(ids), WINDOW):
        window = ids[i : i + WINDOW]
        chunks.append((tok.decode(window), len(window)))
    return chunks or [(text, 0)]


def score(model, text):
    return float(model.predict(text)[ATTR])


def sentence_max(model, text):
    """Max score over sentence-level chunks (the span-scorer granularity)."""
    sents = [s.strip() for s in _SENT_RE.findall(text) if s.strip()]
    if not sents:
        return score(model, text)
    return max(score(model, s) for s in sents)


def aggregate(model, text):
    direct = score(model, text)
    chunks = token_chunks(model, text)
    scores = [score(model, c) for c, _ in chunks]
    weights = [w for _, w in chunks]
    mx = max(scores)
    mean = sum(scores) / len(scores)
    total_w = sum(weights) or 1
    wmean = sum(s * w for s, w in zip(scores, weights, strict=True)) / total_w
    return {
        "tokens": sum(weights),
        "n_chunks": len(chunks),
        "direct": direct,
        "max": mx,
        "mean": mean,
        "wmean": wmean,
        "sent_max": sentence_max(model, text),
    }


CASES = {
    "short toxic": TOXIC,
    "short clean": "I really enjoyed the weather today and the food was lovely.",
    "toxic tail after long benign": BENIGN * 60 + TOXIC,
    "toxic head then long benign": TOXIC + ". " + BENIGN * 60,
    "long all-benign": BENIGN * 60,
    "long all-toxic": (TOXIC + ". ") * 30,
    "1 toxic sentence in long benign": BENIGN * 30 + TOXIC + ". " + BENIGN * 30,
}


def main():
    model = Detoxify("original")
    header = (
        f"{'case':<34}{'tok':>5}{'ch':>4}{'direct':>9}"
        f"{'win_max':>9}{'mean':>9}{'wmean':>9}{'sent_max':>9}"
    )
    print(header)
    print("-" * len(header))
    for name, text in CASES.items():
        r = aggregate(model, text)
        print(
            f"{name:<34}{r['tokens']:>5}{r['n_chunks']:>4}"
            f"{r['direct']:>9.4f}{r['max']:>9.4f}{r['mean']:>9.4f}"
            f"{r['wmean']:>9.4f}{r['sent_max']:>9.4f}"
        )


if __name__ == "__main__":
    main()
