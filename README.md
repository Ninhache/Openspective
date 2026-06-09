# openspective

> Self-hosted, open-source, drop-in replacement for the Google Perspective API — toxicity scoring over a REST API you control.

[![CI](https://github.com/Ninhache/Openspective/actions/workflows/ci.yml/badge.svg)](https://github.com/Ninhache/Openspective/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Google's [Perspective API](https://perspectiveapi.com/) is **sunsetting in December 2026**.
`openspective` is a self-hostable replacement that mirrors Perspective's JSON request/response
schema, so existing integrations can swap the endpoint URL with minimal code changes. Scoring is
powered by the [Detoxify](https://github.com/unitaryai/detoxify) model (XLM-RoBERTa multilingual).

- **Drop-in schema** — same `comments:analyze` request/response shape as Perspective.
- **Self-hosted** — runs entirely on your infrastructure via Docker Compose. No external calls.
- **Language-agnostic** — plain REST + JSON; call it from anything.
- **Multilingual** — EN / FR / ES / IT / PT / TR / RU out of the box.
- **Cached** — Redis-backed score cache with graceful degradation if Redis is down.

---

## Quick start

```bash
git clone https://github.com/Ninhache/Openspective.git
cd Openspective
docker compose up --build
```

The API is then available at `http://localhost:8080`.

> **First build is slow and large.** It downloads the CPU PyTorch wheel and pre-bakes the
> multilingual model weights (~1GB) into the image so the container starts offline.

Health check:

```bash
curl http://localhost:8080/healthz
# {"status":"ok","model":"multilingual"}
```

---

## Example request

```bash
curl -X POST http://localhost:8080/v1alpha1/comments:analyze \
  -H "Content-Type: application/json" \
  -d '{
    "comment": { "text": "you are such an idiot" },
    "requestedAttributes": { "TOXICITY": {}, "INSULT": {}, "THREAT": {} },
    "languages": ["en"]
  }'
```

Response:

```json
{
  "attributeScores": {
    "TOXICITY": { "summaryScore": { "value": 0.87, "type": "PROBABILITY" } },
    "INSULT":   { "summaryScore": { "value": 0.72, "type": "PROBABILITY" } },
    "THREAT":   { "summaryScore": { "value": 0.04, "type": "PROBABILITY" } }
  },
  "languages": ["en"],
  "detectedLanguages": ["en"],
  "clientToken": null
}
```

`requestedAttributes` is optional — omit it to receive **all** attributes. `languages` is optional
too; if omitted, the language is auto-detected (via `langdetect`) and returned in
`detectedLanguages`. A supplied `languages` value overrides the reported `languages` while
`detectedLanguages` still reflects what was detected. Undetectable input falls back to
`["unknown"]`.

### Span scores

Set `spanAnnotations: true` to also receive per-sentence scores for each attribute (like
Perspective's `spanScores`). Each span carries character offsets into the original text:

```bash
curl -X POST http://localhost:8080/v1alpha1/comments:analyze \
  -H "Content-Type: application/json" \
  -d '{
    "comment": { "text": "Nice to meet you. You are an idiot." },
    "requestedAttributes": { "TOXICITY": { "scoreThreshold": 0.5 } },
    "spanAnnotations": true
  }'
```

```json
{
  "attributeScores": {
    "TOXICITY": {
      "summaryScore": { "value": 0.91, "type": "PROBABILITY" },
      "spanScores": [
        { "begin": 18, "end": 35, "score": { "value": 0.91, "type": "PROBABILITY" } }
      ]
    }
  }
}
```

Notes:
- A per-attribute `scoreThreshold` (or the `OPENSPECTIVE_SCORE_THRESHOLD` default) filters which
  spans are returned — spans below the threshold are omitted (the summary score is always returned).
- Sentence splitting is a lightweight, dependency-free `.!?` splitter — good for highlighting toxic
  sentences, not a perfect linguistic tokenizer.
- Span scoring runs one inference **per span**, so it is more expensive than a summary-only call and
  is **not cached**.

### Endpoints

| Method | Path                              | Description                                  |
|--------|-----------------------------------|----------------------------------------------|
| POST   | `/v1alpha1/comments:analyze`      | Score a comment (Perspective-compatible)     |
| GET    | `/healthz`                        | Liveness probe (always 200 while serving)    |
| GET    | `/readyz`                         | Readiness probe (200 only once model loaded) |
| GET    | `/v1/models`                      | List variants + currently loaded model       |
| GET    | `/metrics`                        | Prometheus metrics (requests, inference, cache) |

---

## Configuration

All settings are environment variables (prefix `OPENSPECTIVE_`):

| Variable                   | Default                 | Description                                        |
|----------------------------|-------------------------|----------------------------------------------------|
| `OPENSPECTIVE_MODEL`       | `multilingual`          | Built-in variant (`original`/`unbiased`/`multilingual`) **or** a checkpoint path |
| `OPENSPECTIVE_REDIS_URL`   | `redis://redis:6379`    | Redis connection URL for the score cache            |
| `OPENSPECTIVE_CACHE_TTL`   | `3600`                  | Cache entry TTL in seconds                          |
| `OPENSPECTIVE_LOG_LEVEL`   | `INFO`                  | `DEBUG` / `INFO` / `WARNING` / `ERROR`              |
| `OPENSPECTIVE_WORKERS`     | `1`                     | Worker count (informational; set in your runner)    |
| `OPENSPECTIVE_API_TOKENS`  | _(empty)_               | Comma-separated Bearer tokens; empty disables auth  |
| `OPENSPECTIVE_RATE_LIMIT`  | `0`                     | Max requests per window; `0` disables rate limiting |
| `OPENSPECTIVE_RATE_LIMIT_WINDOW` | `60`              | Rate-limit window length in seconds                 |
| `OPENSPECTIVE_SCORE_THRESHOLD` | `0.0`             | Default min score for returned span scores          |

See [`.env.example`](.env.example) for a starter file.

### Custom / fine-tuned models

Set `OPENSPECTIVE_MODEL` to a path to a fine-tuned Detoxify checkpoint
(`.ckpt`/`.pt`/`.pth`/`.bin`) instead of a built-in variant name to serve your own weights:

```yaml
# docker-compose.yml (api service)
environment:
  - OPENSPECTIVE_MODEL=/models/my-finetuned.ckpt
volumes:
  - ./models:/models:ro
```

The model is loaded once at startup; `/v1/models` and `/healthz` report it as `custom:<filename>`.

### Authentication

Authentication is **off by default** so openspective is a frictionless drop-in. To require
auth, set `OPENSPECTIVE_API_TOKENS` to one or more comma-separated tokens:

```bash
OPENSPECTIVE_API_TOKENS=tok_abc123,tok_def456
```

Clients then send a Bearer token on the analyze endpoint:

```bash
curl -X POST http://localhost:8080/v1alpha1/comments:analyze \
  -H "Authorization: Bearer tok_abc123" \
  -H "Content-Type: application/json" \
  -d '{ "comment": { "text": "hi" } }'
```

Operational endpoints (`/healthz`, `/readyz`, `/metrics`, `/v1/models`) stay unauthenticated so
health probes and metric scrapers can always reach them. Invalid/missing tokens return a structured
`401 {"error":"unauthorized","detail":"..."}`.

### Rate limiting

Rate limiting is **off by default**. Set `OPENSPECTIVE_RATE_LIMIT` (and optionally
`OPENSPECTIVE_RATE_LIMIT_WINDOW`) to enable a Redis-backed fixed window:

```bash
OPENSPECTIVE_RATE_LIMIT=100          # 100 requests...
OPENSPECTIVE_RATE_LIMIT_WINDOW=60    # ...per 60 seconds, per client
```

Clients are keyed by their Bearer token when present, otherwise by IP. Allowed responses carry
`X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset` headers; exceeding the limit
returns `429 {"error":"rate_limited","detail":"..."}` with a `Retry-After` header. If Redis is
unavailable, the limiter **fails open** (requests are allowed) — the same graceful degradation as
the cache.

---

## Attribute mapping

Detoxify scores all attributes on every call; openspective renames them to Perspective's
attribute names and filters to the ones you requested.

| Detoxify key      | Perspective attribute |
|-------------------|-----------------------|
| `toxicity`        | `TOXICITY`            |
| `severe_toxicity` | `SEVERE_TOXICITY`     |
| `obscene`         | `OBSCENE`             |
| `threat`          | `THREAT`              |
| `insult`          | `INSULT`              |
| `identity_attack` | `IDENTITY_ATTACK`     |

---

## ⚠️ Detoxify: official repo vs HuggingFace

This project installs Detoxify from the **official GitHub repository at a pinned tag**
(`git+https://github.com/unitaryai/detoxify@v0.5.2`), **not** from the PyPI package or the
HuggingFace model card. The official implementation and the HuggingFace-hosted variant **diverge**
(preprocessing and weights differ), which produces different scores. Installing from the official
source keeps results consistent with upstream Detoxify. See the [Dockerfile](Dockerfile).

---

## Why this exists

The Perspective API is being shut down in **December 2026**. Teams that depend on it for comment
moderation need a path off it that doesn't involve rewriting integrations or shipping user text to
a third party. openspective keeps the same wire format while running entirely on your own hardware.

---

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# torch + detoxify are not in pyproject (see note above); for full local runs:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "git+https://github.com/unitaryai/detoxify@v0.5.2"

pytest            # the test suite stubs out the model, so torch/detoxify aren't required for tests
```

The test suite mocks the classifier and Redis, so `pytest` runs fast and offline without the model.

---

## Roadmap

- Smaller runtime image (multi-stage build).
- Richer language detection signals and per-language model routing.

## Security

Found a vulnerability? Please report it privately — see [SECURITY.md](SECURITY.md). Hardening tips
for internet-facing deployments are in that file and in the [Authentication](#authentication) and
[Rate limiting](#rate-limiting) sections above.

## License

[MIT](LICENSE) © Ninhache
