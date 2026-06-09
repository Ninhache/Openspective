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
| `OPENSPECTIVE_MODEL`       | `multilingual`          | Detoxify variant: `original`, `unbiased`, `multilingual` |
| `OPENSPECTIVE_REDIS_URL`   | `redis://redis:6379`    | Redis connection URL for the score cache            |
| `OPENSPECTIVE_CACHE_TTL`   | `3600`                  | Cache entry TTL in seconds                          |
| `OPENSPECTIVE_LOG_LEVEL`   | `INFO`                  | `DEBUG` / `INFO` / `WARNING` / `ERROR`              |
| `OPENSPECTIVE_WORKERS`     | `1`                     | Worker count (informational; set in your runner)    |
| `OPENSPECTIVE_API_TOKENS`  | _(empty)_               | Comma-separated Bearer tokens; empty disables auth  |

See [`.env.example`](.env.example) for a starter file.

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

## Roadmap (v0.2)

- Rate limiting (per-token / per-IP).
- Span / per-sentence scores.
- Fine-tuning hooks and configurable score thresholds.

## License

[MIT](LICENSE) © Ninhache
