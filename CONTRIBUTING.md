# Contributing to openspective

Thanks for your interest in improving openspective! This document covers how to get set up and
the conventions we follow.

## Getting started

```bash
git clone https://github.com/Ninhache/Openspective.git
cd Openspective
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install   # runs `ruff check` on staged files before each commit (matches CI)
```

The test suite stubs out the Detoxify model and Redis, so you can run it without downloading
weights or running Redis:

```bash
pytest
ruff check .
```

For an end-to-end run with the real model, use Docker Compose:

```bash
docker compose up --build
```

## Project layout

```
app/
├── main.py            # FastAPI app + lifespan (loads the model once)
├── config.py          # pydantic-settings configuration
├── models.py          # Pydantic request/response schemas (Perspective-compatible)
├── routers/           # analyze + meta (health/models) endpoints
├── services/          # classifier (Detoxify), normalizer, cache (Redis)
└── tests/             # pytest suite (model + Redis mocked)
```

## Conventions

- **Style**: `ruff` enforces formatting and lint rules (`ruff check .`). Line length 100.
- **Types**: prefer type hints on public functions; add docstrings describing behaviour.
- **Tests**: add or update tests for any behaviour change. Keep the suite offline — mock the
  model and Redis rather than requiring them.
- **Schema fidelity**: the request/response field names intentionally match Perspective's exact
  camelCase keys. Do not rename them.
- **Commits**: use conventional-commit-style prefixes (`feat:`, `fix:`, `docs:`, `chore:` …).

## Pull requests

1. Fork and create a feature branch.
2. Make your change with tests and docs.
3. Ensure `pytest` and `ruff check .` pass.
4. Open a PR describing the change and linking any related issue.

## Reporting bugs / requesting features

Open an issue with a clear description and, for bugs, a minimal reproduction (request body +
observed vs expected response).
