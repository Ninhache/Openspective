#!/usr/bin/env bash
# Local development server: hot-reload + DEBUG logs + permissive CORS + Swagger docs.
#
# Usage:
#   ./scripts/dev.sh                 # serves on http://127.0.0.1:8080
#   PORT=9000 ./scripts/dev.sh       # custom port
#   OPENSPECTIVE_MODEL=multilingual ./scripts/dev.sh
#
# Defaults to the lighter `original` model so reloads (which re-load the model)
# stay quick while you iterate.
set -euo pipefail

export OPENSPECTIVE_DEV_MODE=1
export OPENSPECTIVE_MODEL="${OPENSPECTIVE_MODEL:-original}"

# Prefer the project virtualenv if present, otherwise rely on PATH (an activated
# venv or a system install).
if [ -x ".venv/bin/uvicorn" ]; then
  UVICORN=".venv/bin/uvicorn"
else
  UVICORN="uvicorn"
fi

exec "$UVICORN" app.main:app --reload --host 127.0.0.1 --port "${PORT:-8080}"
