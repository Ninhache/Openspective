# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Builder stage: install everything (incl. build-only tools like git) and bake
# the model weights into caches we control, then hand only the results to the
# slim runtime stage. This keeps git, pip caches, and build artefacts OUT of the
# final image.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Pin the Detoxify release from the OFFICIAL repository. The PyPI package and the
# HuggingFace model card diverge from the official implementation, so we install
# straight from GitHub at a fixed tag for reproducible builds.
ARG DETOXIFY_REF=v0.5.2
ARG OPENSPECTIVE_MODEL=multilingual

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Self-contained venv so the runtime stage can copy one tree.
    PATH="/opt/venv/bin:$PATH" \
    # Bake model weights into fixed cache dirs we copy into the runtime stage.
    TORCH_HOME=/opt/model-cache/torch \
    HF_HOME=/opt/model-cache/hf

# git is needed only to install Detoxify from source (build-time only).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv

WORKDIR /app

# 1. CPU-only PyTorch wheel (avoids pulling multi-GB CUDA/NVIDIA libraries).
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Pinned official Detoxify.
RUN pip install "git+https://github.com/unitaryai/detoxify@${DETOXIFY_REF}"

# 3. Application + its declared dependencies.
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install .

# 4. Pre-download the model weights into TORCH_HOME/HF_HOME so the runtime image
#    starts without network access.
RUN python -c "from detoxify import Detoxify; Detoxify('${OPENSPECTIVE_MODEL}')"

# ---------------------------------------------------------------------------
# Runtime stage: slim image with only the venv, the baked model cache, and the
# app source. No git, no build tools, no pip caches.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ARG OPENSPECTIVE_MODEL=multilingual

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    TORCH_HOME=/opt/model-cache/torch \
    HF_HOME=/opt/model-cache/hf \
    # Weights are pre-baked, so load from cache only — no HF Hub calls at startup.
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OPENSPECTIVE_MODEL=${OPENSPECTIVE_MODEL}

# Run as a non-root user for defence in depth.
RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv
# Owned by the runtime user so HF/transformers can write lock files if needed.
COPY --from=builder --chown=appuser:appuser /opt/model-cache /opt/model-cache
COPY --from=builder /app/app /app/app

WORKDIR /app
USER appuser

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
