# syntax=docker/dockerfile:1

FROM python:3.11-slim

# --- Build configuration -----------------------------------------------------
# Pin the Detoxify release from the OFFICIAL repository. The PyPI package and the
# HuggingFace model card diverge from the official implementation, so we install
# straight from GitHub at a fixed tag for reproducible builds.
ARG DETOXIFY_REF=v0.5.2
ARG OPENSPECTIVE_MODEL=multilingual

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENSPECTIVE_MODEL=${OPENSPECTIVE_MODEL}

# git is needed to install Detoxify from source.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install the CPU-only PyTorch wheel first (keeps the image ~GBs smaller than
#    the default CUDA build, and avoids pulling NVIDIA libraries we cannot use).
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install the pinned official Detoxify.
RUN pip install "git+https://github.com/unitaryai/detoxify@${DETOXIFY_REF}"

# 3. Install application dependencies (declared in pyproject.toml).
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install .

# 4. Pre-download the model weights at build time so the container starts without
#    needing network access.
RUN python -c "from detoxify import Detoxify; Detoxify('${OPENSPECTIVE_MODEL}')"

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
