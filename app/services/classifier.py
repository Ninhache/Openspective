"""Detoxify model wrapper.

The model is loaded once at startup (from the FastAPI lifespan, not on first
request) and inference is run inside a single-threaded ``ThreadPoolExecutor`` so
the synchronous PyTorch call never blocks the asyncio event loop.

Inference errors are deliberately *not* swallowed here — they propagate to the
router, which maps them to an HTTP 500 with a structured error body.
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from app.config import VALID_MODEL_VARIANTS
from app.services.chunking import MODEL_WINDOW, chunk_text

logger = logging.getLogger("openspective.classifier")

# File extensions recognised as a fine-tuned Detoxify checkpoint.
_CHECKPOINT_SUFFIXES = (".ckpt", ".pt", ".pth", ".bin")

# Module-level singletons, initialised by ``load_model``.
_model = None
_model_variant: str | None = None
# Detoxify/PyTorch inference is synchronous; run it off the event loop. A single
# worker keeps memory bounded and is sufficient for one model instance.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detoxify")


def _is_checkpoint(variant: str) -> bool:
    """Return True if ``variant`` refers to a fine-tuned checkpoint file rather than
    one of the built-in variant names."""
    return variant.endswith(_CHECKPOINT_SUFFIXES) or os.path.isfile(variant)


def load_model(variant: str) -> None:
    """Load the Detoxify model into the module-level singleton.

    Called once from the application lifespan. Importing Detoxify is done lazily
    inside this function so the rest of the app (and the test suite) does not pull
    in torch unless a model is actually being loaded.

    ``variant`` is either a built-in variant name (``original``/``unbiased``/
    ``multilingual``) or a path to a fine-tuned Detoxify checkpoint file
    (``.ckpt``/``.pt``/...), enabling custom models via ``OPENSPECTIVE_MODEL``.

    :param variant: A built-in variant name or a checkpoint file path.
    :raises ValueError: If ``variant`` is neither a known variant nor a checkpoint.
    """
    global _model, _model_variant

    from detoxify import Detoxify  # local import: avoids importing torch at module load

    if _is_checkpoint(variant):
        logger.info("Loading Detoxify from checkpoint=%s", variant)
        _model = Detoxify(checkpoint=variant, device="cpu")
        _model_variant = f"custom:{os.path.basename(variant)}"
    elif variant in VALID_MODEL_VARIANTS:
        logger.info("Loading Detoxify model variant=%s", variant)
        _model = Detoxify(variant)
        _model_variant = variant
    else:
        raise ValueError(
            f"Unknown model {variant!r}; expected one of {VALID_MODEL_VARIANTS} "
            f"or a path to a checkpoint file ({', '.join(_CHECKPOINT_SUFFIXES)})"
        )
    logger.info("Detoxify model loaded")


def is_loaded() -> bool:
    """Return ``True`` once a model has been loaded."""
    return _model is not None


def model_name() -> str | None:
    """Return the name of the currently loaded model variant (or ``None``)."""
    return _model_variant


def available_variants() -> tuple[str, ...]:
    """Return the tuple of selectable Detoxify model variants."""
    return VALID_MODEL_VARIANTS


def _count_tokens(text: str) -> int:
    """Token count of ``text`` under the model tokenizer (excluding special tokens)."""
    return len(_model.tokenizer(text, add_special_tokens=False)["input_ids"])


def _predict_sync(text: str) -> dict[str, float]:
    """Run blocking Detoxify inference on a single chunk and coerce to plain floats.

    ``text`` is assumed to fit the model window; callers that may exceed it must go
    through :func:`_predict_pooled_sync`.
    """
    if _model is None:
        raise RuntimeError("Model not loaded; call load_model() during startup")
    raw = _model.predict(text)
    # Detoxify may return numpy float types; cast to built-in float for JSON safety.
    return {key: float(value) for key, value in raw.items()}


def _predict_pooled_sync(text: str) -> dict[str, float]:
    """Score ``text`` correctly regardless of length.

    Short text (within the model window) is scored in a single pass — the common
    case, no overhead. Longer text is split into window-sized chunks, each scored,
    and the per-attribute **maximum** is returned, so a toxic passage anywhere in a
    long comment surfaces instead of being silently truncated.
    """
    if _model is None:
        raise RuntimeError("Model not loaded; call load_model() during startup")
    if _count_tokens(text) <= MODEL_WINDOW:
        return _predict_sync(text)

    from app.services.metrics import CHUNKED_REQUESTS

    chunks = chunk_text(text, _count_tokens, MODEL_WINDOW)
    CHUNKED_REQUESTS.inc()
    logger.debug("Long input pooled over %d chunks (%d tokens)", len(chunks), _count_tokens(text))
    pooled: dict[str, float] = {}
    for chunk in chunks:
        for key, value in _predict_sync(chunk).items():
            pooled[key] = max(pooled.get(key, 0.0), value)
    return pooled


async def predict(text: str) -> dict[str, float]:
    """Score ``text`` across all Detoxify attributes.

    Inference runs in the thread pool so the event loop stays responsive. Text that
    exceeds the model's token window is chunked and pooled (see
    :func:`_predict_pooled_sync`) rather than silently truncated.

    :param text: The (already normalised) text to score.
    :returns: Mapping of Detoxify attribute keys to probabilities in ``[0, 1]``.
    """
    from app.services.metrics import INFERENCE_LATENCY

    loop = asyncio.get_running_loop()
    with INFERENCE_LATENCY.time():
        return await loop.run_in_executor(_executor, _predict_pooled_sync, text)


def shutdown() -> None:
    """Release the inference thread pool (called from the lifespan shutdown)."""
    _executor.shutdown(wait=False, cancel_futures=True)
