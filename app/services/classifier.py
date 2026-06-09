"""Detoxify model wrapper.

The model is loaded once at startup (from the FastAPI lifespan, not on first
request) and inference is run inside a single-threaded ``ThreadPoolExecutor`` so
the synchronous PyTorch call never blocks the asyncio event loop.

Inference errors are deliberately *not* swallowed here — they propagate to the
router, which maps them to an HTTP 500 with a structured error body.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from app.config import VALID_MODEL_VARIANTS

logger = logging.getLogger("openspective.classifier")

# Module-level singletons, initialised by ``load_model``.
_model = None
_model_variant: str | None = None
# Detoxify/PyTorch inference is synchronous; run it off the event loop. A single
# worker keeps memory bounded and is sufficient for one model instance.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detoxify")


def load_model(variant: str) -> None:
    """Load the Detoxify model for ``variant`` into the module-level singleton.

    Called once from the application lifespan. Importing Detoxify is done lazily
    inside this function so the rest of the app (and the test suite) does not pull
    in torch unless a model is actually being loaded.

    :param variant: One of ``original``, ``unbiased``, ``multilingual``.
    :raises ValueError: If ``variant`` is not a known Detoxify variant.
    """
    global _model, _model_variant

    if variant not in VALID_MODEL_VARIANTS:
        raise ValueError(
            f"Unknown model variant {variant!r}; expected one of {VALID_MODEL_VARIANTS}"
        )

    from detoxify import Detoxify  # local import: avoids importing torch at module load

    logger.info("Loading Detoxify model variant=%s", variant)
    _model = Detoxify(variant)
    _model_variant = variant
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


def _predict_sync(text: str) -> dict[str, float]:
    """Run blocking Detoxify inference and coerce scores to plain floats."""
    if _model is None:
        raise RuntimeError("Model not loaded; call load_model() during startup")
    raw = _model.predict(text)
    # Detoxify may return numpy float types; cast to built-in float for JSON safety.
    return {key: float(value) for key, value in raw.items()}


async def predict(text: str) -> dict[str, float]:
    """Score ``text`` across all Detoxify attributes.

    Inference runs in the thread pool so the event loop stays responsive.

    :param text: The (already normalised) text to score.
    :returns: Mapping of Detoxify attribute keys to probabilities in ``[0, 1]``.
    """
    from app.services.metrics import INFERENCE_LATENCY

    loop = asyncio.get_running_loop()
    with INFERENCE_LATENCY.time():
        return await loop.run_in_executor(_executor, _predict_sync, text)


def shutdown() -> None:
    """Release the inference thread pool (called from the lifespan shutdown)."""
    _executor.shutdown(wait=False, cancel_futures=True)
