"""Language detection for populating ``detectedLanguages``.

Perspective auto-detects the comment language when the caller does not supply one.

We prefer **fastText `lid.176`** (a ~1 MB model, sub-millisecond, 176 languages) when it
is available — it is far more accurate than ``langdetect`` on short text (e.g. Spanish is
no longer mistaken for Italian) and exposes a confidence we threshold on. It is optional:
install the ``langid`` extra and provide the model (``OPENSPECTIVE_LANGID_MODEL`` or
auto-download). When fastText or its model is unavailable we fall back to ``langdetect``
so detection always works offline.

Detection is best-effort: on short / empty / undetectable input (or low confidence) we
return ``["unknown"]`` rather than failing the request.
"""

import logging
import urllib.request
from pathlib import Path
from threading import Lock

from langdetect import DetectorFactory, LangDetectException, detect

from app.config import get_settings

logger = logging.getLogger("openspective.detector")

# langdetect samples randomly; fix the seed so the fallback is deterministic.
DetectorFactory.seed = 0

UNKNOWN = "unknown"
_LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"

# Lazily-loaded fastText model. ``_attempted`` makes the (possibly failing) load a
# one-shot: after the first try we either have a model or commit to the langdetect path.
_model = None
_attempted = False
_lock = Lock()


def _model_path() -> Path:
    """Resolve the fastText model path (explicit setting or the default cache location)."""
    configured = get_settings().langid_model
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "openspective" / "lid.176.ftz"


def ensure_model(download: bool = False) -> None:
    """Best-effort: fetch the fastText model to the cache if missing.

    Called from the app lifespan when ``OPENSPECTIVE_LANGID_AUTODOWNLOAD`` is on, so the
    download happens once at startup instead of on a request.
    """
    if not download:
        return
    path = _model_path()
    if path.exists():
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading fastText language model to %s", path)
        urllib.request.urlretrieve(_LID_URL, path)  # noqa: S310
    except Exception as exc:  # noqa: BLE001 — optional feature, degrade to langdetect
        logger.warning("Could not download fastText model (%s); using langdetect", exc)


def _load_model():
    """Return the loaded fastText model, or ``None`` to use the langdetect fallback."""
    global _model, _attempted
    if _attempted:
        return _model
    with _lock:
        if _attempted:
            return _model
        _attempted = True
        path = _model_path()
        if not path.exists():
            logger.info("No fastText model at %s; using langdetect", path)
            return None
        try:
            import fasttext  # optional dependency (the ``langid`` extra)

            _model = fasttext.load_model(str(path))
            logger.info("fastText language detection enabled")
        except Exception as exc:  # noqa: BLE001 — fall back to langdetect
            logger.warning("fastText unavailable (%s); using langdetect", exc)
            _model = None
        return _model


def _detect_fasttext(model, text: str) -> list[str] | None:
    """Detect via fastText, applying the confidence threshold. ``None`` => caller falls back."""
    try:
        # Low-level call: the high-level .predict() is broken under numpy 2.x.
        preds = model.f.predict(text, 1, 0.0, "strict")
    except Exception:  # noqa: BLE001
        logger.debug("fastText predict failed; falling back to langdetect")
        return None
    if not preds:
        return [UNKNOWN]
    prob, label = preds[0]
    if prob < get_settings().langid_min_confidence:
        return [UNKNOWN]  # e.g. a bare username has no real language
    return [label.replace("__label__", "")]


def detect_language(text: str) -> list[str]:
    """Best-effort language detection (fastText if available, else langdetect).

    :param text: The comment text (raw or normalised both work).
    :returns: A single-element list with an ISO 639-1 code, or ``["unknown"]``.
    """
    stripped = text.strip().replace("\n", " ")
    if not stripped:
        return [UNKNOWN]

    model = _load_model()
    if model is not None:
        result = _detect_fasttext(model, stripped)
        if result is not None:
            return result

    try:
        return [detect(stripped)]
    except LangDetectException:
        logger.debug("Language detection failed for input of length %d", len(stripped))
        return [UNKNOWN]
