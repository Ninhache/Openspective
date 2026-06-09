"""Prometheus metrics definitions and helpers.

Metrics are registered on the default Prometheus registry and exposed at ``/metrics``
(see routers/meta.py). Other modules import the metric objects from here and update
them, keeping the instrumentation declarations in one place.
"""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# HTTP request counters/timers. ``path`` uses the matched route template (not the raw
# URL) to keep label cardinality bounded.
REQUEST_COUNT = Counter(
    "openspective_requests_total",
    "Total HTTP requests.",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "openspective_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "path"],
)

# Model inference timing (the Detoxify forward pass).
INFERENCE_LATENCY = Histogram(
    "openspective_inference_duration_seconds",
    "Detoxify inference latency in seconds.",
)

# Cache effectiveness.
CACHE_HITS = Counter("openspective_cache_hits_total", "Score cache hits.")
CACHE_MISSES = Counter("openspective_cache_misses_total", "Score cache misses.")


def render() -> tuple[bytes, str]:
    """Return the current metrics exposition and its content type.

    :returns: ``(payload, content_type)`` suitable for a FastAPI ``Response``.
    """
    return generate_latest(), CONTENT_TYPE_LATEST
