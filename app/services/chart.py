"""Hand-built SVG rendering of attribute scores as circular gauges.

No external charting dependency: we emit a self-contained SVG string with one
donut gauge per attribute (percentage in the centre, colour graded by score),
sorted by score descending — visually similar to Perspective's score diagram.
"""

import html
import math

# Gauge geometry / layout constants (in SVG user units ~ pixels).
_RADIUS = 46
_STROKE = 12
_CELL_W = 168
_CELL_H = 168
_COLS = 3
_HEADER_H = 78
_PAD = 24

# Score colour bands (low / medium / high toxicity).
_HIGH = 0.70
_MEDIUM = 0.40
_COLOR_HIGH = "#c62828"
_COLOR_MEDIUM = "#f9a825"
_COLOR_LOW = "#2e7d32"
_TRACK = "#e6e6e6"
_TEXT = "#222222"
_SUBTLE = "#777777"


def _color(value: float) -> str:
    """Return the gauge colour for a score in ``[0, 1]``."""
    if value >= _HIGH:
        return _COLOR_HIGH
    if value >= _MEDIUM:
        return _COLOR_MEDIUM
    return _COLOR_LOW


def _gauge(cx: float, cy: float, label: str, value: float) -> str:
    """Render a single donut gauge centred at ``(cx, cy)``."""
    circumference = 2 * math.pi * _RADIUS
    # stroke-dashoffset hides the remaining (1 - value) portion of the ring.
    offset = circumference * (1 - max(0.0, min(1.0, value)))
    pct = round(value * 100)
    color = _color(value)
    return f"""
  <g>
    <circle cx="{cx}" cy="{cy}" r="{_RADIUS}" fill="none" stroke="{_TRACK}"
            stroke-width="{_STROKE}"/>
    <circle cx="{cx}" cy="{cy}" r="{_RADIUS}" fill="none" stroke="{color}" stroke-width="{_STROKE}"
            stroke-linecap="round" stroke-dasharray="{circumference:.2f}"
            stroke-dashoffset="{offset:.2f}" transform="rotate(-90 {cx} {cy})"/>
    <text x="{cx}" y="{cy + 6}" text-anchor="middle" font-size="22" font-weight="700"
          fill="{_TEXT}" font-family="system-ui, sans-serif">{pct}%</text>
    <text x="{cx}" y="{cy + _RADIUS + 22}" text-anchor="middle" font-size="11"
          fill="{_SUBTLE}" font-family="system-ui, sans-serif">{html.escape(label)}</text>
  </g>"""


def render_svg(text: str, scores: dict[str, float], language: str | None = None) -> str:
    """Render attribute ``scores`` as an SVG of circular gauges.

    :param text: The analysed comment (shown, truncated, as the title).
    :param scores: Mapping of Perspective attribute name -> probability.
    :param language: Optional detected language code, shown as a subtitle.
    :returns: A complete ``<svg>...</svg>`` document string.
    """
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    count = max(1, len(items))
    cols = min(_COLS, count)
    rows = math.ceil(count / cols)

    width = cols * _CELL_W + 2 * _PAD
    height = _HEADER_H + rows * _CELL_H + _PAD

    snippet = text if len(text) <= 80 else text[:77] + "…"
    subtitle = f"detected language: {html.escape(language)}" if language else ""

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="openspective scores">',
        f'  <rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'  <text x="{_PAD}" y="34" font-size="16" font-weight="700" fill="{_TEXT}" '
        f'font-family="system-ui, sans-serif">openspective</text>',
        f'  <text x="{_PAD}" y="56" font-size="13" fill="{_TEXT}" '
        f'font-family="system-ui, sans-serif">“{html.escape(snippet)}”</text>',
    ]
    if subtitle:
        parts.append(
            f'  <text x="{_PAD}" y="72" font-size="11" fill="{_SUBTLE}" '
            f'font-family="system-ui, sans-serif">{subtitle}</text>'
        )

    for index, (label, value) in enumerate(items):
        col = index % cols
        row = index // cols
        cx = _PAD + col * _CELL_W + _CELL_W / 2
        cy = _HEADER_H + row * _CELL_H + _RADIUS + _STROKE
        parts.append(_gauge(cx, cy, label, value))

    parts.append("</svg>")
    return "\n".join(parts)
