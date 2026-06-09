"""Hand-built SVG rendering of attribute scores.

No external charting dependency: we emit self-contained SVG strings. Three styles:

- ``gauges``: one donut gauge per attribute (default).
- ``radar``:  a single radar / spider chart over all attributes.
- ``polar``:  a single polar-area (rose) chart over all attributes.

All styles are visually in the spirit of Perspective's score diagram.
"""

import html
import math

# Supported chart styles.
STYLES = ("gauges", "radar", "polar")

# --- Gauge geometry / layout constants (in SVG user units ~ pixels). ---------
_RADIUS = 46
_STROKE = 12
_CELL_W = 168
_CELL_H = 168
_COLS = 3
_HEADER_H = 78
_PAD = 24

# --- Radar / polar geometry. -------------------------------------------------
_R = 130  # outer radius of radar/polar plot
_R_MIN = 18  # inner base radius: a score of 0 maps here (not the centre) so a single
# high attribute still draws a visible shape instead of collapsing to a point.
_PLOT_W = 520  # canvas width for radar/polar
_LABEL_GAP = 18  # distance from outer ring to axis labels
_GRID_LEVELS = (0.25, 0.5, 0.75, 1.0)

# --- Score colour bands (low / medium / high toxicity). ----------------------
_HIGH = 0.70
_MEDIUM = 0.40
_COLOR_HIGH = "#c62828"
_COLOR_MEDIUM = "#f9a825"
_COLOR_LOW = "#2e7d32"
_TRACK = "#e6e6e6"
_GRID = "#dddddd"
_TEXT = "#222222"
_SUBTLE = "#777777"
_FONT = 'font-family="system-ui, sans-serif"'


def _color(value: float) -> str:
    """Return the colour for a score in ``[0, 1]``."""
    if value >= _HIGH:
        return _COLOR_HIGH
    if value >= _MEDIUM:
        return _COLOR_MEDIUM
    return _COLOR_LOW


def _point(cx: float, cy: float, angle_deg: float, radius: float) -> tuple[float, float]:
    """Cartesian point at ``angle_deg`` (0 = east, clockwise) and ``radius``."""
    a = math.radians(angle_deg)
    return (cx + radius * math.cos(a), cy + radius * math.sin(a))


def _anchor(x: float, cx: float) -> str:
    """Pick a text-anchor so labels don't overlap the plot."""
    if x < cx - 5:
        return "end"
    if x > cx + 5:
        return "start"
    return "middle"


def _header(text: str, language: str | None) -> list[str]:
    """Title / snippet / language lines shared by every style."""
    snippet = text if len(text) <= 80 else text[:77] + "…"
    parts = [
        f'  <text x="{_PAD}" y="34" font-size="16" font-weight="700" fill="{_TEXT}" '
        f'{_FONT}>openspective</text>',
        f'  <text x="{_PAD}" y="56" font-size="13" fill="{_TEXT}" {_FONT}>'
        f'“{html.escape(snippet)}”</text>',
    ]
    if language:
        parts.append(
            f'  <text x="{_PAD}" y="72" font-size="11" fill="{_SUBTLE}" {_FONT}>'
            f'detected language: {html.escape(language)}</text>'
        )
    return parts


# --- Gauges ------------------------------------------------------------------


def _gauge(cx: float, cy: float, label: str, value: float) -> str:
    """Render a single donut gauge centred at ``(cx, cy)``."""
    circumference = 2 * math.pi * _RADIUS
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
          fill="{_TEXT}" {_FONT}>{pct}%</text>
    <text x="{cx}" y="{cy + _RADIUS + 22}" text-anchor="middle" font-size="11"
          fill="{_SUBTLE}" {_FONT}>{html.escape(label)}</text>
  </g>"""


def _gauges_body(items: list[tuple[str, float]]) -> tuple[int, int, list[str]]:
    """Render gauges sorted by score descending. Returns (width, height, parts)."""
    items = sorted(items, key=lambda kv: kv[1], reverse=True)
    count = max(1, len(items))
    cols = min(_COLS, count)
    rows = math.ceil(count / cols)
    width = cols * _CELL_W + 2 * _PAD
    height = _HEADER_H + rows * _CELL_H + _PAD
    parts = []
    for index, (label, value) in enumerate(items):
        col = index % cols
        row = index // cols
        cx = _PAD + col * _CELL_W + _CELL_W / 2
        cy = _HEADER_H + row * _CELL_H + _RADIUS + _STROKE
        parts.append(_gauge(cx, cy, label, value))
    return width, height, parts


# --- Radar -------------------------------------------------------------------


def _axis_angles(n: int) -> list[float]:
    """Evenly spaced angles starting at the top (-90 deg), going clockwise."""
    return [-90 + i * 360 / n for i in range(n)]


def _radar_radius(value: float) -> float:
    """Map a score in ``[0, 1]`` to a radius, offset by ``_R_MIN`` so a 0 still sits
    on a visible inner ring rather than the exact centre."""
    return _R_MIN + max(0.0, min(1.0, value)) * (_R - _R_MIN)


def _radar_body(items: list[tuple[str, float]]) -> tuple[int, int, list[str]]:
    """Render a single round radar/spider chart over all attributes."""
    n = len(items)
    width = _PLOT_W
    cx = width / 2
    cy = _HEADER_H + _R + 28
    height = int(cy + _R + 56)
    angles = _axis_angles(n)
    parts: list[str] = []

    # Round grid: concentric reference circles (incl. the inner base ring at 0%).
    parts.append(
        f'  <circle cx="{cx}" cy="{cy}" r="{_R_MIN:.1f}" fill="none" '
        f'stroke="{_GRID}" stroke-width="1"/>'
    )
    for level in _GRID_LEVELS:
        parts.append(
            f'  <circle cx="{cx}" cy="{cy}" r="{_radar_radius(level):.1f}" fill="none" '
            f'stroke="{_GRID}" stroke-width="1"/>'
        )
    # Axis spokes.
    for a in angles:
        x, y = _point(cx, cy, a, _R)
        parts.append(
            f'  <line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" '
            f'stroke="{_GRID}" stroke-width="1"/>'
        )

    # Data polygon. Colour by the headline TOXICITY score (or the max).
    headline = dict(items).get("TOXICITY", max(v for _, v in items))
    color = _color(headline)
    data_pts = " ".join(
        f"{x:.1f},{y:.1f}"
        for x, y in (
            _point(cx, cy, a, _radar_radius(v)) for a, (_, v) in zip(angles, items, strict=True)
        )
    )
    parts.append(
        f'  <polygon points="{data_pts}" fill="{color}" fill-opacity="0.30" '
        f'stroke="{color}" stroke-width="2"/>'
    )

    # Vertices + labels (attribute name on top line, percentage below).
    for a, (label, value) in zip(angles, items, strict=True):
        vx, vy = _point(cx, cy, a, _radar_radius(value))
        lx, ly = _point(cx, cy, a, _R + _LABEL_GAP)
        anchor = _anchor(lx, cx)
        parts.append(f'  <circle cx="{vx:.1f}" cy="{vy:.1f}" r="3" fill="{color}"/>')
        parts.append(
            f'  <text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" font-size="11" '
            f'font-weight="600" fill="{_TEXT}" {_FONT}>{html.escape(label)}</text>'
        )
        parts.append(
            f'  <text x="{lx:.1f}" y="{ly + 13:.1f}" text-anchor="{anchor}" font-size="10" '
            f'fill="{_SUBTLE}" {_FONT}>{round(value * 100)}%</text>'
        )
    return width, height, parts


# --- Polar area (rose) -------------------------------------------------------


def _polar_body(items: list[tuple[str, float]]) -> tuple[int, int, list[str]]:
    """Render a single polar-area (rose) chart: one wedge per attribute."""
    n = len(items)
    width = _PLOT_W
    cx = width / 2
    cy = _HEADER_H + _R + 28
    height = int(cy + _R + 56)
    angles = _axis_angles(n)
    half = 360 / n / 2
    parts: list[str] = []

    # Reference grid circles.
    for level in _GRID_LEVELS:
        parts.append(
            f'  <circle cx="{cx}" cy="{cy}" r="{_R * level:.1f}" fill="none" '
            f'stroke="{_GRID}" stroke-width="1"/>'
        )

    # One wedge per attribute, radius proportional to its score.
    for a, (label, value) in zip(angles, items, strict=True):
        radius = _R * max(0.0, min(1.0, value))
        x0, y0 = _point(cx, cy, a - half, radius)
        x1, y1 = _point(cx, cy, a + half, radius)
        color = _color(value)
        parts.append(
            f'  <path d="M {cx} {cy} L {x0:.1f} {y0:.1f} '
            f'A {radius:.1f} {radius:.1f} 0 0 1 {x1:.1f} {y1:.1f} Z" '
            f'fill="{color}" fill-opacity="0.55" stroke="{color}" stroke-width="1"/>'
        )
        lx, ly = _point(cx, cy, a, _R + _LABEL_GAP)
        anchor = _anchor(lx, cx)
        parts.append(
            f'  <text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" font-size="11" '
            f'font-weight="600" fill="{_TEXT}" {_FONT}>{html.escape(label)}</text>'
        )
        parts.append(
            f'  <text x="{lx:.1f}" y="{ly + 13:.1f}" text-anchor="{anchor}" font-size="10" '
            f'fill="{_SUBTLE}" {_FONT}>{round(value * 100)}%</text>'
        )
    return width, height, parts


# --- Dispatcher --------------------------------------------------------------

_BODIES = {"gauges": _gauges_body, "radar": _radar_body, "polar": _polar_body}


def render_svg(
    text: str, scores: dict[str, float], language: str | None = None, style: str = "gauges"
) -> str:
    """Render attribute ``scores`` as an SVG in the chosen ``style``.

    :param text: The analysed comment (shown, truncated, as the title).
    :param scores: Mapping of Perspective attribute name -> probability.
    :param language: Optional detected language code, shown as a subtitle.
    :param style: One of ``gauges`` (default), ``radar``, ``polar``.
    :returns: A complete ``<svg>...</svg>`` document string.
    """
    body_fn = _BODIES.get(style, _gauges_body)
    # Gauges sort internally; radar/polar keep the given (canonical) attribute order.
    items = list(scores.items())
    if not items:
        items = [("", 0.0)]
    width, height, body = body_fn(items)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="openspective scores">',
        f'  <rect width="{width}" height="{height}" fill="#ffffff"/>',
        *_header(text, language),
        *body,
        "</svg>",
    ]
    return "\n".join(parts)
