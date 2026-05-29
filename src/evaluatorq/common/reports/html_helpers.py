"""HTML formatting helpers shared across report renderers.

Brand colors, the parameterized CSS loader, and small HTML primitives
(``esc``, ``html_table``, ``pct``, ``truncate``) live here. Chart helpers
that wrap Plotly + kaleido degrade gracefully when those packages are
unavailable.
"""

from __future__ import annotations

import html
from pathlib import Path
from string import Template
from typing import Any


# ---------------------------------------------------------------------------
# Brand colors
# ---------------------------------------------------------------------------

COLORS: dict[str, str] = {
    "orange_300": "#ff8f34",
    "teal_400": "#025558",
    "teal_500": "#01483d",
    "ink_700": "#25232e",
    "ink_800": "#1a1921",
    "sand_100": "#f9f8f6",
    "sand_400": "#e4e2df",
    "success_400": "#2ebd85",
    "yellow_400": "#f2b600",
    "red_400": "#d92d20",
    "blue_400": "#4fd2ff",
}

STATUS_COLORS: dict[str, str] = {
    "success": COLORS["success_400"],
    "warning": COLORS["yellow_400"],
    "failure": COLORS["red_400"],
}


# ---------------------------------------------------------------------------
# CSS loading
# ---------------------------------------------------------------------------

_CSS_CACHE: dict[Path, str] = {}


def load_css(css_path: Path | None = None) -> str:
    """Load and interpolate the shared report.css with the brand color palette.

    Uses ``string.Template`` (``$color_name``) so bare ``%`` characters in CSS
    (e.g. ``opacity: 50%``) don't raise ``ValueError`` like ``%``-formatting
    would.

    Args:
        css_path: Path to a ``.css`` file with ``$color_name`` placeholders.
            Defaults to the bundled ``common/reports/report.css``.

    Returns:
        The CSS text with brand colors substituted in.
    """
    path = css_path or Path(__file__).with_name("report.css")
    cached = _CSS_CACHE.get(path)
    if cached is not None:
        return cached
    text = Template(path.read_text(encoding="utf-8")).safe_substitute(COLORS)
    _CSS_CACHE[path] = text
    return text


# ---------------------------------------------------------------------------
# Small HTML primitives
# ---------------------------------------------------------------------------


def esc(text: str) -> str:
    """HTML-escape text."""
    return html.escape(str(text))


def pct(rate: float) -> str:
    """Format a float rate as a percentage string."""
    return f"{rate:.0%}"


def truncate(text: str, max_chars: int = 800) -> str:
    """Truncate long text with a plain-text marker (no Markdown)."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[truncated — full text in report JSON]"


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render an HTML table. Cell strings may contain inline HTML (e.g. badges)."""
    parts = ["<table>", "<thead><tr>"]
    parts.extend(f"<th>{esc(h)}</th>" for h in headers)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        parts.extend(f"<td>{cell}</td>" for cell in row)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Chart helpers (Plotly + kaleido optional)
# ---------------------------------------------------------------------------


def charts_available() -> bool:
    """Check whether plotly and kaleido are importable."""
    try:
        import kaleido  # noqa: F401
        import plotly  # noqa: F401
        return True
    except ImportError:
        return False


def try_render_svg(fig: Any) -> str | None:
    """Render a Plotly figure as inline SVG, or ``None`` on failure."""
    try:
        svg_bytes = fig.to_image(format="svg", engine="kaleido")
        return svg_bytes.decode("utf-8") if isinstance(svg_bytes, bytes) else svg_bytes
    except Exception:
        return None


def render_donut_chart(
    *,
    labels: list[str],
    values: list[int],
    colors: list[str],
    title: str,
) -> str:
    """Render a donut chart and wrap the SVG in a ``chart-container`` div.

    Returns an empty string when Plotly is unavailable, all values are zero,
    or the SVG render fails.
    """
    if not charts_available():
        return ""

    filtered = [(lbl, v, c) for lbl, v, c in zip(labels, values, colors, strict=False) if v > 0]
    if not filtered:
        return ""
    labels_f, values_f, colors_f = zip(*filtered, strict=False)

    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Pie(
        labels=list(labels_f),
        values=list(values_f),
        hole=0.5,
        marker=dict(colors=list(colors_f)),
        textinfo="label+percent",
        textfont=dict(size=12),
    )])
    fig.update_layout(
        width=400, height=300,
        margin=dict(t=30, b=30, l=30, r=30),
        showlegend=False,
        title=dict(text=title, font=dict(size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    svg = try_render_svg(fig)
    return f'<div class="chart-container">{svg}</div>' if svg else ""


def render_horizontal_bar_chart(
    *,
    labels: list[str],
    values: list[float],
    color: str,
    title: str,
    x_title: str,
    value_suffix: str = "",
) -> str:
    """Render a horizontal bar chart with values displayed outside the bars."""
    if not charts_available() or not labels:
        return ""

    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker_color=color,
        text=[f"{v:.0f}{value_suffix}" for v in values],
        textposition="outside",
    )])
    fig.update_layout(
        width=500, height=max(250, len(labels) * 35 + 80),
        margin=dict(t=40, b=40, l=80, r=50),
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=x_title,
        xaxis=dict(range=[0, max(max(values) * 1.2, 5) if values else 100]),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    svg = try_render_svg(fig)
    return f'<div class="chart-container">{svg}</div>' if svg else ""
