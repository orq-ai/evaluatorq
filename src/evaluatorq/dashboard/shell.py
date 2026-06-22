"""Dashboard HTML shell: ORQ-branded page wrapper.

``page(title, body_html, *, active_surface)`` wraps rendered body content in
the shared dashboard chrome (header with inline logo, nav bar, main element).
The CSS comes from ``evaluatorq.common.reports.load_css()`` so the report body
fragments look identical to their standalone exports.
"""

from __future__ import annotations

from evaluatorq.common.reports import load_css, load_logo_svg
from evaluatorq.dashboard.surfaces import ADAPTERS
from evaluatorq.dashboard.view import head_assets

# Surfaces that have a nav entry, in display order.
_NAV_SURFACES: list[tuple[str, str]] = [
    ("redteam", "Red Team"),
    ("sim", "Simulation"),
]


def _nav_html(active_surface: str | None = None) -> str:
    items: list[str] = []
    for key, label in _NAV_SURFACES:
        if key not in ADAPTERS:
            continue
        active_cls = ' class="active"' if key == active_surface else ""
        items.append(f'<li><a href="/"{active_cls}>{label}</a></li>')
    return f'<nav aria-label="Surface navigation"><ul>{"".join(items)}</ul></nav>'


def _header_html() -> str:
    logo = load_logo_svg()
    logo_html = f'<span class="nav-logo">{logo}</span>' if logo else ""
    return (
        f'<header class="dashboard-header">'
        f'<a href="/" class="brand-link">{logo_html}<span class="brand-name">evaluatorq</span></a>'
        f"</header>"
    )


def page(
    title: str,
    body_html: str,
    *,
    active_surface: str | None = None,
) -> str:
    """Render a complete HTML page in the dashboard shell.

    Args:
        title: ``<title>`` element text.
        body_html: Pre-rendered HTML fragment for the ``<main>`` body.
        active_surface: Surface key (``'redteam'`` | ``'sim'``) to highlight
            in the navigation bar, or ``None`` for the index page.

    Returns:
        A complete HTML document string starting with ``<!DOCTYPE html>``.
    """
    css = load_css()
    header = _header_html()
    nav = _nav_html(active_surface)
    # Render vendored JS script tags from head_assets() as plain HTML strings.
    scripts = "".join(str(a) for a in head_assets())

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title} — evaluatorq</title>\n"
        f"<style>\n{css}\n</style>\n"
        f"{scripts}\n"
        "</head>\n"
        "<body>\n"
        f"{header}\n"
        '<div class="dashboard-layout">\n'
        f"{nav}\n"
        '<main class="dashboard-main">\n'
        f"{body_html}\n"
        "</main>\n"
        "</div>\n"
        "</body>\n"
        "</html>\n"
    )
