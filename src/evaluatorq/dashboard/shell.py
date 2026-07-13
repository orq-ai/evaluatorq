"""Dashboard HTML shell: ORQ-branded sidebar layout.

``page(title, body_html, *, active_surface, active_nav, actions_html)`` wraps
rendered body content in the shared dashboard chrome — a left sidebar (brand +
nav) and a main column (topbar + content), matching the v1 design's
``Shell.jsx``.

The CSS layers, in cascade order, are:

1. ``load_css()`` (``common.reports``) — brand tokens + report-body styles, so
   embedded report fragments look identical to their standalone exports.
2. ``EDITORIAL_CSS`` (``theme``) — the v1 editorial skin tokens for the chrome.
3. ``DASHBOARD_CSS`` (``styles``) — the shell / landing / run-list rules that
   consume those tokens.
"""

from __future__ import annotations

import base64
from pathlib import Path

from evaluatorq.common.reports import esc, load_css
from evaluatorq.dashboard.styles import DASHBOARD_CSS
from evaluatorq.dashboard.theme import EDITORIAL_CSS
from evaluatorq.dashboard.view import head_assets

# Sidebar collapse: runs at body-top so the class lands on <html> before the
# sidebar paints (no flash). State persists in localStorage across the full-page
# navigations between reports. ponytail: inline over a /static file so it works
# in tests and static exports too.
_SIDEBAR_TOGGLE_SCRIPT = (
    '<script>'
    "(function(){try{if(localStorage.getItem('eq-sidebar-collapsed')==='1')"
    "document.documentElement.classList.add('sidebar-collapsed');}catch(e){}})();"
    'function eqToggleSidebar(){'
    "var c=document.documentElement.classList.toggle('sidebar-collapsed');"
    "try{localStorage.setItem('eq-sidebar-collapsed',c?'1':'0');}catch(e){}}"
    "document.addEventListener('keydown',function(e){"
    "if((e.metaKey||e.ctrlKey)&&!e.shiftKey&&!e.altKey&&e.key.toLowerCase()==='b')"
    '{e.preventDefault();eqToggleSidebar();}});'
    # Non-Mac shows "Ctrl B" instead of the ⌘B glyph on the hotkey hint.
    'if(!/Mac|iPhone|iPad/.test(navigator.platform)){'
    "document.addEventListener('DOMContentLoaded',function(){"
    "document.querySelectorAll('.sidebar-toggle-kbd').forEach("
    "function(k){k.textContent='Ctrl B';});});}"
    '</script>\n'
)

# The v1 brand mark (orq ink-nodes logomark), vendored from the design system.
# Inlined rather than served via /static/ so it renders in tests and exports too.
_MARK_PATH = Path(__file__).parent / 'static' / 'orq-mark.svg'
_mark_cache: str | None = None


def _load_mark() -> str:
    global _mark_cache
    if _mark_cache is None:
        try:
            _mark_cache = _MARK_PATH.read_text(encoding='utf-8')
        except OSError:
            _mark_cache = ''
    return _mark_cache


# Green orq favicon, inlined as a base64 data-URI <link> so it works live, in
# tests, and in static exports without depending on the /static/ route.
_FAVICON_PATH = Path(__file__).parent / 'static' / 'orq-favicon.svg'
_favicon_cache: str | None = None


def _favicon_link() -> str:
    global _favicon_cache
    if _favicon_cache is None:
        try:
            svg = _FAVICON_PATH.read_bytes()
            b64 = base64.b64encode(svg).decode('ascii')
            _favicon_cache = f'<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,{b64}">\n'
        except OSError:
            _favicon_cache = ''
    return _favicon_cache


# Sidebar nav: (key, label, href, inline-SVG icon path data).  Keys match the
# ``active_nav`` resolution below; hrefs reuse the existing index routes so the
# run lists stay at ``/?surface=…``.
_NAV: list[tuple[str, str, str, str]] = [
    (
        'dashboard',
        'Dashboard',
        '/',
        (
            '<rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/>'
            '<rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/>'
        ),
    ),
    (
        'redteam',
        'Red Team',
        '/?surface=redteam',
        '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M12 8v4"/><path d="M12 16h.01"/>',
    ),
    (
        'sim',
        'Agent Sim',
        '/?surface=sim',
        '<path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>',
    ),
    (
        'settings',
        'Settings',
        '/settings',
        '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
    ),
]


def _icon(path_data: str) -> str:
    return (
        '<svg class="nav-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f'{path_data}</svg>'
    )


def _sidebar_html(active_nav: str) -> str:
    # The v1 brand lockup is the orq mark (not the full wordmark) + the
    # "evaluatorq" wordmark with an orange "q".
    mark = _load_mark()
    logo_html = f'<span class="nav-mark">{mark}</span>' if mark else ''
    items: list[str] = []
    for key, label, href, icon in _NAV:
        active = ' active' if key == active_nav else ''
        items.append(f'<a class="nav-item{active}" href="{href}">{_icon(icon)}<span>{esc(label)}</span></a>')
    return (
        '<aside class="app-sidebar">'
        f'<a class="app-brand" href="/">{logo_html}'
        '<span class="brand-name">evaluator<span class="brand-q">q</span></span></a>'
        f'<nav class="app-nav" aria-label="Primary">{"".join(items)}</nav>'
        '<button class="sidebar-toggle" type="button" aria-label="Toggle sidebar"'
        ' onclick="eqToggleSidebar()">'
        '<svg class="nav-icon" width="16" height="16" viewBox="0 0 24 24" fill="none"'
        ' stroke="currentColor" stroke-width="2" stroke-linecap="round"'
        ' stroke-linejoin="round"><polyline points="15 18 9 12 15 6"/></svg>'
        '<span>Collapse</span>'
        '<kbd class="sidebar-toggle-kbd">⌘ B</kbd></button>'
        '</aside>'
    )


def _resolve_nav(active_surface: str | None, active_nav: str | None) -> str:
    if active_nav:
        return active_nav
    if active_surface in ('redteam', 'sim'):
        return active_surface
    return 'dashboard'


def page(
    title: str,
    body_html: str,
    *,
    active_surface: str | None = None,
    active_nav: str | None = None,
    actions_html: str = '',
    back_html: str = '',
) -> str:
    """Render a complete HTML page in the dashboard sidebar shell.

    Args:
        title: ``<title>`` text and the topbar heading.
        body_html: Pre-rendered HTML fragment for the ``<main>`` body.
        active_surface: Surface key (``'redteam'`` | ``'sim'``) for the report
            view, used to highlight the matching nav item.
        active_nav: Explicit nav key (``'dashboard'`` | ``'redteam'`` | ``'sim'``
            | ``'settings'``) overriding the surface-derived default.
        actions_html: Optional pre-rendered HTML for the topbar action area
            (e.g. export buttons on a report view).

    Returns:
        A complete HTML document string starting with ``<!DOCTYPE html>``.
    """
    css = load_css()
    nav_key = _resolve_nav(active_surface, active_nav)
    sidebar = _sidebar_html(nav_key)
    scripts = ''.join(str(a) for a in head_assets())
    # On report pages the run name is the hero H1, so the topbar carries the
    # back link instead of repeating the title.
    topbar_lead = back_html or f'<h1 class="app-title">{esc(title)}</h1>'

    return (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'{_favicon_link()}'
        f'<title>{esc(title)} — evaluatorq</title>\n'
        f'<style>\n{css}\n</style>\n'
        f'<style>\n{EDITORIAL_CSS}\n</style>\n'
        f'<style>\n{DASHBOARD_CSS}\n</style>\n'
        f'{scripts}\n'
        '</head>\n'
        '<body class="eq-dashboard">\n'
        f'{_SIDEBAR_TOGGLE_SCRIPT}'
        '<div class="app-shell">\n'
        f'{sidebar}\n'
        '<div class="app-main">\n'
        '<header class="app-topbar">\n'
        f'{topbar_lead}\n'
        f'<div class="app-actions">{actions_html}</div>\n'
        '</header>\n'
        '<main class="app-content">\n'
        f'{body_html}\n'
        '</main>\n'
        '</div>\n'
        '</div>\n'
        '</body>\n'
        '</html>\n'
    )
