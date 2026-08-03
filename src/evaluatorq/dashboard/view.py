"""View helpers: HTML fragments and FastHTML header assets for the dashboard.

``head_assets()`` returns a tuple of FastHTML header elements (Script/Link
tags) for the vendored JS (htmx, vega trio, dashboard.js) under ``/static/``,
served by the app from ``dashboard/static/``.

``landing_body(data)`` renders the combined Dashboard screen; ``runs_screen_body``
renders the per-kind run lists; ``settings_body`` / ``report_not_found`` render
the remaining pure-HTML fragments injected into the shell via ``shell.page()``.

``render_filter_form(rid, surface, opts, selections)`` renders the HTMX
filter sidebar form.  ``filter_fragment(rid, surface, body_html, form_html)``
combines a re-rendered body and a re-rendered form into the HTMX swap target.

``render_message_list(messages, *, role_labels, class_prefix)`` renders a
role-labeled message list as a series of ``<div>`` elements.  Shared by
``sim_views`` (and any other surface that uses the simple flat layout).
"""

from __future__ import annotations

from datetime import datetime
from itertools import starmap
from typing import TYPE_CHECKING, Any

from fasthtml.common import Script

from evaluatorq.common.reports import esc
from evaluatorq.simulation.metrics import TURN_METRICS

if TYPE_CHECKING:
    from evaluatorq.dashboard.library import ReportCard
    from evaluatorq.dashboard.metrics import Landing, RedTeamOverview, RunRow, SimOverview

# Surface key → display label, used for run-list titles + kind badges.
SURFACE_LABELS: dict[str, str] = {'redteam': 'Red Team', 'sim': 'Agent Sim'}

# Allow-listed run-overview page sizes (first entry is the default). Shared with
# app.py so the query parser and the size picker agree.
RUN_PAGE_SIZES: tuple[int, ...] = (8, 15, 25)


def head_assets() -> tuple[Script, ...]:
    """Return FastHTML header elements for vendored JS assets.

    The ``/static/`` files are vendored under ``dashboard/static/`` and served
    by the app; these Script tags wire them into every report page.
    """
    return (
        Script(src='/static/htmx.min.js'),
        Script(src='/static/vega.min.js'),
        Script(src='/static/vega-lite.min.js'),
        Script(src='/static/vega-embed.min.js'),
        # defer: dashboard.js runs at document.body-level (event delegation) — without
        # defer it executes during <head> parse when document.body is still null and
        # throws, aborting all its handlers (incl. the sim-entity modal). vega stays
        # non-deferred: inline body scripts call vegaEmbed during parse.
        Script(src='/static/dashboard.js', defer=True),
    )


# ---------------------------------------------------------------------------
# Combined landing + run lists (the Dashboard / Red Team / Agent Sim screens)
# ---------------------------------------------------------------------------


def _score_cls(score: float | None) -> str:
    if score is None:
        return 'none'
    return 'good' if score >= 0.8 else 'warn'


def _fmt_score(score: float | None) -> str:
    return '—' if score is None else f'{score:.2f}'


def _kind_badge(surface: str) -> str:
    label = SURFACE_LABELS.get(surface, surface)
    return f'<span class="kind-badge {esc(surface)}">{esc(label)}</span>'


def _status_badge(status: str) -> str:
    # Map lifecycle status → an existing pill color class (finished→passed green,
    # error→failed red) so the run-list badges stay styled.
    label, cls = {
        'finished': ('Finished', 'passed'),
        'completed': ('Finished', 'passed'),
        'error': ('Error', 'failed'),
        'running': ('Running', 'warning'),
        'cancelled': ('Cancelled', 'cancelled'),
    }.get(status, (status.title(), status))
    return f'<span class="status-badge {esc(cls)}"><span class="dot"></span>{esc(label)}</span>'


def _run_row(row: RunRow, *, show_badge: bool = True) -> str:
    err = '<span class="card-error" title="failed to load">error</span>' if row.error else ''
    # The surface badge disambiguates mixed lists (the combined dashboard). In a
    # per-surface run overview every row shares the surface, so it carries no
    # information — omit it there.
    badge = _kind_badge(row.surface) if show_badge else ''
    return (
        f'<a class="run-row" href="/r/{esc(row.id)}">'
        f'<span class="run-id">'
        f'<span class="run-name-line"><span class="run-name">{esc(row.name)}</span>{badge}{err}</span>'
        f'<span class="run-meta">{esc(row.headline)} · {esc(row.when)}</span>'
        f'</span>'
        f'<span class="run-score {_score_cls(row.score)}">{_fmt_score(row.score)}</span>'
        f'{_status_badge(row.status)}'
        f'</a>'
    )


# Surface → inline glyph (lucide: shield-alert / messages-square) for the Type column.
_SURFACE_ICONS: dict[str, str] = {
    'redteam': (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 '
        '1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 '
        '3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="M12 8v4M12 16h.01"/></svg>'
    ),
    'sim': (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M14 9a2 2 0 0 1-2 2H6l-4 4V4a2 2 0 0 1 '
        '2-2h8a2 2 0 0 1 2 2z"/><path d="M18 9h2a2 2 0 0 1 2 2v11l-4-4h-6a2 2 0 0 1-2-2v-1"/></svg>'
    ),
}


def _type_cell(surface: str) -> str:
    """Type column: surface glyph + label (replaces the inline kind bubble)."""
    icon = _SURFACE_ICONS.get(surface, '')
    label = SURFACE_LABELS.get(surface, surface)
    return f'<span class="type-cell {esc(surface)}">{icon}{esc(label)}</span>'


def _recent_runs_table(rows: list[RunRow]) -> str:
    """Landing 'Recent runs' as an airy column list (not a boxed table):
    Type · Job · Cases · Time · Score · Status, aligned but borderless."""
    from evaluatorq.common.reports.html_helpers import status_badge

    head = (
        '<div class="rr-head"><span>Type</span><span>Job</span><span>Cases</span>'
        '<span>Time</span><span>Score</span><span>Status</span></div>'
    )
    row_html: list[str] = [head]
    for r in rows:
        label, status = _LIFECYCLE_PILL.get(r.status, (r.status.title(), 'neutral'))
        err = '<span class="card-error" title="failed to load">error</span>' if r.error else ''
        row_html.append(
            f'<a class="rr-row" href="/r/{esc(r.id)}">'
            f'<span class="rr-type">{_type_cell(r.surface)}</span>'
            f'<span class="rr-job">{esc(r.name)}{err}</span>'
            f'<span class="rr-meta">{esc(r.headline)}</span>'
            f'<span class="rr-meta">{esc(r.when)}</span>'
            f'<span class="run-score {_score_cls(r.score)}">{_fmt_score(r.score)}</span>'
            f'{status_badge(label, status)}'
            f'</a>'
        )
    return f'<div class="recent-runs">{"".join(row_html)}</div>'


def _stat_tile(label: str, value: str, unit: str = '') -> str:
    unit_html = f'<span class="stat-unit">{esc(unit)}</span>' if unit else ''
    return (
        f'<div class="stat-tile"><div class="stat-label">{esc(label)}</div>'
        f'<div class="stat-value">{esc(value)}{unit_html}</div></div>'
    )


def _panel(title: str, sub: str, inner: str, *, cls: str = '') -> str:
    cls_attr = f' {cls}' if cls else ''
    return (
        f'<div class="panel{cls_attr}"><div class="panel-title">{esc(title)}</div>'
        f'<div class="panel-sub">{esc(sub)}</div>{inner}</div>'
    )


def _bars(rows: list[tuple[str, float]], colors: list[str], *, total_label: str = 'Total', fmt: str = '{}') -> str:
    total = sum(v for _, v in rows) or 1
    parts: list[str] = ['<div class="bars">']
    for i, (name, val) in enumerate(rows):
        pct = round(val / total * 100)
        color = colors[i % len(colors)]
        parts.append(
            f'<div class="bar-row"><div class="bar-head">'
            f'<span class="bar-name">{esc(name)}</span>'
            f'<span class="bar-val">{esc(fmt.format(val))} <span class="bar-pct">· {pct}%</span></span>'
            f'</div><div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{color}"></div></div></div>'
        )
    parts.append(
        f'<div class="bars-total"><span class="t-label">{esc(total_label)}</span>'
        f'<span class="t-val">{esc(fmt.format(sum(v for _, v in rows)))}</span></div></div>'
    )
    return ''.join(parts)


def landing_body(data: Landing) -> str:
    """Render the combined Dashboard landing as an HTML fragment."""
    if data.total_runs == 0:
        return (
            '<section class="dash-wrap"><div class="runs-empty">'
            'No reports found. Run a red team or simulation job to generate reports.'
            '</div></section>'
        )

    total_tokens = sum(n for _, n in data.tokens_by_kind)
    avg_cost = data.total_cost / data.total_runs if data.total_runs else 0.0
    band = (
        '<div class="stat-band">'
        + _stat_tile('Jobs run', str(data.total_runs))
        + _stat_tile('Avg cost / job', _fmt_cost(avg_cost))
        + _stat_tile('Total spend', _fmt_cost(data.total_cost))
        + _stat_tile('Total tokens', _fmt_compact(total_tokens))
        + '</div>'
    )

    # One colour per surface kind; _bars cycles, so a third kind would
    # otherwise repeat the first one's colour.
    teal_jade = ['var(--chart-1)', 'var(--chart-2)', 'var(--chart-3)']
    severity_colors = ['var(--red-600)', 'var(--orange-500)', 'var(--amber-600)', 'var(--green-600)']

    # Derived from the kinds actually present, so the subtitle cannot claim a
    # surface the panel is not showing (or omit one it is).
    by_kind_sub = ' · '.join(k.lower() for k, _ in data.by_kind) or 'no runs yet'
    by_kind_panel = _panel('Runs by type', by_kind_sub, _bars(data.by_kind, teal_jade))

    sev_rows = [(s.title(), n) for s, n in data.severity]
    sev_inner = _bars(sev_rows, severity_colors) if sev_rows else '<p class="rt-panel-loading">No findings.</p>'
    severity_panel = _panel('Findings by severity', 'Vulnerabilities found', sev_inner)
    # Spend by job type — real dollars (cost_usd recorded upstream).
    spend_inner = (
        _bars(data.cost_by_kind, teal_jade, fmt='${:.4f}')
        if data.cost_by_kind
        else '<p class="rt-panel-loading">No cost recorded.</p>'
    )
    spend_panel = _panel('Spend by job type', 'Real cost across runs', spend_inner)

    tok_inner = (
        _bars(data.tokens_by_kind, teal_jade, fmt='{:,}')
        if data.tokens_by_kind
        else '<p class="rt-panel-loading">No token usage recorded.</p>'
    )
    tokens_panel = _panel('Token usage', 'By job type', tok_inner)

    # 2x2 grid of the four remaining panels.
    grid = (
        f'<div class="dash-row-eq">{by_kind_panel}{severity_panel}</div>'
        f'<div class="dash-row-eq">{spend_panel}{tokens_panel}</div>'
    )

    recent_panel = _panel('Recent runs', 'Latest jobs', _recent_runs_table(data.recent))

    return f'<section class="dash-wrap">{band}{grid}{recent_panel}</section>'


_OUTCOME_PILL: dict[str, tuple[str, str]] = {
    'passed': ('Passed', 'pass'),
    'warning': ('Warning', 'warn'),
    'failed': ('Failed', 'fail'),
    'error': ('Error', 'warn'),
}

# Run lifecycle → (label, pill-kind) for the Status column. Score carries quality;
# Status carries whether the run completed. 'running' reserved for future live jobs.
_LIFECYCLE_PILL: dict[str, tuple[str, str]] = {
    'finished': ('Finished', 'pass'),
    'error': ('Error', 'fail'),
    'running': ('Running', 'warn'),
    'cancelled': ('Cancelled', 'neutral'),
}


def _fmt_cost(v: float) -> str:
    """Format a dollar amount: cents-precision for readable sums, more digits
    for the sub-cent per-item costs typical of a single sim/attack."""
    if v >= 1:
        return f'${v:,.2f}'
    if v > 0:
        return f'${v:.4f}'
    return '$0.00'


# Inline target-kind glyphs (lucide: bot / cpu / rocket), sized to the pill.
_TARGET_ICONS: dict[str, str] = {
    'agent': (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4M8 2h8"/>'
        '<rect x="4" y="8" width="16" height="12" rx="2"/><path d="M2 14h2M20 14h2"/>'
        '<path d="M9 13v2M15 13v2"/></svg>'
    ),
    'llm': (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M9.94 15.5A2 2 0 0 0 8.5 14.06l-6.14-1.58a.5.5 '
        '0 0 1 0-.96L8.5 9.94A2 2 0 0 0 9.94 8.5l1.58-6.14a.5.5 0 0 1 .96 0L14.06 8.5A2 2 0 0 0 15.5 '
        '9.94l6.14 1.58a.5.5 0 0 1 0 .96L15.5 14.06a2 2 0 0 0-1.44 1.44l-1.58 6.14a.5.5 0 0 1-.96 0z"/>'
        '<path d="M20 3v4M22 5h-4M4 17v2M5 18H3"/></svg>'
    ),
    'deployment': (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 '
        '5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="M12 15l-3-3a22 22 0 0 1 '
        '2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/>'
        '<path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5z"/></svg>'
    ),
    'external': (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/>'
        '<path d="M3 12h18M12 3a14.9 14.9 0 0 1 0 18M12 3a14.9 14.9 0 0 0 0 18"/></svg>'
    ),
}


_CHEVRON = (
    '<svg class="row-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18l6-6-6-6"/></svg>'
)


def _target_pill(label: str, kind: str) -> str:
    """Target cell rendered exclusively from the saved run's target kind.

    Older JSON and extensions may carry raw storage kinds rather than the
    display categories emitted by metrics, so normalize here too. Unknown
    targets are external entities, not green status dots.
    """
    display_kind = {
        'orq_agent': 'agent',
        'orq_deployment': 'deployment',
        'openai_model': 'llm',
        'vercel': 'deployment',
        'callback': 'external',
        'other': 'external',
    }.get(kind, kind if kind in _TARGET_ICONS else 'external')
    return f'<span class="target-pill" data-kind="{esc(display_kind)}">{_TARGET_ICONS[display_kind]}{esc(label)}</span>'


# Run lifecycle → label for the solid status pill (mirrors the platform's
# success/error spans). 'finished' reads as 'success'.
_STATUS_LABEL: dict[str, str] = {'finished': 'success', 'error': 'error', 'running': 'running'}


def _run_status_pill(status: str) -> str:
    """Solid filled status pill matching the main platform (green success /
    red error, white lowercase text)."""
    return f'<span class="run-status {esc(status)}">{esc(_STATUS_LABEL.get(status, status))}</span>'


def _run_grid(rows: list[Any]) -> str:
    """Run-level table as a clickable anchor-grid — the WHOLE row links to the
    run's report. Borderless (no boxed table chrome), aligned columns.
    Shared by the Red Team + Agent Sim surfaces (identical row shape)."""
    head = (
        '<div class="runs-grid-head"><span>Job</span><span>Target</span><span>Status</span>'
        '<span>Score</span><span>Cases</span><span>Cost</span><span></span></div>'
    )
    parts: list[str] = [head]
    for r in rows:
        err = '<span class="card-error" title="failed to load">error</span>' if r.error else ''
        parts.append(
            f'<a class="runs-grid-row" href="/r/{esc(r.rid)}">'
            f'<span class="rg-job"><span class="rg-name">{esc(r.name)}{err}</span>'
            f'<span class="rg-sub">{esc(_ago(r.when))}</span></span>'
            f'<span class="rg-targets">{"".join(starmap(_target_pill, r.targets))}</span>'
            f'<span>{_run_status_pill(r.status)}</span>'
            f'<span class="run-score {_score_cls(r.score)}">{_fmt_score(r.score)}</span>'
            f'<span class="rg-num">{esc(str(r.cases))}</span>'
            f'<span class="rg-num">{esc(_fmt_cost(r.cost))}</span>'
            f'{_CHEVRON}'
            f'</a>'
        )
    return f'<div class="runs-grid">{"".join(parts)}</div>'


def _run_pager(surface: str, page: int, total: int, per_page: int) -> str:
    """Prev/Next page nav + page-size picker for the run overview. Full-page GET
    links (no JS), shared by both surfaces. Renders just the size picker when
    everything fits on one page."""
    last = max(1, -(-total // per_page))  # ceil division
    page = min(page, last)

    def link(label: str, *, pg: int, pp: int, disabled: bool = False, active: bool = False) -> str:
        cls = 'runs-pager-link'
        if active:
            cls += ' is-active'
        if disabled:
            return f'<span class="{cls} is-disabled">{label}</span>'
        return f'<a class="{cls}" href="/?surface={esc(surface)}&page={pg}&per_page={pp}">{label}</a>'

    sizes = ''.join(link(str(sz), pg=1, pp=sz, active=(sz == per_page)) for sz in RUN_PAGE_SIZES)
    left = f'<span class="runs-pager-sizes">Per page: {sizes}</span>'

    if total <= per_page:
        count = f'<span class="runs-pager-count">Total {total} runs</span>'
        return f'<div class="runs-pager">{left}<span class="runs-pager-right">{count}</span></div>'

    lo = (page - 1) * per_page + 1
    hi = min(page * per_page, total)
    count = f'<span class="runs-pager-count">{lo}&ndash;{hi} of {total}</span>'
    nav = link('&lsaquo; Prev', pg=page - 1, pp=per_page, disabled=page <= 1) + link(
        'Next &rsaquo;', pg=page + 1, pp=per_page, disabled=page >= last
    )
    return (
        f'<div class="runs-pager">{left}'
        f'<span class="runs-pager-right">{count}<span class="runs-pager-nav">{nav}</span></span></div>'
    )


def _ago(when: datetime) -> str:
    """Relative time: '46m ago', '7h ago', '3d ago'."""
    now = datetime.now(tz=when.tzinfo) if when.tzinfo else datetime.now()  # noqa: DTZ005
    secs = max(0, int((now - when).total_seconds()))
    if secs < 3600:
        return f'{secs // 60}m ago'
    if secs < 86400:
        return f'{secs // 3600}h ago'
    return f'{secs // 86400}d ago'


def _fmt_compact(n: int) -> str:
    """Compact large counts: 1_157_563 -> '1.2M', 4_200 -> '4.2K'."""
    if n >= 1_000_000:
        return f'{n / 1_000_000:.1f}M'
    if n >= 1_000:
        return f'{n / 1_000:.1f}K'
    return str(n)


def sim_overview_body(data: SimOverview) -> str:
    """Render the Agent Sim surface as the design's rich overview: 4 KPI cards
    plus an item-level 'Recent simulations' table (RES-1022)."""
    from evaluatorq.common.reports.html_helpers import kpi_cards, pct

    if data.simulations_run == 0:
        return runs_screen_body([], 'sim')

    gc = '—' if data.goal_completion is None else pct(data.goal_completion)
    gc_status = (
        'neutral'
        if data.goal_completion is None
        else ('pass' if data.goal_completion >= 0.8 else 'warn' if data.goal_completion >= 0.5 else 'fail')
    )
    avg_turns = '—' if data.avg_turns is None else f'{data.avg_turns:.1f}'
    # Real per-sim dollar cost (cost_usd is recorded upstream). Falls back to
    # avg tokens/sim only when no cost data is present.
    if data.avg_cost is not None:
        cost_label, cost_value = 'Avg cost/sim', _fmt_cost(data.avg_cost)
    else:
        cost_label = 'Avg tokens/sim'
        cost_value = '—' if data.avg_tokens is None else f'{data.avg_tokens:,.0f}'
    band = kpi_cards([
        {'label': 'Simulations run', 'value': str(data.simulations_run)},
        {'label': 'Goal completion', 'value': gc, 'status': gc_status},
        {'label': 'Avg turns', 'value': avg_turns},
        {'label': cost_label, 'value': cost_value},
    ])

    panel = _panel(
        'Recent runs',
        'Latest agent simulation runs',
        _run_grid(data.recent) + _run_pager('sim', data.page, data.total_runs, data.per_page),
    )
    return f'<section class="dash-wrap">{band}{panel}</section>'


def search_results(cards: list[ReportCard], query: str) -> str:
    """Render the ⌘K global-search dropdown fragment: report links whose name or
    surface matches ``query`` (case-insensitive). Empty query → empty."""
    q = query.strip().lower()
    if not q:
        return ''
    hits = [c for c in cards if q in c.name.lower() or q in SURFACE_LABELS.get(c.surface, c.surface).lower()]
    if not hits:
        return '<div class="search-empty">No matching reports.</div>'
    items = []
    for c in hits[:10]:
        label = SURFACE_LABELS.get(c.surface, c.surface)
        items.append(
            f'<a class="search-hit" href="/r/{esc(c.id)}">'
            f'<span class="search-hit-kind">{esc(label)}</span>'
            f'<span class="search-hit-name">{esc(c.name)}</span></a>'
        )
    return ''.join(items)


def redteam_overview_body(data: RedTeamOverview) -> str:
    """Render the Red Team surface as the design's rich overview: 4 KPI cards
    plus an item-level 'Recent attacks' table (RES-1021)."""
    from evaluatorq.common.reports.html_helpers import kpi_cards, pct

    if data.attacks_run == 0:
        return runs_screen_body([], 'redteam')

    asr = '—' if data.break_rate is None else pct(data.break_rate)
    asr_status = (
        'neutral'
        if data.break_rate is None
        else ('fail' if data.break_rate >= 0.25 else 'warn' if data.break_rate > 0 else 'pass')
    )
    band = kpi_cards([
        {'label': 'Attacks run', 'value': str(data.attacks_run)},
        {'label': 'ASR', 'value': asr, 'status': asr_status},
        {
            'label': 'Critical findings',
            'value': str(data.critical_findings),
            'status': 'fail' if data.critical_findings else 'pass',
        },
        {'label': 'Total spend', 'value': _fmt_cost(data.total_cost)},
    ])

    panel = _panel(
        'Recent runs',
        'Latest red team runs',
        _run_grid(data.recent) + _run_pager('redteam', data.page, data.total_runs, data.per_page),
    )
    return f'<section class="dash-wrap">{band}{panel}</section>'


def runs_screen_body(rows: list[RunRow], surface: str) -> str:
    """Render a per-kind run-list screen (Red Team / Agent Sim)."""
    label = SURFACE_LABELS.get(surface, 'Reports')
    if not rows:
        return (
            '<section class="runs-screen"><div class="runs-card">'
            '<div class="runs-empty">No reports found for this surface. '
            f'Run a {esc(label.lower())} job to generate one.</div>'
            '</div></section>'
        )
    head = '<div class="runs-head"><span>Job</span><span>Score</span><span>Status</span><span></span></div>'
    body = ''.join(_run_row(r, show_badge=False) for r in rows)
    return (
        f'<section class="runs-screen"><div class="runs-card">{head}<div class="run-list">{body}</div></div></section>'
    )


_ARROW_LEFT = (
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"'
    ' stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="m12 19-7-7 7-7"/><path d="M19 12H5"/></svg>'
)
_DOWNLOAD_ICON = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"'
    ' stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
    '<path d="M7 10l5 5 5-5"/><path d="M12 15V3"/></svg>'
)


def report_back_link(surface: str) -> str:
    """Render the 'back to run list' link shown above a report (matches v1)."""
    target = f'/?surface={esc(surface)}' if surface in ('redteam', 'sim', 'pairwise') else '/'
    label = {'redteam': 'Red team runs', 'sim': 'Agent sim runs', 'pairwise': 'Pairwise runs'}.get(surface, 'All runs')
    return f'<a class="report-back" href="{target}">{_ARROW_LEFT} {esc(label)}</a>'


def report_actions(rid: str) -> str:
    """Render the topbar action area for a report view (Export)."""
    return f'<a class="btn-secondary" href="/r/{esc(rid)}/export.html">{_DOWNLOAD_ICON} Export</a>'


def settings_body(config: list[tuple[str, str | list[str]]]) -> str:
    """Render the Settings screen: read-only runtime configuration (run stores,
    default model, API-key presence, Orq host/workspace).

    There's nothing editable here anymore — deep-links derive host + workspace
    per-run from each run's ``experiment_url``; the host/workspace rows are just
    the env fallback for runs with no experiment.
    """

    def val_html(v: str | list[str]) -> str:
        # A list renders one item per line; a scalar is a single line.
        items = v if isinstance(v, list) else [v]
        return ''.join(f'<span class="config-val-item">{esc(item)}</span>' for item in items)

    rows = ''.join(
        f'<div class="config-row"><span class="config-key">{esc(k)}</span>'
        f'<span class="config-val">{val_html(v)}</span></div>'
        for k, v in config
    )
    config_panel = _panel('Configuration', 'What this dashboard is reading', f'<div class="config-list">{rows}</div>')
    return f'<section class="dash-wrap">{config_panel}</section>'


def report_not_found(rid: str) -> str:
    """Render a 404-style body fragment for an unknown report ID."""
    return (
        '<section class="report-not-found">'
        '<h1>Report not found</h1>'
        f'<p>No report with id <code>{esc(rid)}</code> could be located.</p>'
        '<p><a href="/">Back to reports</a></p>'
        '</section>'
    )


def report_in_flight(manifest: Any) -> str:
    """Render a minimal status page for an in-flight run (no report on disk yet).

    Kept deliberately small: an in-flight run has no results to show, so this
    surfaces the run's lifecycle status, current stage, and per-stage timing
    from the manifest rather than a full transcript view.
    """
    status = manifest.status.value if hasattr(manifest.status, 'value') else str(manifest.status)
    label = SURFACE_LABELS.get(manifest.surface.value, manifest.surface.value)
    stage = manifest.stage or '—'
    rows: list[str] = []
    for rec in manifest.stages:
        st = rec.status.value if hasattr(rec.status, 'value') else str(rec.status)
        dur = f'{rec.duration_seconds:.1f}s' if rec.duration_seconds is not None else '…'
        target = f' ({esc(rec.target)})' if rec.target else ''
        rows.append(
            f'<div class="config-row"><span class="config-key">{esc(rec.name)}{target}</span>'
            f'<span class="config-val">{esc(st)} · {esc(dur)}</span></div>'
        )
    stages_html = ''.join(rows) or '<div class="config-row"><span class="config-val">No stages recorded.</span></div>'
    err_html = f'<p class="report-broken-detail">{esc(manifest.error)}</p>' if manifest.error else ''
    return (
        '<section class="report-in-flight">'
        f'<h1>{esc(manifest.run_name)}</h1>'
        f'<p>{esc(label)} run — {_status_badge(status)} · current stage: <code>{esc(stage)}</code></p>'
        f'{err_html}'
        f'<div class="config-list">{stages_html}</div>'
        '<p><a href="/">Back to reports</a></p>'
        '</section>'
    )


def report_broken(rid: str, filename: str, detail: str) -> str:
    """Render an error page fragment for a report that sniffed OK but failed to load."""
    return (
        '<section class="report-broken">'
        '<h1>Report could not be loaded</h1>'
        f'<p class="report-broken-filename">File: <code>{esc(filename)}</code></p>'
        f'<p class="report-broken-detail">{esc(detail)}</p>'
        '<p><a href="/">Back to reports</a></p>'
        '</section>'
    )


# ---------------------------------------------------------------------------
# Filter form + HTMX fragment helpers
# ---------------------------------------------------------------------------

# Dimensions that behave as a radio (single-select).  All others are
# multiselect checkboxes.
_RADIO_DIMS: frozenset[str] = frozenset({'result', 'goal_outcome'})


_SLIDERS_ICON = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"'
    ' stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/>'
    '<line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/>'
    '<line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/>'
    '<line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/>'
    '<line x1="17" y1="16" x2="23" y2="16"/>'
    '</svg>'
)
_FILTER_CHEVRON = (
    '<svg class="filter-dd-chevron" width="12" height="12" viewBox="0 0 24 24" fill="none"'
    ' stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="m6 9 6 6 6-6"/></svg>'
)

# Goal-outcome chip dot colors (spec: Achieved green-600, Not achieved red-600).
_GOAL_OUTCOME_DOT_CLASS: dict[str, str] = {'Achieved': 'chip-dot-green', 'Not achieved': 'chip-dot-red'}


def _chip(name: str, value: str, *, checked: bool, dot_cls: str) -> str:
    """One chip toggle: a <label>-wrapped visually-hidden checkbox."""
    checked_attr = ' checked' if checked else ''
    active_cls = ' is-active' if checked else ''
    return (
        f'<label class="filter-chip{active_cls}">'
        f'<input type="checkbox" class="filter-chip-input" name="{esc(name)}" value="{esc(value)}"{checked_attr}>'
        f'<span class="filter-chip-dot {dot_cls}"></span>'
        f'<span class="filter-chip-check {dot_cls}">✓</span>'
        f'<span class="filter-chip-label">{esc(value)}</span>'
        f'</label>'
    )


def _dropdown(dim: str, label: str, dim_opts: list[str], sel: list[str]) -> str:
    """One Persona/Scenario <details> dropdown with a stable id.

    ``sel`` empty means "all selected" (the default filter-route behaviour);
    the trigger status/label reflect that.
    """
    selected_set = set(sel) if sel else set(dim_opts)
    n_selected = len(selected_set)
    n_total = len(dim_opts)
    if n_selected == n_total:
        status_cls, status_label, value_cls = 'is-all', 'All', 'filter-dd-value'
    elif n_selected == 0:
        status_cls, status_label, value_cls = 'is-none', 'None', 'filter-dd-value is-none'
    else:
        status_cls, status_label, value_cls = 'is-partial', f'{n_selected} selected', 'filter-dd-value is-engaged'

    rows = ''.join(
        f'<label class="filter-dd-row">'
        f'<input type="checkbox" name="{esc(dim)}" value="{esc(opt)}"'
        f'{" checked" if opt in selected_set else ""}>'
        f'<span>{esc(opt)}</span>'
        f'</label>'
        for opt in dim_opts
    )
    return (
        f'<details id="filter-dd-{esc(dim)}" class="filter-dd">'
        f'<summary class="filter-dd-trigger">'
        f'<span class="filter-dd-status {status_cls}"></span>'
        f'<span class="filter-dd-name">{esc(label)}</span>'
        f'<span class="{value_cls}">{esc(status_label)}</span>'
        f'{_FILTER_CHEVRON}'
        f'</summary>'
        f'<div class="filter-dd-menu">{rows}</div>'
        f'</details>'
    )


def _fmt_range_value(value: float, step: str) -> str:
    """Format a range input's numeric attribute: integers for raw counts
    (``step="1"``), compact decimals for score thresholds (``step="0.05"``)."""
    if step == '1':
        return str(int(value))
    return f'{value:g}'


def _range_engaged(sel: list[str], *, min_val: float, max_val: float, floor: bool) -> bool:
    """True when a range control is actively constraining results, i.e. its
    value differs from the no-op default bound (``min_val`` for floors,
    ``max_val`` for ceilings). Absent/unparseable selections are not engaged."""
    if not sel:
        return False
    try:
        value = float(sel[0])
    except (ValueError, TypeError):
        return False
    return value != (min_val if floor else max_val)


def _range_control(
    dim: str,
    label: str,
    *,
    min_val: float,
    max_val: float,
    step: str,
    sel: list[str],
    floor: bool,
    show_max: bool = False,
) -> str:
    """One min/max range control shared by raw-count and score thresholds.

    ``floor`` selects both the unset ("no-op") end of the range — ``min_val``
    for a floor (e.g. min turns), ``max_val`` for a ceiling (e.g. max goal
    score) — and the readout glyph (``≥`` for floors, ``≤`` for ceilings).
    An absent selection renders the default bound numerically (never "all").
    ``show_max`` appends the slider's upper bound beside the readout (e.g.
    min-turns shows the run's max turns). The readout gains ``is-engaged`` when
    the slider is moved off its no-op bound, so an active filter reads at a
    glance without comparing numbers.
    """
    raw = sel[0] if sel else None
    default = min_val if floor else max_val
    try:
        value = float(raw) if raw is not None else default
    except (ValueError, TypeError):
        value = default
    engaged = _range_engaged(sel, min_val=min_val, max_val=max_val, floor=floor)
    readout_cls = 'filter-slider-readout is-engaged' if engaged else 'filter-slider-readout'
    glyph = '≥' if floor else '≤'
    readout = f'{glyph} {_fmt_range_value(value, step)}'
    max_span = f'<span class="filter-slider-max">/ {_fmt_range_value(max_val, step)}</span>' if show_max else ''
    # data-glyph / data-default let dashboard.js update the readout live on
    # `input` (while dragging), before the HTMX `change` round-trip.
    return (
        f'<div class="filter-group" data-dim="{esc(dim)}">'
        f'<label class="filter-label">{esc(label)}</label>'
        f'<div class="filter-slider-row">'
        f'<input type="range" class="filter-slider" name="{esc(dim)}" min="{_fmt_range_value(min_val, step)}"'
        f' max="{_fmt_range_value(max_val, step)}" step="{esc(step)}" value="{_fmt_range_value(value, step)}"'
        f' data-glyph="{glyph}" data-default="{_fmt_range_value(default, step)}">'
        f'<span class="{readout_cls}">{esc(readout)}</span>'
        f'{max_span}'
        f'</div>'
        f'</div>'
    )


def _render_sim_filter_rail(
    rid: str,
    opts: dict[str, list[str]],
    selections: dict[str, list[str]],
    *,
    shown: int | None,
    total: int | None,
) -> str:
    """Render the right-side sim filter rail: chips + dropdowns + counter.

    Goal outcome / Terminated by render as chip toggles; Persona / Scenario
    render as ``<details>`` dropdowns with stable ids (``filter-dd-persona``,
    ``filter-dd-scenario``) so a later JS task can persist their open state
    across the HTMX outerHTML swap.  The rule-violation chip, goal-score
    ceiling, and min-turns controls render directly in the rail; min total
    tokens and every *available* per-turn quality/risk metric threshold are
    tucked behind the ``filter-dd-more`` expander (omitted entirely when it
    would otherwise be empty).
    """
    from evaluatorq.dashboard.filters import sim_metric_dim_key

    goal_opts = [o for o in opts.get('goal_outcome', []) if o in _GOAL_OUTCOME_DOT_CLASS]
    goal_sel = set(selections.get('goal_outcome', []))
    goal_chips = ''.join(
        _chip('goal_outcome', opt, checked=(opt in goal_sel), dot_cls=_GOAL_OUTCOME_DOT_CLASS[opt]) for opt in goal_opts
    )

    term_opts = opts.get('terminated_by', [])
    term_sel = set(selections.get('terminated_by', [])) or set(term_opts)
    term_chips = ''.join(
        _chip('terminated_by', opt, checked=(opt in term_sel), dot_cls='chip-dot-jade') for opt in term_opts
    )

    persona_dd = _dropdown('persona', 'Persona', opts.get('persona', []), selections.get('persona', []))
    scenario_dd = _dropdown('scenario', 'Scenario', opts.get('scenario', []), selections.get('scenario', []))

    # Rule-violation chip: opt-in, so unchecked unless explicitly selected.
    rule_sel = set(selections.get('rule_broken', []))
    rule_chip = _chip('rule_broken', 'yes', checked=('yes' in rule_sel), dot_cls='chip-dot-red')

    # Goal-score ceiling (ceiling control: unset default is 1.0, i.e. "all").
    goal_score_ctrl = _range_control(
        'max_goal_score',
        'Max Goal Score',
        min_val=0.0,
        max_val=1.0,
        step='0.05',
        sel=selections.get('max_goal_score', []),
        floor=False,
    )

    # Min turns (floor, raw run-length integer — never normalized). Hidden as a
    # no-op control when every conversation has a single turn (max == 1), matching
    # the redteam rail.
    max_turns = int(opts.get('max_turns', ['1'])[0] or 1)
    min_turns_ctrl = ''
    if max_turns > 1:
        min_turns_ctrl = _range_control(
            'min_turns',
            'Min. Turns',
            min_val=1,
            max_val=max_turns,
            step='1',
            sel=selections.get('min_turns', []),
            floor=True,
            show_max=True,
        )

    # More expander: min total tokens (raw floor) + every available metric
    # threshold. Skipped entirely (no empty expander) when there is nothing
    # to show.
    more_rows: list[str] = []
    more_active = 0  # controls hidden inside the expander that are actively filtering
    max_total_tokens = int(opts.get('max_total_tokens', ['0'])[0] or 0)
    if max_total_tokens > 0:
        tok_sel = selections.get('min_total_tokens', [])
        more_active += _range_engaged(tok_sel, min_val=0, max_val=max_total_tokens, floor=True)
        more_rows.append(
            _range_control(
                'min_total_tokens',
                'Min. Total Tokens',
                min_val=0,
                max_val=max_total_tokens,
                step='1',
                sel=tok_sel,
                floor=True,
            )
        )
    available_metrics = set(opts.get('metrics', []))
    for metric in TURN_METRICS:
        if metric.key not in available_metrics:
            continue
        dim_key = sim_metric_dim_key(metric.key, high_is_risky=metric.high_is_risky)
        prefix = 'Min.' if metric.high_is_risky else 'Max.'
        metric_sel = selections.get(dim_key, [])
        more_active += _range_engaged(metric_sel, min_val=0.0, max_val=1.0, floor=metric.high_is_risky)
        more_rows.append(
            _range_control(
                dim_key,
                f'{prefix} {metric.label.title()}',
                min_val=0.0,
                max_val=1.0,
                step='0.05',
                sel=metric_sel,
                floor=metric.high_is_risky,
            )
        )

    more_dd = ''
    if more_rows:
        # Surface any active filters hidden in the collapsed panel so they are
        # never silently narrowing results (audit finding: the expander gave no
        # trace of engaged controls when closed).
        badge = f'<span class="filter-dd-more-badge">{more_active}</span>' if more_active else ''
        more_dd = (
            f'<details id="filter-dd-more" class="filter-dd filter-dd-more">'
            f'<summary class="filter-dd-trigger">'
            f'<span class="filter-dd-name">More filters</span>'
            f'{badge}'
            f'{_FILTER_CHEVRON}'
            f'</summary>'
            f'<div class="filter-dd-more-body">{"".join(more_rows)}</div>'
            f'</details>'
        )

    if shown is not None and total is not None and shown < total:
        counter = f'{shown} of {total} shown'
    else:
        counter = 'Showing all results'

    inner = (
        f'<div class="filter-rail-header">{_SLIDERS_ICON}<span class="filter-rail-title">Filters</span></div>'
        f'<div class="filter-group" data-dim="rule_broken">'
        f'<label class="filter-label">Rule Violations</label>'
        f'<div class="filter-chip-row">{rule_chip}</div>'
        f'</div>'
        f'<div class="filter-group" data-dim="goal_outcome">'
        f'<label class="filter-label">Goal Outcome</label>'
        f'<div class="filter-chip-row">{goal_chips}</div>'
        f'</div>'
        f'<div class="filter-group" data-dim="terminated_by">'
        f'<label class="filter-label">Terminated By</label>'
        f'<div class="filter-chip-row">{term_chips}</div>'
        f'</div>'
        f'{goal_score_ctrl}'
        f'{min_turns_ctrl}'
        f'<div class="filter-group" data-dim="persona">{persona_dd}</div>'
        f'<div class="filter-group" data-dim="scenario">{scenario_dd}</div>'
        f'<div class="filter-rail-footer">{esc(counter)}</div>'
        + (f'<div class="filter-group filter-group--more" data-dim="more">{more_dd}</div>' if more_dd else '')
    )
    return (
        f'<form id="filter-form" class="filter-form filter-form--sim"'
        f' hx-post="/r/{esc(rid)}/filter"'
        f' hx-trigger="change"'
        f' hx-target="#filter-swap"'
        f' hx-swap="outerHTML">'
        f'{inner}'
        f'</form>'
    )


# Outcome chip dot colors (spec: Vulnerable red-600, Resistant green-600, Error amber-600).
_RESULT_DOT_CLASS: dict[str, str] = {
    'Vulnerable': 'chip-dot-red',
    'Resistant': 'chip-dot-green',
    'Error': 'chip-dot-amber',
}

# Severity chip dot colors, aligned with report_kit.severity_pill tones.
_SEVERITY_DOT_CLASS: dict[str, str] = {
    'critical': 'chip-dot-red',
    'high': 'chip-dot-amber',
    'medium': 'chip-dot-gray',
    'low': 'chip-dot-green',
}


def _render_redteam_filter_rail(
    rid: str,
    opts: dict[str, list[str]],
    selections: dict[str, list[str]],
    *,
    shown: int | None,
    total: int | None,
) -> str:
    """Render the right-side redteam filter rail: chips + slider + dropdowns + counter.

    Outcome / Severity render as chip toggles; a min-turns slider is shown
    only when the run has any multi-turn attacks (``opts['max_turns'] > 1``);
    Category / Agent render as ``<details>`` dropdowns (Agent only when there
    is more than one agent); Technique / Delivery Method / Vulnerability are
    tucked behind a ``<details id="filter-dd-more">`` "More filters" expander
    to keep the rail short for the common single-agent, single-category run.
    """
    result_opts = opts.get('result', [])
    result_sel = set(selections.get('result', []))
    result_chips = ''.join(
        _chip('result', opt, checked=(opt in result_sel), dot_cls=_RESULT_DOT_CLASS.get(opt, 'chip-dot-gray'))
        for opt in result_opts
    )

    severity_opts = opts.get('severity', [])
    severity_sel = set(selections.get('severity', [])) or set(severity_opts)
    severity_chips = ''.join(
        _chip(
            'severity',
            opt,
            checked=(opt in severity_sel),
            dot_cls=_SEVERITY_DOT_CLASS.get(opt.lower(), 'chip-dot-gray'),
        )
        for opt in severity_opts
    )

    # Shared range control: numeric readout (never "all"), is-engaged highlight,
    # and the run's max shown beside the slider — consistent with the sim rail.
    max_turns = int(opts.get('max_turns', ['1'])[0] or 1)
    slider_html = ''
    if max_turns > 1:
        slider_html = _range_control(
            'min_turns',
            'Min. Turns',
            min_val=1,
            max_val=max_turns,
            step='1',
            sel=selections.get('min_turns', []),
            floor=True,
            show_max=True,
        )

    category_dd = _dropdown('category', 'Category', opts.get('category', []), selections.get('category', []))

    agent_opts = opts.get('agent', [])
    agent_dd = ''
    if len(agent_opts) > 1:
        agent_dd = _dropdown('agent', 'Agent', agent_opts, selections.get('agent', []))

    more_rows = ''.join([
        _dropdown('technique', 'Technique', opts.get('technique', []), selections.get('technique', [])),
        _dropdown(
            'delivery_method', 'Delivery Method', opts.get('delivery_method', []), selections.get('delivery_method', [])
        ),
        _dropdown('vulnerability', 'Vulnerability', opts.get('vulnerability', []), selections.get('vulnerability', [])),
    ])
    more_dd = (
        f'<details id="filter-dd-more" class="filter-dd filter-dd-more">'
        f'<summary class="filter-dd-trigger">'
        f'<span class="filter-dd-name">More filters</span>'
        f'{_FILTER_CHEVRON}'
        f'</summary>'
        f'<div class="filter-dd-more-body">{more_rows}</div>'
        f'</details>'
    )

    if shown is not None and total is not None and shown < total:
        counter = f'{shown} of {total} shown'
    else:
        counter = 'Showing all results'

    inner = (
        f'<div class="filter-rail-header">{_SLIDERS_ICON}<span class="filter-rail-title">Filters</span></div>'
        f'<div class="filter-group" data-dim="result">'
        f'<label class="filter-label">Outcome</label>'
        f'<div class="filter-chip-row">{result_chips}</div>'
        f'</div>'
        f'<div class="filter-group" data-dim="severity">'
        f'<label class="filter-label">Severity</label>'
        f'<div class="filter-chip-row">{severity_chips}</div>'
        f'</div>'
        f'{slider_html}'
        f'<div class="filter-group" data-dim="category">{category_dd}</div>'
        + (f'<div class="filter-group" data-dim="agent">{agent_dd}</div>' if agent_dd else '')
        + f'<div class="filter-rail-footer">{esc(counter)}</div>'
        f'<div class="filter-group filter-group--more" data-dim="more">{more_dd}</div>'
    )
    return (
        f'<form id="filter-form" class="filter-form filter-form--redteam"'
        f' hx-post="/r/{esc(rid)}/filter"'
        f' hx-trigger="change"'
        f' hx-target="#filter-swap"'
        f' hx-swap="outerHTML">'
        f'{inner}'
        f'</form>'
    )


def render_filter_form(
    rid: str,
    surface: str,
    opts: dict[str, list[str]],
    selections: dict[str, list[str]],
    *,
    shown: int | None = None,
    total: int | None = None,
) -> str:
    """Render the HTMX filter sidebar form as an HTML fragment.

    The form POSTs the full set of dimensions on every change (``hx-trigger``
    on the containing form's ``change`` event).  Missing dimensions default to
    "all" inside the filter route handler.

    Args:
        rid:        Report ID (used to construct the POST URL).
        surface:    Surface key (``'redteam'`` | ``'sim'``).
        opts:       Option lists per dimension (from ``FilterDef.options``).
        selections: Currently active selections per dimension.
        shown:      Count of currently-shown results (sim rail footer counter).
        total:      Total unfiltered result count (sim rail footer counter).
                    Ignored by the redteam branch.

    Returns:
        An HTML ``<form>`` fragment suitable for injection into the sidebar.
    """
    if surface == 'sim':
        return _render_sim_filter_rail(rid, opts, selections, shown=shown, total=total)
    if surface == 'redteam':
        return _render_redteam_filter_rail(rid, opts, selections, shown=shown, total=total)

    from evaluatorq.dashboard.filters import FILTERS

    filter_def = FILTERS.get(surface)
    if filter_def is None or not filter_def.dimensions:
        # A surface with no dimensions (pairwise) would otherwise render an
        # empty rail titled "Filters" with nothing under it.
        return ''

    parts: list[str] = []
    for dim in filter_def.dimensions:
        dim_opts = opts.get(dim, [])
        if not dim_opts:
            continue

        sel = selections.get(dim, [])

        label = dim.replace('_', ' ').title()
        parts.extend([
            f'<div class="filter-group" data-dim="{esc(dim)}">',
            f'<label class="filter-label">{esc(label)}</label>',
        ])

        if dim in _RADIO_DIMS:
            # Radio buttons: single-select dimension
            current = sel[0] if sel else dim_opts[0]
            for opt in dim_opts:
                checked = ' checked' if opt == current else ''
                parts.append(
                    f'<label class="filter-radio">'
                    f'<input type="radio" name="{esc(dim)}" value="{esc(opt)}"{checked}>'
                    f' {esc(opt)}'
                    f'</label>'
                )
        else:
            # Checkboxes: multiselect dimension.  Default = all selected.
            selected_set = set(sel) if sel else set(dim_opts)
            for opt in dim_opts:
                checked = ' checked' if opt in selected_set else ''
                parts.append(
                    f'<label class="filter-checkbox">'
                    f'<input type="checkbox" name="{esc(dim)}" value="{esc(opt)}"{checked}>'
                    f' {esc(opt)}'
                    f'</label>'
                )

        parts.append('</div>')

    inner = ''.join(parts)
    return (
        f'<form id="filter-form" class="filter-form"'
        f' hx-post="/r/{esc(rid)}/filter"'
        f' hx-trigger="change"'
        f' hx-target="#filter-swap"'
        f' hx-swap="outerHTML">'
        f'<div class="filter-sidebar">'
        f'<h3 class="filter-title">Filters</h3>'
        f'{inner}'
        f'</div>'
        f'</form>'
    )


def filter_fragment(
    rid: str,
    surface: str,
    body_html: str,
    form_html: str,
) -> str:
    """Wrap the re-rendered body + filter form in the HTMX swap container.

    The outer ``<div id="filter-swap">`` is the element that HTMX replaces on
    each POST.  It contains both the filter form (sidebar) and the report body
    so that a single swap updates both simultaneously.

    Args:
        rid:       Report ID (for data-attribute bookkeeping).
        surface:   Surface key.
        body_html: Re-rendered report body HTML fragment.
        form_html: Re-rendered filter form HTML fragment.

    Returns:
        An HTML string starting with ``<div id="filter-swap">``.
    """
    return (
        f'<div id="filter-swap" class="filter-swap-container"'
        f' data-rid="{esc(rid)}">'
        f'<div class="report-body-area">'
        f'{body_html}'
        f'</div>'
        f'{form_html}'
        f'</div>'
    )


def report_view_with_filters(
    rid: str,
    surface: str,
    body_html: str,
    form_html: str,
) -> str:
    """Render the full report view including the filter swap container.

    Used by ``GET /r/{rid}`` to produce the initial page body that already
    contains the HTMX-wired filter form.
    """
    swap = filter_fragment(rid, surface, body_html, form_html)
    return f'<section class="report-view">{swap}</section>'


def render_message_list(
    messages: list[Any],
    *,
    role_labels: dict[str, str],
    class_prefix: str,
) -> str:
    """Render a role-labeled message list as a series of chat-bubble ``<div>``s.

    Each message produces:

    .. code-block:: html

        <div class="{class_prefix}-msg {class_prefix}-msg-{css_role}">
          <span class="{class_prefix}-msg-avatar">{label}</span>
          <div class="{class_prefix}-msg-bubble">
            <span class="{class_prefix}-msg-role">{label}</span>
            <pre class="{class_prefix}-msg-content">{esc(content)}</pre>
          </div>
        </div>

    Where ``{css_role}`` is the raw role value when it is one of
    ``user``, ``assistant``, ``system``, ``tool``; otherwise ``unknown``.
    The ``{class_prefix}-msg-avatar`` span is a short glyph (e.g. "USR" /
    "AGT" / "SYS" / "TOOL") taken straight from *role_labels* — callers that
    want a compact avatar should pass short labels; the ``{css_role}`` class
    is what CSS uses to put user/system on the left and assistant/tool on
    the right (spec §Transcripts), so no separate "side" marker is needed.

    Every content string is passed through ``esc()`` (HTML-escaping) to
    prevent stored-XSS vectors. This is purely additive over the prior
    shape (avatar span + bubble wrapper added); the ``{class_prefix}-msg
    {class_prefix}-msg-{css_role}`` outer class and the ``-msg-role`` /
    ``-msg-content`` elements are unchanged, so existing callers keep working.

    Args:
        messages:     Sequence of message dicts (``role`` / ``content`` keys)
                      or objects with ``.role`` / ``.content`` attributes.
        role_labels:  Mapping from role name → display label.  Falls back to
                      the raw role string when a role is not in the map.
        class_prefix: CSS class namespace (e.g. ``"sim"`` → ``sim-msg``).

    Returns:
        Concatenated HTML string (empty string when *messages* is empty).
    """
    known_roles = frozenset({'user', 'assistant', 'system', 'tool'})

    parts: list[str] = []
    for msg in messages:
        # Support both dict-style and attribute-style message objects.
        if isinstance(msg, dict):
            role = str(msg.get('role', 'unknown'))
            raw_content = msg.get('content', '')
        else:
            role = str(getattr(msg, 'role', 'unknown'))
            raw_content = getattr(msg, 'content', '')

        label = role_labels.get(role, role)
        content_text: str = raw_content if isinstance(raw_content, str) else str(raw_content or '')
        safe_content = esc(content_text)
        css_role = role if role in known_roles else 'unknown'
        safe_label = esc(label)

        parts.append(
            f'<div class="{class_prefix}-msg {class_prefix}-msg-{esc(css_role)}">'
            f'<span class="{class_prefix}-msg-avatar">{safe_label}</span>'
            f'<div class="{class_prefix}-msg-bubble">'
            f'<span class="{class_prefix}-msg-role">{safe_label}</span>'
            f'<pre class="{class_prefix}-msg-content">{safe_content}</pre>'
            f'</div>'
            f'</div>'
        )

    return ''.join(parts)


def _sim_rowlist_wrapper(rid: str, inner: str) -> str:
    """Wrap *inner* HTML in a plain container div for the sim row-list.

    Double-fetch removal (spec §Transcripts): ``POST /r/{rid}/filter`` already
    re-renders the whole (filtered) report body — including this row list —
    in a single response.  This wrapper previously *also* self-refetched via
    ``hx-get``/``hx-trigger="orq:filter-changed"``, firing a redundant second
    request for content the body swap had just delivered (about to get
    heavier now that each row is a lazy-loading conversation card).  That
    self-refetch is removed; the div is now an inert container and the
    ``/filter`` POST body swap is the single refresh path.

    ``GET /r/{rid}/sim/row-list`` stays registered as a route — it backs the
    transcript endpoint's filtered-index contract for direct use — it is
    simply no longer auto-triggered from here.

    Args:
        rid:   Report ID (URL-safe).
        inner: Already-rendered row-list HTML to embed inside the div.

    Returns:
        An HTML ``<div>`` string wrapping *inner*.
    """
    safe_rid = esc(rid)
    return f'<div id="sim-row-list-{safe_rid}">{inner}</div>'


def sim_interactive_panels(rid: str, entries: list[Any]) -> str:
    """Render the interactive sim panel section (conversation list).

    Embeds the sim row list with lazy drawer-trigger conversation rows.
    Parity: Streamlit ``_render_transcripts`` (dashboard.py:316-390).

    The row-list itself is a static container: it is refreshed wholesale by
    the ``POST /r/{rid}/filter`` response body swap (see
    ``_sim_rowlist_wrapper``), not by its own ``hx-trigger``. Entries retain
    their full-run indexes so each conversation drawer URL resolves directly
    to its transcript regardless of the active filter.

    Args:
        rid:     Report ID (URL-safe).
        entries: Individual-result entries with stable full-run indexes.

    Returns:
        An HTML ``<section class="sim-interactive-panels">`` fragment.
    """
    from evaluatorq.dashboard.sim_views import render_sim_row_list

    # ponytail: the initial render (and the /filter body-swap that reuses this)
    # always starts at page 1 / default sort. Sort + page live on the
    # /sim/row-list GET; carrying them through a filter change would mean
    # threading them into the POST too — add that only if losing sort-on-filter
    # actually annoys anyone.
    row_list = render_sim_row_list(rid, entries)
    return (
        f'<section class="sim-interactive-panels">'
        f'<h2 class="sim-panels-title">Conversations</h2>'
        f'{_sim_rowlist_wrapper(rid, row_list)}'
        f'</section>'
    )
