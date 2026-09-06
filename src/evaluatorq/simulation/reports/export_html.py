"""HTML renderer for agent simulation reports.

``export_html(results)`` produces a self-contained HTML document styled with
the shared report CSS. All charts are Vega-Lite specs rendered to SVG via
``vl-convert-python``; when that package is absent charts are omitted and
the report degrades to a tables-only layout (this module imports no plotly).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from evaluatorq.common.reports import (
    esc as _esc,
)
from evaluatorq.common.reports import (
    format_date as _format_date,
)
from evaluatorq.common.reports import (
    html_table as _html_table,
)
from evaluatorq.common.reports import (
    kpi_cards as _kpi_cards,
)
from evaluatorq.common.reports import (
    load_css as _load_css,
)
from evaluatorq.common.reports import (
    pct as _pct,
)
from evaluatorq.common.reports import (
    render_body as _render_body,
)
from evaluatorq.common.reports import (
    render_heatmap as _render_heatmap,
)
from evaluatorq.common.reports import (
    render_histogram as _render_histogram,
)
from evaluatorq.common.reports import (
    render_line_chart as _render_line_chart,
)
from evaluatorq.common.reports import (
    scale_color as _scale_color,
)
from evaluatorq.common.reports import (
    status_badge as _status_badge,
)
from evaluatorq.common.reports import (
    svg_bar as _svg_bar,
)
from evaluatorq.common.reports.palette import COLORS
from evaluatorq.dashboard.trace_links import trace_link_button
from evaluatorq.simulation.reports.sections import build_report_sections
from evaluatorq.simulation.reports.token_usage import build_token_usage_rows

if TYPE_CHECKING:
    from evaluatorq.contracts import ReportSection, Usage
    from evaluatorq.simulation.types import SimulationRecommendation, SimulationResult

# Heatmap colour direction:
# ``ORQ_SCALE_GOOD_BAD`` is green at 0.0 -> red at 1.0 (i.e. good == low).
# For these reports "good" means HIGH (a passing criterion / high success
# rate), so we use an explicit green-high scale: red at 0.0 -> green at 1.0.
# Renderers pass the raw pass-rate / success-rate value through it, so a fully
# passing cell is green and a failing cell red.
# Green endpoint darkened from brand success_400 so white text on the greenest
# heatmap cells / score chips clears WCAG AA (4.5:1). Red end already passes.
_SCALE_GREEN_HIGH: list[list[float | str]] = [
    [0.0, COLORS['red_400']],
    [1.0, '#157f57'],
]

# Evaluators whose score is "better when lower" (risk/cost-style). Matched as
# case-insensitive substrings against the evaluator name. Everything else is
# treated as higher-is-better, which covers the built-in quality evaluators.
_LOWER_IS_BETTER = ('risk', 'hallucinat', 'toxic', 'latency', 'cost', 'error', 'violation')

# Plain-language display names for evaluator / metric keys. The raw key is kept
# in a tooltip so engineers can still map back. Unknown keys are title-cased.
_EVALUATOR_LABELS = {
    'goal_achieved': 'Resolved the issue',
    'criteria_met': 'Met success criteria',
    'turn_efficiency': 'Efficient (few turns)',
    'conversation_quality': 'Conversation quality',
    'response_quality': 'Response quality',
    'tone_appropriateness': 'Appropriate tone',
    'factual_accuracy': 'Factual accuracy',
    'hallucination_risk': 'Hallucination risk',
}


def _pretty_evaluator(name: str) -> str:
    return _EVALUATOR_LABELS.get(name, name.replace('_', ' ').capitalize())


def _dir_arrow(name: str) -> str:
    return '▼' if _score_is_lower_better(name) else '▲'


# Above this many conversations the per-conversation turn bar is replaced by the
# compact turn-count distribution (it would otherwise grow taller than a screen).
_MAX_PER_CONV_BARS = 12

# Failure tables longer than this scroll inside a fixed-height box.
_FAILURES_SCROLL_AFTER = 20


def _score_is_lower_better(name: str) -> bool:
    low = name.lower()
    return any(token in low for token in _LOWER_IS_BETTER)


def _score_chip(value: float, *, lower_is_better: bool) -> str:
    """A score value tinted red→green by how good it is (direction-aware)."""
    goodness = (1.0 - value) if lower_is_better else value
    color = _scale_color(max(0.0, min(1.0, goodness)), _SCALE_GREEN_HIGH)
    return f'<span class="score-chip" style="background:{color}">{value:.2f}</span>'


def _build_verdict_line(sections: list[ReportSection], sd: dict[str, Any]) -> str:
    """One-sentence plain-language verdict for the hero: overall + worst cohort."""
    total = sd.get('total_conversations', 0)
    if not total:
        return ''
    achieved = sd.get('goals_achieved', 0)
    verdict = sd.get('verdict', 'neutral')
    word = {'pass': 'STRONG', 'warn': 'MIXED'}.get(verdict, 'FAILING')
    cls = {'pass': 'pass', 'warn': 'warn'}.get(verdict, 'fail')

    def _worst(kind: str, key: str) -> str | None:
        sec = next((s for s in sections if s.kind == kind), None)
        rows = sec.data.get('rows', []) if sec else []
        rows = [r for r in rows if r.get('conversations')]
        if not rows:
            return None
        w = min(rows, key=lambda r: r.get('success_rate', 0.0))
        return f'{w[key]} {_pct(w.get("success_rate", 0.0))}'

    bits = [f'{word}: {_pct(sd.get("success_rate", 0.0))} success ({achieved}/{total})']
    worst_persona = _worst('persona_breakdown', 'persona')
    worst_scenario = _worst('scenario_breakdown', 'scenario')
    weak = [b for b in (worst_persona, worst_scenario) if b]
    if weak:
        bits.append('weakest: ' + ', '.join(weak))
    return f'<p class="verdict-line verdict-line--{cls}">{_esc(". ".join(bits))}.</p>'


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_summary_html(section: ReportSection) -> str:
    # The headline numbers (success rate, avg score, conversations, errors)
    # already live in the hero KPI band, so the summary section renders no
    # standalone card. Empty input still surfaces a no-data note so the report
    # never looks truncated.
    if not section.data.get('total_conversations', 0):
        return (
            '<section class="report-card"><h2>'
            f'{_esc(section.title)}</h2><p>No conversations to summarize.</p></section>'
        )
    narrative = section.data.get('narrative')
    if not narrative:
        return ''
    confidence = section.data.get('confidence', '')
    confidence_note = section.data.get('confidence_note', '')
    pill = _status_badge(f'{confidence} CONFIDENCE', 'warn') if confidence else ''  # ponytail: cosmetic pill
    note = f'<br><span style="font-size:.8em;opacity:.7">{_esc(confidence_note)}</span>' if confidence_note else ''
    return (
        '<section class="report-card exec-summary-narrative"><h2>'
        f'{_esc(section.title)} {pill}</h2>'
        f'<p style="font-size:.98em;line-height:1.55">{_esc(str(narrative))}{note}</p></section>'
    )


def _render_overview_html(section: ReportSection) -> str:
    d = section.data
    personas = d.get('personas', [])
    scenarios = d.get('scenarios', [])
    if not personas and not scenarios:
        return ''
    intro = (
        f'<p>This report evaluates the target agent across <strong>{len(personas)}</strong> '
        f'persona(s) and <strong>{len(scenarios)}</strong> scenario(s), for '
        f'<strong>{d.get("total_conversations", 0)}</strong> simulated conversation(s). '
        'In each, a simulated user (the persona) pursues the scenario goal while a judge '
        'scores success criteria and per-turn quality. The sections below lead with failures.</p>'
    )

    def _persona_item(p: dict[str, Any]) -> str:
        traits = p.get('traits')
        background = p.get('background')
        parts = [
            (
                f'<li><h4 class="intro-name">{_esc(p["name"])} '
                f'<span class="intro-count">· {p["conversations"]} conv.</span></h4>'
            )
        ]
        if isinstance(traits, dict):
            trait_rows = [
                ['Patience', str(traits.get('patience', '?'))],
                ['Assertiveness', str(traits.get('assertiveness', '?'))],
                ['Politeness', str(traits.get('politeness', '?'))],
                ['Technical level', str(traits.get('technical_level', '?'))],
            ]
            if traits.get('communication_style'):
                trait_rows.append(['Style', _esc(str(traits['communication_style']))])
            parts.append(f'<div class="intro-traits">{_html_table(["Trait", "Value"], trait_rows)}</div>')
        if background:
            parts.append(f'<div class="intro-meta">{_esc(str(background))}</div>')
        parts.append('</li>')
        return ''.join(parts)

    persona_items = ''.join(_persona_item(p) for p in personas)

    def _scenario_item(s: dict[str, Any]) -> str:
        goal = s.get('goal')
        context = s.get('context')
        tags = ''.join(
            f'<span class="crit-tag crit-tag--{"mustnot" if c["type"] == "must_not_happen" else "must"}">'
            f'{"✗ must not" if c["type"] == "must_not_happen" else "✓ must"}: {_esc(c["description"])}</span>'
            for c in s.get('criteria', [])
        )
        parts = [f'<li><h4 class="intro-name">{_esc(s["name"])}</h4>']
        if goal:
            parts.append(f'<div class="intro-meta"><strong>Goal:</strong> {_esc(str(goal))}</div>')
        if context:
            parts.append(f'<div class="intro-meta"><strong>Context:</strong> {_esc(str(context))}</div>')
        if tags:
            parts.append(f'<div class="intro-criteria">{tags}</div>')
        parts.append('</li>')
        return ''.join(parts)

    scenario_items = ''.join(_scenario_item(s) for s in scenarios)
    grid = (
        '<div class="intro-grid">'
        f'<div><h3>Personas</h3><ul class="intro-list">{persona_items}</ul></div>'
        f'<div><h3>Scenarios</h3><ul class="intro-list">{scenario_items}</ul></div>'
        '</div>'
    )
    return f'<section class="report-card"><h2>{_esc(section.title)}</h2>{intro}{grid}</section>'


_CRIT_CARET = (
    '<svg class="crit-caret" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>'
)


def _criteria_dots(criteria: list[dict[str, Any]]) -> str:
    """Collapsed pass/fail dots that unfold to the full criteria text.

    Summary is a hover-lit pill of dots (each dot titled for a quick tooltip) plus
    a caret; opening reveals a tinted panel listing every criterion. Height animates
    both ways via ``::details-content`` where supported and snaps otherwise.
    """
    if not criteria:
        return '<span class="crit-empty">—</span>'
    n_fail = sum(1 for c in criteria if not c['passed'])
    n_unknown = sum(1 for c in criteria if c.get('state') == 'unknown')
    dots, items = [], []
    for c in criteria:
        state = c.get('state', 'pass' if c['passed'] else 'fail')
        # An unaudited criterion is neither a green dot nor a red one — the judge
        # never reported on it (RES-1308).
        cls = 'safety' if (c.get('safety') and not c['passed']) else state
        suffix = ' (not audited)' if state == 'unknown' else ''
        dots.append(f'<span class="crit-dot crit-dot--{cls}" title="{_esc(c["description"] + suffix)}"></span>')
        items.append(f'<li class="crit-li crit-li--{cls}">{_esc(c["description"] + suffix)}</li>')
    label = f'{n_fail} of {len(criteria)} criteria failed'
    if n_unknown:
        label += f', {n_unknown} not audited'
    label += ' — show details'
    return (
        f'<details class="crit-cell">'
        f'<summary class="crit-summary" aria-label="{_esc(label)}" title="{_esc(label)}">'
        f'<span class="crit-dots">{"".join(dots)}</span>{_CRIT_CARET}</summary>'
        f'<ul class="crit-list">{"".join(items)}</ul></details>'
    )


def _cap(text: str, max_chars: int = 90) -> str:
    """Single-line cap for table cells; full text lives in the title tooltip."""
    text = ' '.join(text.split())
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + '…'


def _render_failures_first_html(section: ReportSection) -> str:
    rows = section.data.get('rows', [])
    if not rows:
        return '<section class="report-card"><h2>Failures</h2><p>No failed conversations.</p></section>'
    trs = [
        f'<tr><td>{_esc(r["scenario"])}</td>'
        f'<td>{_esc(r["persona"])}</td>'
        f'<td class="fail-why" title="{_esc(r.get("reason", ""))}">{_esc(_cap(r.get("reason", "")))}</td>'
        f'<td>{_criteria_dots(r["criteria"])}</td>'
        f'<td>{r["score"]:.2f}</td></tr>'
        for r in rows
    ]
    table = (
        '<table><thead><tr><th>Scenario</th><th>Persona</th>'
        '<th>Why</th><th>Criteria</th><th>Score</th></tr></thead>'
        f'<tbody>{"".join(trs)}</tbody></table>'
    )
    # Long failure lists get a scroll container with a sticky header so the
    # section can't dominate the report.
    if len(rows) > _FAILURES_SCROLL_AFTER:
        table = f'<div class="scroll-table">{table}</div>'
    note = (
        f'<p class="chart-note">{len(rows)} failures, scroll within the box.</p>'
        if len(rows) > _FAILURES_SCROLL_AFTER
        else ''
    )
    return f'<section class="report-card"><h2>Failures</h2>{table}{note}</section>'


def _render_persona_scenario_heatmap_html(section: ReportSection) -> str:
    d = section.data
    personas, scenarios = d['personas'], d['scenarios']
    if not personas or not scenarios:
        return ''
    # A 1x1 grid is one colored cell restating the headline success rate.
    if len(personas) < 2 and len(scenarios) < 2:
        return ''
    lookup = {(c['persona'], c['scenario']): c for c in d['cells']}
    # cells[row=scenario][col=persona] = success-rate (good=high -> green-high scale)
    cells = [[lookup.get((p, s), {}).get('success_rate', -1.0) for p in personas] for s in scenarios]
    heat = _render_heatmap(
        x_labels=personas,
        y_labels=scenarios,
        cells=cells,
        scale=_SCALE_GREEN_HIGH,
        title=section.title,
        value_fmt=lambda v: '-' if v < 0 else f'{v:.0%}',
    )
    return f'<section class="report-card">{heat}</section>'


def _render_score_distribution_html(section: ReportSection) -> str:
    scores = section.data.get('scores', [])
    # A histogram of one or two values renders as a single squished bar that
    # reads as broken — state the scores directly instead.
    if len(scores) < 3:
        if len(scores) < 2:
            return ''
        listed = ', '.join(f'{v:.2f}' for v in scores)
        return (
            f'<section class="report-card"><h2>{_esc(section.title)}</h2>'
            f'<p>Only {len(scores)} conversation(s) — goal score(s): <strong>{listed}</strong>. '
            'A distribution needs more runs.</p></section>'
        )
    hist = _render_histogram(values=scores, bins=10, title=section.title)
    return f'<section class="report-card">{hist}</section>' if hist else ''


def _render_turn_quality_timeline_html(section: ReportSection) -> str:
    d = section.data
    turns = d.get('turns', [])
    if not turns:
        return ''
    series = [(_pretty_evaluator(name), vals) for name, vals in d['series'].items() if any(v is not None for v in vals)]
    if not series:
        return ''
    # A timeline needs at least two turns; the per-turn averages already appear
    # in Turn Metrics, so a single-point line chart adds nothing but confusion.
    if len(turns) < 2:
        return ''
    chart = _render_line_chart(x_labels=[str(t) for t in turns], series=series, title=section.title)
    return f'<section class="report-card">{chart}</section>'


def _render_persona_breakdown_html(section: ReportSection) -> str:
    rows = section.data.get('rows', [])
    if not rows:
        return f'<section class="report-card"><h2>{_esc(section.title)}</h2><p>No persona data.</p></section>'
    # A single conversation has no cohorts to break down.
    if sum(r.get('conversations', 0) for r in rows) < 2:
        return ''
    table_rows = [
        [
            _esc(r['persona']),
            str(r['conversations']),
            str(r['goals_achieved']),
            _pct(r['success_rate']),
            f'{r["avg_goal_completion_score"]:.2f}',
            f'{r["total_tokens"]:,}',
        ]
        for r in rows
    ]
    table = _html_table(
        ['Persona', 'Conversations', 'Achieved', 'Success', 'Avg Score', 'Tokens'],
        table_rows,
    )
    return f'<section class="report-card"><h2>{_esc(section.title)}</h2>{table}</section>'


def _render_scenario_breakdown_html(section: ReportSection) -> str:
    rows = section.data.get('rows', [])
    if not rows:
        return f'<section class="report-card"><h2>{_esc(section.title)}</h2><p>No scenario data.</p></section>'
    # A single conversation has no cohorts to break down.
    if sum(r.get('conversations', 0) for r in rows) < 2:
        return ''
    table_rows = [
        [
            _esc(r['scenario']),
            str(r['conversations']),
            str(r['goals_achieved']),
            _pct(r['success_rate']),
            f'{r["avg_goal_completion_score"]:.2f}',
            f'{r["avg_turn_count"]:.1f}',
        ]
        for r in rows
    ]
    table = _html_table(
        ['Scenario', 'Conversations', 'Achieved', 'Success', 'Avg Score', 'Avg Turns'],
        table_rows,
    )
    return f'<section class="report-card"><h2>{_esc(section.title)}</h2>{table}</section>'


def _render_judge_verdicts_html(section: ReportSection) -> str:
    data = section.data
    terminated_by = data.get('terminated_by', {})
    # A single termination reason (e.g. every conversation "judge"-ended) is a
    # tautology that adds no signal — only show the breakdown when it varies.
    if len(terminated_by) <= 1:
        return ''
    rows = [[_esc(r), str(c)] for r, c in sorted(terminated_by.items(), key=lambda kv: -kv[1])]
    return (
        f'<section class="report-card"><h2>{_esc(section.title)}</h2>'
        f'<h3>Terminated By</h3>{_html_table(["Reason", "Count"], rows)}</section>'
    )


def _render_turn_metrics_html(section: ReportSection) -> str:
    data = section.data
    per_conv = data.get('per_conversation', [])
    dist = data.get('turn_count_distribution', {})
    qualities = data.get('avg_quality_metrics', {})

    parts = [f'<section class="report-card"><h2>{_esc(section.title)}</h2>']

    # One bar per conversation reads well for small runs, but grows unbounded
    # and duplicates the distribution table at scale — so past a threshold show
    # the compact turn-count distribution instead.
    # A one-conversation bar chart is a single full-width bar — skip it; the
    # turn count is already in the conversation header.
    if len(per_conv) >= 2 and len(per_conv) <= _MAX_PER_CONV_BARS:
        parts.extend((
            _svg_bar(
                rows=[(c['label'], float(c['turns'])) for c in per_conv],
                title='Turns per Conversation',
                label_w=240,
                value_fmt=lambda v: f'{v:.0f}',
            ),
            '<p class="chart-note">Full persona · scenario names appear in Individual Conversations (#n).</p>',
        ))
    elif dist:
        parts.append(
            _svg_bar(
                rows=[(f'{t} turns', float(c)) for t, c in sorted(dist.items())],
                title='Conversations by Turn Count',
                label_w=110,
                value_fmt=lambda v: f'{v:.0f}',
            )
        )

    if qualities:
        parts.extend((
            '<h3>Average Per-Turn Quality Metrics</h3>',
            _html_table(
                ['Metric', 'Avg Score'],
                [
                    [f'{_esc(_pretty_evaluator(k))} <span class="dir">{_dir_arrow(k)}</span>', f'{v:.2f}']
                    for k, v in qualities.items()
                ],
            ),
        ))

    parts.append('</section>')
    return ''.join(parts)


def _render_failure_mode_html(section: ReportSection) -> str:
    rows = section.data.get('rows', [])
    # One failure mode = one full-width bar restating the Failures table — skip.
    if len(rows) < 2:
        return ''
    bar = _svg_bar(
        rows=[(label, float(count)) for label, count in rows],
        title=section.title,
        width=680,
        label_w=340,
        value_fmt=lambda v: f'{v:.0f}',
    )
    return f'<section class="report-card">{bar}</section>' if bar else ''


def _render_evaluator_scores_html(section: ReportSection) -> str:
    rows = section.data.get('rows', [])
    if not rows:
        return ''
    table_rows = []
    for r in rows:
        raw = r['evaluator']
        lower = _score_is_lower_better(raw)
        arrow = '▼' if lower else '▲'
        hint = 'lower is better' if lower else 'higher is better'
        dropped = r.get('dropped', 0)
        dropped_cell = str(dropped)
        if dropped and r.get('first_error'):
            dropped_cell = f'<span title="{_esc(str(r["first_error"]))}">{dropped}</span>'
        # None statistics mean every run of this evaluator failed — render a dash
        # rather than a chip, so a dead evaluator cannot read as a 0.00 score.
        mean = r['mean_score']
        table_rows.append([
            f'{_esc(_pretty_evaluator(raw))} <span class="dir" title="{_esc(raw)} · {hint}">{arrow}</span>',
            str(r['runs']),
            dropped_cell,
            '—' if mean is None else _score_chip(mean, lower_is_better=lower),
            '—' if r['min_score'] is None else f'{r["min_score"]:.2f}',
            '—' if r['max_score'] is None else f'{r["max_score"]:.2f}',
        ])
    table = _html_table(['Evaluator', 'Runs', 'Dropped', 'Mean', 'Min', 'Max'], table_rows)
    legend = (
        '<p class="score-legend">'
        '<span class="dir">▲</span> higher is better &nbsp;·&nbsp; '
        '<span class="dir">▼</span> lower is better &nbsp;·&nbsp; '
        'mean shaded <span class="score-chip score-chip--bad">worse</span> → '
        '<span class="score-chip score-chip--good">better</span>'
        '</p>'
    )
    return f'<section class="report-card"><h2>{_esc(section.title)}</h2>{table}{legend}</section>'


def _render_token_usage_html(section: ReportSection) -> str:
    table = _html_table(['Metric', 'Value'], build_token_usage_rows(section.data))
    return f'<section class="report-card"><h2>{_esc(section.title)}</h2>{table}</section>'


def _render_errors_html(section: ReportSection) -> str:
    data = section.data
    total = data.get('total_errored', 0)
    by_message = data.get('by_message', {})
    parts = [
        f'<section class="report-card"><h2>{_esc(section.title)}</h2>',
        f'<p>Total errored conversations: <strong>{total}</strong></p>',
    ]
    if by_message:
        rows = [[_esc(m), str(c)] for m, c in by_message.items()]
        parts.append(_html_table(['Error', 'Count'], rows))
    parts.append('</section>')
    return ''.join(parts)


def _render_individual_results_html(section: ReportSection) -> str:
    entries = section.data.get('entries', [])
    if not entries:
        return f'<section class="report-card"><h2>{_esc(section.title)}</h2><p>No conversations.</p></section>'

    parts = [f'<section class="report-card"><h2>{_esc(section.title)}</h2>']
    for entry in entries:
        anchor = f'conv-{entry["index"] + 1}'
        verdict = 'ACHIEVED' if entry['goal_achieved'] else 'NOT ACHIEVED'
        badge = _status_badge(verdict, 'pass' if entry['goal_achieved'] else 'fail')
        title = (
            f'#{entry["index"] + 1}: {_esc(entry["persona"])} / '
            f'{_esc(entry["scenario"])} {badge}'
            f' ({entry["turn_count"]} turn{"s" if entry["turn_count"] != 1 else ""}, '
            f'score {entry["goal_completion_score"]:.2f})'
        )

        meta_rows = []
        model = entry.get('target_model')
        if model:
            meta_rows.append(['Model', _esc(model)])
        meta_rows.extend((['Terminated by', _esc(entry['terminated_by'])], ['Tokens', f'{entry["total_tokens"]:,}']))

        criteria_rows = entry.get('criteria', [])
        if criteria_rows:
            # `state` (not `passed`) drives the colour: a criterion the judge never
            # audited is neutral, never green — see `_criteria_rows` (RES-1308).
            badges = ' '.join(
                _status_badge(
                    c['description'] + (' (not audited)' if c.get('state') == 'unknown' else ''),
                    {'pass': 'pass', 'fail': 'fail'}.get(c.get('state', 'pass' if c['passed'] else 'fail'), 'neutral'),
                )
                for c in criteria_rows
            )
            meta_rows.append(['Criteria', badges])
            if entry.get('criteria_verified') is False:
                meta_rows.append([
                    'Criteria audit',
                    _esc(
                        'Unverified — the judge returned no per-criterion audit, so these verdicts '
                        'are defaults and criteria_met scored this run 0.0.'
                    ),
                ])
            evidence_items = [
                f'<li>{_esc(c["description"])}: &ldquo;{_esc(c["evidence"])}&rdquo;</li>'
                for c in criteria_rows
                if c.get('evidence')
            ]
            if evidence_items:
                meta_rows.append(['Evidence', f'<ul class="crit-evidence">{"".join(evidence_items)}</ul>'])

        if entry['evaluator_scores']:
            meta_rows.append([
                'Evaluator scores',
                _esc(', '.join(f'{k}={v:.2f}' for k, v in entry['evaluator_scores'].items())),
            ])
        if entry['error']:
            meta_rows.append(['Error', _esc(entry['error'])])
        meta_rows.append(['Judge reason', _esc(entry['judge_reason'])])

        transcript_html = []
        for msg in entry['transcript']:
            role = _esc(msg['role'])
            raw = msg.get('content', '')
            # Truncate individual very long messages so the file stays
            # manageable, but keep the whole transcript visible in the
            # <details> block (no "full text in report JSON" indirection).
            if len(raw) > 4000:
                raw = raw[:4000] + '\n\n[message truncated]'
            content = _esc(raw).replace('\n', '<br>')
            transcript_html.append(f'<div class="transcript-message"><em>{role}:</em><br>{content}</div>')

        parts.append(
            f'<div id="{anchor}"><details><summary>{title}</summary>'
            f'{_html_table(["Field", "Value"], meta_rows)}'
            f'<h4>Transcript</h4>{"".join(transcript_html)}'
            f'</details></div>'
        )

    parts.append('</section>')
    return ''.join(parts)


def _render_recommendations_html(section: ReportSection) -> str:
    rows = section.data.get('rows', [])
    if not rows:
        return ''
    parts = [
        f'<section class="report-card"><h2>{_esc(section.title)}</h2>',
        (
            '<p>LLM-generated fixes for conversations the judge flagged with a '
            'concrete, remediable issue. Benign failures (e.g. plain max-turns) '
            'are not analyzed.</p>'
        ),
    ]
    for r in rows:
        datapoint = f' · datapoint <code>{_esc(str(r["datapoint_id"]))}</code>' if r.get('datapoint_id') else ''
        flagged = ''.join(_status_badge(t, 'fail') for t in r.get('triggers', []))
        fixes = ''.join(f'<li>{_esc(s)}</li>' for s in r.get('suggestions', []))
        parts.append(
            f'<div class="recommendation-entry"><h3><a href="#{r["anchor"]}">#{r["index"]}</a> '
            f'{_esc(r["persona"])} / {_esc(r["scenario"])}{datapoint}</h3>'
            f'<p class="rec-flagged"><strong>Triggered by:</strong> {flagged}</p>'
            f'<ol class="rec-fixes">{fixes}</ol></div>'
        )
    parts.append('</section>')
    return ''.join(parts)


_SECTION_RENDERERS = {
    'summary': _render_summary_html,
    'overview': _render_overview_html,
    'failures_first': _render_failures_first_html,
    'recommendations': _render_recommendations_html,
    'persona_scenario_heatmap': _render_persona_scenario_heatmap_html,
    'score_distribution': _render_score_distribution_html,
    'turn_quality_timeline': _render_turn_quality_timeline_html,
    'persona_breakdown': _render_persona_breakdown_html,
    'scenario_breakdown': _render_scenario_breakdown_html,
    'judge_verdicts': _render_judge_verdicts_html,
    'turn_metrics': _render_turn_metrics_html,
    'failure_mode': _render_failure_mode_html,
    'evaluator_scores': _render_evaluator_scores_html,
    'token_usage': _render_token_usage_html,
    'errors': _render_errors_html,
    'individual_results': _render_individual_results_html,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_report_body(
    results: list[SimulationResult],
    *,
    target: str = 'agent',
    run_date: datetime | None = None,
    executive_summary: str | None = None,
    experiment_url: str | None = None,
    recommendations: list[SimulationRecommendation] | None = None,
    run_token_usage_total: Usage | None = None,
) -> str:
    """Render simulation results as an HTML body fragment (no ``<html>`` or ``<head>`` wrapper).

    This is the body-only counterpart to ``export_html``.  It assembles the
    same hero header, KPI band, and section content but does not wrap the
    output in a full HTML document.  Use this function when you need to embed
    the report inside another shell (e.g. a FastHTML dashboard page).

    Args:
        results: Simulation results to render.
        target: Display name for the target agent.
        run_date: Report generation timestamp (defaults to now).
        executive_summary: Optional narrative summary for the hero header.
        experiment_url: Optional absolute URL to the uploaded Orq experiment
            run; when set, renders an "Open experiment in Orq" button in the
            hero header.
        recommendations: Pre-generated remediation suggestions
            (``SimulationRun.recommendations``); rendered as their own
            section when non-empty.
        run_token_usage_total: ``SimulationRun.token_usage_total`` — pass this
            when *results* is the run's full result set so the Token Usage
            section also shows the whole-run figure (simulation + generation +
            executive summary), labelled distinctly from the simulation-only
            total. Omit for a filtered/partial *results* list.

    Returns:
        An HTML fragment string (no ``<!DOCTYPE>``, ``<html>``, or ``<head>``).
    """
    sections = build_report_sections(
        results,
        executive_summary=executive_summary,
        recommendations=recommendations,
        run_token_usage_total=run_token_usage_total,
    )
    summary_data = next((s.data for s in sections if s.kind == 'summary'), {})

    sd = summary_data
    verdict = sd.get('verdict', 'neutral')
    success_status = 'pass' if verdict == 'pass' else ('warn' if verdict == 'warn' else 'fail')
    errors = sd.get('errors', 0)
    kpis = _kpi_cards([
        {
            'label': 'Success Rate',
            'value': _pct(sd.get('success_rate', 0.0)),
            'status': success_status,
        },
        {
            'label': 'Avg Score',
            'value': f'{sd.get("avg_goal_completion_score", 0.0):.2f}',
            'status': 'neutral',
        },
        {
            'label': 'Conversations',
            'value': str(sd.get('total_conversations', 0)),
            'status': 'neutral',
        },
        {
            # "Runtime Errors" = crashes, not goal failures — distinct from the
            # success rate so a low score isn't masked by "0 Errors".
            'label': 'Runtime Errors',
            'value': str(errors),
            'status': 'warn' if errors else 'neutral',
        },
    ])

    verdict_html = _build_verdict_line(sections, sd)
    summary_section = next((s for s in sections if s.kind == 'summary'), None)
    narrative_html = _render_summary_html(summary_section) if summary_section is not None else ''
    experiment_link_html = trace_link_button(experiment_url, 'Open experiment in Orq')
    header_html = (
        '<header class="hero"><h1>Agent Simulation Report</h1>'
        f'<p><strong>Target:</strong> {_esc(target)} &nbsp;|&nbsp; '
        f'<strong>Date:</strong> {_format_date(run_date or datetime.now(tz=timezone.utc))}</p>'
        f'{experiment_link_html}{verdict_html}{narrative_html}{kpis}</header>'
    )

    return _render_body(
        [section for section in sections if section.kind != 'summary'],
        renderers=_SECTION_RENDERERS,
        body_header=header_html,
        body_footer='<footer><p class="footer">Generated by evaluatorq agent simulation suite.</p></footer>',
    )


def export_html(
    results: list[SimulationResult],
    *,
    target: str = 'agent',
    run_date: datetime | None = None,
    executive_summary: str | None = None,
    experiment_url: str | None = None,
    recommendations: list[SimulationRecommendation] | None = None,
    run_token_usage_total: Usage | None = None,
) -> str:
    """Render a list of simulation results as a self-contained HTML document."""
    head = (
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>Agent Simulation Report</title>\n'
        f'<style>\n{_load_css()}\n</style>'
    )
    body_html = render_report_body(
        results,
        target=target,
        run_date=run_date,
        executive_summary=executive_summary,
        experiment_url=experiment_url,
        recommendations=recommendations,
        run_token_usage_total=run_token_usage_total,
    )
    return (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        f'<head>\n{head}\n</head>\n'
        '<body>\n<div class="container">\n'
        f'{body_html}\n'
        '</div>\n</body>\n'
        '</html>\n'
    )
