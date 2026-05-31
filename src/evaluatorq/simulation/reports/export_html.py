"""HTML renderer for agent simulation reports.

``export_html(results)`` produces a self-contained HTML document styled with
the shared report CSS. All charts are hand-authored SVG/HTML primitives from
``evaluatorq.common.reports`` so the simulation report renders identically
whether or not plotly/kaleido are installed (this module imports no plotly).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

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
    render_heatmap as _render_heatmap,
)
from evaluatorq.common.reports import (
    render_histogram as _render_histogram,
)
from evaluatorq.common.reports import (
    render_html as _render_html_doc,
)
from evaluatorq.common.reports import (
    render_line_chart as _render_line_chart,
)
from evaluatorq.common.reports import (
    render_sparkline as _render_sparkline,
)
from evaluatorq.common.reports import (
    status_badge as _status_badge,
)
from evaluatorq.common.reports import (
    svg_bar as _svg_bar,
)
from evaluatorq.common.reports import (
    svg_donut as _svg_donut,
)
from evaluatorq.common.reports.palette import COLORS
from evaluatorq.simulation.reports.sections import build_report_sections

if TYPE_CHECKING:
    from evaluatorq.contracts import ReportSection
    from evaluatorq.simulation.types import SimulationResult

# Heatmap colour direction:
# ``ORQ_SCALE_GOOD_BAD`` is green at 0.0 -> red at 1.0 (i.e. good == low).
# For these reports "good" means HIGH (a passing criterion / high success
# rate), so we use an explicit green-high scale: red at 0.0 -> green at 1.0.
# Renderers pass the raw pass-rate / success-rate value through it, so a fully
# passing cell is green and a failing cell red.
_SCALE_GREEN_HIGH: list[list[float | str]] = [
    [0.0, COLORS['red_400']],
    [1.0, COLORS['success_400']],
]


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_summary_html(section: ReportSection) -> str:
    data = section.data
    achieved = data.get('goals_achieved', 0)
    failed = data.get('goals_failed', 0)
    errors = data.get('errors', 0)

    donut = _svg_donut(
        labels=['Achieved', 'Not Achieved', 'Errored'],
        values=[achieved, failed, errors],
        colors=[COLORS['success_400'], COLORS['red_400'], COLORS['orange_300']],
        center_label=_pct(data.get('success_rate', 0.0)),
        title='Goal Outcomes',
    )
    body = donut or '<p>No conversations to summarize.</p>'
    return f'<section class="report-card"><h2>{_esc(section.title)}</h2>{body}</section>'


def _render_failures_first_html(section: ReportSection) -> str:
    rows = section.data.get('rows', [])
    if not rows:
        return '<section class="report-card"><h2>Failures</h2><p>No failed conversations.</p></section>'
    trs = []
    for r in rows:
        badges = ''.join(_status_badge(v, 'fail') for v in r['violated']) or '—'
        safety = _status_badge('SAFETY', 'fail') if r['has_safety'] else ''
        trs.append(
            f'<tr><td><a href="#{r["anchor"]}">#{r["index"]}</a></td>'
            f'<td>{_esc(r["persona"])}</td><td>{_esc(r["scenario"])}</td>'
            f'<td>{badges} {safety}</td><td>{r["score"]:.2f}</td>'
            f'<td>{_esc(r["terminated_by"])}</td></tr>'
        )
    return (
        '<section class="report-card"><h2>Failures</h2>'
        '<table><thead><tr><th>#</th><th>Persona</th><th>Scenario</th>'
        '<th>Violated criteria</th><th>Score</th><th>Ended</th></tr></thead>'
        f'<tbody>{"".join(trs)}</tbody></table></section>'
    )


def _render_criteria_heatmap_html(section: ReportSection) -> str:
    d = section.data
    if not d.get('y_labels'):
        return ''
    heat = _render_heatmap(
        x_labels=d['x_labels'],
        y_labels=d['y_labels'],
        cells=d['cells'],
        scale=_SCALE_GREEN_HIGH,
        title=section.title,
        value_fmt=lambda v: '—' if v < 0 else ('PASS' if v >= 0.5 else 'FAIL'),
        safety_mask=d['safety'],
    )
    return f'<section class="report-card">{heat}</section>'


def _render_persona_scenario_heatmap_html(section: ReportSection) -> str:
    d = section.data
    personas, scenarios = d['personas'], d['scenarios']
    if not personas or not scenarios:
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
        value_fmt=lambda v: '—' if v < 0 else f'{v:.0%}',
    )
    return f'<section class="report-card">{heat}</section>'


def _render_score_distribution_html(section: ReportSection) -> str:
    hist = _render_histogram(values=section.data.get('scores', []), bins=10, title=section.title)
    return f'<section class="report-card">{hist}</section>' if hist else ''


def _render_turn_quality_timeline_html(section: ReportSection) -> str:
    d = section.data
    turns = d.get('turns', [])
    if not turns:
        return ''
    series = [(name, vals) for name, vals in d['series'].items() if any(vals)]
    if not series:
        return ''
    chart = _render_line_chart(x_labels=[str(t) for t in turns], series=series, title=section.title)
    return f'<section class="report-card">{chart}</section>'


def _render_persona_breakdown_html(section: ReportSection) -> str:
    rows = section.data.get('rows', [])
    if not rows:
        return f'<section class="report-card"><h2>{_esc(section.title)}</h2><p>No persona data.</p></section>'
    table_rows = [
        [
            _esc(r['persona']),
            str(r['conversations']),
            str(r['goals_achieved']),
            f'{_pct(r["success_rate"])} {_render_sparkline([r["success_rate"], 1.0 - r["success_rate"]])}',
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
    table_rows = [
        [
            _esc(r['scenario']),
            str(r['conversations']),
            str(r['goals_achieved']),
            f'{_pct(r["success_rate"])} {_render_sparkline([r["success_rate"], 1.0 - r["success_rate"]])}',
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

    parts = [f'<section class="report-card"><h2>{_esc(section.title)}</h2>']

    if terminated_by:
        rows = [[_esc(r), str(c)] for r, c in sorted(terminated_by.items(), key=lambda kv: -kv[1])]
        parts.extend(('<h3>Terminated By</h3>', _html_table(['Reason', 'Count'], rows)))

    parts.append('</section>')
    return ''.join(parts)


def _render_turn_metrics_html(section: ReportSection) -> str:
    data = section.data
    dist = data.get('turn_count_distribution', {})
    qualities = data.get('avg_quality_metrics', {})

    parts = [f'<section class="report-card"><h2>{_esc(section.title)}</h2>']

    if dist:
        bar = _svg_bar(
            rows=[(f'{t} turns', float(c)) for t, c in dist.items()],
            title='Conversations by Turn Count',
        )
        parts.extend((bar, _html_table(['Turns', 'Conversations'], [[str(t), str(c)] for t, c in dist.items()])))

    if qualities:
        parts.extend((
            '<h3>Average Per-Turn Quality Metrics</h3>',
            _html_table(['Metric', 'Avg Score'], [[_esc(k), f'{v:.2f}'] for k, v in qualities.items()]),
        ))

    parts.append('</section>')
    return ''.join(parts)


def _render_failure_mode_html(section: ReportSection) -> str:
    rows = section.data.get('rows', [])
    if not rows:
        return ''
    bar = _svg_bar(rows=[(label, float(count)) for label, count in rows], title=section.title)
    return f'<section class="report-card">{bar}</section>' if bar else ''


def _render_evaluator_scores_html(section: ReportSection) -> str:
    rows = section.data.get('rows', [])
    if not rows:
        return ''
    table_rows = [
        [
            _esc(r['evaluator']),
            str(r['runs']),
            f'{r["mean_score"]:.2f}',
            f'{r["min_score"]:.2f}',
            f'{r["max_score"]:.2f}',
        ]
        for r in rows
    ]
    table = _html_table(['Evaluator', 'Runs', 'Mean', 'Min', 'Max'], table_rows)
    return f'<section class="report-card"><h2>{_esc(section.title)}</h2>{table}</section>'


def _render_token_usage_html(section: ReportSection) -> str:
    data = section.data
    rows = [
        ['Prompt Tokens (total)', f'{data.get("prompt_tokens", 0):,}'],
        ['Completion Tokens (total)', f'{data.get("completion_tokens", 0):,}'],
        ['Total Tokens', f'{data.get("total_tokens", 0):,}'],
        ['Avg Total / Conversation', f'{data.get("avg_total_per_conversation", 0):,.0f}'],
        ['Avg Prompt / Conversation', f'{data.get("avg_prompt_per_conversation", 0):,.0f}'],
        [
            'Avg Completion / Conversation',
            f'{data.get("avg_completion_per_conversation", 0):,.0f}',
        ],
    ]
    table = _html_table(['Metric', 'Value'], rows)
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
            f'{_esc(entry["scenario"])} — {badge}'
            f' ({entry["turn_count"]} turns, '
            f'score {entry["goal_completion_score"]:.2f})'
        )

        meta_rows = []
        model = entry.get('target_model')
        if model:
            meta_rows.append(['Model', _esc(model)])
        meta_rows.extend((['Terminated by', _esc(entry['terminated_by'])], ['Tokens', f'{entry["total_tokens"]:,}']))

        criteria_rows = entry.get('criteria', [])
        if criteria_rows:
            badges = ' '.join(_status_badge(c['description'], 'pass' if c['passed'] else 'fail') for c in criteria_rows)
            meta_rows.append(['Criteria', badges])

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


_SECTION_RENDERERS = {
    'summary': _render_summary_html,
    'failures_first': _render_failures_first_html,
    'persona_scenario_heatmap': _render_persona_scenario_heatmap_html,
    'criteria_heatmap': _render_criteria_heatmap_html,
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


def export_html(
    results: list[SimulationResult],
    *,
    target: str = 'agent',
    run_date: datetime | None = None,
) -> str:
    """Render a list of simulation results as a self-contained HTML document."""
    sections = build_report_sections(results)
    summary_data = next((s.data for s in sections if s.kind == 'summary'), {})

    head = (
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>Agent Simulation Report</title>\n'
        f'<style>\n{_load_css()}\n</style>'
    )

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
            'label': 'Errors',
            'value': str(errors),
            'status': 'warn' if errors else 'neutral',
        },
    ])

    header_html = (
        '<header class="hero"><h1>Agent Simulation Report</h1>'
        f'<p><strong>Target:</strong> {_esc(target)} &nbsp;|&nbsp; '
        f'<strong>Date:</strong> {_format_date(run_date or datetime.now(tz=timezone.utc))}</p>'
        f'{kpis}</header>'
    )

    return _render_html_doc(
        sections,
        renderers=_SECTION_RENDERERS,
        head=head,
        body_header=header_html,
        body_footer='<footer><p class="footer">Generated by evaluatorq agent simulation suite.</p></footer>',
    )
