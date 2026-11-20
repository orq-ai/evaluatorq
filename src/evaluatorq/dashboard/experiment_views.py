"""Report body for a saved ``evaluatorq()`` experiment run (the dashboard's ``experiment`` surface).

Pure HTML builders, like ``report_kit``: no FastHTML, so the offline export in
``common.reports.standalone`` renders the same body. Every dynamic string goes
through ``esc()``. Long inputs and outputs are cut short in the cell only; the
full text stays in the saved run and opens from the cell.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from evaluatorq.common.output_adapters import output_to_text
from evaluatorq.common.reports import esc, kpi_cards
from evaluatorq.common.reports.html_helpers import humanize_duration, parse_http_url, pct
from evaluatorq.dashboard.report_kit import bar_rows, meta_grid, panel, pill
from evaluatorq.experiment_run import numeric_score

if TYPE_CHECKING:
    from evaluatorq.experiment_run import EvaluatorStats, ExperimentRun
    from evaluatorq.types import DataPointResult, EvaluatorScore, JobResult

CELL_PREVIEW_CHARS = 160
"""Characters of an input or output shown in a table cell before the rest folds away."""


def _clip(text: str, limit: int = CELL_PREVIEW_CHARS) -> str:
    """Show *text* in full when short, else a preview that expands to the full text."""
    if len(text) <= limit:
        return f'<span class="exp-text">{esc(text)}</span>'
    return (
        f'<details class="exp-clip"><summary>{esc(text[:limit].rstrip())}…</summary>'
        f'<pre class="exp-full">{esc(text)}</pre></details>'
    )


def _inputs_text(result: DataPointResult) -> str:
    inputs = result.data_point.inputs
    if len(inputs) == 1:
        value = next(iter(inputs.values()))
        if isinstance(value, str):
            return value
    return json.dumps(inputs, ensure_ascii=False, default=str)


def _score_cell(score: EvaluatorScore | None) -> str:
    if score is None:
        return '<span class="exp-none">—</span>'
    if score.error is not None:
        return f'<span title="{esc(score.error)}">{pill("error", tone="warn", dot=True)}</span>'
    result = score.score
    value = result.value
    number = numeric_score(value)
    if isinstance(value, bool):
        shown = 'true' if value else 'false'
    elif number is not None:
        shown = f'{number:.3g}'
    elif isinstance(value, str):
        shown = value if len(value) <= 40 else f'{value[:40].rstrip()}…'
    else:
        shown = 'structured'
    title = f' title="{esc(result.explanation)}"' if result.explanation else ''
    verdict = ''
    if result.pass_ is not None:
        verdict = pill('pass', tone='good') if result.pass_ else pill('fail', tone='danger')
    return f'<span class="exp-score"{title}><span class="exp-score-value">{esc(shown)}</span>{verdict}</span>'


def _status_cell(error: str | None) -> str:
    if error is None:
        return pill('ok', tone='good', dot=True)
    return f'<span title="{esc(error)}">{pill("error", tone="warn", dot=True)}</span>'


def _output_cell(job: JobResult) -> str:
    if job.error is not None:
        text = output_to_text(job.output)
        detail = f'<div class="exp-output-after-error">{_clip(text)}</div>' if text else ''
        return f'<div class="exp-error">{_clip(job.error)}</div>{detail}'
    return _clip(output_to_text(job.output))


def _row_table(run: ExperimentRun, names: list[str]) -> str:
    if not run.results:
        return '<p class="exp-empty">This run has no datapoints.</p>'
    head = ''.join(f'<th>{esc(h)}</th>' for h in ('#', 'Inputs', 'Job', 'Output', *names, 'Status'))
    rows: list[str] = []
    for index, result in enumerate(run.results, start=1):
        inputs = _clip(_inputs_text(result))
        jobs = result.job_results or []
        if not jobs:
            empty_scores = ''.join('<td><span class="exp-none">—</span></td>' for _ in names)
            output = f'<div class="exp-error">{_clip(result.error)}</div>' if result.error else '—'
            rows.append(
                f'<tr class="{"exp-row--error" if result.error else ""}"><td>{index}</td><td>{inputs}</td>'
                f'<td>—</td><td>{output}</td>{empty_scores}<td>{_status_cell(result.error)}</td></tr>'
            )
            continue
        for job in jobs:
            by_name = {s.evaluator_name: s for s in job.evaluator_scores or []}
            scores = ''.join(f'<td>{_score_cell(by_name.get(name))}</td>' for name in names)
            error = job.error or result.error
            rows.append(
                f'<tr class="{"exp-row--error" if error else ""}"><td>{index}</td><td>{inputs}</td>'
                f'<td class="exp-job">{esc(job.job_name)}</td><td>{_output_cell(job)}</td>{scores}<td>{_status_cell(error)}</td></tr>'
            )
    return f'<div class="exp-table-wrap"><table class="exp-rows"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _evaluator_table(stats: list[EvaluatorStats]) -> str:
    if not stats:
        return '<p class="exp-empty">No evaluators ran in this run.</p>'
    rows = []
    for s in stats:
        average = '—' if s.average is None else f'{s.average:.3g}'
        pass_rate = '—' if s.pass_rate is None else f'{pct(s.pass_rate)} ({s.passed}/{s.pass_set})'
        rows.append(
            f'<tr><td>{esc(s.name)}</td><td>{esc(average)}</td><td>{esc(pass_rate)}</td>'
            f'<td>{s.scored}</td><td>{s.non_numeric}</td><td>{s.errors}</td></tr>'
        )
    head = ''.join(
        f'<th>{h}</th>' for h in ('Evaluator', 'Average', 'Pass rate', 'Numeric scores', 'Non-numeric', 'Errors')
    )
    return f'<table class="exp-evaluators"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def _hero(run: ExperimentRun) -> str:
    when = run.started_at.strftime('%Y-%m-%d %H:%M %Z')
    duration = 'under 1s' if run.duration_seconds < 1 else humanize_duration(run.duration_seconds)
    description = f' · {esc(run.description)}' if run.description else ''
    link = ''
    if (url := parse_http_url(run.experiment_url)) is not None:
        link = f' · <a href="{esc(url.geturl())}" target="_blank" rel="noopener">Open in Orq</a>'
    return (
        '<header class="report-hero">'
        '<p class="report-hero-kicker">Experiment</p>'
        f'<h2 class="report-hero-title">{esc(run.run_name)}</h2>'
        f'<p class="report-hero-sub">{esc(when)} · {esc(duration)}{description}{link}</p>'
        '</header>'
    )


def experiment_report_body(run: ExperimentRun) -> str:
    """Render a saved experiment run: KPIs, per-evaluator averages and pass rates, and one row per job result."""
    stats = run.evaluator_stats()
    names = [s.name for s in stats]
    counts = run.row_counts()
    errored = counts.errored
    pass_rate = run.pass_rate()
    tokens, cost = run.judge_usage()
    kpis = kpi_cards([
        {'label': 'Datapoints', 'value': str(len(run.results))},
        {'label': 'Job results', 'value': str(counts.job_results)},
        {
            'label': 'Errored rows',
            'value': str(errored),
            'status': 'warn' if errored else 'neutral',
            'sub': 'a job or datapoint that raised',
        },
        {
            'label': 'Pass rate',
            'value': pct(pass_rate),
            'status': 'neutral' if pass_rate is None else ('pass' if pass_rate >= 0.8 else 'warn'),
            'sub': 'scores that set pass' if pass_rate is not None else 'no evaluator set pass',
        },
        {
            'label': 'Judge tokens',
            'value': f'{tokens:,}',
            'sub': f'${cost:.4f}' if cost is not None else 'no cost reported',
        },
    ])
    pass_bars = [(s.name, s.pass_rate * 100) for s in stats if s.pass_rate is not None]
    chart = bar_rows(pass_bars, width=520, label_w=180, fmt=lambda v: f'{v:.0f}%', max_value=100.0) if pass_bars else ''
    evaluators = panel(
        'Evaluators',
        _evaluator_table(stats) + (f'<div class="exp-chart">{chart}</div>' if chart else ''),
        sub='Average over numeric scores; pass rate over scores that set pass. Errored scores count in neither.',
    )
    rows = panel(
        'Rows',
        _row_table(run, names),
        sub='One row per job result. Long inputs and outputs are shortened here; open a cell to read all of it.',
    )
    details = panel(
        'Run details',
        meta_grid([
            ('Run id', run.run_id),
            ('Dataset id', run.dataset_id),
            ('Started', run.started_at.strftime('%Y-%m-%d %H:%M:%S %Z')),
            ('Ended', run.ended_at.strftime('%Y-%m-%d %H:%M:%S %Z')),
            ('Duration', f'{run.duration_seconds:.1f}s'),
            ('evaluatorq version', run.evaluatorq_version),
        ]),
    )
    return f'<div class="report-aligned exp-report">{_hero(run)}{kpis}{evaluators}{rows}{details}</div>'
