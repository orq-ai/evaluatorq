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
from evaluatorq.common.reports import esc
from evaluatorq.common.reports.html_helpers import humanize_duration, parse_http_url, pct
from evaluatorq.dashboard.report_kit import meta_grid, panel, pill
from evaluatorq.experiment_run import combined_pass_rate, numeric_score

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
        f'<details class="exp-clip"><summary><span class="exp-preview">{esc(text[:limit].rstrip())}…</span>'
        '<span class="exp-toggle exp-toggle--more">Show all</span><span class="exp-toggle exp-toggle--less">Show less</span>'
        f'</summary><pre class="exp-full">{esc(text)}</pre></details>'
    )


def _inputs_text(result: DataPointResult) -> str:
    inputs = result.data_point.inputs
    if len(inputs) == 1:
        value = next(iter(inputs.values()))
        if isinstance(value, str):
            return value
    return json.dumps(inputs, ensure_ascii=False, default=str)


def _number(value: float) -> str:
    """Three significant digits below 1000, else the whole number: ``.3g`` alone shows 12345 as ``1.23e+04``."""
    return f'{value:.3g}' if abs(value) < 1000 else f'{value:,.0f}'


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
        shown = _number(number)
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


def _duration(run: ExperimentRun) -> str:
    return 'under 1s' if run.duration_seconds < 1 else humanize_duration(run.duration_seconds)


def _evaluator_scores(stats: list[EvaluatorStats]) -> str:
    """One row per evaluator: its pass rate on the shared score track, then the figure and the average."""
    if not stats:
        return '<p class="exp-empty">No evaluators ran in this run.</p>'
    rows = []
    for s in stats:
        rate = s.pass_rate
        hint = [f'{s.errors} errored' if s.errors else '', f'{s.non_numeric} non-numeric' if s.non_numeric else '']
        hint_text = ' · '.join(h for h in hint if h)
        hint_html = f'<span class="cmp-score-hint">{esc(hint_text)}</span>' if hint_text else ''
        if rate is None:
            track = '<span class="cmp-track"><span class="cmp-track-na">no pass set</span></span>'
            figure = '—'
        else:
            track = f'<span class="cmp-track"><span class="cmp-track-fill cmp-track-fill--a" style="width:{rate * 100:.1f}%"></span></span>'
            figure = f'{pct(rate)} ({s.passed}/{s.pass_set})'
        average = '—' if s.average is None else f'avg {_number(s.average)}'
        rows.append(
            f'<div class="cmp-score-row"><span class="cmp-score-label">{esc(s.name)}{hint_html}</span>'
            f'<span class="cmp-score-tracks" aria-hidden="true">{track}</span>'
            f'<span class="cmp-score-ab">{esc(figure)}</span><span class="cmp-score-ab">{esc(average)}</span></div>'
        )
    return f'<div class="cmp-scores">{"".join(rows)}</div>'


def _hero(run: ExperimentRun, *, errored: int) -> str:
    """Report hero plus the Insights-style status line: when, how long, how many, and where it lives in Orq."""
    tokens, cost = run.judge_usage()
    chips = [
        run.started_at.strftime('%Y-%m-%d %H:%M %Z').strip(),
        _duration(run),
        f'{len(run.results):,} datapoints',
        f'{run.row_counts().job_results:,} job results',
        f'{tokens:,} judge tokens' + (f' · ${cost:.4f}' if cost is not None else ''),
    ]
    status = ''.join(f'<span class="cmp-chip">{esc(chip)}</span>' for chip in chips)
    if errored:
        status += f'<span class="cmp-chip cmp-chip--warn">{errored:,} errored</span>'
    if run.description:
        status += f'<span class="cmp-run-meta">{esc(run.description)}</span>'
    if (url := parse_http_url(run.experiment_url)) is not None:
        status += f'<a href="{esc(url.geturl())}" target="_blank" rel="noopener">Open in Orq</a>'
    return (
        '<header class="report-hero"><p class="report-hero-kicker">Experiment</p>'
        f'<h2 class="report-hero-title">{esc(run.run_name)}</h2></header>'
        f'<div class="cmp-status">{status}</div>'
    )


def _headline(number: str, finding: str, why: str, *, unit: str = '', tone: str = '') -> str:
    """One Insights-style headline row: a big number, the finding, and why it reads that way."""
    unit_html = f' <span class="rk-head-unit">{esc(unit)}</span>' if unit else ''
    tone_cls = f' rk-head--{tone}' if tone else ''
    return (
        f'<div class="rk-head{tone_cls}"><span class="rk-head-number">{esc(number)}{unit_html}</span>'
        f'<span><span class="rk-head-finding">{esc(finding)}</span><span class="rk-head-why">{esc(why)}</span></span>'
        '<span></span></div>'
    )


def _headlines(stats: list[EvaluatorStats], *, errored: int, rows: int) -> str:
    """The overall pass rate, the weakest evaluator when there are several, and errored rows when any."""
    heads: list[str] = []
    passed, pass_set = sum(s.passed for s in stats), sum(s.pass_set for s in stats)
    rate = combined_pass_rate(stats)
    if rate is not None:
        heads.append(
            _headline(
                pct(rate),
                'Pass rate across evaluators',
                f'{passed:,} of {pass_set:,} scores that set pass passed',
                tone='good' if rate >= 0.8 else 'warn',
            )
        )
    graded = [s for s in stats if s.pass_rate is not None]
    if len(graded) > 1:
        weakest = min(graded, key=lambda s: s.pass_rate or 0.0)
        weak_rate = weakest.pass_rate or 0.0
        heads.append(
            _headline(
                pct(weak_rate),
                f'Weakest evaluator: {weakest.name}',
                f'{weakest.passed:,} of {weakest.pass_set:,} passed',
                tone='bad' if weak_rate < 0.8 else '',
            )
        )
    if errored:
        heads.append(
            _headline(
                f'{errored:,}',
                'Rows errored',
                f'of {rows:,}; a job or datapoint raised, so those rows have no scores',
                unit='rows',
                tone='warn',
            )
        )
    return f'<section class="rk-heads" aria-label="Headlines">{"".join(heads)}</section>' if heads else ''


def experiment_report_body(run: ExperimentRun, *, total: int | None = None) -> str:
    """Render a saved experiment run the way the Insights page reads: status line, headlines, evaluators, rows.

    *total* is the unfiltered datapoint count when *run* holds a filtered subset; it feeds the "Showing x of y" bar.
    """
    stats = run.evaluator_stats()
    names = [s.name for s in stats]
    counts = run.row_counts()
    errored = counts.errored
    shown = len(run.results)
    total = shown if total is None else total
    fbar = (
        '<div class="rk-fbar" role="status">'
        f'<span class="rk-fbar-count">Showing <b>{shown:,}</b> of <b>{total:,}</b> datapoints</span>'
        '<span class="rk-fbar-hint">Use the filters to narrow the rows to errored or failing datapoints.</span></div>'
    )
    evaluators = panel(
        'Evaluators',
        _evaluator_scores(stats),
        sub='Pass rate over scores that set pass; average over numeric scores. Errored scores count in neither.',
    )
    details = panel(
        'Run details',
        meta_grid([
            ('Run id', run.run_id),
            ('Dataset id', run.dataset_id),
            ('Started', run.started_at.strftime('%Y-%m-%d %H:%M:%S %Z')),
            ('Ended', run.ended_at.strftime('%Y-%m-%d %H:%M:%S %Z')),
            ('Duration', _duration(run)),
            ('evaluatorq version', run.evaluatorq_version),
        ]),
    )
    rows = panel(
        'Rows',
        _row_table(run, names),
        sub='One row per job result. Long inputs and outputs are shortened here; open a cell to read all of it.',
    )
    return (
        f'<div class="report-aligned cmp-report exp-report">{_hero(run, errored=errored)}{fbar}'
        f'{_headlines(stats, errored=errored, rows=counts.rows)}'
        f'<div class="cmp-grid">{evaluators}{details}</div>{rows}</div>'
    )
