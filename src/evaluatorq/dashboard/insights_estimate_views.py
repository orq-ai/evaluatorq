"""Render an Insights run estimate: the step-bar line and the review step's full block."""

from __future__ import annotations

from typing import TYPE_CHECKING

from evaluatorq.common.reports import esc

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from evaluatorq.insights.estimate import RunEstimate, StageEstimate

_CODING_TITLE = 'Coding questions (extra classify call)'


def _money(value: float) -> str:
    if value == 0:
        return '$0.00'
    return '<$0.01' if value < 0.01 else f'${value:,.2f}'


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f'{round(seconds)} s'
    if seconds < 5400:
        return f'{round(seconds / 60)} min'
    return f'{seconds / 3600:.1f} h'


def _span(low: float, high: float, fmt: Callable[[float], str], *, prefix: str = '') -> str:
    """One ceiling when the figures agree, else the range they span."""
    if fmt(low) == fmt(high):
        return f'{prefix}{fmt(high)}' if prefix else f'up to {fmt(high)}'
    return f'{prefix}{fmt(low)} to {fmt(high)}'


def _traces_value(estimate: RunEstimate) -> str:
    bound = estimate.bound
    noun = 'trace' if bound.n == 1 else 'traces'
    return f'{bound.n:,} {noun}' if bound.exact else f'up to {bound.n:,} {noun}'


def _cost_value(estimate: RunEstimate) -> str:
    if estimate.cost_low is None or estimate.cost_high is None:
        return 'unknown'
    text = _span(estimate.cost_low, estimate.cost_high, _money)
    return f'{text} (priced stages only)' if estimate.cost_partial else text


def _time_value(estimate: RunEstimate) -> str:
    if estimate.seconds_low is None or estimate.seconds_high is None:
        return estimate.time_note or 'unknown'
    text = _span(estimate.seconds_low, estimate.seconds_high, _duration, prefix='about ')
    return f'{text} (timed stages only)' if estimate.time_partial else text


def render_compact_estimate(estimate: RunEstimate) -> str:
    """The one-line summary shown in the step bar."""
    traces = 'traces unknown' if estimate.bound.n is None else _traces_value(estimate)
    return esc(f'{traces} · cost {_cost_value(estimate)} · time {_time_value(estimate)}')


def render_estimate_unavailable(message: str) -> str:
    """The step-bar line while the form cannot be estimated yet."""
    return esc(f'Estimate unavailable: {message}')


def _title(row: StageEstimate, titles: dict[str, str]) -> str:
    return _CODING_TITLE if row.name == 'label_coding' else titles.get(row.name, row.name)


def _row(row: StageEstimate, titles: dict[str, str]) -> str:
    traces = (
        f'{row.traces_high:,}' if row.traces_low == row.traces_high else f'{row.traces_low:,} to {row.traces_high:,}'
    )
    cost = 'unknown' if row.cost_low is None or row.cost_high is None else _span(row.cost_low, row.cost_high, _money)
    time = (
        'unknown'
        if row.seconds_low is None or row.seconds_high is None
        else _span(row.seconds_low, row.seconds_high, _duration, prefix='about ')
    )
    return (
        f'<tr><td>{esc(_title(row, titles))}</td><td>{esc(traces)}</td><td>{esc(cost)}</td><td>{esc(time)}</td>'
        f'<td class="irf-estimate-basis">{esc(row.basis)}</td></tr>'
    )


def render_estimate(estimate: RunEstimate, stages: Sequence[tuple[str, str]]) -> str:
    """The review step's full block: totals, each number's basis, the per-stage table and what is unknown."""
    titles = dict(stages)
    traces = 'unknown' if estimate.bound.n is None else _traces_value(estimate)
    lines = (
        f'<li><strong>Traces</strong> {esc(traces)} <small>{esc(estimate.bound.basis)}</small></li>'
        f'<li><strong>Cost</strong> {esc(_cost_value(estimate))} '
        '<small>traces x tokens per stage x each stage&#x27;s model price from the Orq catalogue</small></li>'
        f'<li><strong>Time</strong> {esc(_time_value(estimate))} <small>{esc(estimate.time_basis)}</small></li>'
    )
    note = f'<p class="irf-hint">{esc(estimate.time_note)}.</p>' if estimate.time_note and estimate.seconds_high else ''
    unknowns = ''.join(f'<li>{esc(item)}</li>' for item in estimate.unknowns)
    unknown_block = f'<h5>Unknown</h5><ul class="irf-estimate-unknowns">{unknowns}</ul>' if unknowns else ''
    head = '<th>Stage</th><th>Traces</th><th>Cost</th><th>Time</th><th>Basis</th>'
    body = ''.join(_row(row, titles) for row in estimate.rows)
    excluded = f'<p class="irf-hint">Not included: {esc("; ".join(estimate.excluded))}.</p>'
    return (
        '<section class="irf-estimate"><h4>Estimate</h4>'
        f'<ul class="irf-estimate-totals">{lines}</ul>{note}'
        f'<table class="irf-estimate-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
        f'{unknown_block}{excluded}</section>'
    )
