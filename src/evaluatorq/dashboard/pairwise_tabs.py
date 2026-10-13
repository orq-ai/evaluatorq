"""Tabbed pairwise report for the dashboard.

Overview, Judges, Comparisons and Config, in the same CSS-only tab group the
simulation and red-team reports use. The standalone export
(``pairwise_reports.export_html``) is a separate, flat rendering of the same
run and is deliberately left alone; this module reads the numbers from
``pairwise_reports.stats`` and the judge section data from
``pairwise_reports.sections``, so the page and the export cannot disagree about
what a judge did.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from itertools import starmap
from typing import TYPE_CHECKING, Any

from evaluatorq.common.reports import esc, kpi_cards
from evaluatorq.dashboard.pairwise_palette import pw_side_colour, verdict_eyebrow
from evaluatorq.dashboard.pairwise_views import render_pairwise_comparisons, valid_focus
from evaluatorq.dashboard.report_kit import bar_rows, donut, meta_grid, panel
from evaluatorq.dashboard.report_tabs import _tabs
from evaluatorq.pairwise_reports.export_html import _bias_cell, _consistency_cell, _rate
from evaluatorq.pairwise_reports.sections import (
    INCONCLUSIVE_WARN,
    POSITION_BIAS_WARN,
    build_report_sections,
    judge_order,
    observed_swap,
)
from evaluatorq.pairwise_reports.stats import judge_usage, trust_signals, verdict

if TYPE_CHECKING:
    from evaluatorq.pairwise_reports.stats import TrustSignal, Verdict
    from evaluatorq.pairwise_run import PairwiseEntry, PairwiseRun

_EMPTY_FILTERED = '<p class="pw-note">No comparisons match the current filter.</p>'
_EMPTY_RUN = '<p class="pw-note">No comparisons in this run.</p>'

_BIAS_HEADER_TIP = (
    'Share of completed pairs where the judge changed its answer when the sides were swapped. '
    f'Flagged at {POSITION_BIAS_WARN:g} or more.'
)
_SHRUNK_TIP = 'reliability weight, shrunk toward the panel mean'
_RAW_TIP = 'how often the judge agreed with itself across repeated passes'


def _aggregation(run: PairwiseRun) -> str:
    """The aggregation the stored report was built with."""
    return 'bt-sigma' if run.report is not None and run.report.bt_sigma is not None else 'plurality'


def with_entries(run: PairwiseRun, entries: list[PairwiseEntry]) -> PairwiseRun:
    """A copy of *run* carrying only *entries*, with the rollup recomputed.

    The stored report describes the full run, so it would misreport a filtered
    view. Recomputed once here rather than nulled: ``rollup()`` is called by the
    header and by several section builders, so leaving the field empty would
    roll the same entries up once per caller.
    """
    aggregation = _aggregation(run)
    filtered_run = run.model_copy(update={'entries': list(entries), 'report': None})
    rolled = filtered_run.rollup(aggregation=aggregation)
    return filtered_run.model_copy(update={'report': rolled})


def _swatch(side: str, label: str) -> str:
    cls = 'pw-side-a' if side == 'A' else 'pw-side-b'
    return f'<span class="pw-swatch" style="background:{pw_side_colour(side)}"></span><span class="{cls}">{esc(label)}</span>'


def _swap_text(run: PairwiseRun) -> str:
    if not run.swap:
        return 'off'
    return 'on' if observed_swap(run) else 'on (never observed)'


def _hero(run: PairwiseRun, hero_actions_html: str) -> str:
    judges = len(run.judges) or len(judge_order(run))
    criteria = f' · criteria: {esc(run.criteria)}' if run.criteria else ''
    actions = f'<div class="report-hero-actions">{hero_actions_html}</div>' if hero_actions_html else ''
    return (
        '<header class="report-hero">'
        '<p class="report-hero-kicker">Pairwise</p>'
        f'<h2 class="report-hero-title">{esc(run.run_name)}</h2>'
        f'<p class="report-hero-sub">{_swatch("A", run.label_a)} vs {_swatch("B", run.label_b)}'
        f' · {judges} judges · swap {esc(_swap_text(run))}{criteria}</p>'
        f'{actions}</header>'
    )


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def _verdict_text(v: Verdict) -> tuple[str, str, str]:
    """``(eyebrow, headline, explanation)`` for the verdict card."""
    left_out = (
        f'{v.ties} {"tie" if v.ties == 1 else "ties"} and {v.inconclusive} inconclusive '
        f'{"comparison" if v.inconclusive == 1 else "comparisons"} are left out of the win rate.'
    )
    if v.state == 'no_decided' or v.low is None or v.high is None or v.win_rate is None:
        detail = (
            f'Of {v.total} comparisons, {v.ties} ended in a tie and {v.inconclusive} were inconclusive.'
            if v.total
            else 'This run has no comparisons.'
        )
        return (verdict_eyebrow(v.state), 'No comparison was decided, so there is no winner to report.', detail)
    interval = f'95% range {v.low:.0%}\u2013{v.high:.0%} over {v.decided} decided comparisons'
    if v.state == 'clear':
        return (
            verdict_eyebrow(v.state),
            f'{v.leader_label} wins {v.win_rate:.0%} of decided comparisons.',
            f'{interval} stays above 50%. {left_out}',
        )
    crossing = f'{interval} crosses 50%, so the gap could still go either way. More comparisons would settle it. '
    headline = (
        'The two sides won the same number of decided comparisons.'
        if v.leader is None
        else f'{v.leader_label} leads with {v.win_rate:.0%} of decided comparisons, but this run can\u2019t call it yet.'
    )
    return (verdict_eyebrow(v.state), headline, crossing + left_out)


def _interval_bar(v: Verdict) -> str:
    if v.state == 'no_decided' or v.low is None or v.high is None or v.win_rate is None:
        return ''
    colour = pw_side_colour(v.leader or 'tie')
    who = v.leader_label or 'Either side'
    description = (
        f'{who} won {v.win_rate:.0%} of {v.decided} decided comparisons. '
        f'95% uncertainty range: {v.low:.0%} to {v.high:.0%}. 50% means equal wins.'
    )
    return (
        f'<figure class="pw-ci-figure" style="--c:{colour}">'
        f'<figcaption class="pw-ci-title">{esc(who)} win rate</figcaption>'
        '<div class="pw-ci-readout">'
        f'<span><i class="pw-ci-key-dot" aria-hidden="true"></i>Observed <strong>{v.win_rate:.0%}</strong></span>'
        f'<span><i class="pw-ci-key-range" aria-hidden="true"></i>95% range '
        f'<strong>{v.low:.0%}\u2013{v.high:.0%}</strong></span></div>'
        f'<div class="pw-ci" role="img" aria-label="{esc(description)}">'
        f'<div class="pw-ci-range" style="left:{v.low * 100:.1f}%;width:{(v.high - v.low) * 100:.1f}%"></div>'
        '<div class="pw-ci-mid"></div>'
        f'<div class="pw-ci-dot" style="left:{v.win_rate * 100:.1f}%"></div></div>'
        '<div class="pw-ci-cap"><span>0%</span><span>50% · equal wins</span><span>100%</span></div>'
        '</figure>'
    )


def _side_card(label: str, side: str, wins: int, decided: int) -> str:
    rate = f'{wins / decided:.0%}' if decided else '—'
    return (
        f'<div class="pw-side-card" style="--c:{pw_side_colour(side)}"><div class="nm">{esc(label)}</div>'
        '<div class="mets">'
        f'<div class="met"><span>wins</span><b>{wins}</b></div>'
        f'<div class="met"><span>win rate</span><b>{rate}</b></div></div></div>'
    )


def _verdict_card(run: PairwiseRun, v: Verdict) -> str:
    eyebrow, headline, explanation = _verdict_text(v)
    is_open = '' if v.state == 'clear' else ' is-open'
    return (
        f'<div class="pw-verdict{is_open}">'
        f'<p class="pw-verdict-eyebrow">{esc(eyebrow)}</p>'
        f'<p class="pw-verdict-head">{esc(headline)}</p>'
        f'<p class="pw-verdict-expl">{esc(explanation)}</p>'
        f'{_interval_bar(v)}'
        '<div class="pw-verdict-sides">'
        f'{_side_card(run.label_a, "A", v.wins_a, v.decided)}{_side_card(run.label_b, "B", v.wins_b, v.decided)}'
        '</div></div>'
    )


def _kpis(run: PairwiseRun, v: Verdict) -> str:
    usage = judge_usage(run)
    inconclusive_status = 'warn' if v.total and v.inconclusive / v.total > INCONCLUSIVE_WARN else 'neutral'
    if usage.with_usage == 0:
        spend = {'label': 'Judge cost', 'value': 'not recorded', 'status': 'neutral'}
    elif usage.cost is not None:
        spend = {'label': 'Judge cost', 'value': f'${usage.cost:.2f}', 'status': 'neutral'}
    else:
        spend = {'label': 'Judge tokens', 'value': f'{usage.total_tokens:,}', 'status': 'neutral'}
    return kpi_cards([
        {'label': 'Comparisons', 'value': str(v.total), 'status': 'neutral'},
        {'label': 'Decided', 'value': str(v.decided), 'status': 'neutral'},
        {'label': 'Ties', 'value': str(v.ties), 'status': 'neutral'},
        {'label': 'Inconclusive', 'value': str(v.inconclusive), 'status': inconclusive_status},
        spend,
    ])


def _outcomes_panel(run: PairwiseRun, v: Verdict) -> str:
    segments = [
        {'label': run.label_a, 'value': v.wins_a, 'color': pw_side_colour('A')},
        {'label': run.label_b, 'value': v.wins_b, 'color': pw_side_colour('B')},
        {'label': 'tie', 'value': v.ties, 'color': pw_side_colour('tie')},
        {'label': 'inconclusive', 'value': v.inconclusive, 'color': pw_side_colour('inconclusive')},
    ]
    return panel('Outcomes', donut(segments, str(v.total), 'comparisons'))


def _trust_panel(run: PairwiseRun) -> str:
    rows = ''.join(_trust_row(s) for s in trust_signals(run))
    return panel(
        'How much to trust this',
        f'<table class="pw-trust"><tbody>{rows}</tbody></table>',
        info='Signals about the run, not about either side. Amber means read the verdict with care.',
    )


def _trust_row(signal: TrustSignal) -> str:
    cls = f' class="pw-tone-{signal.tone}"' if signal.tone else ''
    return f'<tr><td>{esc(signal.label)}</td><td{cls}><b>{esc(signal.value)}</b> {esc(signal.detail)}</td></tr>'


def _tokens_panel(run: PairwiseRun) -> str:
    usage = judge_usage(run)
    if usage.with_usage == 0:
        return ''
    average = usage.total_tokens / usage.with_usage
    n = len(run.entries)
    if usage.with_usage < n:
        sub = f'{usage.total_tokens:,} total · {average:,.0f} per comparison with usage ({usage.with_usage} of {n})'
    else:
        sub = f'{usage.total_tokens:,} total · {average:,.0f}/comparison'
    if usage.cost is not None:
        sub += f' · ${usage.cost:.2f}'
    return panel(
        'Judge tokens',
        bar_rows(
            [('Input', usage.input_tokens), ('Output', usage.output_tokens)],
            width=520,
            label_w=70,
            color='var(--chart-2)',
            fmt=lambda value: f'{int(value):,}',
        ),
        sub=sub,
        info='Answer-generation tokens are not stored on a pairwise run, so only the judges are counted.',
    )


def _overview(run: PairwiseRun, *, filtered_empty: bool = False) -> str:
    if filtered_empty:
        return _EMPTY_FILTERED
    v = verdict(run)
    return (
        _verdict_card(run, v)
        + _kpis(run, v)
        + f'<div class="pw-grid-2">{_outcomes_panel(run, v)}{_trust_panel(run)}</div>'
        + _tokens_panel(run)
    )


# ---------------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------------


def _th(label_html: str, plain: str, tip: str | None = None) -> str:
    title = f' title="{esc(tip)}"' if tip else ''
    return f'<th{title} data-label="{esc(plain)}">{label_html}</th>'


def _judge_notes(data: dict[str, Any]) -> str:
    notes: list[str] = []
    if not data.get('observed_swap'):
        notes.append(
            'No pair completed both orderings, so position bias could not be measured for this run despite '
            'swapping being enabled.'
            if data.get('swap')
            else 'This run used a single ordering, so position bias was not measured.'
        )
    if not data.get('observed_consistency'):
        notes.append(
            'Repeated passes ran but no judge completed two decisive passes of the same ordering, so judge '
            'self-consistency could not be measured for this run.'
            if data.get('repeated')
            else 'This run used a single pass per ordering, so judge self-consistency was not measured.'
        )
    return ''.join(f'<p class="pw-note">{esc(n)}</p>' for n in notes)


def _judges_panel(run: PairwiseRun, data: dict[str, Any]) -> str:
    rows: list[dict[str, Any]] = data.get('rows', [])
    sub = (
        f'Position bias is flagged at {POSITION_BIAS_WARN:.2f} or more: '
        f'the judge changed its pick on at least {POSITION_BIAS_WARN:.0%} of swapped pairs.'
    )
    if not rows:
        return panel('Judges', '<p class="pw-note">No judge voted in this run.</p>', sub=sub)
    failed: dict[str, int] = {}
    for entry in run.entries:
        for vote in entry.comparison.votes:
            failed[vote.model] = failed.get(vote.model, 0) + vote.repetition_failures
    observed = bool(data.get('observed_swap'))
    show_consistency = bool(data.get('observed_consistency'))
    columns: list[tuple[str, str, str | None]] = [
        ('Judge', 'Judge', None),
        (f'<span class="pw-side-a">{esc(run.label_a)}</span> rate', f'{run.label_a} rate', None),
        (f'<span class="pw-side-b">{esc(run.label_b)}</span> rate', f'{run.label_b} rate', None),
        ('Tie rate', 'Tie rate', None),
        ('Position bias ⓘ', 'Position bias', _BIAS_HEADER_TIP),
        ('Failed passes', 'Failed passes', None),
    ]
    if show_consistency:
        columns.extend((
            ('Consistency (shrunk)', 'Consistency (shrunk)', _SHRUNK_TIP),
            ('Consistency (raw)', 'Consistency (raw)', _RAW_TIP),
        ))
    heads = list(starmap(_th, columns))
    body: list[str] = []
    for r in rows:
        cells = [
            esc(r['model']) + (' <span class="pw-tag">stand-in</span>' if r.get('replacement') else ''),
            _rate(r.get('a_rate')),
            _rate(r.get('b_rate')),
            _rate(r.get('tie_rate')),
            _bias_cell(r, observed_swap=observed),
            str(failed.get(r['model'], 0)),
        ]
        if show_consistency:
            cells.extend((_consistency_cell(r.get('consistency')), _consistency_cell(r.get('consistency_raw'))))
        body.append(
            '<tr>'
            + ''.join(f'<td data-label="{esc(column[1])}">{c}</td>' for column, c in zip(columns, cells, strict=True))
            + '</tr>'
        )
    table = f'<table><thead><tr>{"".join(heads)}</tr></thead><tbody>{"".join(body)}</tbody></table>'
    return panel('Judges', table + _judge_notes(data), sub=sub)


def _weighted_panel(run: PairwiseRun, consensus: dict[str, Any]) -> str:
    weighted = consensus.get('bt_sigma')
    if weighted is None:
        return ''
    bars = ''
    for label, side, rate in (
        (run.label_a, 'A', weighted.get('a_win_rate')),
        (run.label_b, 'B', weighted.get('b_win_rate')),
    ):
        if rate is not None:
            bars += bar_rows(
                [(label, rate)],
                width=520,
                label_w=110,
                color=pw_side_colour(side),
                fmt=lambda value: f'{value:.0%}',
                max_value=1.0,
            )
    warnings = ''.join(f'<li>{esc(w)}</li>' for w in weighted.get('fit_warnings', []))
    warning_html = f'<ul class="pw-warnings">{warnings}</ul>' if warnings else ''
    return panel(
        'Reliability-weighted consensus',
        bars + warning_html,
        sub='Each judge\u2019s vote is weighted by how consistent it was with itself (BT-sigma).',
    )


def _judges(run: PairwiseRun, *, filtered_empty: bool = False) -> str:
    if filtered_empty:
        return _EMPTY_FILTERED
    by_kind = {s.kind: s for s in build_report_sections(run)}
    judges = by_kind['pairwise_judges'].data
    consensus = by_kind['pairwise_consensus'].data
    return _judges_panel(run, judges) + _weighted_panel(run, consensus)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _passes_per_ordering(run: PairwiseRun) -> int:
    seen = [o.repetition for e in run.entries for v in e.comparison.votes for o in v.observations]
    return max(seen) + 1 if seen else 1


def _evaluatorq_version() -> str:
    try:
        return version('evaluatorq')
    except PackageNotFoundError:
        return 'unknown'


_METHOD_HTML = (
    '<div class="rk-panel pw-method-panel"><details class="pw-method">'
    '<summary>How this was judged</summary>'
    '<p>Every judge sees each pair twice, once in each order. A judge that picks different sides in the two '
    'orders has its vote thrown out for that pair; this is what the position-bias number counts. The panel '
    'winner is the plurality of the remaining votes; with no votes left the comparison is inconclusive.</p>'
    '<p>The 95% range on the headline win rate is a Wilson interval over decided comparisons only. When it '
    'crosses 50% the page says the run is too close to call rather than naming a winner.</p>'
    '</details></div>'
)


def _config(run: PairwiseRun) -> str:
    panel_judges = list(run.judges) or judge_order(run)
    aggregation = _aggregation(run)
    grid = meta_grid([
        ('Side A', run.label_a),
        ('Side B', run.label_b),
        ('Judges', ', '.join(panel_judges)),
        ('Criteria', run.criteria),
        ('Swap orderings', _swap_text(run)),
        ('Passes per ordering', str(_passes_per_ordering(run))),
        ('Aggregation', aggregation),
        ('Created', run.created_at.strftime('%Y-%m-%d %H:%M')),
        ('Comparisons', str(len(run.entries))),
    ])
    fingerprint = f'Run fingerprint · evaluatorq {_evaluatorq_version()} · judges: {", ".join(panel_judges)}'
    return panel('Run configuration', grid) + _METHOD_HTML + f'<p class="pw-note">{esc(fingerprint)}</p>'


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def pairwise_report_tabs(
    rid: str,
    run: PairwiseRun,
    entries: list[PairwiseEntry] | None = None,
    *,
    hero_actions_html: str = '',
    focus: int | None = None,
) -> str:
    """Render the pairwise report body as tabs.

    Pass ``entries`` to render a filtered subset: every number on the page is
    then computed over a copy of the run carrying only those entries.
    The Comparisons tab lists the same entries.

    ``focus`` is a 1-based comparison number: when it numbers one of the listed
    entries, the Comparisons tab is the active tab, on the page holding that row
    with the row open.
    """
    view = run if entries is None else with_entries(run, entries)
    focus = valid_focus(run, view.entries, focus)
    if view.entries:
        comparisons = render_pairwise_comparisons(rid, run, view.entries, focus=focus)
    else:
        comparisons = _EMPTY_FILTERED if entries is not None else _EMPTY_RUN
    filtered_empty = entries is not None and not view.entries
    tabs = _tabs(
        'pwtab',
        [
            ('Overview', _overview(view, filtered_empty=filtered_empty)),
            ('Judges', _judges(view, filtered_empty=filtered_empty)),
            ('Comparisons', comparisons, f'Comparisons <span class="tab-count">{len(view.entries)}</span>'),
            ('Config', _config(view)),
        ],
        active='Comparisons' if focus is not None else None,
    )
    return f'<div class="report-aligned pw-report">{_hero(view, hero_actions_html)}{tabs}</div>'
