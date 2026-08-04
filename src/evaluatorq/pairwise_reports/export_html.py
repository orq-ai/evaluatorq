"""HTML renderer for pairwise run reports.

Body-fragment and standalone-document renderers built from the shared
``evaluatorq.common.reports`` primitives, dispatching by section kind exactly
as the simulation and red-team renderers do.
"""

from __future__ import annotations

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
    render_body as _render_body,
)
from evaluatorq.common.reports import (
    status_badge as _status_badge,
)
from evaluatorq.common.reports.palette import COLORS
from evaluatorq.pairwise_reports.sections import INCONCLUSIVE_WARN, build_report_sections
from evaluatorq.pairwise_reports.sections import observed_swap as _observed_swap

if TYPE_CHECKING:
    from evaluatorq.contracts import ReportSection
    from evaluatorq.pairwise_run import PairwiseRun

# Two brand hues far enough apart that the side bars and the vote chips read as
# two distinct things at a glance. Neither is a good/bad colour: a side winning
# is not a pass or a failure, so the semantic pass/fail palette is wrong here.
_A_COLOR = COLORS['blue_400']
_B_COLOR = COLORS['purple_400']


def _rate(value: float | None) -> str:
    """Percentage, or an explicit dash when the metric had no denominator."""
    return '-' if value is None else f'{value:.0%}'


def _num(value: float | None) -> str:
    return '-' if value is None else f'{value:.2f}'


def _side_bar(label: str, rate: float | None, color: str) -> str:
    """One labelled, full-width bar for a side of the comparison."""
    width = 0.0 if rate is None else max(0.0, min(1.0, rate)) * 100
    return (
        '<div class="pw-side">'
        f'<div class="pw-side__label">{_esc(label)}</div>'
        '<div class="pw-side__track">'
        f'<div class="pw-side__fill" style="width:{width:.1f}%;background:{color}"></div>'
        '</div>'
        f'<div class="pw-side__value">{_rate(rate)}</div>'
        '</div>'
    )


def _render_consensus_html(section: ReportSection) -> str:
    d = section.data
    total = d.get('comparisons', 0)
    decided = d.get('decided', 0)
    bars = _side_bar(d['label_a'], d.get('a_win_rate'), _A_COLOR) + _side_bar(
        d['label_b'], d.get('b_win_rate'), _B_COLOR
    )
    # Stated out loud because both win rates share a denominator that excludes
    # ties and inconclusive comparisons.
    caption = f'<p class="pw-caption">Win rates over the {decided} of {total} comparisons the panel decided.</p>'
    stats = _kpi_cards([
        {'label': 'Ties', 'value': _rate(d.get('tie_rate')), 'status': 'neutral'},
        {
            'label': 'Inconclusive',
            'value': _rate(d.get('inconclusive_rate')),
            'status': 'warn' if (d.get('inconclusive_rate') or 0) > INCONCLUSIVE_WARN else 'neutral',
        },
        {'label': 'Mean agreement', 'value': _num(d.get('mean_agreement')), 'status': 'neutral'},
    ])
    return (
        f'<section id="{section.kind}"><h2>{_esc(section.title)}</h2>'
        f'<div class="pw-sides">{bars}</div>{caption}{stats}</section>'
    )


def _bias_cell(row: dict[str, Any], *, observed_swap: bool) -> str:
    bias = row.get('position_bias')
    if bias is None:
        # Blaming the wrong party is the failure to avoid here. When no judge in
        # the run completed a pair, that is a run-level fact and saying "this
        # judge" would pin a run's shape on each judge in turn.
        reason = (
            'This judge never completed both orderings of a pair, so no flip could be observed'
            if observed_swap
            else 'No pair completed both orderings, so no flip was observable in this run'
        )
        return f'<span class="pw-muted" title="{_esc(reason)}">n/a</span>'
    width = max(0.0, min(1.0, bias)) * 100
    warn = ' pw-bias__fill--warn' if row.get('biased') else ''
    marker = ' ⚠' if row.get('biased') else ''
    return (
        '<span class="pw-bias">'
        f'<span class="pw-bias__track"><span class="pw-bias__fill{warn}" style="width:{width:.1f}%"></span></span>'
        f'{bias:.2f}{marker}</span>'
    )


def _render_judges_html(section: ReportSection) -> str:
    d = section.data
    observed = bool(d.get('observed_swap'))
    headers = ['Judge', f'{d["label_a"]} rate', f'{d["label_b"]} rate', 'Tie rate', 'Position bias']
    rows = [
        [
            _esc(r['model']) + (' <span class="pw-tag">stand-in</span>' if r.get('replacement') else ''),
            _rate(r.get('a_rate')),
            _rate(r.get('b_rate')),
            _rate(r.get('tie_rate')),
            _bias_cell(r, observed_swap=observed),
        ]
        for r in d.get('rows', [])
    ]
    if not rows:
        return ''
    # The caption keys off what ran, not what was configured: a run saved with
    # the default swap=True but executed single-ordering would otherwise show a
    # clean table with nothing saying the column is empty for a reason.
    if observed:
        note = ''
    elif d.get('swap'):
        note = (
            '<p class="pw-caption">No pair completed both orderings, so position bias could not be '
            'measured for this run despite swapping being enabled.</p>'
        )
    else:
        note = '<p class="pw-caption">This run used a single ordering, so position bias was not measured.</p>'
    return f'<section id="{section.kind}"><h2>{_esc(section.title)}</h2>{_html_table(headers, rows)}{note}</section>'


_WINNER_STATUS = {'inconclusive': 'warn', 'tie': 'neutral'}


def _vote_chip(vote: dict[str, Any]) -> str:
    value = vote.get('vote')
    if not vote.get('present') or value is None:
        # An abstention is a recorded outcome, not missing data.
        return '<span class="pw-chip pw-chip--abstain" title="abstained">&middot;</span>'
    tone = 'a' if value == 'A' else ('b' if value == 'B' else 'tie')
    flip = ' pw-chip--flip' if vote.get('flipped') else ''
    return f'<span class="pw-chip pw-chip--{tone}{flip}" title="{_esc(vote["model"])}">{_esc(vote["label"])}</span>'


def _detail_html(row: dict[str, Any], labels: tuple[str, str]) -> str:
    label_a, label_b = labels
    responses = (
        '<div class="pw-responses">'
        f'<div class="pw-response"><h4>{_esc(label_a)}</h4><pre>{_esc(row["response_a"])}</pre></div>'
        f'<div class="pw-response"><h4>{_esc(label_b)}</h4><pre>{_esc(row["response_b"])}</pre></div>'
        '</div>'
    )
    votes = ''.join(_vote_row(v) for v in row.get('votes', []))
    return f'{responses}<ul class="pw-votes">{votes}</ul>'


_FLIP_TAG = '<span class="pw-tag pw-tag--flip">flipped</span>'


def _vote_row(vote: dict[str, Any]) -> str:
    flip = _FLIP_TAG if vote.get('flipped') else ''
    return (
        '<li class="pw-vote">'
        f'<span class="pw-vote__model">{_esc(vote["model"])}</span>'
        f'{_vote_chip(vote)}{flip}'
        f'<span class="pw-vote__why">{_esc(vote.get("explanation") or "")}</span>'
        '</li>'
    )


def _render_comparisons_html(section: ReportSection) -> str:
    d = section.data
    labels = (d['label_a'], d['label_b'])
    rows = d.get('rows', [])
    if not rows:
        return (
            f'<section id="{section.kind}"><h2>{_esc(section.title)}</h2><p>No comparisons in this run.</p></section>'
        )

    parts = [f'<section id="{section.kind}"><h2>{_esc(section.title)}</h2>']
    for row in rows:
        chips = ''.join(_vote_chip(v) for v in row.get('votes', []))
        status = _WINNER_STATUS.get(row['winner'], 'pass')
        mark = '<span class="pw-split" title="judges split or panel undecided">⚠</span>' if row.get('split') else ''
        parts.append(
            '<details class="pw-cmp">'
            '<summary>'
            f'<span class="pw-cmp__idx">{row["index"]}</span>'
            f'<span class="pw-cmp__q">{_esc(row["question"])}</span>'
            f'{_status_badge(row["winner_label"], status)}'
            f'<span class="pw-cmp__chips">{chips}</span>{mark}'
            '</summary>'
            f'{_detail_html(row, labels)}'
            '</details>'
        )
    parts.append('</section>')
    return ''.join(parts)


_SECTION_RENDERERS = {
    'pairwise_consensus': _render_consensus_html,
    'pairwise_judges': _render_judges_html,
    'pairwise_comparisons': _render_comparisons_html,
}

_PAIRWISE_CSS = """
.pw-sides{display:flex;flex-direction:column;gap:10px;margin:16px 0 8px}
.pw-side{display:flex;align-items:center;gap:12px}
.pw-side__label{width:160px;font-weight:600;text-align:right}
.pw-side__track{flex:1;height:22px;background:rgba(127,127,127,.15);border-radius:4px;overflow:hidden}
.pw-side__fill{height:100%;border-radius:4px}
.pw-side__value{width:56px;font-variant-numeric:tabular-nums}
.pw-caption{opacity:.7;font-size:.9em;margin:4px 0 12px}
.pw-muted{opacity:.6}
.pw-bias{display:inline-flex;align-items:center;gap:8px;font-variant-numeric:tabular-nums}
.pw-bias__track{width:60px;height:8px;background:rgba(127,127,127,.15);border-radius:4px;overflow:hidden}
.pw-bias__fill{display:block;height:100%;background:#7f8c9a}
.pw-bias__fill--warn{background:#d9534f}
.pw-tag{font-size:.75em;padding:1px 6px;border-radius:10px;background:rgba(127,127,127,.18);margin-left:6px}
.pw-tag--flip{background:rgba(217,83,79,.18)}
.pw-chip{display:inline-block;min-width:34px;text-align:center;padding:1px 8px;margin-right:4px;
 border-radius:10px;font-size:.8em;background:rgba(127,127,127,.15)}
.pw-chip--a{background:rgba(59,111,212,.22)}
.pw-chip--b{background:rgba(130,87,197,.22)}
.pw-chip--tie{background:rgba(127,127,127,.2)}
.pw-chip--abstain{opacity:.5}
.pw-chip--flip{outline:1px dashed rgba(217,83,79,.6)}
.pw-split{margin-left:8px;color:#d9534f}
.pw-cmp{border:1px solid rgba(127,127,127,.25);border-radius:6px;margin:6px 0;padding:6px 10px}
.pw-cmp summary{display:flex;align-items:center;gap:10px;cursor:pointer}
.pw-cmp__idx{opacity:.6;width:28px;font-variant-numeric:tabular-nums}
/* Wraps rather than clipping to one line: the question is the only thing
   identifying a row, and an ellipsis made a long one readable nowhere. */
.pw-cmp__q{flex:1;min-width:0;overflow-wrap:anywhere}
.pw-cmp__chips{white-space:nowrap}
.pw-responses{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0}
.pw-response pre{white-space:pre-wrap;word-break:break-word;background:rgba(127,127,127,.08);
 padding:8px;border-radius:4px;margin:4px 0 0}
.pw-votes{list-style:none;padding:0;margin:8px 0 4px}
.pw-vote{display:flex;align-items:baseline;gap:10px;padding:6px 0;border-top:1px solid rgba(127,127,127,.15)}
.pw-vote__model{width:200px;font-weight:600;flex:none}
.pw-vote__why{flex:1;opacity:.85}
@media (max-width:720px){.pw-responses{grid-template-columns:1fr}.pw-side__label{width:100px}}
"""


def _swap_summary(run: PairwiseRun) -> str:
    """Report swapping as configured, qualified by whether it was ever observed.

    A bare "on" taken from the flag is the claim a reader relies on when
    deciding whether the position-bias column means anything, so it says so when
    the votes never bore it out.
    """
    if not run.swap:
        return 'off'
    if _observed_swap(run):
        return 'on'
    return 'on <span class="pw-muted">(never observed)</span>'


def _header_html(run: PairwiseRun) -> str:
    report = run.rollup()
    meta = (
        f'<p><strong>Comparing:</strong> {_esc(run.label_a)} vs {_esc(run.label_b)} &nbsp;|&nbsp; '
        f'<strong>Judges:</strong> {len(run.judges) or len(report.per_judge)} &nbsp;|&nbsp; '
        f'<strong>Swap:</strong> {_swap_summary(run)} &nbsp;|&nbsp; '
        f'<strong>Date:</strong> {_format_date(run.created_at)}</p>'
    )
    criteria = f'<p class="pw-caption"><strong>Criteria:</strong> {_esc(run.criteria)}</p>' if run.criteria else ''
    return f'<header class="hero"><h1>{_esc(run.run_name)}</h1>{meta}{criteria}</header>'


def render_report_body(run: PairwiseRun) -> str:
    """Render a pairwise run as an HTML body fragment (no ``<html>`` / ``<head>``)."""
    return _render_body(
        build_report_sections(run),
        renderers=_SECTION_RENDERERS,
        body_header=f'<style>{_PAIRWISE_CSS}</style>{_header_html(run)}',
        body_footer='<footer><p class="footer">Generated by evaluatorq pairwise judging.</p></footer>',
    )


def export_html(run: PairwiseRun) -> str:
    """Render a pairwise run as a self-contained HTML document."""
    head = (
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<title>{_esc(run.run_name)} - Pairwise Report</title>\n'
        f'<style>\n{_load_css()}\n</style>'
    )
    return (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        f'<head>\n{head}\n</head>\n'
        '<body>\n<div class="container">\n'
        f'{render_report_body(run)}\n'
        '</div>\n</body>\n'
        '</html>\n'
    )
