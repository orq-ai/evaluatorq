"""Tests for the tabbed pairwise report: tabs, verdict, trust table, tokens, routing."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from evaluatorq.contracts import Usage
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import report_id
from evaluatorq.dashboard.pairwise_palette import pw_side_colour, pw_side_label
from evaluatorq.dashboard.pairwise_tabs import (
    pairwise_report_tabs,
    with_entries,
)
from evaluatorq.pairwise import PairwiseComparison, PairwiseVote
from evaluatorq.pairwise_run import PairwiseRun, new_run

JUDGES = ['judge-one', 'judge-two', 'judge-three']


def _comparison(values: list[str | None], winner: str, *, usage: Usage | None = None) -> PairwiseComparison:
    votes = [
        PairwiseVote(model=m, vote=v, flipped=False, completed=True, explanation=f'{m} says {v}')
        for m, v in zip(JUDGES, values)
    ]
    return PairwiseComparison(winner=winner, votes=votes, token_usage=usage)


def _run(winners: list[str], *, usage: Usage | None = None, **kw: str) -> PairwiseRun:
    r = new_run(run_name='v2 vs v3', label_a='prompt-v2', label_b='prompt-v3', judges=JUDGES, criteria='accuracy', **kw)
    for i, w in enumerate(winners):
        values: list[str | None] = [None] * 3 if w == 'inconclusive' else [w] * 3
        r.add(_comparison(values, w, usage=usage), question=f'q{i}', response_a=f'a{i}', response_b=f'b{i}')
    return r


@pytest.fixture
def run() -> PairwiseRun:
    return _run(['A', 'tie', 'B', 'inconclusive'])


def _kpis(html: str) -> dict[str, str]:
    return {
        label: value
        for value, label in re.findall(r'<div class="kpi-value">(.*?)</div><div class="kpi-label">(.*?)</div>', html)
    }


def test_the_four_tabs_render_in_order(run: PairwiseRun) -> None:
    html = pairwise_report_tabs('rid', run)
    labels = re.findall(r'<label class="tab-label"[^>]*>(.*?)</label>', html)
    assert labels == ['Overview', 'Judges', 'Comparisons <span class="tab-count">4</span>', 'Config']


@pytest.mark.parametrize(
    ('winners', 'eyebrow'),
    [
        (['A'] * 10, '✓ Clear winner'),
        (['A', 'A', 'B', 'B', 'A'], '⚠ Too close to call'),
        (['tie', 'inconclusive'], 'No decided comparisons'),
    ],
)
def test_verdict_eyebrow_matches_the_state(winners: list[str], eyebrow: str) -> None:
    html = pairwise_report_tabs('rid', _run(winners))
    assert f'<p class="pw-verdict-eyebrow">{eyebrow}</p>' in html
    assert ('pw-verdict is-open' in html) == (eyebrow != '✓ Clear winner')
    assert ('class="pw-ci"' in html) == (eyebrow != 'No decided comparisons')


def test_trust_table_lists_the_six_signals(run: PairwiseRun) -> None:
    html = pairwise_report_tabs('rid', run)
    table = html[html.index('class="pw-trust"') :]
    labels = re.findall(r'<tr><td>(.*?)</td>', table[: table.index('</table>')])
    assert labels == [
        'Decided',
        'Votes thrown out',
        'Judge agreement',
        'Position bias',
        'Length',
        'Judge failures',
    ]


def test_kpi_tiles_show_the_counts(run: PairwiseRun) -> None:
    kpis = _kpis(pairwise_report_tabs('rid', run))
    assert (kpis['Comparisons'], kpis['Decided'], kpis['Ties'], kpis['Inconclusive']) == ('4', '2', '1', '1')
    assert kpis['Judge cost'] == 'not recorded'


def test_entries_narrow_every_number(run: PairwiseRun) -> None:
    html = pairwise_report_tabs('rid', run, entries=run.entries[:1])
    assert _kpis(html)['Comparisons'] == '1'
    assert 'Comparisons <span class="tab-count">1</span>' in html
    # The caller's run is not mutated by the narrowing.
    assert len(run.entries) == 4


def test_with_entries_recomputes_the_rollup(run: PairwiseRun) -> None:
    narrowed = with_entries(run, run.entries[:1])
    assert narrowed.report is not None
    assert narrowed.report.comparisons == 1
    assert len(run.entries) == 4


def test_judge_tokens_panel_is_absent_without_usage(run: PairwiseRun) -> None:
    assert 'Judge tokens' not in pairwise_report_tabs('rid', run)


def test_judge_tokens_panel_sums_recorded_usage() -> None:
    usage = Usage(input_tokens=1000, output_tokens=200, total_tokens=1200)
    html = pairwise_report_tabs('rid', _run(['A', 'B'], usage=usage))
    assert 'Judge tokens' in html
    assert '2,400 total' in html
    assert _kpis(html)['Judge tokens'] == '2,400'


def test_priced_usage_shows_the_cost() -> None:
    usage = Usage(input_tokens=1000, output_tokens=200, total_tokens=1200, total_cost=0.25)
    html = pairwise_report_tabs('rid', _run(['A', 'B'], usage=usage))
    assert _kpis(html)['Judge cost'] == '$0.50'


def test_run_name_and_labels_are_escaped() -> None:
    r = new_run(run_name='<script>alert(1)</script>', label_a='<script>a</script>', label_b='b', judges=JUDGES)
    r.add(_comparison(['A'] * 3, 'A'), question='q', response_a='a', response_b='b')
    html = pairwise_report_tabs('rid', r, hero_actions_html='<a>go</a>')
    assert '<script>' not in html
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in html
    assert '<div class="report-hero-actions"><a>go</a></div>' in html


def test_empty_run_still_renders_every_tab() -> None:
    html = pairwise_report_tabs('rid', new_run(run_name='empty'))
    assert html.count('class="tab-label"') == 4
    assert 'No comparisons in this run.' in html


def test_comparisons_tab_lists_the_entries(run: PairwiseRun) -> None:
    html = pairwise_report_tabs('rid', run)
    assert 'id="cmp-4"' in html
    assert 'No comparisons in this run.' not in html


def test_filtered_render_changes_the_verdict(run: PairwiseRun) -> None:
    # Full run: one A win and one B win, so the sides are level.
    assert 'won the same number of decided comparisons' in pairwise_report_tabs('rid', run)
    only_b = [e for e in run.entries if e.comparison.winner == 'B']
    html = pairwise_report_tabs('rid', run, entries=only_b)
    assert 'won the same number' not in html
    assert 'prompt-v3 leads with 100%' in html
    # Original numbering is kept: the B win is comparison 3 of the full run.
    assert 'id="cmp-3"' in html and 'id="cmp-1"' not in html


def test_token_subline_says_when_only_some_comparisons_have_usage() -> None:
    r = _run(['A', 'B'], usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
    r.add(_comparison(['A', 'A', 'A'], 'A'), question='q9', response_a='a', response_b='b')
    assert '(2 of 3)' in pairwise_report_tabs('rid', r)
    assert 'of 2)' not in pairwise_report_tabs(
        'rid', _run(['A', 'B'], usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
    )


def test_single_ordering_run_explains_the_missing_position_bias() -> None:
    r = new_run(run_name='single', judges=JUDGES, swap=False)
    r.add(
        PairwiseComparison(winner='A', votes=[PairwiseVote(model=m, vote='A', completed=False) for m in JUDGES]),
        question='q',
        response_a='a',
        response_b='b',
    )
    html = pairwise_report_tabs('rid', r)
    assert 'swap off' in html
    assert 'n/a' in html
    assert (
        'Position bias'
        not in html[html.index('class="pw-trust"') : html.index('</table>', html.index('class="pw-trust"'))]
    )


def test_config_tab_reports_the_run_setup(run: PairwiseRun) -> None:
    html = pairwise_report_tabs('rid', run)
    assert 'Passes per ordering' in html
    assert 'plurality' in html
    assert 'How this was judged' in html
    assert 'evaluatorq ' in html


def test_side_helpers(run: PairwiseRun) -> None:
    assert [pw_side_colour(v) for v in ('A', 'B', 'tie', 'inconclusive', None)] == [
        'var(--pw-a)',
        'var(--pw-b)',
        'var(--pw-tie)',
        'var(--pw-none)',
        'var(--pw-none)',
    ]
    assert [pw_side_label(v, run) for v in ('A', 'B', 'tie', 'inconclusive', None)] == [
        'prompt-v2',
        'prompt-v3',
        'tie',
        'inconclusive',
        'abstain',
    ]


# ---------------------------------------------------------------------------
# Routes


@pytest.fixture
def client(run: PairwiseRun, tmp_path: Path) -> tuple[TestClient, str]:
    runs_dir = tmp_path / 'pairwise-runs'
    runs_dir.mkdir()
    path = run.save(runs_dir / 'run.json')
    return TestClient(build_app(roots=[runs_dir]), raise_server_exceptions=True), report_id(path)


def test_report_route_serves_the_tabbed_page(client: tuple[TestClient, str]) -> None:
    c, rid = client
    resp = c.get(f'/r/{rid}')
    assert resp.status_code == 200
    assert 'class="report-aligned pw-report"' in resp.text
    assert 'pw-verdict' in resp.text


def test_filter_route_returns_the_tabbed_body(client: tuple[TestClient, str]) -> None:
    c, rid = client
    resp = c.post(f'/r/{rid}/filter', data={})
    assert resp.status_code == 200
    assert 'class="report-aligned pw-report"' in resp.text
    assert _kpis(resp.text)['Comparisons'] == '4'


def test_report_page_has_the_comparisons_list(client: tuple[TestClient, str]) -> None:
    c, rid = client
    html = c.get(f'/r/{rid}').text
    assert 'pw-row-list-' in html
    assert 'id="cmp-1"' in html


def test_filter_by_winner_narrows_the_comparisons_tab(client: tuple[TestClient, str]) -> None:
    c, rid = client
    html = c.post(f'/r/{rid}/filter', data={'winner': 'prompt-v3'}).text
    assert 'Comparisons <span class="tab-count">1</span>' in html
    assert 'id="cmp-3"' in html and 'id="cmp-1"' not in html
    assert 'prompt-v3 leads with 100%' in html


def test_compare_control_needs_a_second_pairwise_run(run: PairwiseRun, tmp_path: Path) -> None:
    runs_dir = tmp_path / 'runs'
    runs_dir.mkdir()
    rid = report_id(run.save(runs_dir / 'one.json'))
    c = TestClient(build_app(roots=[runs_dir]))
    assert 'action="/compare/pairwise"' not in c.get(f'/r/{rid}').text
    run.save(runs_dir / 'two.json')
    c = TestClient(build_app(roots=[runs_dir]))
    assert 'action="/compare/pairwise"' in c.get(f'/r/{rid}').text
