"""Tests for the pairwise dashboard surface.

Covers the persisted run artifact, the file-scanner sniff, the surface
adapter, and the rendered sections.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import report_id, scan, sniff_kind
from evaluatorq.dashboard.surfaces import ADAPTERS
from evaluatorq.pairwise import PairwiseComparison, PairwiseVote
from evaluatorq.pairwise_reports.export_html import render_report_body
from evaluatorq.pairwise_reports.sections import build_report_sections
from evaluatorq.pairwise_run import PairwiseRun, new_run

JUDGES = ['judge-one', 'judge-two', 'judge-three']


def _vote(model: str, value: str | None, *, flipped: bool = False, replacement: bool = False) -> PairwiseVote:
    return PairwiseVote(
        model=model,
        vote=value,
        flipped=flipped,
        completed=True,
        replacement=replacement,
        explanation=f'{model} says {value}',
    )


def _comparison(values: list[str | None], winner: str, **kw: bool) -> PairwiseComparison:
    return PairwiseComparison(
        winner=winner,
        votes=[_vote(m, v, **kw) for m, v in zip(JUDGES, values)],
    )


@pytest.fixture
def run() -> PairwiseRun:
    """A run with a decided pair, a split pair, a tie and an inconclusive pair."""
    r = new_run(
        run_name='v2 vs v3',
        label_a='prompt-v2',
        label_b='prompt-v3',
        judges=JUDGES,
        criteria='accuracy',
    )
    r.add(_comparison(['A', 'A', 'A'], 'A'), question='q1', response_a='a1', response_b='b1')
    r.add(_comparison(['A', 'B', 'tie'], 'tie'), question='q2', response_a='a2', response_b='b2')
    r.add(_comparison(['B', 'B', 'B'], 'B'), question='q3', response_a='a3', response_b='b3')
    r.add(_comparison([None, None, None], 'inconclusive', flipped=True), question='q4', response_a='a4', response_b='b4')
    return r


@pytest.fixture
def saved(run: PairwiseRun, tmp_path: Path) -> tuple[Path, Path]:
    """The run written into a runs dir; returns (runs_dir, file)."""
    runs_dir = tmp_path / 'pairwise-runs'
    runs_dir.mkdir()
    return runs_dir, run.save(runs_dir / 'run.json')


# ---------------------------------------------------------------------------
# Run artifact
# ---------------------------------------------------------------------------


def test_save_round_trips_through_json(saved: tuple[Path, Path], run: PairwiseRun) -> None:
    _, path = saved
    loaded = PairwiseRun.load(path)
    assert loaded.run_name == run.run_name
    assert loaded.label_a == 'prompt-v2'
    assert loaded.label_b == 'prompt-v3'
    assert len(loaded.entries) == 4
    assert loaded.entries[0].question == 'q1'
    assert loaded.entries[0].comparison.winner == 'A'


def test_save_computes_and_stores_the_rollup(saved: tuple[Path, Path]) -> None:
    _, path = saved
    loaded = PairwiseRun.load(path)
    assert loaded.report is not None
    assert loaded.report.comparisons == 4
    # Two decided comparisons: one A, one B.
    assert loaded.report.a_win_rate == pytest.approx(0.5)
    assert loaded.report.b_win_rate == pytest.approx(0.5)
    assert loaded.report.tie_rate == pytest.approx(0.25)
    assert loaded.report.inconclusive_rate == pytest.approx(0.25)


def test_label_defaults_are_the_slot_letters() -> None:
    r = new_run(run_name='unnamed sides')
    assert (r.label_a, r.label_b) == ('A', 'B')


def test_rollup_recomputes_when_no_report_is_stored(run: PairwiseRun) -> None:
    assert run.report is None
    assert run.rollup().comparisons == 4


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


def test_sniff_identifies_a_pairwise_file() -> None:
    assert sniff_kind({'judging': 'pairwise'}) == 'pairwise'


def test_sniff_leaves_the_existing_surfaces_alone() -> None:
    assert sniff_kind({'mode': 'run'}) == 'sim'
    assert sniff_kind({'pipeline': {}}) == 'redteam'
    assert sniff_kind({'something': 'else'}) is None


def test_scan_lists_the_run_with_a_comparison_headline(saved: tuple[Path, Path]) -> None:
    runs_dir, path = saved
    cards = [c for c in scan([runs_dir]) if c.surface == 'pairwise']
    assert len(cards) == 1
    assert cards[0].name == 'v2 vs v3'
    assert cards[0].headline == '4 comparisons'
    assert cards[0].error is None
    assert cards[0].id == report_id(path)


def test_scan_flags_a_run_missing_its_entries(tmp_path: Path) -> None:
    runs_dir = tmp_path / 'pairwise-runs'
    runs_dir.mkdir()
    (runs_dir / 'broken.json').write_text(json.dumps({'judging': 'pairwise', 'run_name': 'broken'}))
    card = next(c for c in scan([runs_dir]) if c.surface == 'pairwise')
    assert card.error == 'missing required field: entries'


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def test_adapter_reads_name_and_timestamp(saved: tuple[Path, Path], run: PairwiseRun) -> None:
    _, path = saved
    adapter = ADAPTERS['pairwise']
    loaded = adapter.load(path)
    assert adapter.name(loaded) == 'v2 vs v3'
    assert adapter.created_at(loaded) == run.created_at


def test_adapter_rows_resolve_every_vote_to_a_label(run: PairwiseRun) -> None:
    adapter = ADAPTERS['pairwise']
    assert adapter.rows is not None
    rows = adapter.rows(run, run.entries)
    assert len(rows) == 4
    assert rows[0]['Question'] == 'q1'
    assert rows[0]['Winner'] == 'prompt-v2'
    assert rows[0]['judge-one'] == 'prompt-v2'
    assert rows[2]['Winner'] == 'prompt-v3'
    assert rows[3]['Winner'] == 'inconclusive'
    assert rows[3]['judge-one'] == 'abstained'


def test_adapter_body_from_results_rescopes_the_rollup(run: PairwiseRun) -> None:
    """A filtered body must not report the full run's numbers."""
    adapter = ADAPTERS['pairwise']
    body = adapter.body_from_results(run, run.entries[:1])
    assert 'the 1 of 1 comparisons' in body


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def test_sections_are_built_in_display_order(run: PairwiseRun) -> None:
    kinds = [s.kind for s in build_report_sections(run)]
    assert kinds == ['pairwise_consensus', 'pairwise_judges', 'pairwise_comparisons']


def test_consensus_reports_the_decided_count_not_the_total(run: PairwiseRun) -> None:
    consensus = build_report_sections(run)[0]
    assert consensus.data['comparisons'] == 4
    assert consensus.data['decided'] == 2


def test_split_and_inconclusive_comparisons_are_marked(run: PairwiseRun) -> None:
    rows = build_report_sections(run)[2].data['rows']
    assert [r['split'] for r in rows] == [False, True, False, True]


def test_judge_columns_keep_a_fixed_order_across_rows(run: PairwiseRun) -> None:
    section = build_report_sections(run)[2]
    assert section.data['judges'] == JUDGES
    for row in section.data['rows']:
        assert [v['model'] for v in row['votes']] == JUDGES


def test_position_bias_is_unavailable_when_the_run_did_not_swap() -> None:
    r = new_run(run_name='single ordering', judges=JUDGES, swap=False)
    r.add(_comparison(['A', 'A', 'A'], 'A'), question='q', response_a='a', response_b='b')
    rows = build_report_sections(r)[1].data['rows']
    # None, not 0.0: nothing was flippable, so there is no measurement to show.
    assert all(row['position_bias'] is None for row in rows)
    assert all(row['biased'] is False for row in rows)


def test_position_bias_is_flagged_above_the_threshold() -> None:
    r = new_run(run_name='biased panel', judges=['flipper'])
    for i in range(4):
        flipped = i < 2
        r.add(
            PairwiseComparison(winner='A', votes=[_vote('flipper', 'A', flipped=flipped)]),
            question=f'q{i}',
            response_a='a',
            response_b='b',
        )
    row = build_report_sections(r)[1].data['rows'][0]
    assert row['position_bias'] == pytest.approx(0.5)
    assert row['biased'] is True


def test_replacement_judges_are_tagged() -> None:
    r = new_run(run_name='with stand-in', judges=['primary'])
    r.add(
        PairwiseComparison(winner='A', votes=[_vote('stand-in', 'A', replacement=True)]),
        question='q',
        response_a='a',
        response_b='b',
    )
    row = build_report_sections(r)[1].data['rows'][0]
    assert row['replacement'] is True


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_body_uses_the_side_labels_not_the_slot_letters(run: PairwiseRun) -> None:
    body = render_report_body(run)
    assert 'prompt-v2 rate' in body
    assert 'prompt-v3 rate' in body
    assert '>A rate<' not in body


def test_body_renders_every_section(run: PairwiseRun) -> None:
    body = render_report_body(run)
    for anchor in ('pairwise_consensus', 'pairwise_judges', 'pairwise_comparisons'):
        assert f'id="{anchor}"' in body


def test_detail_includes_both_responses_and_every_explanation(run: PairwiseRun) -> None:
    body = render_report_body(run)
    assert 'a2' in body
    assert 'b2' in body
    for judge in JUDGES:
        assert f'{judge} says' in body


def test_abstention_renders_distinctly_from_a_decided_vote(run: PairwiseRun) -> None:
    body = render_report_body(run)
    assert 'pw-chip--abstain' in body


def test_single_ordering_run_says_bias_was_not_measured() -> None:
    r = new_run(run_name='single ordering', judges=JUDGES, swap=False)
    r.add(_comparison(['A', 'A', 'A'], 'A'), question='q', response_a='a', response_b='b')
    body = render_report_body(r)
    assert 'n/a' in body
    assert 'position bias was not measured' in body


def test_empty_run_renders_without_error() -> None:
    body = render_report_body(new_run(run_name='nothing here'))
    assert 'No comparisons in this run.' in body


def test_fully_inconclusive_run_renders_without_error() -> None:
    r = new_run(run_name='all noise', judges=JUDGES)
    r.add(_comparison([None, None, None], 'inconclusive'), question='q', response_a='a', response_b='b')
    body = render_report_body(r)
    # No decided comparison, so both win rates are None and render as a dash.
    assert 'the 0 of 1 comparisons' in body


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@pytest.fixture
def client(saved: tuple[Path, Path]) -> tuple[TestClient, str]:
    runs_dir, path = saved
    return TestClient(build_app(roots=[runs_dir]), raise_server_exceptions=True), report_id(path)


def test_report_route_renders_the_run(client: tuple[TestClient, str]) -> None:
    c, rid = client
    r = c.get(f'/r/{rid}')
    assert r.status_code == 200
    assert 'prompt-v2' in r.text
    assert 'Comparisons' in r.text


def test_run_list_route_renders(client: tuple[TestClient, str]) -> None:
    c, _ = client
    r = c.get('/?surface=pairwise')
    assert r.status_code == 200
    assert 'v2 vs v3' in r.text


def test_export_routes_all_succeed(client: tuple[TestClient, str]) -> None:
    c, rid = client
    for suffix in ('export.html', 'export.json', 'export.csv'):
        resp = c.get(f'/r/{rid}/{suffix}')
        assert resp.status_code == 200, suffix
        assert resp.text


def test_no_filter_rail_is_rendered_for_pairwise(client: tuple[TestClient, str]) -> None:
    """Pairwise has no dimensions, so an empty 'Filters' panel must not appear."""
    c, rid = client
    r = c.get(f'/r/{rid}')
    # 'filter-rail' also appears in the always-inlined stylesheet, so assert on
    # the rail's markup and its POST wiring rather than the bare class name.
    assert 'class="filter-rail"' not in r.text
    assert 'hx-post' not in r.text
