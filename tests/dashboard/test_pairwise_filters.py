"""Tests for the pairwise filter dimensions and the pairwise filter rail."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.filters import FILTERS, apply_or_all
from evaluatorq.dashboard.library import report_id
from evaluatorq.pairwise import PairwiseComparison, PairwiseVote
from evaluatorq.pairwise_run import PairwiseRun, new_run

JUDGES = ['judge-one', 'judge-two', 'judge-three']


def _vote(model: str, value: str | None, *, flipped: bool = False, failures: int = 0) -> PairwiseVote:
    return PairwiseVote(
        model=model,
        vote=value,
        flipped=flipped,
        completed=True,
        repetition_failures=failures,
        explanation=f'{model} says {value}',
    )


def _comparison(values: list[str | None], winner: str, **kw: bool) -> PairwiseComparison:
    return PairwiseComparison(
        winner=winner,
        votes=[_vote(m, v, **kw) for m, v in zip(JUDGES, values)],
    )


@pytest.fixture
def run() -> PairwiseRun:
    """q1 unanimous A, q2 split tie, q3 unanimous B, q4 thrown-out inconclusive."""
    r = new_run(run_name='v2 vs v3', label_a='prompt-v2', label_b='prompt-v3', judges=JUDGES, criteria='accuracy')
    r.add(_comparison(['A', 'A', 'A'], 'A'), question='q1', response_a='a1', response_b='b1')
    r.add(_comparison(['A', 'B', 'tie'], 'tie'), question='q2', response_a='a2', response_b='b2')
    r.add(_comparison(['B', 'B', 'B'], 'B'), question='q3', response_a='a3', response_b='b3')
    r.add(
        _comparison([None, None, None], 'inconclusive', flipped=True), question='q4', response_a='a4', response_b='b4'
    )
    return r


def _questions(entries: list) -> list[str]:
    return [e.question for e in entries]


def _apply(run: PairwiseRun, selections: dict[str, list[str]]) -> list:
    return FILTERS['pairwise'].apply(run, selections)


def test_dimensions_are_declared_in_rail_order() -> None:
    assert FILTERS['pairwise'].dimensions == ['winner', 'split', 'thrown_out', 'judge_failed', 'judge']


def test_options_list_labels_then_outcomes_and_judges(run: PairwiseRun) -> None:
    opts = FILTERS['pairwise'].options(run)
    assert opts['winner'] == ['prompt-v2', 'prompt-v3', 'tie', 'inconclusive']
    assert opts['judge'] == JUDGES
    # Opt-in chips carry no option list.
    for dim in ('split', 'thrown_out', 'judge_failed'):
        assert dim not in opts


def test_equal_labels_fall_back_to_slot_letters() -> None:
    r = new_run(run_name='same', label_a='x', label_b='x', judges=JUDGES)
    assert FILTERS['pairwise'].options(r)['winner'] == ['A', 'B', 'tie', 'inconclusive']


def test_empty_selections_return_every_entry_in_order(run: PairwiseRun) -> None:
    out = _apply(run, {})
    assert _questions(out) == ['q1', 'q2', 'q3', 'q4']
    assert _apply(run, {'winner': [], 'judge': []}) == out
    assert FILTERS['pairwise'].results(run) == out


def test_apply_preserves_entry_identity(run: PairwiseRun) -> None:
    out = _apply(run, {'winner': ['prompt-v3', 'tie']})
    assert [id(e) for e in out] == [id(run.entries[1]), id(run.entries[2])]
    assert out[0] is run.entries[1]


def test_winner_maps_labels_back_to_sides(run: PairwiseRun) -> None:
    assert _questions(_apply(run, {'winner': ['prompt-v2']})) == ['q1']
    assert _questions(_apply(run, {'winner': ['prompt-v3']})) == ['q3']
    assert _questions(_apply(run, {'winner': ['tie']})) == ['q2']
    assert _questions(_apply(run, {'winner': ['inconclusive']})) == ['q4']
    assert _questions(_apply(run, {'winner': ['prompt-v2', 'inconclusive']})) == ['q1', 'q4']


def test_all_four_winners_selected_means_all(run: PairwiseRun) -> None:
    sel = {'winner': ['prompt-v2', 'prompt-v3', 'tie', 'inconclusive']}
    assert _questions(_apply(run, sel)) == ['q1', 'q2', 'q3', 'q4']


def test_winner_with_equal_labels_uses_slot_letters() -> None:
    r = new_run(run_name='same', label_a='x', label_b='x', judges=JUDGES)
    r.add(_comparison(['A', 'A', 'A'], 'A'), question='qa', response_a='a', response_b='b')
    r.add(_comparison(['B', 'B', 'B'], 'B'), question='qb', response_a='a', response_b='b')
    assert _questions(_apply(r, {'winner': ['B']})) == ['qb']


def test_split_keeps_split_and_inconclusive_panels(run: PairwiseRun) -> None:
    # An inconclusive comparison counts as split (stats.is_split).
    assert _questions(_apply(run, {'split': ['yes']})) == ['q2', 'q4']


def test_thrown_out_keeps_entries_with_a_flipped_vote(run: PairwiseRun) -> None:
    assert _questions(_apply(run, {'thrown_out': ['yes']})) == ['q4']


def test_judge_failed_covers_repetition_failures_and_missing_judges() -> None:
    r = new_run(run_name='fail', judges=JUDGES)
    r.add(_comparison(['A', 'A', 'A'], 'A'), question='ok', response_a='a', response_b='b')
    failing = PairwiseComparison(
        winner='A',
        votes=[_vote('judge-one', 'A', failures=1), _vote('judge-two', 'A'), _vote('judge-three', 'A')],
    )
    r.add(failing, question='retried', response_a='a', response_b='b')
    missing = PairwiseComparison(winner='A', votes=[_vote('judge-one', 'A'), _vote('judge-two', 'A')])
    r.add(missing, question='absent', response_a='a', response_b='b')
    assert _questions(_apply(r, {'judge_failed': ['yes']})) == ['retried', 'absent']


def test_judge_keeps_entries_where_a_selected_judge_voted() -> None:
    r = new_run(run_name='judges', judges=JUDGES)
    r.add(_comparison(['A', 'A', 'A'], 'A'), question='all', response_a='a', response_b='b')
    only_two = PairwiseComparison(winner='A', votes=[_vote('judge-two', 'A')])
    r.add(only_two, question='two', response_a='a', response_b='b')
    only_one = PairwiseComparison(winner='A', votes=[_vote('judge-one', 'A')])
    r.add(only_one, question='one', response_a='a', response_b='b')
    assert _questions(_apply(r, {'judge': ['judge-two']})) == ['all', 'two']
    assert _questions(_apply(r, {'judge': ['judge-one', 'judge-two']})) == ['all', 'two', 'one']
    assert _questions(_apply(r, {'judge': JUDGES})) == ['all', 'two', 'one']


def test_judge_filter_needs_a_cast_vote_on_real_shaped_runs() -> None:
    # run_pairwise records a vote object for every panel judge; abstentions carry vote=None.
    r = new_run(run_name='judges', judges=JUDGES)
    r.add(_comparison(['A', 'A', 'A'], 'A'), question='voted', response_a='a', response_b='b')
    r.add(_comparison(['A', 'A', None], 'A'), question='abstained', response_a='a', response_b='b')
    assert _questions(_apply(r, {'judge': ['judge-three']})) == ['voted']
    assert _questions(_apply(r, {'judge': ['judge-one']})) == ['voted', 'abstained']


def test_winner_and_split_intersect(run: PairwiseRun) -> None:
    assert _questions(_apply(run, {'winner': ['tie'], 'split': ['yes']})) == ['q2']
    assert _apply(run, {'winner': ['prompt-v2'], 'split': ['yes']}) == []


def test_apply_or_all_dispatches_through_the_registry(run: PairwiseRun) -> None:
    assert _questions(apply_or_all(run, 'pairwise', {'thrown_out': ['yes']})) == ['q4']


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@pytest.fixture
def client(run: PairwiseRun, tmp_path: Path) -> tuple[TestClient, str]:
    runs_dir = tmp_path / 'pairwise-runs'
    runs_dir.mkdir()
    path = run.save(runs_dir / 'run.json')
    return TestClient(build_app(roots=[runs_dir]), raise_server_exceptions=True), report_id(path)


def test_report_page_renders_the_filter_rail(client: tuple[TestClient, str]) -> None:
    c, rid = client
    r = c.get(f'/r/{rid}')
    assert r.status_code == 200
    assert 'id="filter-form"' in r.text
    assert f'hx-post="/r/{rid}/filter"' in r.text
    assert 'Showing all comparisons' in r.text


def test_filter_route_reports_the_narrowed_count(client: tuple[TestClient, str]) -> None:
    c, rid = client
    r = c.post(f'/r/{rid}/filter', data={'split': 'yes'})
    assert r.status_code == 200
    assert '2 of 4 comparisons' in r.text
    # The chip stays checked after the swap.
    assert 'name="split" value="yes" checked' in r.text


def test_filter_route_without_selections_shows_all(client: tuple[TestClient, str]) -> None:
    c, rid = client
    r = c.post(f'/r/{rid}/filter', data={})
    assert r.status_code == 200
    assert 'Showing all comparisons' in r.text


def test_side_label_colliding_with_an_outcome_falls_back_to_slot_letters() -> None:
    r = new_run(run_name='collide', label_a='tie', label_b='B2', judges=JUDGES)
    r.add(_comparison(['A', 'A', 'A'], 'A'), question='qa', response_a='a', response_b='b')
    r.add(_comparison(['A', 'B', 'tie'], 'tie'), question='qt', response_a='a', response_b='b')
    assert FILTERS['pairwise'].options(r)['winner'] == ['A', 'B', 'tie', 'inconclusive']
    assert _questions(_apply(r, {'winner': ['tie']})) == ['qt']
