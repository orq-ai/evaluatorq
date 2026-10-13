"""Tests for the run-vs-run pairwise comparison page."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import report_id
from evaluatorq.dashboard.pairwise_compare import (
    match_comparisons,
    pairwise_run_compare_control,
    render_pairwise_compare_page,
)
from evaluatorq.pairwise import PairwiseComparison, PairwiseVote
from evaluatorq.pairwise_run import PairwiseRun, new_run

JUDGES = ['j1', 'j2']


def _cmp(winner: str) -> PairwiseComparison:
    return PairwiseComparison(
        winner=winner,
        votes=[PairwiseVote(model=m, vote=winner, flipped=False, completed=True, explanation='x') for m in JUDGES],
    )


def _run(name: str, rows: list[tuple[str, str]], *, a: str = 'old', b: str = 'new') -> PairwiseRun:
    r = new_run(run_name=name, label_a=a, label_b=b, judges=JUDGES, criteria='c')
    for q, w in rows:
        r.add(_cmp(w), question=q, response_a='ra', response_b='rb')
    return r


def test_match_pairs_by_question_and_reports_leftovers() -> None:
    a = _run('a', [('q1', 'A'), ('q2', 'B'), ('only-a', 'A')])
    b = _run('b', [('q2', 'A'), ('q1', 'B'), ('only-b', 'B')])
    m = match_comparisons(a, b)
    assert sorted(m.matched) == [(0, 1), (1, 0)]
    assert m.a_only == [2]
    assert m.b_only == [2]
    assert m.match_rate == pytest.approx(2 / 3)


def test_duplicate_questions_pair_positionally() -> None:
    a = _run('a', [('q', 'A'), ('q', 'B'), ('q', 'A')])
    b = _run('b', [('q', 'A'), ('q', 'B')])
    m = match_comparisons(a, b)
    assert m.matched == [(0, 0), (1, 1)]
    assert m.a_only == [2]
    assert m.b_only == []


def test_changed_winner_is_tagged_by_side_label() -> None:
    a = _run('a', [('q1', 'A'), ('q2', 'A')], a='old', b='new')
    # Slot letters differ but the winning label ('old') is the same for q1.
    b = _run('b', [('q1', 'B'), ('q2', 'A')], a='new', b='old')
    html = render_pairwise_compare_page('ra', 'rb', a, b)
    assert html.count('pw-changed') == 1  # q2: old -> new
    assert 'pw-compare' in html


def test_user_strings_are_escaped() -> None:
    a = _run('<b>x</b>', [('<script>q</script>', 'A')])
    b = _run('y', [('<script>q</script>', 'A')])
    html = render_pairwise_compare_page('ra', 'rb', a, b)
    assert '<script>' not in html
    assert '&lt;script&gt;' in html


@pytest.fixture
def client_and_ids(tmp_path: Path) -> tuple[TestClient, str, str, Path]:
    d = tmp_path / 'runs'
    d.mkdir()
    pa = _run('alpha run', [('q1', 'A'), ('q2', 'B')]).save(d / 'a.json')
    pb = _run('beta run', [('q1', 'B'), ('q2', 'B')]).save(d / 'b.json')
    return TestClient(build_app([d])), report_id(pa), report_id(pb), d


def test_route_renders_both_run_names(client_and_ids: tuple[TestClient, str, str, Path]) -> None:
    client, a, b, _ = client_and_ids
    resp = client.get(f'/compare/pairwise?a={a}&b={b}')
    assert resp.status_code == 200
    assert 'alpha run' in resp.text and 'beta run' in resp.text
    assert f'/r/{a}#cmp-1' in resp.text


def test_route_400_when_b_missing_or_same(client_and_ids: tuple[TestClient, str, str, Path]) -> None:
    client, a, _, _ = client_and_ids
    assert client.get(f'/compare/pairwise?a={a}').status_code == 400
    assert client.get(f'/compare/pairwise?a={a}&b={a}').status_code == 400


def test_route_404_for_unknown_or_sim_run(client_and_ids: tuple[TestClient, str, str, Path]) -> None:
    client, a, _, d = client_and_ids
    assert client.get(f'/compare/pairwise?a={a}&b=deadbeef').status_code == 404
    sim = d / 'sim.json'
    sim.write_text(json.dumps({'mode': 'run', 'run_name': 's', 'entries': []}))
    assert client.get(f'/compare/pairwise?a={a}&b={report_id(sim)}').status_code == 404


def test_route_422_for_corrupt_run(client_and_ids: tuple[TestClient, str, str, Path]) -> None:
    client, a, _, d = client_and_ids
    bad = d / 'bad.json'
    bad.write_text('{"judging": "pairwise", ')
    assert client.get(f'/compare/pairwise?a={a}&b={report_id(bad)}').status_code == 422


def test_control_omits_current_run_and_hides_when_alone() -> None:
    html = pairwise_run_compare_control('r1', [('r1', 'one'), ('r2', 'two')])
    assert 'action="/compare/pairwise"' in html
    assert 'value="r2"' in html and 'value="r1">' not in html.split('<select')[1]
    assert pairwise_run_compare_control('r1', [('r1', 'one')]) == ''
