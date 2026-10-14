"""Tests for the run-vs-run pairwise comparison page."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import report_id
from evaluatorq.dashboard.pairwise_compare import match_comparisons, render_pairwise_compare_page
from evaluatorq.dashboard.view import run_compare_control
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


def test_duplicate_same_entry_object_is_paired_by_position() -> None:
    run = _run('same', [('q', 'A')])
    entry = run.entries[0]
    run.entries = [entry, entry]
    other = _run('other', [('q', 'B')])
    other.entries = [other.entries[0], other.entries[0]]

    matching = match_comparisons(run, other)
    assert matching.matched == [(0, 0), (1, 1)]
    assert matching.a_only == []
    assert matching.b_only == []



def test_renamed_labels_do_not_change_same_winning_slot() -> None:
    a = _run('a', [('q', 'A')], a='old', b='new')
    b = _run('b', [('q', 'A')], a='renamed', b='different')
    html = render_pairwise_compare_page('ra', 'rb', a, b)
    assert 'old' in html and 'renamed' in html
    assert 'pw-changed' not in html


def test_swapped_labels_do_not_hide_changed_winning_slot() -> None:
    a = _run('a', [('q', 'A')], a='old', b='new')
    b = _run('b', [('q', 'B')], a='new', b='old')
    html = render_pairwise_compare_page('ra', 'rb', a, b)
    assert html.count('pw-changed') == 1


def test_tie_label_collision_distinguishes_side_from_tie() -> None:
    a = _run('a', [('q', 'A')], a='tie')
    b = _run('b', [('q', 'tie')], a='tie')
    html = render_pairwise_compare_page('ra', 'rb', a, b)
    assert 'tie (A)' in html
    assert html.count('pw-changed') == 1


def test_empty_runs_show_empty_state_without_low_overlap_warning() -> None:
    html = render_pairwise_compare_page('ra', 'rb', _run('a', []), _run('b', []))
    assert 'No question appears in both runs.' in html
    assert 'low overlap' not in html

def test_equal_side_labels_are_slot_qualified() -> None:
    a = _run('a', [('q', 'A')], a='same', b='same')
    b = _run('b', [('q', 'B')], a='same', b='same')
    html = render_pairwise_compare_page('ra', 'rb', a, b)
    assert 'same (A)' in html and 'same (B)' in html
    assert html.count('pw-changed') == 1



def test_user_strings_are_escaped() -> None:
    a = _run('<b>x</b>', [('<script>q</script>', 'A')], a='a&b <i>')
    b = _run('y', [('<script>q</script>', 'A')])
    html = render_pairwise_compare_page('ra', 'rb', a, b)
    assert '<script>' not in html
    assert '&lt;script&gt;' in html
    assert 'a&amp;b &lt;i&gt;' in html
    assert '&amp;amp;' not in html


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
    assert f'/r/{a}?cmp=1' in resp.text
    assert f'/r/{b}?cmp=2' in resp.text


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


def test_route_422_for_invalid_utf8(client_and_ids: tuple[TestClient, str, str, Path]) -> None:
    client, a, _, d = client_and_ids
    bad = d / 'invalid-utf8.json'
    bad.write_bytes(b'\xff\xfe')
    response = client.get(f'/compare/pairwise?a={a}&b={report_id(bad)}')
    assert response.status_code == 422


def test_empty_runs_keep_the_actual_zero_comparison_total() -> None:
    a, b = _run('empty a', []), _run('empty b', [])
    matching = match_comparisons(a, b)
    assert matching.total == 0
    assert matching.match_rate == 0


def test_control_omits_current_run_and_hides_when_alone() -> None:
    html = run_compare_control('r1', [('r1', 'one'), ('r2', 'two')], '/compare/pairwise')
    assert 'action="/compare/pairwise"' in html
    assert 'value="r2"' in html and 'value="r1">' not in html.split('<select')[1]
    assert run_compare_control('r1', [('r1', 'one')], '/compare/pairwise') == ''


def test_control_posts_to_the_given_action() -> None:
    html = run_compare_control('r1', [('r1', 'one'), ('r2', 'two')], '/compare/sim')
    assert 'action="/compare/sim"' in html


def test_compare_links_open_the_report_on_that_comparison(tmp_path: Path) -> None:
    d = tmp_path / 'runs'
    d.mkdir()
    rows = [(f'q{i}', 'A') for i in range(30)]
    pa = _run('alpha', rows).save(d / 'a.json')
    pb = _run('beta', rows).save(d / 'b.json')
    client = TestClient(build_app([d]))
    page = client.get(f'/compare/pairwise?a={report_id(pa)}&b={report_id(pb)}').text
    link = f'/r/{report_id(pa)}?cmp=30'
    assert link in page
    report = client.get(link).text
    assert 'id="cmp-30"' in report


def _tied_run() -> PairwiseRun:
    return _run('tied', [('q1', 'A'), ('q2', 'B')])


def test_run_without_a_leader_uses_the_reports_verdict_wording() -> None:
    from evaluatorq.dashboard.pairwise_palette import verdict_eyebrow
    from evaluatorq.pairwise_reports.stats import verdict

    v = verdict(_tied_run())
    assert v.state == 'too_close'
    assert v.leader is None
    html = render_pairwise_compare_page('ra', 'rb', _tied_run(), _tied_run())
    assert verdict_eyebrow(v.state) in html


def test_corrupt_pairwise_file_gets_the_compare_error_page(client_and_ids: tuple[TestClient, str, str, Path]) -> None:
    client, a, _, d = client_and_ids
    bad = d / 'invalid.json'
    bad.write_text('{"judging": "pairwise", "entries": "nope"}')
    resp = client.get(f'/compare/pairwise?a={a}&b={report_id(bad)}')
    assert resp.status_code == 422
