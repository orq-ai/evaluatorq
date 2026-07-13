"""Side-by-side comparison of two sim runs (RES-1085).

Covers the KPI delta math, (persona, scenario) matching incl. unmatched
handling, the compare page + transcript routes, the graceful-error paths, and
the compare picker on the sim overview.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import report_id

# ---------------------------------------------------------------------------
# Fixture builders (real SimulationRun objects — the compare path model_validates)
# ---------------------------------------------------------------------------


def _result(
    persona: str,
    scenario: str,
    *,
    goal: bool,
    score: float,
    turns: int,
    terminated: str = 'judge',
    msgs: list[tuple[str, str]] | None = None,
):
    from evaluatorq.contracts import Message, TokenUsage
    from evaluatorq.simulation.types import SimulationResult, TerminatedBy

    return SimulationResult(
        messages=[Message(role=r, content=c) for r, c in (msgs or [])],
        terminated_by=TerminatedBy(terminated),
        reason='done',
        goal_achieved=goal,
        goal_completion_score=score,
        rules_broken=[],
        turn_count=turns,
        turn_metrics=[],
        token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        metadata={'persona': persona, 'scenario': scenario},
    )


def _run(name: str, results: list, averages: dict[str, float]):
    from evaluatorq.simulation.types import SimulationRun

    return SimulationRun(
        run_name=name,
        created_at=datetime.now(tz=timezone.utc),
        mode='run',
        target_kind='orq_agent',
        evaluator_names=['goal_achieved'],
        total_results=len(results),
        scorer_averages=averages,
        results=results,
    )


def _run_a():
    return _run(
        'run-A',
        [
            _result(
                'alice', 'billing', goal=True, score=0.9, turns=4, msgs=[('user', 'hi A'), ('assistant', 'hello A')]
            ),
            _result('bob', 'refund', goal=False, score=0.2, turns=6),
            _result('carol', 'onlyA', goal=True, score=1.0, turns=2),
        ],
        {'goal_achieved': 0.66, 'criteria_met': 0.5},
    )


def _run_b():
    return _run(
        'run-B',
        [
            # alice/billing flips True -> False, score drops
            _result(
                'alice', 'billing', goal=False, score=0.4, turns=5, msgs=[('user', 'hi B'), ('assistant', 'hello B')]
            ),
            _result('bob', 'refund', goal=True, score=0.8, turns=3, terminated='max_turns'),
            _result('dave', 'onlyB', goal=True, score=0.7, turns=2),
        ],
        {'goal_achieved': 0.66, 'safety': 1.0},
    )


@pytest.fixture
def roots(tmp_path: Path) -> list[Path]:
    sim = tmp_path / 'sim-runs'
    sim.mkdir()
    (sim / 'a.json').write_text(_run_a().model_dump_json())
    (sim / 'b.json').write_text(_run_b().model_dump_json())
    return [tmp_path / 'runs', sim]


def _rids(roots: list[Path]) -> tuple[str, str]:
    sim = roots[1]
    return report_id(sim / 'a.json'), report_id(sim / 'b.json')


# ---------------------------------------------------------------------------
# Unit: KPI deltas
# ---------------------------------------------------------------------------


def test_compare_kpis_covers_aggregates_scorers_and_terminated():
    from evaluatorq.dashboard.sim_compare import compare_kpis
    from evaluatorq.simulation.reports.sections import individual_entries

    a, b = _run_a(), _run_b()
    rows = compare_kpis(individual_entries(a.results), individual_entries(b.results), a, b)
    by_label = {r.label: r for r in rows}

    assert by_label['Conversations'].a == 3.0
    assert by_label['Conversations'].b == 3.0
    # A: 2/3 achieved; B: 2/3 achieved.
    assert by_label['Goal-achieved rate'].a == pytest.approx(2 / 3)
    assert by_label['Goal-achieved rate'].b == pytest.approx(2 / 3)
    # per-scorer union: goal_achieved (both), criteria_met (A only), safety (B only)
    assert by_label['Scorer: criteria_met'].a == 0.5
    assert by_label['Scorer: criteria_met'].b == 0.0
    assert by_label['Scorer: safety'].b == 1.0
    # terminated-by distribution present
    assert by_label['Terminated: judge'].a == 3.0  # A: all judge
    assert by_label['Terminated: max_turns'].b == 1.0
    assert by_label['Terminated: max_turns'].delta == 1.0


# ---------------------------------------------------------------------------
# Unit: matching by (persona, scenario)
# ---------------------------------------------------------------------------


def test_match_entries_pairs_and_flags_unmatched():
    from evaluatorq.dashboard.sim_compare import match_entries
    from evaluatorq.simulation.reports.sections import individual_entries

    m = match_entries(individual_entries(_run_a().results), individual_entries(_run_b().results))
    matched_keys = {(p.a.persona, p.a.scenario) for p in m.matched}
    assert matched_keys == {('alice', 'billing'), ('bob', 'refund')}
    assert [(e.persona, e.scenario) for e in m.a_only] == [('carol', 'onlyA')]
    assert [(e.persona, e.scenario) for e in m.b_only] == [('dave', 'onlyB')]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def test_compare_page_renders_kpis_diffs_and_flip(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    resp = client.get(f'/compare/sim?a={rid_a}&b={rid_b}')
    assert resp.status_code == 200
    html = resp.text
    assert 'KPI deltas' in html
    assert 'Outcome diffs' in html
    assert 'run-A' in html and 'run-B' in html
    # alice/billing flips True->False → a flip marker is present
    assert 'flip' in html
    # unmatched conversations surfaced, not dropped
    assert 'Only in A' in html and 'Only in B' in html
    assert 'carol' in html and 'dave' in html
    # inline-SVG charts render server-side (no JS/static dependency)
    assert '<svg' in html
    assert 'Outcomes' in html and 'Scorer averages' in html and 'How conversations ended' in html
    assert 'Per-conversation score change' in html


def test_compare_transcript_route_shows_both_sides(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    # alice/billing is index 0 in both runs
    resp = client.get(f'/compare/sim/transcript?a={rid_a}&b={rid_b}&ia=0&ib=0')
    assert resp.status_code == 200
    html = resp.text
    assert 'hello A' in html and 'hello B' in html
    assert 'cmp-transcript-grid' in html


def test_compare_missing_param_is_400(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, _ = _rids(roots)
    resp = client.get(f'/compare/sim?a={rid_a}')
    assert resp.status_code == 400


def test_compare_unknown_rid_is_404(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, _ = _rids(roots)
    resp = client.get(f'/compare/sim?a={rid_a}&b=deadbeef')
    assert resp.status_code == 404


def test_compare_bar_on_sim_overview(roots: list[Path]):
    client = TestClient(build_app(roots))
    resp = client.get('/?surface=sim')
    assert resp.status_code == 200
    html = resp.text
    assert 'action="/compare/sim"' in html
    assert 'name="a"' in html and 'name="b"' in html
