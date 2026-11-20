"""Target cost in the sim and red-team report headers and on the sim compare page.

The target agent's own cost sits next to the run's total cost. A run saved before
target usage was recorded shows "unknown", never $0, and a partly priced figure is
marked as a lower bound.
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from starlette.testclient import TestClient

from evaluatorq.contracts import TokenUsage
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import report_id
from evaluatorq.redteam.contracts import ExecutionDetails, RedTeamReport
from evaluatorq.redteam.reports.converters import compute_report_summary
from evaluatorq.simulation.contracts import SimulationResult, SimulationRun, TerminatedBy
from tests.dashboard.conftest import _make_rt_result


def _usage(cost: float, *, calls: int = 2, priced: int = 2) -> TokenUsage:
    return TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120, total_cost=cost, calls=calls, priced_calls=priced)


def _sim_run(name: str, target_cost: float | None) -> SimulationRun:
    results = [
        SimulationResult(
            messages=[],
            terminated_by=TerminatedBy.judge,
            reason='done',
            goal_achieved=True,
            goal_completion_score=1.0,
            rules_broken=[],
            turn_count=2,
            turn_metrics=[],
            token_usage=_usage(0.5),
            target_token_usage=None if target_cost is None else _usage(target_cost),
            metadata={'persona': persona, 'scenario': 'billing'},
        )
        for persona in ('alice', 'bob')
    ]
    return SimulationRun(
        run_name=name,
        created_at=datetime(2026, 11, 1, tzinfo=timezone.utc),
        mode='run',
        target_kind='callback',
        evaluator_names=[],
        total_results=2,
        scorer_averages={},
        results=results,
        token_usage_total=_usage(1.0, calls=4, priced=4),
        target_token_usage=None if target_cost is None else _usage(2 * target_cost, calls=4, priced=4),
    )


def _write_legacy(run: SimulationRun, path: Path) -> None:
    """Save *run* the way a version without target usage did: no target fields at all."""
    data = run.model_dump(mode='json')
    data.pop('target_token_usage')
    for r in data['results']:
        r.pop('target_token_usage')
    path.write_text(json.dumps(data))


def _hero_cost(page: str) -> str:
    match = re.search(r'<p class="report-hero-cost">(.*?)</p>', page, re.S)
    assert match, 'hero cost line missing'
    return html.unescape(re.sub(r'<[^>]+>', ' ', match.group(1)))


def _client(root: Path) -> TestClient:
    return TestClient(build_app(roots=[root]), raise_server_exceptions=True)


def test_sim_header_shows_total_and_target_cost(tmp_path: Path) -> None:
    path = tmp_path / 'a.json'
    path.write_text(_sim_run('A', 0.01).model_dump_json())

    text = _hero_cost(_client(tmp_path).get(f'/r/{report_id(path)}').text)

    assert 'Total cost $1.0000' in ' '.join(text.split())
    assert 'Target cost $0.0200' in ' '.join(text.split())


def test_sim_header_shows_unknown_target_cost_for_an_old_run(tmp_path: Path) -> None:
    path = tmp_path / 'old.json'
    _write_legacy(_sim_run('Old', 0.01), path)

    page = _client(tmp_path).get(f'/r/{report_id(path)}').text
    text = ' '.join(_hero_cost(page).split())

    assert 'Target cost unknown' in text
    assert '$0.0000' not in text
    assert 'did not record the target agent' in html.unescape(page)


def test_rt_header_marks_a_partly_priced_target_cost_as_a_lower_bound(tmp_path: Path) -> None:
    results = [
        _make_rt_result(
            attack_id='ASI01-1',
            category='ASI01',
            execution=ExecutionDetails(turns=1, token_usage=_usage(0.3), token_usage_target=_usage(0.004, calls=2, priced=1)),
        )
    ]
    report = RedTeamReport(
        created_at=datetime(2026, 11, 1, tzinfo=timezone.utc),
        description='rt',
        pipeline='dynamic',
        categories_tested=['ASI01'],
        tested_agents=['agent-a'],
        total_results=1,
        results=results,
        summary=compute_report_summary(results),
    )
    path = tmp_path / 'rt.json'
    path.write_text(report.model_dump_json())

    text = ' '.join(_hero_cost(_client(tmp_path).get(f'/r/{report_id(path)}').text).split())

    assert 'Total cost $0.3000' in text
    assert 'Target cost $0.0040 (1 of 2 calls)' in text


def test_rt_header_shows_unknown_target_cost_for_an_old_report(tmp_path: Path) -> None:
    results = [_make_rt_result(attack_id='ASI01-1', category='ASI01', execution=ExecutionDetails(turns=1, token_usage=_usage(0.3)))]
    report = RedTeamReport(
        created_at=datetime(2026, 11, 1, tzinfo=timezone.utc),
        description='rt',
        pipeline='dynamic',
        categories_tested=['ASI01'],
        tested_agents=['agent-a'],
        total_results=1,
        results=results,
        summary=compute_report_summary(results),
    )
    path = tmp_path / 'rt.json'
    path.write_text(report.model_dump_json())

    text = ' '.join(_hero_cost(_client(tmp_path).get(f'/r/{report_id(path)}').text).split())

    assert 'Target cost unknown' in text


def _target_tile_html(page: str) -> tuple[str, str]:
    """The target cost tile's value and its raw sub-line HTML (the pill's colour lives in its style)."""
    match = re.search(r'<div class="kpi-card[^"]*"><div class="kpi-value">([^<]*)</div><div class="kpi-label">Target cost / conv\.</div>(.*?)</div></div>', page, re.S)
    assert match, 'target cost tile missing'
    return match.group(1), match.group(2)


def _target_tile(page: str) -> str:
    value, sub = _target_tile_html(page)
    return ' '.join(html.unescape(value + ' ' + re.sub(r'<[^>]+>', ' ', sub)).split())


def _tone(tone: str) -> str:
    """The inline style a pill of *tone* carries."""
    from evaluatorq.dashboard.report_kit import pill

    match = re.search(r'style="[^"]*"', pill('x', tone=tone))
    assert match
    return match.group(0)


def _compare(tmp_path: Path, a_run: SimulationRun, b_run: SimulationRun) -> str:
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    a.write_text(a_run.model_dump_json())
    b.write_text(b_run.model_dump_json())
    return _client(tmp_path).get(f'/compare/sim?a={report_id(a)}&b={report_id(b)}').text


def _drop_target_usage(run: SimulationRun, index: int) -> SimulationRun:
    """*run* with conversation *index* reporting no target usage, the run total summed over the rest."""
    run.results[index].target_token_usage = None
    kept = [r.target_token_usage for r in run.results if r.target_token_usage is not None]
    run.target_token_usage = _usage(sum(u.total_cost or 0.0 for u in kept), calls=2 * len(kept), priced=2 * len(kept))
    return run


def test_compare_shows_the_target_cost_per_conversation_from_a_to_b(tmp_path: Path) -> None:
    page = _compare(tmp_path, _sim_run('A', 0.01), _sim_run('B', 0.004))

    tile = _target_tile(page)
    _, sub = _target_tile_html(page)

    # two conversations each: the run totals are $0.0200 and $0.0080, the averages half that
    assert tile.startswith('$0.0040')
    assert '$0.0100 → $0.0040' in tile
    assert '▼ -$0.0060' in tile
    # equal coverage keeps lower-is-better colouring: a cheaper B is good
    assert _tone('good') in sub
    assert 'conversations reported usage' not in tile
    assert 'average over the 2 of 2 conversations that reported target usage' in html.unescape(sub)


def test_compare_colours_a_costlier_b_as_worse_when_coverage_is_equal(tmp_path: Path) -> None:
    _, sub = _target_tile_html(_compare(tmp_path, _sim_run('A', 0.004), _sim_run('B', 0.01)))

    assert '▲ +$0.0060' in html.unescape(sub)
    assert _tone('danger') in sub


def test_compare_averages_over_the_conversations_that_reported_usage(tmp_path: Path) -> None:
    # A: one of two conversations reported $0.01; B: both reported $0.01. The run totals
    # ($0.0100 vs $0.0200) would read as B doubling its cost; per conversation it held.
    page = _compare(tmp_path, _drop_target_usage(_sim_run('A', 0.01), 1), _sim_run('B', 0.01))

    tile = _target_tile(page)

    assert '$0.0100 → $0.0100' in tile
    assert '▲' not in tile and '▼' not in tile
    assert 'A 1 of 2, B 2 of 2 conversations reported usage' in tile
    assert 'lower bound' not in tile


def test_unequal_coverage_shows_the_change_in_a_neutral_pill(tmp_path: Path) -> None:
    b_run = _sim_run('B', 0.004)
    page = _compare(tmp_path, _drop_target_usage(_sim_run('A', 0.01), 1), b_run)

    tile = _target_tile(page)
    _, sub = _target_tile_html(page)

    assert '$0.0100 → $0.0040' in tile
    assert '▼ -$0.0060' in tile
    assert _tone('neutral') in sub
    assert _tone('good') not in sub and _tone('danger') not in sub
    assert 'A 1 of 2, B 2 of 2 conversations reported usage' in tile
    assert 'A: average over the 1 of 2 conversations' in html.unescape(sub)


def test_compare_with_an_old_run_shows_unknown_and_no_delta(tmp_path: Path) -> None:
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    _write_legacy(_sim_run('Old', 0.01), a)
    b.write_text(_sim_run('B', 0.004).model_dump_json())

    tile = _target_tile(_client(tmp_path).get(f'/compare/sim?a={report_id(a)}&b={report_id(b)}').text)

    assert 'unknown → $0.0040' in tile
    assert 'n/a' in tile
    assert '▲' not in tile and '▼' not in tile
    # an unknown side has no coverage to compare
    assert 'conversations reported usage' not in tile


def test_a_partly_priced_target_cost_is_a_lower_bound_on_compare(tmp_path: Path) -> None:
    a_run = _sim_run('A', 0.01)
    for r in a_run.results:
        r.target_token_usage = _usage(0.01, calls=2, priced=1)
    a_run.target_token_usage = _usage(0.02, calls=4, priced=2)

    tile = _target_tile(_compare(tmp_path, a_run, _sim_run('B', 0.004)))

    assert 'lower bound' in tile


def test_the_report_header_marks_a_missing_conversation_as_a_lower_bound(tmp_path: Path) -> None:
    run = _drop_target_usage(_sim_run('A', 0.01), 1)
    a = tmp_path / 'a.json'
    a.write_text(run.model_dump_json())

    header = ' '.join(_hero_cost(_client(tmp_path).get(f'/r/{report_id(a)}').text).split())

    assert 'Target cost $0.0100 (1 of 2 conversations)' in header


def test_cost_figure_names_both_coverage_gaps_in_one_qualifier() -> None:
    from evaluatorq.dashboard.report_kit import cost_figure

    fig = cost_figure(_usage(0.004, calls=2, priced=1), unknown_tip='', missing=1, total=4, noun='attacks')

    assert (fig.value, fig.qualifier) == ('$0.0040', '(1 of 2 calls, 3 of 4 attacks)')
    assert fig.tip == 'Lower bound: the cost covers only 1 of 2 calls, 3 of 4 attacks.'
    assert cost_figure(None, unknown_tip='why').unknown
