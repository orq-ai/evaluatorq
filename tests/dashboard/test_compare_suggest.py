"""Which earlier runs the report's compare menu suggests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from evaluatorq.dashboard.compare_suggest import CompareCandidate, candidates, is_default_name, name_key, suggest
from evaluatorq.dashboard.library import report_id

T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)


def cand(rid: str, name: str, day: int, target: str | None = None) -> CompareCandidate:
    return CompareCandidate(rid=rid, name=name, when=T0 + timedelta(days=day), target=target, headline='')


@pytest.mark.parametrize(
    ('raw', 'key'),
    [
        ('checkout-agent v3', 'checkout-agent'),
        ('Checkout-Agent', 'checkout-agent'),
        ('checkout-agent-2026-11-18', 'checkout-agent'),
        ('checkout-agent (2)', 'checkout-agent'),
        ('checkout-agent_002', 'checkout-agent'),
        ('checkout agent #4', 'checkout agent'),
        ('checkout-agent-20261118-143000', 'checkout-agent'),
        ('checkout-agent v3 2026-11-18', 'checkout-agent'),
        ('v2', ''),
        ('checkout agent 2', 'checkout agent'),
        ('agent-gpt4', 'agent-gpt4'),  # a number glued to the name is part of it, not a version
    ],
)
def test_name_key(raw: str, key: str) -> None:
    assert name_key(raw) == key


@pytest.mark.parametrize('raw', ['sim', 'red-team', 'simulation', 'simulation-20261118-143000-ab12cd34', 'v2', '  '])
def test_default_names(raw: str) -> None:
    assert is_default_name(raw)


def test_real_name_is_not_default() -> None:
    assert not is_default_name('checkout-agent v3')


def test_name_matches_newest_first_capped_and_earlier_only() -> None:
    current = cand('c', 'agent v4', 10)
    others = [
        cand('v1', 'agent v1', 1),
        cand('v2', 'agent v2', 2),
        cand('v3', 'agent v3', 3),
        cand('v5', 'agent v5', 11),
        cand('o', 'other', 5),
    ]
    assert [c.rid for c in suggest(current, others)] == ['v3', 'v2', 'v1']
    assert [c.rid for c in suggest(current, others, limit=2)] == ['v3', 'v2']


def test_current_run_is_never_suggested() -> None:
    current = cand('c', 'agent v2', 10)
    assert suggest(current, [current, cand('v1', 'agent v1', 1)]) == [cand('v1', 'agent v1', 1)]


def test_target_fallback_for_default_name() -> None:
    current = cand('c', 'sim', 10, target='support-bot')
    others = [cand('s', 'sim', 1, target='support-bot'), cand('x', 'x', 2, target='support-bot'), cand('y', 'y', 3, target='billing')]
    assert [c.rid for c in suggest(current, others)] == ['x', 's']


def test_target_matches_skipped_when_name_matches_exist() -> None:
    current = cand('c', 'agent v2', 10, target='bot')
    others = [cand('n', 'agent v1', 1, target='other'), cand('t', 'unrelated', 2, target='bot')]
    assert [c.rid for c in suggest(current, others)] == ['n']


def test_same_timestamp_is_not_earlier() -> None:
    assert suggest(cand('c', 'agent v2', 5), [cand('d', 'agent v1', 5)]) == []


def test_missing_target_never_matches() -> None:
    assert suggest(cand('c', 'sim', 5), [cand('d', 'sim', 1), cand('e', 'other', 2)]) == []


# ---------------------------------------------------------------------------
# candidates(): read from the run stores
# ---------------------------------------------------------------------------


def _sim(name: str, created: datetime, **extra: object) -> str:
    from evaluatorq.simulation.contracts import SimulationRun

    return SimulationRun(
        run_name=name,
        created_at=created,
        mode='run',
        target_kind='orq_agent',
        evaluator_names=['goal_achieved'],
        total_results=0,
        scorer_averages={'goal_achieved': 1.0},
        results=[],
        **extra,
    ).model_dump_json()


def test_sim_candidates_carry_concrete_targets_only(tmp_path: Path) -> None:
    sim = tmp_path / 'sim'
    sim.mkdir()
    (sim / 'a.json').write_text(_sim('agent v1', T0, target='agent:support-bot'))
    (sim / 'b.json').write_text(_sim('legacy', T0 + timedelta(days=1)))
    by_rid = {c.rid: c for c in candidates('sim', [sim])}
    assert by_rid[report_id(sim / 'a.json')].target == 'support-bot'
    assert by_rid[report_id(sim / 'b.json')].target is None
    assert [c.name for c in candidates('sim', [sim])] == ['legacy', 'agent v1']


def test_pairwise_candidates_use_named_label_pairs(tmp_path: Path) -> None:
    from evaluatorq.pairwise_run import new_run

    pw = tmp_path / 'pw'
    pw.mkdir()
    (pw / 'a.json').write_text(new_run(run_name='x', label_a='gpt', label_b='claude', judges=['j']).model_dump_json())
    (pw / 'b.json').write_text(new_run(run_name='y', judges=['j']).model_dump_json())
    by_rid = {c.rid: c for c in candidates('pairwise', [pw])}
    assert by_rid[report_id(pw / 'a.json')].target == 'gpt vs claude'
    assert by_rid[report_id(pw / 'b.json')].target is None


def test_pairwise_candidates_with_placeholder_labels_name_no_target(tmp_path: Path) -> None:
    from evaluatorq.pairwise_run import new_run

    pw = tmp_path / 'pw'
    pw.mkdir()
    (pw / 'a.json').write_text(new_run(run_name='x', label_a='Baseline', label_b='candidate', judges=['j']).model_dump_json())
    (pw / 'b.json').write_text(new_run(run_name='y', label_a='gpt', label_b='baseline', judges=['j']).model_dump_json())
    by_rid = {c.rid: c for c in candidates('pairwise', [pw])}
    assert by_rid[report_id(pw / 'a.json')].target is None
    assert by_rid[report_id(pw / 'b.json')].target == 'gpt vs baseline'


def test_pairwise_candidates_reread_nothing_while_the_store_is_unchanged(tmp_path: Path, monkeypatch) -> None:
    from evaluatorq.dashboard import library
    from evaluatorq.pairwise_run import new_run

    pw = tmp_path / 'pw'
    pw.mkdir()
    for i in range(3):
        (pw / f'{i}.json').write_text(new_run(run_name=f'r{i}', label_a='x', label_b='y', judges=['j']).model_dump_json())
    candidates('pairwise', [pw])
    reads: list[Path] = []
    real = library.read_json_cached
    monkeypatch.setattr(library, 'read_json_cached', lambda p: reads.append(p) or real(p))
    assert len(candidates('pairwise', [pw])) == 3
    assert reads == []


def test_pairwise_report_deleted_mid_scan_is_skipped(tmp_path: Path, monkeypatch) -> None:
    from evaluatorq.dashboard import library
    from evaluatorq.pairwise_run import new_run

    pw = tmp_path / 'pw'
    pw.mkdir()
    for i in range(2):
        (pw / f'{i}.json').write_text(new_run(run_name=f'r{i}', judges=['j']).model_dump_json())
    real_scan = library.scan

    def scan_then_delete(roots):
        cards = real_scan(roots)
        (pw / '0.json').unlink()  # removed after the scan listed it
        return cards

    monkeypatch.setattr(library, 'scan', scan_then_delete)
    library.read_json.cache_clear()
    assert [c.name for c in candidates('pairwise', [pw])] == ['r1']
