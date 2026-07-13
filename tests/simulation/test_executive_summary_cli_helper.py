# tests/simulation/test_executive_summary_cli_helper.py
from __future__ import annotations

from datetime import datetime, timezone

from evaluatorq.simulation.types import SimulationRun


def _run(results):
    return SimulationRun(
        run_name='test',
        created_at=datetime.now(tz=timezone.utc),
        mode='run',
        target_kind='openai_model',
        target='gpt-4o-mini',
        evaluator_names=['goal_achieved'],
        total_results=len(results),
        scorer_averages={},
        results=results,
    )


def test_helper_disabled_is_noop():
    from evaluatorq.simulation import cli

    run = _run([])
    cli._maybe_generate_executive_summary(run, enabled=False, model='m')
    assert run.executive_summary is None


def test_helper_skips_gracefully_without_credentials(monkeypatch):
    from evaluatorq.simulation import cli
    from evaluatorq.common.llm_client import MissingLLMCredentialsError
    from tests.simulation.test_executive_summary_facts import _result

    def _boom(*a, **k):
        raise MissingLLMCredentialsError('no creds')

    monkeypatch.setattr(cli, 'resolve_llm_client', _boom)
    run = _run([_result(goal=False, rules=['must_not_reveal_pii'])])
    # Must not raise; leaves summary unset.
    cli._maybe_generate_executive_summary(run, enabled=True, model='m')
    assert run.executive_summary is None
