# tests/simulation/test_executive_summary_api_helper.py
"""Shared async exec-summary helper used by the SDK (simulate/generate_and_simulate)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

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


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletions:
    async def create(self, *a, **k):  # noqa: ANN002, ANN003
        class _Resp:
            choices = [_FakeChoice('NARRATIVE from the LLM.')]

        return _Resp()


class _FakeChat:
    completions = _FakeCompletions()


class _FakeClient:
    chat = _FakeChat()


class _Resolved:
    client = _FakeClient()


@pytest.mark.asyncio
async def test_helper_populates_summary_when_enabled():
    from evaluatorq.simulation.reports.executive_summary import populate_run_executive_summary
    from tests.simulation.test_executive_summary_facts import _result

    run = _run([_result(goal=False, rules=['must_not_reveal_pii'])])
    await populate_run_executive_summary(run, enabled=True, model='m', resolve_client=lambda: _Resolved())
    assert run.executive_summary == 'NARRATIVE from the LLM.'


@pytest.mark.asyncio
async def test_helper_disabled_is_noop():
    from evaluatorq.simulation.reports.executive_summary import populate_run_executive_summary
    from tests.simulation.test_executive_summary_facts import _result

    run = _run([_result(goal=True, rules=[])])
    await populate_run_executive_summary(run, enabled=False, model='m', resolve_client=lambda: _Resolved())
    assert run.executive_summary is None


@pytest.mark.asyncio
async def test_helper_skips_without_credentials():
    from evaluatorq.common.llm_client import MissingLLMCredentialsError
    from evaluatorq.simulation.reports.executive_summary import populate_run_executive_summary
    from tests.simulation.test_executive_summary_facts import _result

    def _boom():
        raise MissingLLMCredentialsError('no creds')

    run = _run([_result(goal=False, rules=['must_not_reveal_pii'])])
    # Must not raise; leaves summary unset.
    await populate_run_executive_summary(run, enabled=True, model='m', resolve_client=_boom)
    assert run.executive_summary is None
