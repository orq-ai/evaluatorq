"""The target agent's own token usage is recorded per conversation and per run, and survives a save."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.contracts import AgentResponse, AgentTarget, Message, TokenUsage
from evaluatorq.simulation.contracts import (
    CommunicationStyle,
    Persona,
    Scenario,
    SimulationDatapoint,
    SimulationResult,
    SimulationRun,
    TerminatedBy,
)
from evaluatorq.simulation.runner.simulation import SimulationRunner
from evaluatorq.simulation.utils.run_store import auto_save_run, build_simulation_run


class _Target(AgentTarget):
    """Replies with fixed usage, or with none (a plain callable that reports nothing)."""

    def __init__(self, usage: TokenUsage | None) -> None:
        super().__init__()
        self._usage = usage
        self.calls = 0

    async def respond(self, messages: list[Message]) -> AgentResponse:
        self.calls += 1
        return AgentResponse(text=f'reply {self.calls}', usage=self._usage)

    def new(self) -> _Target:
        return self


def _judgment(*, terminate: bool) -> MagicMock:
    j = MagicMock()
    j.should_terminate = terminate
    j.goal_achieved = terminate
    j.goal_completion_score = 1.0 if terminate else 0.5
    j.rules_broken = []
    j.reason = 'done' if terminate else 'keep going'
    j.response_quality = 0.9
    j.hallucination_risk = 0.1
    j.tone_appropriateness = 0.9
    j.factual_accuracy = 0.9
    return j


def _runner(target: _Target) -> SimulationRunner:
    judge = MagicMock()
    # Two turns: keep going, then stop, so the target is called twice.
    judge.evaluate = AsyncMock(side_effect=[_judgment(terminate=False), _judgment(terminate=True)])
    judge.get_usage = MagicMock(
        return_value=TokenUsage(input_tokens=500, output_tokens=50, total_tokens=550, total_cost=0.5, calls=2, priced_calls=2)
    )
    user_sim = MagicMock()
    user_sim.generate_first_message = AsyncMock(return_value='Hi')
    user_sim.respond_async = AsyncMock(return_value='ok')
    user_sim.get_usage = MagicMock(
        return_value=TokenUsage(input_tokens=200, output_tokens=20, total_tokens=220, total_cost=0.2, calls=1, priced_calls=1)
    )
    return SimulationRunner(target_agent=target, model='test', max_turns=3, user_simulator=user_sim, judge=judge)


def _datapoint() -> SimulationDatapoint:
    persona = Persona(
        name='Tester',
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background='A test user',
    )
    return SimulationDatapoint(
        id='dp', persona=persona, scenario=Scenario(name='S', goal='Get help'), user_system_prompt='sys', first_message='Hi'
    )


@pytest.fixture
def _orq_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')


@pytest.mark.asyncio
@pytest.mark.usefixtures('_orq_key')
async def test_target_usage_is_the_targets_share_of_the_total_and_survives_a_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    per_turn = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15, total_cost=0.001, calls=1, priced_calls=1)
    target = _Target(per_turn)

    result = await _runner(target).run(datapoint=_datapoint())

    assert target.calls == 2
    assert result.target_token_usage is not None
    assert result.target_token_usage.total_tokens == 30
    assert result.target_token_usage.calls == 2
    assert result.target_token_usage.total_cost == pytest.approx(0.002)
    # The combined total still holds the simulator and judge spend on top of the target's.
    assert result.token_usage.total_cost == pytest.approx(0.702)

    run = build_simulation_run(
        run_name='r', mode='run', target_kind='callback', evaluator_names=[], results=[result, result]
    )
    assert run.target_token_usage is not None
    assert run.target_token_usage.total_cost == pytest.approx(0.004)
    assert run.target_token_usage.calls == 4

    monkeypatch.setattr('evaluatorq.simulation.utils.run_store.get_sim_runs_dir', lambda: tmp_path)
    loaded = SimulationRun.model_validate_json(auto_save_run(run=run, run_name='r').read_text())
    assert loaded.target_token_usage == run.target_token_usage
    assert loaded.results[0].target_token_usage == result.target_token_usage


@pytest.mark.asyncio
@pytest.mark.usefixtures('_orq_key')
async def test_a_target_that_reports_no_usage_is_unknown_not_zero() -> None:
    result = await _runner(_Target(None)).run(datapoint=_datapoint())

    assert result.target_token_usage is None
    run = build_simulation_run(run_name='r', mode='run', target_kind='callback', evaluator_names=[], results=[result])
    assert run.target_token_usage is None


def _result(target: TokenUsage | None) -> SimulationResult:
    return SimulationResult(
        messages=[],
        terminated_by=TerminatedBy.judge,
        reason='done',
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        token_usage=TokenUsage(total_tokens=10, total_cost=0.1, calls=1, priced_calls=1),
        target_token_usage=target,
        turn_metrics=[],
    )


def test_run_total_skips_conversations_without_target_usage() -> None:
    known = TokenUsage(total_tokens=4, total_cost=0.01, calls=1, priced_calls=1)
    run = build_simulation_run(
        run_name='r', mode='run', target_kind='callback', evaluator_names=[], results=[_result(known), _result(None)]
    )
    assert run.target_token_usage == known


def test_a_run_saved_before_target_usage_existed_loads_with_none() -> None:
    run = build_simulation_run(
        run_name='r', mode='run', target_kind='callback', evaluator_names=[], results=[_result(None)]
    )
    data = json.loads(run.model_dump_json())
    data.pop('target_token_usage')
    for r in data['results']:
        r.pop('target_token_usage')

    loaded = SimulationRun.model_validate(data)

    assert loaded.target_token_usage is None
    assert loaded.results[0].target_token_usage is None


def test_coverage_counts_the_conversations_without_target_usage() -> None:
    from evaluatorq.common.structured_output import sum_usage_coverage

    known = TokenUsage(total_tokens=4, total_cost=0.01, calls=1, priced_calls=1)

    assert sum_usage_coverage([known, None, known]) == (known + known, 1)
    assert sum_usage_coverage([None, None]) == (None, 2)


def test_markdown_export_shows_target_cost_with_its_coverage() -> None:
    from evaluatorq.simulation.reports import export_markdown

    known = TokenUsage(total_tokens=4, total_cost=0.01, calls=1, priced_calls=1)

    md = export_markdown([_result(known), _result(None)], target='t')

    assert '| Target Agent Cost | $0.0100 (1 of 2 conversations) |' in md


def test_markdown_export_shows_unknown_target_cost_not_zero() -> None:
    from evaluatorq.simulation.reports import export_markdown

    md = export_markdown([_result(None)], target='t')

    assert '| Target Agent Cost | unknown |' in md


def test_html_export_from_bare_results_carries_target_cost_into_the_header() -> None:
    import html
    import re

    from evaluatorq.simulation.reports import export_html

    known = TokenUsage(total_tokens=4, total_cost=0.01, calls=1, priced_calls=1)

    page = export_html([_result(known)], target='t')

    line = re.search(r'<p class="report-hero-cost">(.*?)</p>', page, re.S)
    assert line is not None
    assert 'Target cost $0.0100' in ' '.join(html.unescape(re.sub(r'<[^>]+>', ' ', line.group(1))).split())
