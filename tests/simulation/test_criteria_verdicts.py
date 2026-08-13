"""RES-1308: scenario criteria must be able to fail.

Before the per-criterion audit existed, pass/fail was inferred from the absence of
an id in ``judgment.rules_broken``, so a ``must_happen`` criterion that simply never
occurred came back PASS and ``criteria_met`` returned 1.0 on every run.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from evaluatorq.contracts import AgentResponse, TokenUsage
from evaluatorq.simulation.agents.base import LLMResult
from evaluatorq.simulation.agents.judge import JudgeAgent, JudgeAgentConfig
from evaluatorq.simulation.evaluators.scorers import criteria_met_scorer
from evaluatorq.simulation.runner.simulation import SimulationRunner, _CriteriaTracker
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Criterion,
    Judgment,
    Persona,
    Scenario,
    SimulationResult,
    TerminatedBy,
)

CRITERIA = [
    Criterion(description='Agent writes BANANA', type='must_happen'),
    Criterion(description='Agent mentions a plan', type='must_not_happen'),
]


def _scenario(criteria: list[Criterion] | None = None) -> Scenario:
    return Scenario(name='S', goal='g', criteria=CRITERIA if criteria is None else criteria)


def _persona() -> Persona:
    return Persona(
        name='P',
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background='b',
    )


def _judgment(verdicts: dict[str, bool] | None, *, terminate: bool = False, broken: list[str] | None = None) -> Judgment:
    return Judgment(
        should_terminate=terminate,
        reason='r',
        goal_achieved=False,
        rules_broken=broken or [],
        goal_completion_score=0.0,
        criteria_verdicts=verdicts,
    )


def _occurred(occurrences: dict[str, bool], *, terminate: bool = False) -> Judgment:
    """Judgment carrying an occurrence audit — True means the behaviour appeared,
    for BOTH criterion types. criteria_0 is must_happen, criteria_1 must_not_happen."""
    return _judgment(occurrences, terminate=terminate)


# ---------------------------------------------------------------------------
# Tracker fold
# ---------------------------------------------------------------------------


def test_must_happen_that_never_occurs_fails():
    t = _CriteriaTracker(_scenario())
    t.observe(_occurred({'criteria_0': False, 'criteria_1': False}))
    assert t.resolve(_judgment(None, terminate=True)).rules_broken == ['criteria_0']


def test_must_happen_is_sticky_once_it_occurs():
    t = _CriteriaTracker(_scenario())
    t.observe(_occurred({'criteria_0': True, 'criteria_1': False}))
    t.observe(_occurred({'criteria_0': False, 'criteria_1': False}))
    assert t.resolve(_judgment(None, terminate=True)).rules_broken == []


def test_must_not_happen_violation_survives_later_clean_turns():
    t = _CriteriaTracker(_scenario())
    t.observe(_occurred({'criteria_0': True, 'criteria_1': True}))
    t.observe(_occurred({'criteria_0': True, 'criteria_1': False}))
    assert t.resolve(_judgment(None, terminate=True)).rules_broken == ['criteria_1']


def test_no_audit_at_all_falls_back_and_warns(caplog):
    t = _CriteriaTracker(_scenario())
    t.observe(_judgment(None))
    with caplog.at_level('WARNING'):
        resolved = t.resolve(_judgment(None, terminate=True))
    assert resolved.rules_broken == []
    assert 'unverified' in caplog.text


def test_unaudited_criterion_keeps_honest_default_and_warns(caplog):
    t = _CriteriaTracker(_scenario())
    t.observe(_occurred({'criteria_1': False}))  # criteria_0 never mentioned
    with caplog.at_level('WARNING'):
        resolved = t.resolve(_judgment(None, terminate=True))
    assert resolved.rules_broken == ['criteria_0']
    assert 'criteria_0' in caplog.text


def test_free_text_rules_broken_cannot_override_an_audited_criterion():
    """The audit wins where it has an answer.

    Letting the judge's free-text list flip an audited pass to a fail puts the
    unreliable channel back in charge — and it can only ever add failures.
    """
    t = _CriteriaTracker(_scenario())
    t.observe(_occurred({'criteria_0': True, 'criteria_1': False}))  # both pass
    resolved = t.resolve(_judgment(None, terminate=True, broken=['criteria_0', 'criteria_1']))
    assert resolved.rules_broken == []


def test_free_text_rules_broken_still_rescues_an_unaudited_criterion():
    """...but a criterion the judge never audited has no verdict to defend, so an
    explicit free-text mention is the only evidence available and is honoured."""
    t = _CriteriaTracker(_scenario())
    t.observe(_occurred({'criteria_0': True}))  # criteria_1 never audited
    resolved = t.resolve(_judgment(None, terminate=True, broken=['criteria_1']))
    assert resolved.rules_broken == ['criteria_1']


def test_unknown_ids_in_free_text_are_ignored():
    t = _CriteriaTracker(_scenario())
    t.observe(_occurred({'criteria_0': True, 'criteria_1': False}))
    resolved = t.resolve(_judgment(None, terminate=True, broken=['criteria_9', 'Agent writes BANANA']))
    assert resolved.rules_broken == []


def test_scenario_without_criteria_is_untouched():
    t = _CriteriaTracker(_scenario(criteria=[]))
    j = _judgment(None, terminate=True, broken=['whatever'])
    assert t.resolve(j) is j


# ---------------------------------------------------------------------------
# Judge parsing
# ---------------------------------------------------------------------------


def _judge() -> JudgeAgent:
    return JudgeAgent(JudgeAgentConfig(goal='g', criteria=CRITERIA, client=object()))


def _llm_result(name: str, args: dict[str, object]) -> LLMResult:
    call = SimpleNamespace(function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    return LLMResult(content='', tool_calls=[call])


def test_continue_conversation_reports_mid_run_violation():
    """The continue tool has no rules_broken field; it used to hardcode []."""
    judge = _judge()
    result = _llm_result(
        'continue_conversation',
        {
            'reason': 'r',
            'goal_completion_score': 0.2,
            'criteria_verdicts': [
                {'id': 'criteria_0', 'occurred': False, 'evidence': ''},
                {'id': 'criteria_1', 'occurred': True, 'evidence': 'which plan?'},
            ],
        },
    )
    judgment = judge._parse_judgment(result)  # pyright: ignore[reportPrivateUsage]
    assert judgment.rules_broken == ['criteria_1']  # must_happen not yet met is not a violation
    assert judgment.criteria_verdicts == {'criteria_0': False, 'criteria_1': True}


def test_finish_conversation_unions_reported_and_audited():
    judge = _judge()
    result = _llm_result(
        'finish_conversation',
        {
            'reason': 'r',
            'goal_achieved': False,
            'rules_broken': ['criteria_0'],
            'goal_completion_score': 0.0,
            'criteria_verdicts': [{'id': 'criteria_1', 'occurred': True, 'evidence': 'a plan'}],
        },
    )
    judgment = judge._parse_judgment(result)  # pyright: ignore[reportPrivateUsage]
    assert judgment.rules_broken == ['criteria_0', 'criteria_1']


def test_missing_verdicts_parse_to_none_not_empty(caplog):
    judge = _judge()
    result = _llm_result('finish_conversation', {'reason': 'r', 'goal_achieved': True, 'rules_broken': [], 'goal_completion_score': 1.0})
    with caplog.at_level('WARNING'):
        judgment = judge._parse_judgment(result)  # pyright: ignore[reportPrivateUsage]
    assert judgment.criteria_verdicts is None
    assert 'criteria_verdicts' in caplog.text


def test_malformed_verdict_entries_are_dropped_with_a_warning(caplog):
    """A wrong-typed entry is indistinguishable from one the judge never sent, and
    the run-level fold only complains about ids missing from EVERY turn."""
    judge = _judge()
    result = _llm_result(
        'continue_conversation',
        {
            'reason': 'r',
            'goal_completion_score': 0.0,
            'criteria_verdicts': [
                {'id': 'criteria_0', 'occurred': 'true', 'evidence': ''},  # string, not bool
                {'id': 42, 'occurred': True, 'evidence': ''},  # id not a string
                'not-an-object',
                {'id': 'criteria_1', 'occurred': True, 'evidence': 'a plan'},
            ],
        },
    )
    with caplog.at_level('WARNING'):
        judgment = judge._parse_judgment(result)  # pyright: ignore[reportPrivateUsage]
    assert judgment.criteria_verdicts == {'criteria_1': True}
    assert 'malformed' in caplog.text


def test_both_judge_tools_require_criteria_verdicts():
    from evaluatorq.simulation.agents.judge import JUDGE_TOOLS

    for tool in JUDGE_TOOLS:
        params = tool['function']['parameters']
        assert 'criteria_verdicts' in params['properties'], tool['function']['name']
        assert 'criteria_verdicts' in params['required'], tool['function']['name']


# ---------------------------------------------------------------------------
# End to end through the runner
# ---------------------------------------------------------------------------


class _FakeJudge:
    """Replays a scripted per-turn audit, terminating on the last entry."""

    def __init__(self, script: list[dict[str, bool]], *, terminate: bool = True) -> None:
        self._script = script
        self._terminate = terminate
        self._turns = 0

    async def evaluate(self, messages):  # noqa: ANN001
        verdicts = self._script[min(self._turns, len(self._script) - 1)]
        self._turns += 1
        return _occurred(verdicts, terminate=self._terminate and self._turns >= len(self._script))

    def reset_usage(self) -> None: ...

    def get_usage(self):
        from evaluatorq.contracts import TokenUsage

        return TokenUsage()


class _FakeSimulator:
    def update_context(self, *, persona_context, scenario_context) -> None: ...  # noqa: ANN001

    async def generate_first_message(self) -> str:
        return 'hi'

    async def respond_async(self, messages, *, llm_purpose=None) -> str:  # noqa: ANN001
        return 'and?'

    def reset_usage(self) -> None: ...

    def get_usage(self):
        from evaluatorq.contracts import TokenUsage

        return TokenUsage()


async def _run(script: list[dict[str, bool]], *, max_turns: int = 4, terminate: bool = True):
    async def target(messages):  # noqa: ANN001
        return AgentResponse(text='Which plan are you interested in?')

    runner = SimulationRunner(
        target=target,
        max_turns=max_turns,
        user_simulator=_FakeSimulator(),  # pyright: ignore[reportArgumentType]
        judge=_FakeJudge(script, terminate=terminate),  # pyright: ignore[reportArgumentType]
    )
    try:
        return await runner.run(persona=_persona(), scenario=_scenario())
    finally:
        await runner.close()


@pytest.mark.asyncio
async def test_runner_end_to_end_criteria_can_fail():
    result = await _run([{'criteria_0': False, 'criteria_1': True}] * 2)

    assert result.terminated_by is TerminatedBy.judge
    assert sorted(result.rules_broken) == ['criteria_0', 'criteria_1']
    assert result.criteria_results == {'Agent writes BANANA': False, 'Agent mentions a plan': False}
    assert criteria_met_scorer(result) == 0.0
    assert all(not c['passed'] for c in result.metadata['criteria_meta'])


@pytest.mark.asyncio
async def test_runner_end_to_end_folds_occurrence_across_turns():
    """A violation on turn 1 and a satisfaction on turn 2 both survive to the end,
    even though the final turn's audit alone reports neither."""
    result = await _run([
        {'criteria_0': False, 'criteria_1': True},  # plan mentioned — violation
        {'criteria_0': True, 'criteria_1': False},  # BANANA written; plan not repeated
        {'criteria_0': False, 'criteria_1': False},  # final turn sees neither
    ])

    assert result.rules_broken == ['criteria_1']
    assert result.criteria_results == {'Agent writes BANANA': True, 'Agent mentions a plan': False}
    assert criteria_met_scorer(result) == 0.5


@pytest.mark.asyncio
async def test_runner_end_to_end_max_turns_path_resolves_criteria():
    """The max-turns branch builds its result through a different helper than the
    judge-terminated branch; it must fold the audit too."""
    result = await _run([{'criteria_0': True, 'criteria_1': False}], max_turns=2, terminate=False)

    assert result.terminated_by is TerminatedBy.max_turns
    assert result.rules_broken == []
    assert criteria_met_scorer(result) == 1.0


# ---------------------------------------------------------------------------
# criteria_met on runs that never reached the audit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('terminated_by', [TerminatedBy.error, TerminatedBy.timeout])
def test_criteria_met_is_zero_when_the_run_never_reached_the_audit(terminated_by: TerminatedBy, caplog):
    """A dead target used to score a perfect 1.0 — the same flattering silence
    RES-1308 was filed about, just relocated to crashed runs."""
    result = SimulationResult(
        messages=[],
        terminated_by=terminated_by,
        reason='boom',
        goal_achieved=False,
        goal_completion_score=0.0,
        rules_broken=[],
        turn_count=0,
        turn_metrics=[],
        token_usage=TokenUsage(),
    )
    with caplog.at_level('WARNING'):
        assert criteria_met_scorer(result) == 0.0
    assert terminated_by.value in caplog.text
