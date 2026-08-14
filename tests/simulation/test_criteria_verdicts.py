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
    CriterionVerdict,
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
        criteria_verdicts=None if verdicts is None else _occurred(verdicts),
    )


def _occurred(occurrences: dict[str, bool]) -> list[CriterionVerdict]:
    """An occurrence audit — True means the behaviour appeared, for BOTH criterion
    types. criteria_0 is must_happen, criteria_1 must_not_happen."""
    return [
        CriterionVerdict(criterion_id=cid, occurred=occurred, evidence='q' if occurred else '')
        for cid, occurred in occurrences.items()
    ]


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
    t.observe(None)
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
    """Once an audit exists it is the only input.

    Letting the judge's free-text list flip an audited pass to a fail puts the
    unreliable channel back in charge.
    """
    t = _CriteriaTracker(_scenario())
    t.observe(_occurred({'criteria_0': True, 'criteria_1': False}))  # both pass
    resolved = t.resolve(_judgment(None, terminate=True, broken=['criteria_0', 'criteria_1']))
    assert resolved.rules_broken == []


def test_free_text_rules_broken_does_not_rescue_an_unaudited_criterion():
    """...including for a criterion the audit skipped.

    The not-observed default is the honest reading, and rescuing from free text
    would hand that channel exactly the criteria the audit has no answer for —
    the ones where it is least checkable. Note criteria_1 is must_not_happen, so
    not-observed means it passes.
    """
    t = _CriteriaTracker(_scenario())
    t.observe(_occurred({'criteria_0': True}))  # criteria_1 never audited
    resolved = t.resolve(_judgment(None, terminate=True, broken=['criteria_1']))
    assert resolved.rules_broken == []


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
                {'criterion_id': 'criteria_0', 'occurred': False, 'evidence': ''},
                {'criterion_id': 'criteria_1', 'occurred': True, 'evidence': 'which plan?'},
            ],
        },
    )
    judgment = judge._parse_judgment(result)  # pyright: ignore[reportPrivateUsage]
    assert judgment.rules_broken == ['criteria_1']  # must_happen not yet met is not a violation
    assert [(v.criterion_id, v.occurred) for v in judgment.criteria_verdicts or []] == [
        ('criteria_0', False),
        ('criteria_1', True),
    ]


@pytest.mark.parametrize(
    'result',
    [
        LLMResult(content='I think it should stop.', tool_calls=[]),
        LLMResult(content='', tool_calls=[SimpleNamespace(function=SimpleNamespace(name='finish_conversation', arguments='{not json'))]),
        LLMResult(content='', tool_calls=[SimpleNamespace(function=SimpleNamespace(name='invent_a_tool', arguments='{}'))]),
    ],
    ids=['no-tool-call', 'unparseable-arguments', 'unknown-tool'],
)
def test_judge_safety_terminate_carries_no_audit(result: LLMResult):
    """The three safety-terminate paths return ``rules_broken=[]``, which passes every
    criterion. They must leave ``criteria_verdicts`` at ``None`` so the runner marks
    the run unverified — a malfunctioning judge scoring a perfect 1.0 is the
    RES-1308 bug wearing a different hat."""
    judgment = _judge()._parse_judgment(result)  # pyright: ignore[reportPrivateUsage]
    assert judgment.should_terminate is True
    assert judgment.rules_broken == []
    assert judgment.criteria_verdicts is None


def test_finish_conversation_derives_rules_broken_from_the_audit_alone():
    """A free-text rules_broken is no longer asked for, and one sent anyway is
    ignored — otherwise the two channels have something to disagree about."""
    judge = _judge()
    result = _llm_result(
        'finish_conversation',
        {
            'reason': 'r',
            'goal_achieved': False,
            'rules_broken': ['criteria_0'],  # not in the schema; must not survive
            'goal_completion_score': 0.0,
            'criteria_verdicts': [{'criterion_id': 'criteria_1', 'occurred': True, 'evidence': 'a plan'}],
        },
    )
    judgment = judge._parse_judgment(result)  # pyright: ignore[reportPrivateUsage]
    assert judgment.rules_broken == ['criteria_1']


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
                {'criterion_id': 'criteria_0', 'occurred': 'not-a-bool', 'evidence': ''},  # unparseable as bool
                {'criterion_id': 42, 'occurred': True, 'evidence': ''},  # id not a string
                'not-an-object',
                {'criterion_id': 'criteria_1', 'occurred': True, 'evidence': 'a plan'},
            ],
        },
    )
    with caplog.at_level('WARNING'):
        judgment = judge._parse_judgment(result)  # pyright: ignore[reportPrivateUsage]
    assert [(v.criterion_id, v.occurred) for v in judgment.criteria_verdicts or []] == [('criteria_1', True)]
    assert 'malformed' in caplog.text


def test_a_wholly_unusable_verdicts_payload_is_unknown_not_empty(caplog):
    """``None`` is *unknown* and marks the run unverified; ``[]`` would claim the
    judge audited and found nothing, which is the RES-1308 flattering silence."""
    judge = _judge()
    result = _llm_result(
        'continue_conversation',
        {'reason': 'r', 'goal_completion_score': 0.0, 'criteria_verdicts': 'all fine'},
    )
    with caplog.at_level('WARNING'):
        judgment = judge._parse_judgment(result)  # pyright: ignore[reportPrivateUsage]
    assert judgment.criteria_verdicts is None
    assert 'unusable as a whole' in caplog.text


def test_an_empty_audit_survives_as_audited_and_does_not_become_none():
    """Every criterion settled is the normal end state, and it must stay
    distinguishable from a judge that reported nothing at all."""
    judge = _judge()
    result = _llm_result(
        'continue_conversation',
        {'reason': 'r', 'goal_completion_score': 0.5, 'criteria_verdicts': []},
    )
    judgment = judge._parse_judgment(result)  # pyright: ignore[reportPrivateUsage]
    assert judgment.criteria_verdicts == []


def test_verdicts_outside_the_scenario_criteria_are_dropped_at_parse_time(caplog):
    """A well-formed id the scenario never defined passes shape validation and then
    matches nothing, so only the scenario context can catch it."""
    judge = _judge()
    result = _llm_result(
        'continue_conversation',
        {
            'reason': 'r',
            'goal_completion_score': 0.0,
            'criteria_verdicts': [
                {'criterion_id': 'criteria_5', 'occurred': True, 'evidence': 'nope'},
                {'criterion_id': 'criteria_1', 'occurred': True, 'evidence': 'a plan'},
            ],
        },
    )
    with caplog.at_level('WARNING'):
        judgment = judge._parse_judgment(result)  # pyright: ignore[reportPrivateUsage]
    assert [v.criterion_id for v in judgment.criteria_verdicts or []] == ['criteria_1']
    assert 'out-of-range' in caplog.text


def test_duplicate_verdict_ids_are_collapsed_with_a_warning(caplog):
    """Left alone the last entry silently wins, and nothing says two arrived."""
    judge = _judge()
    result = _llm_result(
        'continue_conversation',
        {
            'reason': 'r',
            'goal_completion_score': 0.0,
            'criteria_verdicts': [
                {'criterion_id': 'criteria_1', 'occurred': False, 'evidence': ''},
                {'criterion_id': 'criteria_1', 'occurred': True, 'evidence': 'a plan'},
            ],
        },
    )
    with caplog.at_level('WARNING'):
        judgment = judge._parse_judgment(result)  # pyright: ignore[reportPrivateUsage]
    assert [(v.criterion_id, v.occurred) for v in judgment.criteria_verdicts or []] == [('criteria_1', True)]
    assert 'duplicate' in caplog.text


def test_verdicts_come_back_sorted_by_criterion_index():
    """Nothing downstream should depend on the order the model happened to emit."""
    judge = _judge()
    result = _llm_result(
        'continue_conversation',
        {
            'reason': 'r',
            'goal_completion_score': 0.0,
            'criteria_verdicts': [
                {'criterion_id': 'criteria_1', 'occurred': False, 'evidence': ''},
                {'criterion_id': 'criteria_0', 'occurred': True, 'evidence': 'BANANA'},
            ],
        },
    )
    judgment = judge._parse_judgment(result)  # pyright: ignore[reportPrivateUsage]
    assert [v.criterion_id for v in judgment.criteria_verdicts or []] == ['criteria_0', 'criteria_1']


def test_unusable_goal_completion_score_falls_back_to_zero_and_keeps_the_audit(caplog):
    """Rejecting the payload over one bad number would safety-terminate the run and
    throw away the criteria audit with it — a far worse trade."""
    judge = _judge()
    result = _llm_result(
        'continue_conversation',
        {
            'reason': 'r',
            'goal_completion_score': 'high',
            'criteria_verdicts': [{'criterion_id': 'criteria_1', 'occurred': True, 'evidence': 'a plan'}],
        },
    )
    with caplog.at_level('WARNING'):
        judgment = judge._parse_judgment(result)  # pyright: ignore[reportPrivateUsage]
    assert judgment.goal_completion_score == 0.0
    assert 'goal_completion_score' in caplog.text
    assert judgment.rules_broken == ['criteria_1']  # the audit survived


def test_a_salvage_that_saves_nothing_is_unknown_not_an_empty_audit(caplog):
    """Every entry malformed leaves the retry with ``[]``, which would claim the judge
    audited and found nothing left to report — the flattering silence of RES-1308.
    Nothing salvageable is *unknown*, so the run is marked unverified."""
    judge = _judge()
    result = _llm_result(
        'continue_conversation',
        {
            'reason': 'r',
            'goal_completion_score': 0.0,
            'criteria_verdicts': [
                {'criterion_id': 'criteria_0', 'occurred': 'not-a-bool', 'evidence': ''},
                {'criterion_id': 42, 'occurred': True, 'evidence': ''},
            ],
        },
    )
    with caplog.at_level('WARNING'):
        judgment = judge._parse_judgment(result)  # pyright: ignore[reportPrivateUsage]
    assert judgment.criteria_verdicts is None
    assert 'unknown, not an empty audit' in caplog.text


def test_a_non_string_reason_is_coerced_instead_of_discarding_the_audit(caplog):
    """`reason` is free text nothing branches on; failing the payload over it would
    safety-terminate the run and throw away the criteria audit with it — the same
    trade `goal_completion_score` already makes."""
    judge = _judge()
    result = _llm_result(
        'continue_conversation',
        {
            'reason': 42,
            'goal_completion_score': 0.5,
            'criteria_verdicts': [{'criterion_id': 'criteria_1', 'occurred': True, 'evidence': 'a plan'}],
        },
    )
    with caplog.at_level('WARNING'):
        judgment = judge._parse_judgment(result)  # pyright: ignore[reportPrivateUsage]
    assert judgment.should_terminate is False  # not a safety termination
    assert judgment.reason == '42'
    assert judgment.rules_broken == ['criteria_1']  # the audit survived
    assert 'non-string reason' in caplog.text


@pytest.mark.parametrize(
    'payload',
    [
        {'reason': 'r', 'goal_achieved': True},
        {'reason': 'r', 'goal_achieved': True, 'goal_completion_score': None},
    ],
    ids=['omitted', 'explicit-null'],
)
def test_an_achieved_finish_without_a_score_reads_as_complete(payload: dict[str, object]):
    """The two spellings mean the same thing to a model, so they must score the same.
    An explicit null lands in `model_fields_set`, which used to skip the 1.0 fallback
    and score an achieved goal 0.0."""
    judge = _judge()
    judgment = judge._parse_judgment(_llm_result('finish_conversation', payload))  # pyright: ignore[reportPrivateUsage]
    assert judgment.goal_achieved is True
    assert judgment.goal_completion_score == 1.0


def test_the_criterion_verdict_item_schema_requires_evidence():
    """`WIRE_REQUIRED` governs the top-level object only, and pydantic's own
    ``required`` drops `evidence` because the parser defaults it to ''. The evidence
    capture the audit depends on must not be optional on the wire."""
    from evaluatorq.simulation.agents.judge import JUDGE_TOOLS

    for tool in JUDGE_TOOLS:
        items = tool['function']['parameters']['properties']['criteria_verdicts']['items']
        assert items['required'] == ['criterion_id', 'evidence', 'occurred'], tool['function']['name']


def test_a_boolean_score_is_not_read_as_a_number():
    """``True`` coerces to 1.0 through float(), which would read as a perfect score
    from a judge that answered the wrong type entirely."""
    judge = _judge()
    result = _llm_result(
        'continue_conversation',
        {'reason': 'r', 'goal_completion_score': 0.5, 'response_quality': True},
    )
    judgment = judge._parse_judgment(result)  # pyright: ignore[reportPrivateUsage]
    assert judgment.response_quality is None


def test_tracker_keeps_the_evidence_from_the_turn_occurrence_first_flipped():
    t = _CriteriaTracker(_scenario())
    t.observe([CriterionVerdict(criterion_id='criteria_0', occurred=True, evidence='BANANA')])
    t.observe([CriterionVerdict(criterion_id='criteria_0', occurred=True, evidence='banana again')])
    assert t.evidence == {'criteria_0': 'BANANA'}


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
    """Replays a scripted per-turn audit, terminating on the last entry.

    A ``None`` script entry is a turn that reported no audit at all — what a custom
    ``judge=`` predating ``criteria_verdicts`` does on every turn.
    """

    def __init__(
        self,
        script: list[dict[str, bool] | None],
        *,
        terminate: bool = True,
        broken: list[str] | None = None,
    ) -> None:
        self._script = script
        self._terminate = terminate
        self._broken = broken
        self._turns = 0

    async def evaluate(self, messages):  # noqa: ANN001
        verdicts = self._script[min(self._turns, len(self._script) - 1)]
        self._turns += 1
        terminate = self._terminate and self._turns >= len(self._script)
        return _judgment(verdicts, terminate=terminate, broken=self._broken if terminate else None)

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


async def _run(
    script: list[dict[str, bool] | None],
    *,
    max_turns: int = 4,
    terminate: bool = True,
    broken: list[str] | None = None,
):
    async def target(messages):  # noqa: ANN001
        return AgentResponse(text='Which plan are you interested in?')

    runner = SimulationRunner(
        target=target,
        max_turns=max_turns,
        user_simulator=_FakeSimulator(),  # pyright: ignore[reportArgumentType]
        judge=_FakeJudge(script, terminate=terminate, broken=broken),  # pyright: ignore[reportArgumentType]
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


@pytest.mark.asyncio
async def test_runner_end_to_end_no_audit_is_reported_as_unverified(caplog):
    """A judge that never audits (a custom ``judge=`` predating this field, or the
    built-in one terminating for safety) must not come back a clean 1.0.

    This is the whole RES-1308 failure mode one layer up: the fallback verdicts
    cannot fail a ``must_happen`` criterion, so an all-green result is unknown, not
    passing, and `criteria_verified` has to say so on the result itself — a log
    line nobody is tailing is not a signal a pipeline can act on.
    """
    with caplog.at_level('WARNING'):
        result = await _run([None, None])

    assert result.criteria_verified is False
    assert result.rules_broken == []  # the fallback cannot fail must_happen
    assert result.criteria_results == {'Agent writes BANANA': True, 'Agent mentions a plan': True}
    assert criteria_met_scorer(result) == 0.0
    assert 'unverified' in caplog.text


@pytest.mark.asyncio
async def test_runner_end_to_end_no_audit_still_honours_free_text_rules_broken():
    """The documented fallback: with no audit to trust, the judge's free-text list
    is the only evidence there is, so it must survive to the result."""
    result = await _run([None, None], broken=['criteria_1'])

    assert result.criteria_verified is False
    assert result.rules_broken == ['criteria_1']


@pytest.mark.asyncio
async def test_runner_end_to_end_partial_audit_counts_as_verified():
    """One audited criterion is enough to score off the audit; the criterion the
    judge never mentioned keeps the honest not-observed default."""
    result = await _run([{'criteria_1': False}] * 2)  # criteria_0 never audited

    assert result.criteria_verified is True
    assert result.rules_broken == ['criteria_0']  # must_happen, never observed
    assert criteria_met_scorer(result) == 0.5


@pytest.mark.asyncio
async def test_runner_end_to_end_max_turns_without_any_judgment_is_unverified():
    """The max-turns branch builds its result from a different helper, and with no
    judgment at all it has no criteria_results — which used to score 1.0."""
    result = await _run([None], max_turns=1, terminate=False)

    assert result.terminated_by is TerminatedBy.max_turns
    assert result.criteria_verified is False
    assert criteria_met_scorer(result) == 0.0


@pytest.mark.asyncio
async def test_runner_end_to_end_marks_which_criteria_were_audited():
    """`criteria_meta` distinguishes a verdict the judge gave from a default."""
    result = await _run([{'criteria_1': False}] * 2)  # criteria_0 never audited

    meta = {m['id']: m for m in result.metadata['criteria_meta']}
    assert meta['criteria_0']['passed'] is False and meta['criteria_0']['audited'] is False
    assert meta['criteria_1']['passed'] is True and meta['criteria_1']['audited'] is True


@pytest.mark.asyncio
async def test_runner_tells_the_judge_which_criteria_have_settled():
    """The saving only lands if the runner actually feeds the tracker's settled set
    back to the judge each turn."""

    class _RecordingJudge(_FakeJudge):
        def __init__(self, script):  # noqa: ANN001
            super().__init__(script)
            self.settled_per_turn: list[frozenset[str]] = []

        def mark_settled(self, ids) -> None:  # noqa: ANN001
            self.settled_per_turn.append(frozenset(ids))

    judge = _RecordingJudge([
        {'criteria_0': True, 'criteria_1': False},  # criteria_0 settles here
        {'criteria_0': False, 'criteria_1': False},
    ])

    async def target(messages):  # noqa: ANN001
        return AgentResponse(text='ok')

    runner = SimulationRunner(
        target=target,
        max_turns=4,
        user_simulator=_FakeSimulator(),  # pyright: ignore[reportArgumentType]
        judge=judge,  # pyright: ignore[reportArgumentType]
    )
    try:
        await runner.run(persona=_persona(), scenario=_scenario())
    finally:
        await runner.close()

    assert judge.settled_per_turn == [frozenset({'criteria_0'}), frozenset({'criteria_0'})]


def test_settled_criteria_are_confirmed_ids_only():
    t = _CriteriaTracker(_scenario())
    assert t.settled_ids == frozenset()
    t.observe(_occurred({'criteria_0': True, 'criteria_1': False}))
    assert t.settled_ids == frozenset({'criteria_0'})
    # Sticky: a later turn reporting False cannot unsettle it.
    t.observe(_occurred({'criteria_0': False, 'criteria_1': False}))
    assert t.settled_ids == frozenset({'criteria_0'})


def test_settled_criteria_are_excluded_from_the_audit_but_stay_in_the_prompt():
    """Occurrence is sticky, so re-auditing a confirmed criterion costs a payload
    entry per turn and can only restate a settled fact. It stays listed — the judge
    still needs it to decide whether to stop — but is marked do-not-report."""
    judge = _judge()
    assert 'ALREADY CONFIRMED' not in judge.system_prompt

    judge.mark_settled({'criteria_0'})
    prompt = judge.system_prompt
    assert 'criteria_0' in prompt and 'criteria_1' in prompt
    assert prompt.count('ALREADY CONFIRMED') == 1
    assert 'criteria_0' in prompt.split('ALREADY CONFIRMED')[0].rsplit('\n', 1)[-1]


def test_mark_settled_rebinds_so_a_shallow_copy_cannot_leak_between_runs():
    """The runner shallow-copies the judge per simulation; a shared mutable set
    would leak one run's occurrence into another running concurrently."""
    import copy

    original = _judge()
    original.mark_settled({'criteria_0'})
    clone = copy.copy(original)
    clone.mark_settled({'criteria_0', 'criteria_1'})

    assert original._settled == frozenset({'criteria_0'})  # pyright: ignore[reportPrivateUsage]
    assert clone._settled == frozenset({'criteria_0', 'criteria_1'})  # pyright: ignore[reportPrivateUsage]


def test_empty_audit_is_silent_once_every_criterion_is_settled(caplog):
    """An empty payload is expected when nothing is left to report, so the
    'no criteria_verdicts' warning must not cry wolf on it."""
    judge = _judge()
    judge.mark_settled({'criteria_0', 'criteria_1'})
    result = _llm_result('continue_conversation', {'reason': 'r', 'goal_completion_score': 0.5})
    with caplog.at_level('WARNING'):
        judge._parse_judgment(result)  # pyright: ignore[reportPrivateUsage]
    assert 'no criteria_verdicts' not in caplog.text


def test_tracker_reports_verified_only_once_an_audit_arrives():
    t = _CriteriaTracker(_scenario())
    assert t.verified is False
    t.observe(None)
    assert t.verified is False
    t.observe(_occurred({'criteria_0': True}))
    assert t.verified is True


def test_tracker_without_criteria_has_nothing_to_verify():
    assert _CriteriaTracker(_scenario([])).verified is True


def test_verdicts_for_ids_the_scenario_never_defined_are_dropped_with_a_warning(caplog):
    """A well-formed entry for a nonexistent id is not 'malformed', so the judge's
    parser lets it through — without this warning it vanishes in silence, and a
    payload that is off by one everywhere just looks unaudited."""
    t = _CriteriaTracker(_scenario())
    with caplog.at_level('WARNING'):
        t.observe(_occurred({'criteria_0': True, 'criteria_7': True}))
    assert 'criteria_7' in caplog.text
    assert t.resolve(_judgment(None, terminate=True)).rules_broken == []


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
