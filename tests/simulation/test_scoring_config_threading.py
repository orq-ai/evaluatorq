"""`scoring=` must survive the whole trip from the public entry point to the score.

`tests/simulation/test_evaluators.py` already covers the two ends of that trip:
`SimulationScoringConfig` validates its own shape, and `get_evaluator(name, cfg)`
binds a config that changes the number. What nothing covered is the middle — that
`simulate(scoring=...)` / `generate_and_simulate(scoring=...)` actually put the
caller's config into the `SimulationConfig` that `_simulate_via_evaluatorq` reads,
so the stamped `evaluator_scores` change when the policy changes.

Every assertion here is on a *number*, not on the arrival of an object: a config
that reached the scorer but was ignored would produce the default score and fail
these tests. The `scoring=None` case is asserted too, because "the default was
applied" and "the field was dropped on the floor" are indistinguishable otherwise.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

# ruff: noqa: S101
import pytest

from evaluatorq.simulation.api import generate_and_simulate, simulate
from evaluatorq.simulation.evaluators.scorers import SimulationScoringConfig
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Judgment,
    Persona,
    Scenario,
    SimulationDatapoint,
    SimulationResult,
    TerminatedBy,
    TokenUsage as _TU,
)

# The conversation every test below runs: the judge lets two turns through and
# terminates on the third with the goal reached, so `turn_count == 3`.
TURNS_UNTIL_GOAL = 3

# Shipped cliffs are ((2, 1.0), (4, 0.9), (6, 0.7)): 3 turns sits past the <=2 step
# and inside the <=4 one.
DEFAULT_EFFICIENCY_AT_3_TURNS = 0.9

# One cliff at 1 turn, then a 0.25/turn decay with no floor in the way:
# 1.0 - (3 - 1) * 0.25 = 0.5. Deliberately far from 0.9 so a rounding change
# cannot make the two indistinguishable.
STEEP_SCORING = SimulationScoringConfig(
    turn_efficiency_cliffs=((1, 1.0),),
    turn_efficiency_decay_per_turn=0.25,
    turn_efficiency_floor=0.0,
)
STEEP_EFFICIENCY_AT_3_TURNS = 0.5

# Weights that make `conversation_quality` ignore everything but the goal verdict.
GOAL_ONLY_SCORING = SimulationScoringConfig(
    goal_achieved_weight=1.0,
    criteria_met_weight=0.0,
    turn_efficiency_weight=0.0,
)

# ... and weights that make it ignore everything but the turn-count proxy.
EFFICIENCY_ONLY_SCORING = SimulationScoringConfig(
    goal_achieved_weight=0.0,
    criteria_met_weight=0.0,
    turn_efficiency_weight=1.0,
)


class _StubUserSim:
    def update_context(self, *, persona_context=None, scenario_context=None) -> None:  # noqa: ANN001
        pass

    async def generate_first_message(self) -> str:
        return "hello"

    async def respond_async(self, messages, *, llm_purpose=None) -> str:  # noqa: ANN001
        return "and then?"

    def reset_usage(self) -> None:
        pass

    def get_usage(self) -> _TU:
        return _TU(prompt_tokens=0, completion_tokens=0, total_tokens=0)


class _GoalReachedAfterNTurns:
    """Judge that terminates with the goal achieved on its ``n``-th verdict.

    Each instance is used by exactly one datapoint (these tests run a single row),
    so the call counter is not shared across concurrent work.
    """

    def __init__(self, n: int = TURNS_UNTIL_GOAL) -> None:
        self._n = n
        self._calls = 0

    async def evaluate(self, messages) -> Judgment:  # noqa: ANN001
        self._calls += 1
        done = self._calls >= self._n
        return Judgment(
            should_terminate=done,
            reason="stub",
            goal_achieved=done,
            rules_broken=[],
            goal_completion_score=1.0 if done else 0.0,
        )

    def reset_usage(self) -> None:
        pass

    def get_usage(self) -> _TU:
        return _TU(prompt_tokens=0, completion_tokens=0, total_tokens=0)


def _datapoint() -> SimulationDatapoint:
    return SimulationDatapoint(
        id="dp1",
        persona=Persona(
            name="p",
            patience=0.5,
            assertiveness=0.5,
            politeness=0.5,
            technical_level=0.5,
            communication_style=CommunicationStyle.casual,
            background="b",
        ),
        scenario=Scenario(name="s", goal="g"),
        user_system_prompt="You are a user.",
        first_message="Hello",
    )


async def _target(messages: list[Any]) -> str:
    return "ok"


def _scores(result: SimulationResult) -> dict[str, float]:
    scores = result.metadata.get("evaluator_scores")
    assert isinstance(scores, dict), f"no evaluator_scores stamped: {result.metadata!r}"
    return scores


async def _run_simulate(scoring: SimulationScoringConfig | None) -> SimulationResult:
    results = await simulate(
        datapoints=[_datapoint()],
        target=_target,
        max_turns=10,  # well above TURNS_UNTIL_GOAL: the judge ends the run, not the cap
        evaluator_names=["turn_efficiency", "conversation_quality", "goal_achieved"],
        scoring=scoring,
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_GoalReachedAfterNTurns(),  # pyright: ignore[reportArgumentType]
        upload_results=False,
        executive_summary=False,
    )
    assert len(results) == 1
    result = results[0]
    # Guard the arithmetic the expected scores are derived from. If the stub
    # conversation ever stops being 3 successful turns, the score constants below
    # are meaningless and this says so instead of silently passing.
    assert result.terminated_by == TerminatedBy.judge
    assert result.goal_achieved is True
    assert result.turn_count == TURNS_UNTIL_GOAL
    return result


@pytest.mark.asyncio
async def test_simulate_without_scoring_uses_the_shipped_policy():
    """`scoring=None` must mean DEFAULT_SCORING_CONFIG, not "no policy at all"."""
    result = await _run_simulate(None)
    assert _scores(result)["turn_efficiency"] == DEFAULT_EFFICIENCY_AT_3_TURNS


@pytest.mark.asyncio
async def test_simulate_scoring_changes_the_turn_efficiency_score():
    """A steeper curve must move the number the report prints.

    0.5 instead of the shipped 0.9 for the same 3-turn conversation. A `scoring=`
    that were accepted and then dropped anywhere between `simulate()` and
    `get_evaluator()` would score 0.9 here.
    """
    result = await _run_simulate(STEEP_SCORING)
    assert _scores(result)["turn_efficiency"] == STEEP_EFFICIENCY_AT_3_TURNS


@pytest.mark.asyncio
async def test_simulate_scoring_changes_the_composite_weights():
    """The composite weights thread too, not only the cliffs.

    With goal_achieved weighted 1.0 the composite is exactly the goal verdict
    (1.0). Under the shipped weights the same run scores 0.97, so this cannot
    pass by accident on the defaults:

        goal_achieved   1.0 * 0.4 = 0.40
        criteria_met    1.0 * 0.3 = 0.30   (the scenario defines no criteria, so
                                            there is nothing to fail)
        turn_efficiency 0.9 * 0.3 = 0.27
                                    ------
                                      0.97
    """
    default_run = await _run_simulate(None)
    default_scores = _scores(default_run)
    assert default_scores["goal_achieved"] == 1.0
    assert default_scores["conversation_quality"] == 0.97

    goal_only_run = await _run_simulate(GOAL_ONLY_SCORING)
    assert _scores(goal_only_run)["conversation_quality"] == 1.0

    # And the mirror image: all the weight on the turn-count proxy collapses the
    # composite onto turn_efficiency's own 0.9. Two different re-weightings, two
    # different numbers — neither is reachable with the shipped weights.
    efficiency_only_run = await _run_simulate(EFFICIENCY_ONLY_SCORING)
    assert _scores(efficiency_only_run)["conversation_quality"] == DEFAULT_EFFICIENCY_AT_3_TURNS


@pytest.mark.asyncio
async def test_generate_and_simulate_threads_scoring_to_the_scorer(monkeypatch):
    """The second public entry point owns its own forwarding; assert its number too.

    Generation is stubbed out (it needs a provider); everything from
    `_simulate_core` down is the real path, so the assertion is still on a score
    the config changed and not on the config object arriving.
    """
    # A key only so the generation-client factory resolves; no call reaches it,
    # since both generation seams below are stubbed.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ORQ_API_KEY", raising=False)

    with (
        patch(
            "evaluatorq.simulation.api._generate_personas_scenarios",
            new=AsyncMock(return_value=([_datapoint().persona], [_datapoint().scenario], None)),
        ),
        patch(
            "evaluatorq.simulation.api._resolve_or_generate_datapoints",
            new=AsyncMock(return_value=[_datapoint()]),
        ),
    ):
        results = await generate_and_simulate(
            agent_description="a test agent",
            target=_target,
            max_turns=10,
            evaluator_names=["turn_efficiency"],
            scoring=STEEP_SCORING,
            user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
            judge=_GoalReachedAfterNTurns(),  # pyright: ignore[reportArgumentType]
            upload_results=False,
            executive_summary=False,
        )

    assert len(results) == 1
    assert results[0].turn_count == TURNS_UNTIL_GOAL
    assert _scores(results[0])["turn_efficiency"] == STEEP_EFFICIENCY_AT_3_TURNS
