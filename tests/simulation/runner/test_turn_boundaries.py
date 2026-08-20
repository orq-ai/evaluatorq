"""The user's line opens the turn it belongs to, not the tail of the previous one."""

import pytest

from evaluatorq.contracts import AgentResponse, TokenUsage
from evaluatorq.simulation.runner.simulation import SimulationRunner
from evaluatorq.simulation.types import CommunicationStyle, Judgment, Persona, Scenario


class _NeverTerminatingJudge:
    async def evaluate(self, messages):  # noqa: ANN001, ARG002
        return Judgment(
            should_terminate=False,
            reason='keep going',
            goal_achieved=False,
            rules_broken=[],
            goal_completion_score=0.0,
        )

    def reset_usage(self) -> None: ...

    def get_usage(self) -> TokenUsage:
        return TokenUsage()


class _RecordingSimulator:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def update_context(self, *, persona_context, scenario_context) -> None: ...  # noqa: ANN001

    async def generate_first_message(self) -> str:
        self._log.append('first_message')
        return 'hi'

    async def respond_async(self, messages, *, llm_purpose=None) -> str:  # noqa: ANN001, ARG002
        self._log.append('user_simulator')
        return 'and?'

    def reset_usage(self) -> None: ...

    def get_usage(self) -> TokenUsage:
        return TokenUsage()


@pytest.mark.asyncio
async def test_user_simulator_runs_at_the_head_of_each_later_turn():
    log: list[str] = []

    async def target(messages):  # noqa: ANN001, ARG001
        log.append('target')
        return AgentResponse(text='ok')

    runner = SimulationRunner(
        target=target,
        max_turns=3,
        user_simulator=_RecordingSimulator(log),  # pyright: ignore[reportArgumentType]
        judge=_NeverTerminatingJudge(),  # pyright: ignore[reportArgumentType]
    )
    try:
        result = await runner.run(
            persona=Persona(
                name='Tester',
                patience=0.5,
                assertiveness=0.5,
                politeness=0.5,
                technical_level=0.5,
                communication_style=CommunicationStyle.casual,
                background='Checks a card charge.',
            ),
            scenario=Scenario(name='Charge check', goal='Understand a charge'),
        )
    finally:
        await runner.close()

    # Turn 1 opens with the generated first message; turns 2 and 3 open with the
    # simulator. No trailing simulator call after the final target response.
    assert log == ['first_message', 'target', 'user_simulator', 'target', 'user_simulator', 'target']
    assert [m.role for m in result.messages] == ['user', 'assistant', 'user', 'assistant', 'user', 'assistant']
