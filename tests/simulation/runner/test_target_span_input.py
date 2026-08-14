"""The target-call span records message text, never a Python repr.

`runner/simulation.py` used `m.content or ''`, which is correct for the `str`
content the runner produces today but renders `list[ContentPart]` content as a
Python repr on the span — the same latent defect the judge/user-simulator span
input was fixed for. Both now route through the one `span_message_text` helper.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.contracts import AgentResponse, InputTextContent, TokenUsage
from evaluatorq.simulation.runner import simulation as runner_mod
from evaluatorq.simulation.runner.simulation import SimulationRunner
from evaluatorq.simulation.types import CommunicationStyle, Persona, Scenario, SimulationDatapoint

pytestmark = pytest.mark.asyncio


def _datapoint() -> SimulationDatapoint:
    return SimulationDatapoint(
        id='dp-span-001',
        persona=Persona(
            name='P',
            patience=0.5,
            assertiveness=0.5,
            politeness=0.5,
            technical_level=0.5,
            communication_style=CommunicationStyle.casual,
            background='b',
        ),
        scenario=Scenario(name='S', goal='g'),
        user_system_prompt='system',
        first_message='Hello, can you help me?',
    )


def _judge() -> MagicMock:
    judgment = MagicMock()
    judgment.should_terminate = True
    judgment.goal_achieved = True
    judgment.goal_completion_score = 1.0
    judgment.rules_broken = []
    judgment.criteria_verdicts = None
    judgment.reason = 'Done'
    for field in ('response_quality', 'hallucination_risk', 'tone_appropriateness', 'factual_accuracy'):
        setattr(judgment, field, 0.5)
    judge = MagicMock()
    judge.evaluate = AsyncMock(return_value=judgment)
    judge.get_usage = MagicMock(return_value=TokenUsage())
    return judge


def _user_simulator() -> MagicMock:
    sim = MagicMock()
    sim.generate_first_message = AsyncMock(return_value='Hello')
    sim.respond_async = AsyncMock(return_value='thanks')
    sim.get_usage = MagicMock(return_value=TokenUsage())
    return sim


async def _run_one_turn() -> None:
    async def target(messages):  # noqa: ANN001, ANN202
        return AgentResponse(text='ok')

    runner = SimulationRunner(
        target=target,
        max_turns=1,
        user_simulator=_user_simulator(),
        judge=_judge(),
    )
    try:
        await runner.run(datapoint=_datapoint())
    finally:
        await runner.close()


async def test_the_target_call_span_flattens_content_through_the_shared_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With `m.content or ''` the helper is never called and multi-part content
    reaches the span as a repr."""
    seen: list[object] = []
    real = runner_mod.span_message_text

    def spy(content):  # noqa: ANN001, ANN202
        seen.append(content)
        return real(content)

    monkeypatch.setattr(runner_mod, 'span_message_text', spy)
    recorded: list[list[dict[str, object]]] = []
    monkeypatch.setattr(runner_mod, 'record_llm_input', lambda _span, payload: recorded.append(payload))

    await _run_one_turn()

    assert seen == ['Hello, can you help me?']
    assert recorded == [[{'role': 'user', 'content': 'Hello, can you help me?'}]]


async def test_the_helper_flattens_multi_part_content_instead_of_repring_it() -> None:
    """The latent defect itself: a list of parts must land as its text."""
    text = runner_mod.span_message_text([
        InputTextContent(type='input_text', text='part one '),
        InputTextContent(type='input_text', text='part two'),
    ])

    assert text == 'part one part two'
    assert 'InputTextContent' not in text
