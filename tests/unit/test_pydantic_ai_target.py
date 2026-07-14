"""Unit tests for the Pydantic AI AgentTarget wrapper (RES-931 edge-case verification).

The wrapper had no test coverage. These exercise its edge cases by mocking the
``agent.run`` boundary with real ``pydantic_ai.messages`` part objects — no LLM
calls: text/tool-call/tool-return extraction and ordering, empty-output fallback,
role validation, internal history threading (incl. the stale-on-failure path),
version-tolerant usage extraction, and parallel-safe ``new()``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip('pydantic_ai')

from pydantic_ai.messages import TextPart, ToolCallPart, ToolReturnPart

from evaluatorq.contracts import Message, ToolCallOutputItem
from evaluatorq.integrations.pydantic_ai_integration import PydanticAITarget
from evaluatorq.redteam.contracts import AgentResponse


def _agent(result: Any) -> MagicMock:
    agent = MagicMock()
    agent.name = 'support'
    agent.model = SimpleNamespace(model_name='gpt-4o')
    agent.run = AsyncMock(return_value=result)
    return agent


def _result(new_parts: list[list[Any]], *, output: object = '', usage: object = None, all_msgs: object = None) -> MagicMock:
    result = MagicMock()
    result.new_messages = MagicMock(return_value=[SimpleNamespace(parts=parts) for parts in new_parts])
    result.all_messages = MagicMock(return_value=all_msgs if all_msgs is not None else ['H'])
    result.output = output
    result.usage = usage
    return result


def _user(text: str = 'hi') -> list[Message]:
    return [Message(role='user', content=text)]


class TestPydanticAIRespond:
    @pytest.mark.asyncio
    async def test_text_part_extracted(self) -> None:
        target = PydanticAITarget(_agent(_result([[TextPart(content='hello back')]])))
        res = await target.respond(_user())
        assert isinstance(res, AgentResponse)
        assert res.text == 'hello back'

    @pytest.mark.asyncio
    async def test_tool_call_and_return_merged(self) -> None:
        parts = [
            ToolCallPart(tool_name='lookup', args={'q': 'x'}, tool_call_id='c1'),
            ToolReturnPart(tool_name='lookup', content='the answer', tool_call_id='c1'),
        ]
        res = await PydanticAITarget(_agent(_result([parts]))).respond(_user())
        tools = [o for o in res.output if isinstance(o, ToolCallOutputItem)]
        assert len(tools) == 1
        assert tools[0].name == 'lookup'
        assert tools[0].result == 'the answer'

    @pytest.mark.asyncio
    async def test_empty_text_part_skipped_then_falls_back_to_output(self) -> None:
        res = await PydanticAITarget(_agent(_result([[TextPart(content='')]], output='fallback'))).respond(_user())
        assert res.text == 'fallback'

    @pytest.mark.asyncio
    async def test_empty_output_fallback_when_output_none(self) -> None:
        res = await PydanticAITarget(_agent(_result([], output=None))).respond(_user())
        assert res.text == ''

    @pytest.mark.asyncio
    async def test_requires_last_message_user(self) -> None:
        target = PydanticAITarget(_agent(_result([[TextPart(content='x')]])))
        with pytest.raises(ValueError, match='user'):
            await target.respond([Message(role='assistant', content='hi')])
        with pytest.raises(ValueError, match='user'):
            await target.respond([])

    @pytest.mark.asyncio
    async def test_history_threaded_across_turns(self) -> None:
        agent = _agent(_result([[TextPart(content='a')]], all_msgs=['m1', 'm2']))
        target = PydanticAITarget(agent)
        await target.respond(_user('first'))
        # first call has no prior history
        assert agent.run.call_args.kwargs['message_history'] is None
        assert target._history == ['m1', 'm2']
        await target.respond(_user('second'))
        # second call re-feeds the accumulated history
        assert agent.run.call_args.kwargs['message_history'] == ['m1', 'm2']

    @pytest.mark.asyncio
    async def test_stale_history_on_all_messages_failure_is_logged_not_raised(self, caplog: pytest.LogCaptureFixture) -> None:
        result = _result([[TextPart(content='ok')]])
        result.all_messages = MagicMock(side_effect=RuntimeError('boom'))
        target = PydanticAITarget(_agent(result))
        target._history = ['prior']
        res = await target.respond(_user())  # must not raise
        assert res.text == 'ok'
        assert target._history == ['prior']  # unchanged (stale, but not corrupted)
        assert any('stale' in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_usage_property_shape(self) -> None:
        usage = SimpleNamespace(input_tokens=12, output_tokens=8, total_tokens=20, requests=1)
        res = await PydanticAITarget(_agent(_result([[TextPart(content='x')]], usage=usage))).respond(_user())
        assert res.usage is not None
        assert (res.usage.prompt_tokens, res.usage.completion_tokens, res.usage.total_tokens) == (12, 8, 20)

    @pytest.mark.asyncio
    async def test_usage_callable_old_version(self) -> None:
        usage = SimpleNamespace(input_tokens=4, output_tokens=6, total_tokens=None, requests=2)
        res = await PydanticAITarget(_agent(_result([[TextPart(content='x')]], usage=lambda: usage))).respond(_user())
        assert res.usage is not None
        assert res.usage.total_tokens == 10  # derived
        assert res.usage.calls == 2

    @pytest.mark.asyncio
    async def test_usage_all_zero_is_none(self) -> None:
        usage = SimpleNamespace(input_tokens=0, output_tokens=0, total_tokens=0, requests=0)
        res = await PydanticAITarget(_agent(_result([[TextPart(content='x')]], usage=usage))).respond(_user())
        assert res.usage is None

    def test_new_resets_history_and_shares_agent(self) -> None:
        agent = _agent(_result([[TextPart(content='x')]]))
        target = PydanticAITarget(agent)
        target._history = ['prior']
        clone = target.new()
        assert clone._history == []
        assert clone._agent is agent
