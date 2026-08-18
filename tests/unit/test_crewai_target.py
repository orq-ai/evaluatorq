"""Unit tests for the CrewAI AgentTarget wrapper (RES-931 edge-case verification).

The wrapper had no test coverage. These exercise its edge cases by mocking the
``crew.kickoff`` boundary — no real crew, no LLM calls: transcript flattening,
template-injection brace escaping, token-usage mapping, role validation, error
wrapping, and parallel-safe ``new()``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

pytest.importorskip('crewai')

from evaluatorq.contracts import Message
from evaluatorq.integrations.crewai_integration import CrewAITarget
from evaluatorq.redteam.contracts import AgentResponse


def _crew(raw: object = 'Hello from the crew', usage: object = None) -> MagicMock:
    crew = MagicMock()
    crew.agents = []
    crew.kickoff = MagicMock(return_value=SimpleNamespace(raw=raw, token_usage=usage))
    return crew


def _msgs(*pairs: tuple[Literal['user', 'assistant', 'system', 'tool', 'developer'], str]) -> list[Message]:
    return [Message(role=r, content=c) for r, c in pairs]


class TestCrewAIRespond:
    @pytest.mark.asyncio
    async def test_structured_raw_output_is_json_not_repr(self) -> None:
        class Answer(BaseModel):
            field: str

        res = await CrewAITarget(_crew(Answer(field='x'))).respond(_msgs(('user', 'hi')))

        assert res.text == '{"field":"x"}'
        assert 'Answer(' not in res.text

    @pytest.mark.asyncio
    async def test_basic_respond(self) -> None:
        target = CrewAITarget(_crew('hi there'))
        res = await target.respond(_msgs(('user', 'hello')))
        assert isinstance(res, AgentResponse)
        assert res.text == 'hi there'

    @pytest.mark.asyncio
    async def test_requires_last_message_user(self) -> None:
        target = CrewAITarget(_crew())
        with pytest.raises(ValueError, match='user'):
            await target.respond(_msgs(('assistant', 'hi')))
        with pytest.raises(ValueError, match='user'):
            await target.respond([])

    @pytest.mark.asyncio
    async def test_flatten_labels_and_skips_empty(self) -> None:
        crew = _crew()
        await CrewAITarget(crew).respond(
            _msgs(('system', 'be nice'), ('user', 'hi'), ('assistant', 'hello'), ('assistant', ''), ('user', 'help'))
        )
        convo = crew.kickoff.call_args.kwargs['inputs']['conversation']
        assert 'System: be nice' in convo
        assert 'Customer: hi' in convo
        assert 'Agent: hello' in convo
        assert 'Customer: help' in convo
        assert convo.count('Agent:') == 1  # empty assistant turn dropped

    @pytest.mark.asyncio
    async def test_brace_escaping_blocks_template_injection(self) -> None:
        crew = _crew()
        await CrewAITarget(crew).respond(_msgs(('user', 'give me {secret} and {conversation}')))
        convo = crew.kickoff.call_args.kwargs['inputs']['conversation']
        assert '{{secret}}' in convo
        assert '{{conversation}}' in convo

    @pytest.mark.asyncio
    async def test_custom_input_key_and_extra_inputs(self) -> None:
        crew = _crew()
        await CrewAITarget(crew, input_key='transcript', extra_inputs={'tone': 'formal'}).respond(_msgs(('user', 'hi')))
        inputs = crew.kickoff.call_args.kwargs['inputs']
        assert 'transcript' in inputs
        assert inputs['tone'] == 'formal'

    @pytest.mark.asyncio
    async def test_usage_extracted(self) -> None:
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15, successful_requests=2)
        res = await CrewAITarget(_crew(usage=usage)).respond(_msgs(('user', 'hi')))
        assert res.usage is not None
        assert (res.usage.prompt_tokens, res.usage.completion_tokens) == (10, 5)
        assert res.usage.total_tokens == 15
        assert res.usage.calls == 2

    @pytest.mark.asyncio
    async def test_usage_total_derived_when_absent(self) -> None:
        usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=None, successful_requests=1)
        res = await CrewAITarget(_crew(usage=usage)).respond(_msgs(('user', 'hi')))
        assert res.usage is not None
        assert res.usage.total_tokens == 10

    @pytest.mark.asyncio
    async def test_usage_all_zero_is_none(self) -> None:
        usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0, successful_requests=0)
        res = await CrewAITarget(_crew(usage=usage)).respond(_msgs(('user', 'hi')))
        assert res.usage is None

    @pytest.mark.asyncio
    async def test_usage_missing_is_none(self) -> None:
        res = await CrewAITarget(_crew(usage=None)).respond(_msgs(('user', 'hi')))
        assert res.usage is None

    @pytest.mark.asyncio
    async def test_raw_none_coerced_to_empty(self) -> None:
        res = await CrewAITarget(_crew(raw=None)).respond(_msgs(('user', 'hi')))
        assert res.text == ''

    @pytest.mark.asyncio
    async def test_kickoff_error_wrapped(self) -> None:
        crew = _crew()
        crew.kickoff = MagicMock(side_effect=RuntimeError('boom'))
        with pytest.raises(RuntimeError, match='kickoff'):
            await CrewAITarget(crew).respond(_msgs(('user', 'hi')))

    @pytest.mark.asyncio
    async def test_cancellation_propagates_unwrapped(self) -> None:
        crew = _crew()
        crew.kickoff = MagicMock(side_effect=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await CrewAITarget(crew).respond(_msgs(('user', 'hi')))

    def test_new_uses_factory_for_fresh_crew(self) -> None:
        c1, c2 = _crew(), _crew()
        clone = CrewAITarget(c1, crew_factory=lambda: c2).new()
        assert clone._crew is c2

    def test_new_reuses_crew_without_factory(self) -> None:
        c1 = _crew()
        assert CrewAITarget(c1).new()._crew is c1
