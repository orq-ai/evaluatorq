"""generate_structured / FirstMessageGenerator tag LLM calls with pipeline metadata.

Covers the gap where persona/scenario/first-message generation LLM calls
bypassed ``apply_pipeline_metadata`` (see ``evaluatorq.common.llm_call``):
under a bound ``evaluatorq_pipeline``, the call kwargs must carry
``metadata={'evaluatorq_pipeline': ...}``; without a bound pipeline, no metadata
is sent.
"""

from __future__ import annotations

# ruff: noqa: S101, SLF001
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from evaluatorq.common.thread_context import evaluatorq_pipeline
from evaluatorq.contracts import LLMCallConfig
from evaluatorq.simulation.agents.base import BaseAgent
from evaluatorq.simulation.generators.first_message_generator import FirstMessageGenerator
from evaluatorq.simulation.types import CommunicationStyle, Message, Persona, Scenario
from evaluatorq.simulation.utils.structured_output import generate_structured

_ORQ_ROUTER_BASE_URL = 'https://my.orq.ai/v3/router'


class SampleResponse(BaseModel):
    value: str


def _orq_routed_client() -> MagicMock:
    client = MagicMock()
    client.base_url = _ORQ_ROUTER_BASE_URL

    parsed_message = MagicMock(refusal=None, parsed=SampleResponse(value='ok'))
    parsed_choice = MagicMock(message=parsed_message)
    parsed_response = MagicMock(choices=[parsed_choice])
    parsed_response.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)

    client.chat.completions.parse = AsyncMock(return_value=parsed_response)
    client.chat.completions.create = AsyncMock()
    return client


def _plain_client() -> MagicMock:
    client = _orq_routed_client()
    client.base_url = 'https://api.openai.com/v1'
    return client


def _persona() -> Persona:
    return Persona(
        name='Test User',
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background='bg',
    )


def _scenario(goal: str = 'fix my bug') -> Scenario:
    return Scenario(name='S', goal=goal)


def _first_message_client(message_content: str) -> MagicMock:
    msg = MagicMock(content=message_content)
    choice = MagicMock(message=msg)
    resp = MagicMock(choices=[choice])
    resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=resp)
    return client


@pytest.mark.asyncio
class TestGenerateStructuredPipelineMetadata:
    async def test_sends_pipeline_metadata_when_bound_and_orq_routed(self) -> None:
        client = _orq_routed_client()
        with evaluatorq_pipeline('agent_simulation'):
            await generate_structured(
                client,
                model='local-model',
                messages=[{'role': 'user', 'content': 'hi'}],
                response_format=SampleResponse,
                temperature=0.0,
                max_tokens=100,
                label='Sample.generate',
            )

        _, kwargs = client.chat.completions.parse.call_args
        assert kwargs.get('metadata') == {'evaluatorq_pipeline': 'agent_simulation'}

    async def test_no_metadata_without_bound_pipeline(self) -> None:
        client = _orq_routed_client()
        await generate_structured(
            client,
            model='local-model',
            messages=[{'role': 'user', 'content': 'hi'}],
            response_format=SampleResponse,
            temperature=0.0,
            max_tokens=100,
            label='Sample.generate',
        )

        _, kwargs = client.chat.completions.parse.call_args
        assert 'metadata' not in kwargs

    async def test_sends_metadata_when_not_orq_routed(self) -> None:
        client = _plain_client()
        with evaluatorq_pipeline('agent_simulation'):
            await generate_structured(
                client,
                model='local-model',
                messages=[{'role': 'user', 'content': 'hi'}],
                response_format=SampleResponse,
                temperature=0.0,
                max_tokens=100,
                label='Sample.generate',
            )

        _, kwargs = client.chat.completions.parse.call_args
        assert kwargs['metadata'] == {'evaluatorq_pipeline': 'agent_simulation'}


@pytest.mark.asyncio
class TestFirstMessageGeneratorPipelineMetadata:
    async def test_sends_pipeline_metadata_when_bound_and_orq_routed(self) -> None:
        client = _first_message_client('hi there')
        client.base_url = _ORQ_ROUTER_BASE_URL
        gen = FirstMessageGenerator(model='gpt-4o', client=client)

        with evaluatorq_pipeline('agent_simulation'):
            await gen.generate(_persona(), _scenario())

        _, kwargs = client.chat.completions.create.call_args
        assert kwargs.get('metadata') == {'evaluatorq_pipeline': 'agent_simulation'}

    async def test_no_metadata_without_bound_pipeline(self) -> None:
        client = _first_message_client('hi there')
        client.base_url = _ORQ_ROUTER_BASE_URL
        gen = FirstMessageGenerator(model='gpt-4o', client=client)

        await gen.generate(_persona(), _scenario())

        _, kwargs = client.chat.completions.create.call_args
        assert 'metadata' not in kwargs

    async def test_sends_metadata_when_not_orq_routed(self) -> None:
        client = _first_message_client('hi there')
        client.base_url = 'https://api.openai.com/v1'
        gen = FirstMessageGenerator(model='gpt-4o', client=client)

        with evaluatorq_pipeline('agent_simulation'):
            await gen.generate(_persona(), _scenario())

        _, kwargs = client.chat.completions.create.call_args
        assert kwargs['metadata'] == {'evaluatorq_pipeline': 'agent_simulation'}


class _ConcreteAgent(BaseAgent):
    """Minimal BaseAgent subclass forced onto the responses API path."""

    @property
    def name(self) -> str:
        return 'TestAgent'

    @property
    def system_prompt(self) -> str:
        return 'You are a test agent.'


def _responses_client() -> MagicMock:
    client = MagicMock()
    client.base_url = _ORQ_ROUTER_BASE_URL

    mock_response = MagicMock()
    mock_response.output = []
    mock_response.usage = None
    client.responses = MagicMock()
    client.responses.create = AsyncMock(return_value=mock_response)
    return client


@pytest.mark.asyncio
class TestCallResponsesPipelineMetadata:
    async def test_sends_pipeline_metadata_when_bound(self) -> None:
        client = _responses_client()
        config = LLMCallConfig(model='gpt-4o', api='responses', client=client)
        agent = _ConcreteAgent(config)

        with evaluatorq_pipeline('agent_simulation'):
            await agent._call_llm([Message(role='user', content='hi')])

        _, kwargs = client.responses.create.call_args
        assert kwargs['extra_body']['metadata'] == {'evaluatorq_pipeline': 'agent_simulation'}

    async def test_no_metadata_without_bound_pipeline(self) -> None:
        client = _responses_client()
        config = LLMCallConfig(model='gpt-4o', api='responses', client=client)
        agent = _ConcreteAgent(config)

        await agent._call_llm([Message(role='user', content='hi')])

        _, kwargs = client.responses.create.call_args
        assert 'metadata' not in kwargs.get('extra_body', {})
