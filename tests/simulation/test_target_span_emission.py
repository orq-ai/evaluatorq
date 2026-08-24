"""Target-call tracing: Responses span type, deployment span, AgentTarget routing.

Each test here guards a defect that shipped silently: a Responses span Orq
rendered as raw JSON, a deployment target that produced no LLM span at all, and
an ``AgentTarget`` passed as ``target=`` that errored every turn without ever
reaching the network.
"""

# ruff: noqa: S101

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluatorq.common.tracing import orq_span_type_for_operation
from evaluatorq.contracts import AgentResponse, AgentTarget, LLMCallConfig
from evaluatorq.deployment import DeploymentResponse
from evaluatorq.openresponses.target import OrqResponsesTarget
from evaluatorq.simulation.adapters import from_orq_deployment
from evaluatorq.simulation.runner.simulation import SimulationRunner
from evaluatorq.simulation.tracing import with_llm_span as simulation_llm_span
from evaluatorq.simulation.types import Message
from tests.simulation.conftest import find_span as _find
from tests.simulation.conftest import new_collector as _provider
from tests.simulation.conftest import span_attrs as _attrs

_deployment_module = importlib.import_module('evaluatorq.deployment')


def test_span_type_only_overridden_for_responses_operations() -> None:
    # 'chat' is already in Orq's ingest table; overriding it would be noise.
    assert orq_span_type_for_operation('chat') is None
    assert orq_span_type_for_operation('invoke') is None
    assert orq_span_type_for_operation('responses') == 'span.responses'
    assert orq_span_type_for_operation('agents.responses') == 'span.responses'


@pytest.mark.asyncio
async def test_responses_target_span_claims_span_responses_type() -> None:
    """The Responses span carries orq.span_type, or Orq renders it as raw JSON."""
    exporter, provider, tracer = _provider()

    part = MagicMock()
    part.type = 'output_text'
    part.text = 'hi'
    item = MagicMock()
    item.type = 'message'
    item.content = [part]
    response = MagicMock()
    response.id = 'resp_1'
    response.output = [item]
    response.usage = MagicMock(input_tokens=5, output_tokens=3)

    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock(return_value=response)
    target = OrqResponsesTarget(LLMCallConfig(model='agent/some-key'), client=client)

    with patch('evaluatorq.common.tracing.get_tracer', return_value=tracer):
        await target.respond([Message(role='user', content='hello')])

    provider.shutdown()
    span = _find(exporter, 'responses ')
    attrs = _attrs(span)
    assert attrs['orq.span_type'] == 'span.responses'
    assert attrs['gen_ai.operation.name'] == 'responses'
    # The transcript renderer reads the unprefixed key; gen_ai.input.messages
    # alone drops every non-message item.
    assert 'openresponses.input' in attrs


@pytest.mark.asyncio
async def test_simulation_responses_span_claims_span_responses_type() -> None:
    """The user-simulator / judge legs build their spans through a third builder.

    `simulation.tracing.with_llm_span` is a separate copy from common's and
    openresponses'; patching only the other two left every simulator and judge
    Responses span rendering as raw JSON.
    """
    exporter, provider, tracer = _provider()

    with patch('evaluatorq.common.tracing.get_tracer', return_value=tracer):
        async with simulation_llm_span(model='openai/gpt-4o-mini', operation='responses', purpose='judge'):
            pass

    provider.shutdown()
    assert _attrs(_find(exporter, 'responses '))['orq.span_type'] == 'span.responses'


@pytest.mark.asyncio
async def test_deployment_target_emits_llm_span() -> None:
    """A deployment target used to make the only untraced target call."""
    exporter, provider, tracer = _provider()
    mock_deployment = AsyncMock(return_value=DeploymentResponse(content='hello', raw=None, usage=None))

    with (
        patch.object(_deployment_module, 'deployment', mock_deployment),
        patch('evaluatorq.common.tracing.get_tracer', return_value=tracer),
    ):
        callback = from_orq_deployment('some-key')
        result = await callback([Message(role='user', content='Hello')])

    provider.shutdown()
    assert result.text == 'hello'
    span = _find(exporter, 'invoke deployment:some-key')
    attrs = _attrs(span)
    assert attrs['gen_ai.provider.name'] == 'orq'
    assert attrs['orq.llm.purpose'] == 'target'


class _RecordingTarget(AgentTarget):
    """Minimal AgentTarget that records whether it was ever called."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def respond(self, messages: list[Message]) -> AgentResponse:
        self.calls += 1
        return AgentResponse(text='ok')

    def new(self) -> _RecordingTarget:
        # Contract: new() returns a fresh instance — a shared one races on state
        # when datapoints run concurrently (contracts.py AgentTarget.new).
        return _RecordingTarget()


def test_agent_target_passed_as_target_routes_to_target_agent() -> None:
    """Passing an AgentTarget as target= must not wrap it in CallableTarget."""
    agent = _RecordingTarget()
    runner = SimulationRunner(target=agent)  # pyright: ignore[reportArgumentType]

    assert runner._effective_target is agent  # noqa: SLF001
    assert runner._target is None  # noqa: SLF001
