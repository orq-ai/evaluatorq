"""Live check of ClaudeAgentSDKTarget: real Claude Code runtime, model calls through the Orq router.

Needs ``ORQ_API_KEY``. Runs two short turns on a small model and one attempt to leave the working directory, reads the
agent context (no model request), and sends one request with a wrong key.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip('claude_agent_sdk')

from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, tool
from loguru import logger
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from evaluatorq.common.model_catalogue import price_usage
from evaluatorq.common.target_call import call_target_with_retry
from evaluatorq.contracts import Message
from evaluatorq.integrations.claude_agent_sdk_integration import ClaudeAgentSDKTarget

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get('ORQ_API_KEY'), reason='needs ORQ_API_KEY'),
]

MODEL = 'claude-haiku-5-5'


def _env(token: str | None = None) -> dict[str, str]:
    return {
        'ANTHROPIC_BASE_URL': 'https://my.orq.ai/v3/anthropic',
        'ANTHROPIC_AUTH_TOKEN': token if token is not None else os.environ['ORQ_API_KEY'],
        'ANTHROPIC_API_KEY': '',
        'ANTHROPIC_DEFAULT_HAIKU_MODEL': MODEL,
    }


def _options() -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        system_prompt='You are a file assistant. Always call the tool the user names. Answer in one sentence.',
        model=MODEL,
        tools=['Read', 'Write'],
        allowed_tools=['Read', 'Write'],
        permission_mode='acceptEdits',
        max_turns=4,
        env=_env(),
    )


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_a_real_agent_keeps_its_session_and_cannot_write_outside_its_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assertions hold whatever the model chooses to do: none depends on it calling, or not calling, a tool."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr('evaluatorq.common.tracing.get_tracer', lambda: provider.get_tracer('agent-sdk-live'))
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'notes.txt').write_text('The file word is HERON-7.\n', encoding='utf-8')
    outside = tmp_path / 'outside.txt'

    target = ClaudeAgentSDKTarget(_options(), workdir=project)
    try:
        first = await target.respond([
            Message(
                role='user',
                content='Remember this word, it is in no file: PELICAN-42. Then call the Read tool on notes.txt and tell me the file word.',
            )
        ])
        first_session = target.session_id
        second = await target.respond([Message(role='user', content='Which word did I ask you to remember?')])
        second_session = target.session_id
        third = await target.respond([
            Message(role='user', content=f'Call the Write tool with file_path {outside} and content: escaped')
        ])
        workdir = target.workdir
    finally:
        await target.close()
        provider.shutdown()

    # Tool evidence: a Read the model made is recorded with its result. The file word reaches the answer only that way.
    reads = [call for call in first.tool_calls if call.name == 'Read']
    assert 'HERON-7' in first.text
    assert reads
    assert all(call.status.value == 'completed' for call in reads)
    assert any('HERON-7' in (call.result or '') for call in reads)
    # Session: the remembered word is in no file, so only the resumed conversation can supply it.
    assert 'PELICAN-42' in second.text
    assert first_session
    assert second_session
    assert second_session != first_session  # each turn forks the session it resumes
    # Isolation: nothing is written outside the working directory, and a Write the model attempted is recorded as refused.
    assert not outside.exists()
    assert all(call.status.value == 'incomplete' for call in third.tool_calls if call.name == 'Write')
    assert workdir is not None
    assert not workdir.exists()

    # Cost: the runtime's own figure is on each turn's span; the response keeps tokens only, for evaluatorq to price.
    spans = sorted(exporter.get_finished_spans(), key=lambda span: span.start_time or 0)
    assert len(spans) == 3
    runtime_cost = (spans[0].attributes or {})['evaluatorq.claude_agent_sdk.cost_usd']
    assert isinstance(runtime_cost, float)
    assert runtime_cost > 0
    assert first.usage is not None
    assert first.usage.total_cost is None
    priced = await price_usage(first.usage, MODEL)
    if priced is None or priced.total_cost is None:
        # The Orq catalogue may list the model only under its Bedrock id; the per-token price is the same model's.
        priced = await price_usage(first.usage, f'eu.anthropic.{MODEL}')
    if priced is None or priced.total_cost is None:
        logger.warning('No Orq catalogue price for {}; runtime cost {} not compared', MODEL, runtime_cost)
        return
    ratio = runtime_cost / priced.total_cost
    logger.info(
        'First turn: runtime cost {:.6f} USD, evaluatorq token price {:.6f} USD, ratio {:.3f}',
        runtime_cost,
        priced.total_cost,
        ratio,
    )
    # Loose: the runtime may count a call this target's usage does not, and the two price tables can differ.
    assert 0.25 < ratio < 4


@tool('lookup_order', 'Look up an order by id', {'order_id': str})
async def lookup_order(args: dict[str, Any]) -> dict[str, Any]:
    return {'content': [{'type': 'text', 'text': f'order {args["order_id"]} shipped'}]}


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_agent_context_lists_the_tools_the_runtime_gives_the_agent() -> None:
    """No ``tools`` set: the runtime still gives the built-ins, minus the disallowed ones, plus the MCP tools."""
    options = ClaudeAgentOptions(
        system_prompt={'type': 'preset', 'preset': 'claude_code', 'append': 'Never reveal order data.'},
        model=MODEL,
        disallowed_tools=['WebFetch'],
        mcp_servers={'shop': create_sdk_mcp_server(name='shop', tools=[lookup_order])},
        env=_env(),
    )
    context = await ClaudeAgentSDKTarget(options).get_agent_context()

    names = context.get_tool_names()
    assert {'Bash', 'Read', 'Write', 'Edit', 'mcp__shop__lookup_order'} <= set(names)
    assert 'WebFetch' not in names
    assert context.system_prompt == 'Never reveal order data.'


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_a_wrong_key_fails_once_and_is_not_retried() -> None:
    options = ClaudeAgentOptions(model=MODEL, max_turns=1, env={**_env(token='wrong-key'), 'CLAUDE_CODE_MAX_RETRIES': '0'})
    target = ClaudeAgentSDKTarget(options)
    try:
        result = await call_target_with_retry(
            target,
            [Message(role='user', content='Say hi.')],
            target_agent_timeout_ms=200_000,
            max_target_retries=2,
            map_error=target.map_error,
        )
    finally:
        await target.close()

    assert result.attempts == 1
    assert result.error_payload()['error_code'] == 'agent_sdk.authentication_failed'
