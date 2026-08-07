"""OrqResponsesTarget sends pipeline metadata as a native Responses field.

Router-only request extensions remain gated on the client route.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.contracts import LLMCallConfig, Message
from evaluatorq.common.thread_context import evaluatorq_pipeline
from evaluatorq.openresponses.target import OrqResponsesTarget


def _resp() -> dict[str, Any]:
    return {
        'id': 'resp_1',
        'object': 'response',
        'model': 'gpt-4o-mini',
        'output': [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'ok'}]}],
    }


def _client(base_url: str) -> MagicMock:
    client = MagicMock()
    client.base_url = base_url
    client.responses.create = AsyncMock(return_value=_resp())
    return client


async def _call(base_url: str, pipeline: str | None) -> dict[str, Any]:
    client = _client(base_url)
    target = OrqResponsesTarget(LLMCallConfig(model='gpt-4o-mini'), client=client)
    if pipeline:
        with evaluatorq_pipeline(pipeline):
            await target.respond([Message(role='user', content='hi')])
    else:
        await target.respond([Message(role='user', content='hi')])
    return client.responses.create.call_args.kwargs


@pytest.mark.asyncio
async def test_tags_metadata_when_routed_through_orq() -> None:
    kwargs = await _call('https://my.orq.ai/v3/router', 'agent_simulation')
    assert kwargs['metadata'] == {'evaluatorq_pipeline': 'agent_simulation'}


@pytest.mark.asyncio
async def test_no_metadata_for_direct_openai_client() -> None:
    kwargs = await _call('https://api.openai.com/v1', 'agent_simulation')
    assert kwargs['metadata'] == {'evaluatorq_pipeline': 'agent_simulation'}
    assert 'extra_body' not in kwargs
