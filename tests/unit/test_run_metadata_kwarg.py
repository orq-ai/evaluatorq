from __future__ import annotations

import json

import httpx
import pytest
from openai import AsyncOpenAI

from evaluatorq.common.llm_call import apply_pipeline_metadata
from evaluatorq.common.thread_context import evaluatorq_pipeline, evaluatorq_run_id


def test_no_op_when_nothing_bound() -> None:
    params: dict = {'model': 'gpt-4o-mini'}
    apply_pipeline_metadata(params)

    assert params == {'model': 'gpt-4o-mini'}


def test_applies_bound_pipeline_and_run_id() -> None:
    params: dict = {'model': 'gpt-4o-mini'}
    with evaluatorq_pipeline('red_teaming'), evaluatorq_run_id('r1'):
        apply_pipeline_metadata(params)

    assert params['metadata'] == {'evaluatorq_pipeline': 'red_teaming', 'evaluatorq_run_id': 'r1'}


def test_applies_regardless_of_endpoint() -> None:
    """`metadata` is a standard OpenAI field, so it is sent off-Orq too.

    This is the behavior the guard used to suppress: a red team pointed at a
    direct OpenAI target got no run correlation even though its trace still
    reached Orq via OTel.
    """
    params: dict = {'model': 'gpt-4o-mini'}
    with evaluatorq_run_id('r1'):
        apply_pipeline_metadata(params)

    assert params['metadata'] == {'evaluatorq_run_id': 'r1'}


def test_caller_supplied_metadata_wins_on_conflict() -> None:
    """A caller's own metadata key survives; the bound keys fill in around it."""
    params = {'metadata': {'evaluatorq_run_id': 'caller-id', 'custom': 'x'}}
    with evaluatorq_pipeline('red_teaming'), evaluatorq_run_id('bound-id'):
        apply_pipeline_metadata(params)

    assert params['metadata'] == {
        'evaluatorq_run_id': 'caller-id',
        'evaluatorq_pipeline': 'red_teaming',
        'custom': 'x',
    }


def test_never_sets_metadata_to_none() -> None:
    """An explicit ``metadata=None`` serializes as ``"metadata": null``.

    The SDK strips an omitted param via its ``omit`` sentinel, but a passed
    ``None`` survives into the request body — so the key must be absent, not None.
    """
    params: dict = {'model': 'gpt-4o-mini'}
    apply_pipeline_metadata(params)

    assert 'metadata' not in params


@pytest.mark.asyncio
async def test_openai_sdk_serializes_metadata_for_chat_and_responses() -> None:
    """The installed OpenAI SDK sends metadata on both native endpoints."""
    request_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.content))
        if request.url.path.endswith('/chat/completions'):
            return httpx.Response(
                200,
                json={
                    'id': 'chatcmpl-test',
                    'object': 'chat.completion',
                    'created': 0,
                    'model': 'gpt-4o-mini',
                    'choices': [
                        {
                            'index': 0,
                            'message': {'role': 'assistant', 'content': 'ok'},
                            'finish_reason': 'stop',
                        }
                    ],
                    'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
                },
            )
        return httpx.Response(
            200,
            json={
                'id': 'resp_test',
                'object': 'response',
                'created_at': 0,
                'status': 'completed',
                'model': 'gpt-4o-mini',
                'output': [],
                'parallel_tool_calls': False,
                'tool_choice': 'auto',
                'temperature': 1,
                'top_p': 1,
                'truncation': 'disabled',
                'usage': {'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2},
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url='https://api.openai.test/v1') as http_client:
        client = AsyncOpenAI(api_key='test-key', http_client=http_client)
        metadata = {'evaluatorq_pipeline': 'red_teaming', 'evaluatorq_run_id': 'run-1'}
        await client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role': 'user', 'content': 'hello'}],
            metadata=metadata,
        )
        await client.responses.create(model='gpt-4o-mini', input='hello', metadata=metadata)

    assert [body['metadata'] for body in request_bodies] == [metadata, metadata]
