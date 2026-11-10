from __future__ import annotations

from typing import Any

import pytest

from evaluatorq.common.reports.executive_summary import generate_executive_summary
from evaluatorq.common.thread_context import evaluatorq_pipeline, evaluatorq_run_id


class _Msg:
    content = 'summary text'


class _Choice:
    message = _Msg()


class _Resp:
    choices = [_Choice()]


class _Completions:
    def __init__(self, sink: dict[str, Any]) -> None:
        self._sink = sink

    async def create(self, **kwargs: Any) -> Any:
        self._sink['kwargs'] = kwargs
        return _Resp()


class _Chat:
    def __init__(self, sink: dict[str, Any]) -> None:
        self.completions = _Completions(sink)


class _Client:
    def __init__(self, sink: dict[str, Any], base_url: str = 'https://my.orq.ai/v3/router') -> None:
        self.chat = _Chat(sink)
        self.base_url = base_url


@pytest.mark.asyncio
async def test_executive_summary_call_carries_run_id() -> None:
    sink: dict[str, Any] = {}
    with evaluatorq_pipeline('red_teaming'), evaluatorq_run_id('rid-123'):
        await generate_executive_summary(
            facts='some facts',
            llm_client=_Client(sink),
            model='gpt-4o',
        )
    assert sink['kwargs']['metadata']['evaluatorq_run_id'] == 'rid-123'
    assert sink['kwargs']['metadata']['evaluatorq_pipeline'] == 'red_teaming'


@pytest.mark.asyncio
async def test_executive_summary_call_omits_metadata_for_direct_openai() -> None:
    """Direct OpenAI rejects `metadata` unless `store` is on, so it is withheld."""
    sink: dict[str, Any] = {}
    with evaluatorq_pipeline('red_teaming'), evaluatorq_run_id('rid-123'):
        await generate_executive_summary(
            facts='some facts',
            llm_client=_Client(sink, base_url='https://api.openai.com/v1'),
            model='gpt-4o',
        )
    assert 'metadata' not in sink['kwargs']
