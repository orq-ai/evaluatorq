"""Tests for the static AgentTarget job's error-field forwarding.

Covers ``_create_static_job_for_agent_target``, which now wraps its
target call in ``call_target_with_retry`` (see
``evaluatorq.common.target_call``) and surfaces the target's
``AgentResponse.error`` in its output dict.
"""

from __future__ import annotations

import pytest

from evaluatorq.contracts import AgentResponse, AgentResponseError, Message
from evaluatorq.types import DataPoint

pytestmark = pytest.mark.asyncio


class _StubTarget:
    def __init__(self, resp: AgentResponse) -> None:
        self._resp = resp

    async def respond(self, messages: list[Message]) -> AgentResponse:
        return self._resp

    async def close(self) -> None:
        pass


def _datapoint(text: str = 'hello there') -> DataPoint:
    return DataPoint(inputs={'messages': [{'role': 'user', 'content': text}]})


async def test_static_job_forwards_error_field() -> None:
    from evaluatorq.redteam.runner import _create_static_job_for_agent_target

    err_resp = AgentResponse(
        text='[ERROR: boom]',
        error=AgentResponseError(message='boom', error_type='target_error', code='x'),
    )
    job = _create_static_job_for_agent_target(lambda: _StubTarget(err_resp), 'lbl')

    wrapped = await job(_datapoint(), 0)
    out = wrapped['output']

    assert out['error'] is not None
    assert out['error'].code == 'x'


async def test_static_job_success_has_none_error() -> None:
    from evaluatorq.redteam.runner import _create_static_job_for_agent_target

    ok = AgentResponse(text='hello')
    job = _create_static_job_for_agent_target(lambda: _StubTarget(ok), 'lbl')

    wrapped = await job(_datapoint(), 0)
    out = wrapped['output']

    assert out['error'] is None
    assert out['response'] == 'hello'
