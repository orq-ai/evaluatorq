"""Tests for the static AgentTarget job's error-field forwarding.

Covers ``_create_static_job_for_agent_target``, which now wraps its
target call in ``call_target_with_retry`` (see
``evaluatorq.common.target_call``) and flattens the target's
``AgentResponse.error`` into the string/type/stage/code fields the report
layer validates.
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


class _RaisingTarget:
    async def respond(self, messages: list[Message]) -> AgentResponse:
        raise RuntimeError('backend exploded')


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

    assert isinstance(out['error'], str)
    assert out['error'].endswith('attempt(s): boom')
    assert out['error_code'] == 'x'
    assert out['error_type'] == 'target_error'
    assert out['error_stage'] == 'target_call'


async def test_static_job_error_payload_survives_report_conversion() -> None:
    """A failed static attack must not take the whole report down with it.

    ``JobOutputPayload.error`` is ``str | None``; handing back the
    ``AgentResponseError`` object made report generation raise ``ValidationError``
    after every attack had already run and been billed.
    """
    from evaluatorq.redteam.reports.converters import _coerce_job_output_payload
    from evaluatorq.redteam.runner import _create_static_job_for_agent_target

    err_resp = AgentResponse(
        text='[ERROR: boom]',
        error=AgentResponseError(message='boom', error_type='target_error', code='x'),
    )
    job = _create_static_job_for_agent_target(lambda: _StubTarget(err_resp), 'lbl')

    payload = _coerce_job_output_payload((await job(_datapoint(), 0))['output'])

    assert payload.error is not None
    assert 'boom' in payload.error
    assert payload.error_code == 'x'


async def test_static_job_success_has_none_error() -> None:
    from evaluatorq.redteam.runner import _create_static_job_for_agent_target

    ok = AgentResponse(text='hello')
    job = _create_static_job_for_agent_target(lambda: _StubTarget(ok), 'lbl')

    wrapped = await job(_datapoint(), 0)
    out = wrapped['output']

    assert out['error'] is None
    assert out['response'] == 'hello'


async def test_shared_static_target_call_uses_the_supplied_error_mapper() -> None:
    from evaluatorq.redteam.runner import _run_static_target_call

    out = await _run_static_target_call(
        _RaisingTarget(),
        'hello there',
        max_target_retries=0,
        target_agent_timeout_ms=1000,
        map_error=lambda _: ('custom.error', 'mapped failure'),
    )

    assert out['error'] is not None
    assert 'mapped failure' in out['error']
    assert out['error_code'] == 'custom.error'
