"""Tests for the static AgentTarget jobs' error-field forwarding.

Covers both static legs — ``_create_static_job_for_agent_target`` and the
hybrid pipeline's ``at_static_job`` — which wrap their target call in
``call_target_with_retry`` (see ``evaluatorq.common.target_call``) and flatten
the target's ``AgentResponse.error`` via ``TargetCallResult.error_payload``
into the string/type/stage/code fields the report layer validates.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.common.target_call import TargetCallResult
from evaluatorq.contracts import AgentResponse, AgentResponseError, Message, Usage
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
    assert payload.error_turn == 1
    # error_details is the only field the old object-shaped payload dropped outright;
    # without it the dashboard's error-details column is blank for every static failure.
    assert payload.error_details is not None
    assert payload.error_details['raw_message'] == 'boom'


async def test_static_job_success_has_none_error() -> None:
    from evaluatorq.redteam.reports.converters import _coerce_job_output_payload
    from evaluatorq.redteam.runner import _create_static_job_for_agent_target

    ok = AgentResponse(text='hello')
    job = _create_static_job_for_agent_target(lambda: _StubTarget(ok), 'lbl')

    wrapped = await job(_datapoint(), 0)
    out = wrapped['output']

    assert out['error'] is None
    assert out['response'] == 'hello'

    # The success payload has to survive the same validation the failure payload does —
    # a type slip on token_usage/finish_reason/model kills the report just as dead.
    payload = _coerce_job_output_payload(out)
    assert payload.response == 'hello'
    # A leaked error_stage on a clean attack makes the dashboard count resistant
    # attacks as failures.
    assert (payload.error_type, payload.error_stage, payload.error_code) == (None, None, None)
    assert (payload.error_turn, payload.error_details) == (None, None)


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


async def test_error_payload_keys_are_present_on_both_branches() -> None:
    """Success returns the same keys as failure, all ``None``.

    Call sites index the payload directly (``output['error'] is not None``), so
    dropping the keys on success would turn a clean attack into a ``KeyError``.
    """
    ok = TargetCallResult(response=AgentResponse(text='hi'), succeeded=True, attempts=1, error=None, error_details=None)
    failed = TargetCallResult(
        response=AgentResponse(text='[ERROR: boom]'),
        succeeded=False,
        attempts=3,
        error=AgentResponseError(message='boom', error_type='timeout', code=None),
        error_details={'attempts': 3},
    )

    assert ok.error_payload().keys() == failed.error_payload().keys()
    assert set(ok.error_payload().values()) == {None}

    fields = failed.error_payload(context=' on turn 2/5', turn=2)
    assert fields['error'] == 'Target agent failed after 3 attempt(s) on turn 2/5: boom'
    assert fields['error_type'] == 'timeout'
    assert fields['error_stage'] == 'target_call'
    # A target that reports no code still needs one — the dashboard groups on it.
    assert fields['error_code'] == 'target_error'
    assert fields['error_turn'] == 2


async def test_static_job_sums_billed_usage_across_retry_attempts() -> None:
    """The static leg's ``token_usage`` must be `call.billed_usage`, not `result.usage`.

    ``runner.py`` changed the static job's ``token_usage`` field from
    ``result.usage`` (the surviving response only) to ``call.billed_usage``
    (every billed attempt). A scripted target burns tokens on a refused first
    attempt and then succeeds on retry; asserting only the final response's
    usage would pass against either implementation, so the assertion sums both
    attempts and checks ``calls`` too — a revert to ``result.usage`` fails this.
    """
    from evaluatorq.contracts import AgentTarget
    from evaluatorq.redteam.runner import _create_static_job_for_agent_target

    burned = AgentResponse(
        text='[refused]',
        usage=Usage(total_tokens=7, calls=1),
        error=AgentResponseError(message='refused', error_type='target_error', code='x'),
    )
    ok = AgentResponse(text='hello there', usage=Usage(total_tokens=11, calls=1))

    class _RetryingTarget(AgentTarget):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def respond(self, messages: list[Message]) -> AgentResponse:
            self.calls += 1
            return burned if self.calls == 1 else ok

        def new(self) -> _RetryingTarget:
            return self

    target = _RetryingTarget()
    job = _create_static_job_for_agent_target(lambda: target, 'lbl')

    wrapped = await job(_datapoint(), 0)
    out = wrapped['output']

    assert target.calls == 2
    usage = out['token_usage']
    assert usage.total_tokens == 7 + 11
    assert usage.calls == 2
    assert out['error'] is None
    assert out['response'] == 'hello there'


async def test_hybrid_static_leg_reports_target_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hybrid static leg must surface target failures, not judge the error text.

    It used to call ``respond()`` directly and return no ``error`` key at all, so
    ``output_error_text`` saw ``None`` and the judge scored the literal
    ``[ERROR: ...]`` marker as a real answer — a timed-out target came back
    RESISTANT. That is worse than the crash on the non-hybrid leg: a plausible
    wrong number instead of no number.
    """
    from evaluatorq.contracts import AgentTarget
    from evaluatorq.redteam.adaptive.capability_classifier import AgentCapabilities
    from evaluatorq.redteam.contracts import Pipeline
    from evaluatorq.redteam.runner import _run_dynamic_or_hybrid
    from evaluatorq.types import DataPointResult, JobResult

    class _FailingTarget(AgentTarget):
        async def respond(self, messages: list[Message]) -> AgentResponse:
            raise RuntimeError('backend exploded')

        def new(self) -> _FailingTarget:
            return _FailingTarget()

    static_datapoint = DataPoint(
        inputs={'id': 'hybrid-static-1', 'category': 'ASI01', 'messages': [{'role': 'user', 'content': 'attack'}]}
    )
    captured: dict[str, Any] = {}

    async def fake_evaluatorq(_name: str, *, data: list[DataPoint], jobs: list[Any], **_kwargs: Any) -> list[Any]:
        static_row = next(dp for dp in data if dp.inputs['hybrid_source'] == 'static')
        job_result = await jobs[0](static_row, 0)
        captured['output'] = job_result['output']
        return [
            DataPointResult(
                data_point=static_row,
                job_results=[JobResult(job_name=job_result['name'], output=job_result['output'])],
            )
        ]

    monkeypatch.setattr('evaluatorq.evaluatorq', fake_evaluatorq)
    monkeypatch.setattr(
        'evaluatorq.redteam.runner.classify_agent_capabilities', AsyncMock(return_value=AgentCapabilities())
    )
    monkeypatch.setattr('evaluatorq.redteam.runner.generate_dynamic_datapoints', AsyncMock(return_value=([], {})))
    monkeypatch.setattr(
        'evaluatorq.redteam.runner.create_dynamic_redteam_job', MagicMock(return_value=AsyncMock(return_value={}))
    )
    monkeypatch.setattr(
        'evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge.load_owasp_agentic_dataset',
        lambda **_kwargs: [static_datapoint],
    )
    monkeypatch.setattr('evaluatorq.redteam.runner.create_dynamic_evaluator', MagicMock(return_value={}))
    monkeypatch.setattr(
        'evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge.create_owasp_evaluator', MagicMock(return_value={})
    )
    monkeypatch.setattr('evaluatorq.redteam.runner._send_cleaned_results', AsyncMock())

    report, _metrics = await _run_dynamic_or_hybrid(
        targets=[],
        agent_targets=[_FailingTarget()],
        mode=Pipeline.HYBRID,
        categories=['ASI01'],
        max_turns=1,
        max_per_category=1,
        attack_model='test-model',
        evaluator_model='test-model',
        parallelism=1,
        generate_strategies=False,
        generated_strategy_count=0,
        max_dynamic_datapoints=None,
        max_static_datapoints=None,
        cleanup_memory=False,
        llm_client=MagicMock(),
        description=None,
        dataset='ignored.json',
        run_id='hybrid-fail-run',
    )

    out = captured['output']
    assert isinstance(out['error'], str)
    assert 'backend exploded' in out['error']
    assert out['error_stage'] == 'target_call'
    # The failure has to reach the report, which is what the judge and the
    # dashboard read — not just the job dict.
    assert report.results[0].error is not None
