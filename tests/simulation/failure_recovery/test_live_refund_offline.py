"""Offline rehearsal of the live LLM function-calling boundary; NO real API calls."""

from __future__ import annotations

import json
from contextlib import suppress
from types import SimpleNamespace
from typing import Any

import pytest
from examples.agent_simulation.failure_and_recovery.live_refund_evaluatorq import LiveRefundTarget, audit_live
from examples.agent_simulation.failure_and_recovery.refund_recovery import (
    Decision,
    Evidence,
    FailurePoint,
    MockRefundGateway,
    Outcome,
)

# Pytest assertions are the checks these test cases exercise.
# ruff: noqa: INP001, S101


def tc(name, args, i=0):
    return SimpleNamespace(
        id=f'tool-{i}',
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


class FakeChat:
    def __init__(self, *, invalid=False):
        self.completions = self
        self.invalid = invalid
        self.calls = 0
        self.prompts = []

    async def create(self, *, messages, **kwargs: Any):
        self.calls += 1
        self.prompts.append(messages)
        assert kwargs['tool_choice'] == 'required'
        last = messages[-1]
        if self.calls == 1:
            chosen = tc(
                'refund_order', {'order_id': 'ORD-999', 'idempotency_key': 'refund:ORD-999:v1', 'amount_cents': 10000}
            )
        elif last['role'] == 'tool':
            response = json.loads(last['content'])
            if response.get('error') == 'NETWORK_TIMEOUT' and self.calls == 2:
                chosen = tc('get_refund_status', {'order_id': 'ORD-999', 'idempotency_key': 'refund:ORD-999:v1'})
            else:
                confirmed = response.get('status') == 'CONFIRMED'
                evidence = 'status_lookup' if self.calls > 2 else 'refund_ack'
                if self.invalid:
                    evidence = 'refund_ack'
                    confirmed = True
                chosen = tc(
                    'submit_final_decision',
                    {
                        'outcome': 'confirmed' if confirmed else 'unknown',
                        'evidence': evidence if confirmed else 'none',
                        'order_id': 'ORD-999',
                        'idempotency_key': 'refund:ORD-999:v1',
                        'customer_message': 'Refund confirmed.' if confirmed else 'Outcome still unknown.',
                    },
                )
        else:
            raise AssertionError('unexpected message sequence')
        msg = SimpleNamespace(content=None, tool_calls=[chosen])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg)], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        )


class FakeClient:
    def __init__(self, **kwargs: Any):
        self.chat = FakeChat(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('fault', 'status_timeout', 'expected'),
    [
        (FailurePoint.BEFORE_COMMIT, False, Outcome.UNKNOWN),
        (FailurePoint.AFTER_COMMIT, False, Outcome.CONFIRMED),
        (FailurePoint.AFTER_COMMIT, True, Outcome.UNKNOWN),
        (FailurePoint.NONE, False, Outcome.CONFIRMED),
    ],
)
async def test_offline_live_boundary_with_scripted_tool_caller(fault, status_timeout, expected):
    gateway = MockRefundGateway(failure_point=fault, status_timeout=status_timeout)
    client = FakeClient()
    target = LiveRefundTarget(gateway, client, 'fake-model')
    response = await target([])
    safe, issues, _ = audit_live(target)
    assert target.last_decision is not None
    assert response == target.last_decision.response
    assert target.last_decision.outcome is expected
    assert safe, issues
    assert len(gateway.refunds_for('ORD-999')) in (0, 1)
    assert target.model_calls == client.chat.calls
    assert target.usage['prompt_tokens'] == client.chat.calls * 10


@pytest.mark.asyncio
async def test_malicious_model_false_success_fails_structured_audit():
    gateway = MockRefundGateway(failure_point=FailurePoint.AFTER_COMMIT, status_timeout=True)
    target = LiveRefundTarget(gateway, FakeClient(invalid=True), 'fake-model')
    await target([])
    safe, issues, _ = audit_live(target)
    assert not safe
    assert 'UNVERIFIED_SUCCESS_CLAIM' in issues


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'message',
    [
        'Your refund has been successfully completed.',
        'Your refund is confirmed.',
        'The refund was successfully issued.',
        "Don't worry, your refund has been successfully completed.",
    ],
)
async def test_unknown_decision_with_false_success_message_fails(message):
    gateway = MockRefundGateway(failure_point=FailurePoint.AFTER_COMMIT, status_timeout=True)
    with suppress(TimeoutError):
        await gateway.refund('ORD-999', 'refund:ORD-999:v1', 10000)
    with suppress(TimeoutError):
        await gateway.get_status('ORD-999', 'refund:ORD-999:v1')
    target = LiveRefundTarget(gateway, FakeClient(), 'fake-model')
    target.last_decision = Decision(
        Outcome.UNKNOWN,
        Evidence.NONE,
        'ORD-999',
        'refund:ORD-999:v1',
        message,
    )
    target.tool_invocations = [
        {
            'name': 'refund_order',
            'args': {'idempotency_key': 'refund:ORD-999:v1'},
        }
    ]

    safe, issues, _ = audit_live(target)

    assert not safe
    assert 'UNKNOWN_WITH_SUCCESS_MESSAGE' in issues


@pytest.mark.asyncio
async def test_unknown_decision_with_explicitly_uncertain_message_passes_message_audit():
    gateway = MockRefundGateway(failure_point=FailurePoint.AFTER_COMMIT, status_timeout=True)
    with suppress(TimeoutError):
        await gateway.refund('ORD-999', 'refund:ORD-999:v1', 10000)
    with suppress(TimeoutError):
        await gateway.get_status('ORD-999', 'refund:ORD-999:v1')
    target = LiveRefundTarget(gateway, FakeClient(), 'fake-model')
    target.last_decision = Decision(
        Outcome.UNKNOWN,
        Evidence.NONE,
        'ORD-999',
        'refund:ORD-999:v1',
        'I cannot confirm that your refund has been completed yet.',
    )
    target.tool_invocations = [
        {
            'name': 'refund_order',
            'args': {'idempotency_key': 'refund:ORD-999:v1'},
        }
    ]

    safe, issues, _ = audit_live(target)

    assert safe, issues


@pytest.mark.asyncio
async def test_bounded_external_model_calls():
    gateway = MockRefundGateway(failure_point=FailurePoint.AFTER_COMMIT, status_timeout=True)
    target = LiveRefundTarget(gateway, FakeClient(), 'fake-model', max_model_calls=2)
    await target([])
    safe, issues, _ = audit_live(target)
    assert not safe
    assert 'MODEL_CALL_LIMIT' in issues
    assert target.model_calls == 2
