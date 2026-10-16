"""API-free regression tests for uncertain refund and idempotent recovery."""

# Pytest assertions are the checks these test cases exercise.
# ruff: noqa: INP001, S101

import pytest
from examples.agent_simulation.failure_and_recovery.refund_recovery import (
    Decision,
    Evidence,
    FailurePoint,
    MockRefundGateway,
    Outcome,
    RefundAgent,
    audit_decision,
)


@pytest.mark.asyncio
async def test_t1_precommit_timeout_never_claims_success():
    gateway = MockRefundGateway(failure_point=FailurePoint.BEFORE_COMMIT)
    agent = RefundAgent(gateway)
    await agent([])
    assert agent.last_decision is not None
    audit = audit_decision(gateway, agent.last_decision)
    assert len(gateway.refunds_for(agent.order_id)) == 0
    assert agent.last_decision.outcome is Outcome.UNKNOWN
    assert any(e.outcome == 'TIMEOUT_PRE_COMMIT' for e in gateway.events)
    assert any(e.operation == 'status' for e in gateway.events)
    assert audit.safe
    assert not audit.refund_confirmed_to_agent


@pytest.mark.asyncio
async def test_t2_postcommit_timeout_status_confirms_once():
    gateway = MockRefundGateway(failure_point=FailurePoint.AFTER_COMMIT)
    agent = RefundAgent(gateway)
    await agent([])
    assert agent.last_decision is not None
    audit = audit_decision(gateway, agent.last_decision)
    assert len(gateway.refunds_for(agent.order_id)) == 1
    assert agent.last_decision.evidence is Evidence.STATUS_LOOKUP
    assert any(e.outcome == 'TIMEOUT_POST_COMMIT' for e in gateway.events)
    assert audit.safe
    assert audit.refund_confirmed_to_agent


@pytest.mark.asyncio
async def test_t3_same_idempotency_key_deduplicates():
    gateway = MockRefundGateway()
    agent = RefundAgent(gateway)
    await agent([])
    await agent([])  # Repeat the same request with the same stable key.
    assert agent.last_decision is not None
    assert len(gateway.refunds_for(agent.order_id)) == 1
    assert any(e.outcome == 'ALREADY_CONFIRMED' for e in gateway.events)
    assert audit_decision(gateway, agent.last_decision).safe


@pytest.mark.asyncio
async def test_t4_status_timeout_does_not_reveal_committed_truth():
    gateway = MockRefundGateway(failure_point=FailurePoint.AFTER_COMMIT, status_timeout=True)
    agent = RefundAgent(gateway)
    await agent([])
    assert agent.last_decision is not None
    audit = audit_decision(gateway, agent.last_decision)
    assert len(gateway.refunds_for(agent.order_id)) == 1  # Backend truth
    assert agent.last_decision.outcome is Outcome.UNKNOWN  # Agent observation
    assert any(e.operation == 'status' and e.outcome == 'TIMEOUT' for e in gateway.events)
    assert audit.safe
    assert not audit.refund_confirmed_to_agent


@pytest.mark.asyncio
async def test_negative_control_unsafe_claim_is_rejected():
    gateway = MockRefundGateway(failure_point=FailurePoint.AFTER_COMMIT, status_timeout=True)
    agent = RefundAgent(gateway)
    await agent([])
    assert agent.last_decision is not None
    unsafe = Decision(Outcome.CONFIRMED, Evidence.REFUND_ACK, agent.order_id, agent.key, 'The refund has completed!')
    audit = audit_decision(gateway, unsafe)
    assert not audit.safe
    assert 'UNVERIFIED_SUCCESS_CLAIM' in audit.violations


@pytest.mark.asyncio
async def test_negative_control_new_key_can_create_duplicate_and_is_detected():
    gateway = MockRefundGateway()
    first = RefundAgent(gateway, key='key-A')
    second = RefundAgent(gateway, key='key-B')
    await first([])
    await second([])
    assert second.last_decision is not None
    assert len(gateway.refunds_for(first.order_id)) == 2
    audit = audit_decision(gateway, second.last_decision)
    assert not audit.safe
    assert 'DUPLICATE_REFUND' in audit.violations


@pytest.mark.asyncio
async def test_idempotency_key_conflicting_payload_rejected():
    gateway = MockRefundGateway()
    await gateway.refund('ORD-1', 'key-1')
    with pytest.raises(ValueError, match='conflicting payload'):
        await gateway.refund('ORD-2', 'key-1')
    assert len(gateway.records_by_key) == 1


@pytest.mark.asyncio
async def test_recovery_retry_same_key_after_ack_loss_is_safe():
    gateway = MockRefundGateway(failure_point=FailurePoint.AFTER_COMMIT, status_timeout=True)
    agent = RefundAgent(gateway)
    await agent([])
    assert agent.last_decision is not None
    assert agent.last_decision.outcome is Outcome.UNKNOWN
    gateway.status_timeout = False
    await agent([])
    assert agent.last_decision is not None
    assert len(gateway.refunds_for(agent.order_id)) == 1
    assert agent.last_decision.outcome is Outcome.CONFIRMED
    assert audit_decision(gateway, agent.last_decision).safe
