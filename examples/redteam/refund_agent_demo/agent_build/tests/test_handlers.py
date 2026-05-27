from agent_build.demo_data import SESSION_USER_ID, DemoState
from agent_build.handlers import (
    handle_get_policy,
    handle_issue_refund,
    handle_lookup_order,
)


def test_lookup_order_owned():
    state = DemoState()
    result = handle_lookup_order(state, order_id='ord_a1')
    assert result['ok']
    assert result['order']['owner_id'] == SESSION_USER_ID
    assert result['order']['id'] == 'ord_a1'


def test_lookup_order_foreign_returns_404():
    state = DemoState()
    result = handle_lookup_order(state, order_id='ord_b1')
    assert result['ok'] is False
    assert result['status_code'] == 404


def test_lookup_order_missing_returns_404():
    state = DemoState()
    result = handle_lookup_order(state, order_id='ord_zzz')
    assert result['ok'] is False
    assert result['status_code'] == 404


def test_issue_refund_in_window_owned_unrefunded():
    state = DemoState()
    result = handle_issue_refund(state, order_id='ord_a1', reason='changed_mind')
    assert result['ok']
    assert result['amount_refunded'] == 24.99
    assert state.is_refunded('ord_a1')


def test_issue_refund_foreign_order_rejected():
    state = DemoState()
    result = handle_issue_refund(state, order_id='ord_b1', reason='changed_mind')
    assert result['ok'] is False
    assert result['status_code'] == 404
    assert not state.is_refunded('ord_b1')


def test_issue_refund_already_refunded():
    state = DemoState()
    result = handle_issue_refund(state, order_id='ord_a4', reason='damaged_in_transit')
    assert result['ok'] is False
    assert result['error'] == 'already_refunded'


def test_issue_refund_past_window_no_flag():
    state = DemoState()
    result = handle_issue_refund(state, order_id='ord_a3', reason='damaged_in_transit')
    assert result['ok'] is False
    assert result['error'] == 'outside_window'
    assert not state.is_refunded('ord_a3')


def test_issue_refund_past_window_with_flag():
    state = DemoState()
    result = handle_issue_refund(
        state,
        order_id='ord_a3',
        reason='damaged_in_transit',
        post_window_exception=True,
    )
    assert result['ok']
    assert state.is_refunded('ord_a3')


def test_get_policy_local_fallback_known_topic():
    result = handle_get_policy(orq_client=None, topic='refund_basics')
    assert result['ok']
    assert '30 days' in result['text']


def test_get_policy_unknown_topic_returns_error():
    result = handle_get_policy(orq_client=None, topic='some_unknown_topic')
    assert result['ok'] is False
