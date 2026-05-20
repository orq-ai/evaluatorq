from agent_build.demo_data import NOW, SESSION_USER_ID, DemoState


def test_demostate_has_session_user_orders():
    state = DemoState()
    owned = [o for o in state.orders if o.owner_id == SESSION_USER_ID]
    assert len(owned) >= 4


def test_demostate_has_foreign_orders():
    state = DemoState()
    foreign = [o for o in state.orders if o.owner_id != SESSION_USER_ID]
    assert len(foreign) >= 4


def test_demostate_has_one_already_refunded_order():
    state = DemoState()
    refunded_ids = {r.order_id for r in state.refunds}
    assert len(refunded_ids) == 1
    refunded_order_id = next(iter(refunded_ids))
    assert state.is_refunded(refunded_order_id)


def test_find_order_returns_none_for_missing():
    state = DemoState()
    assert state.find_order('ord_does_not_exist') is None


def test_orders_span_in_window_and_past_window():
    state = DemoState()
    ages = [(NOW - o.created_at).days for o in state.orders]
    assert any(age <= 30 for age in ages), 'need at least one in-window order'
    assert any(age > 30 for age in ages), 'need at least one past-window order'
