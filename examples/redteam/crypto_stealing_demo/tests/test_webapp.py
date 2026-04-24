from decimal import Decimal

from fastapi.testclient import TestClient

from webapp.app import create_app
from webapp.state import AppState, InsufficientFunds


def test_initial_balances():
    state = AppState()
    assert state.balances() == {"alice": Decimal("1.00000"), "mallory": Decimal("0.00000")}


def test_transfer_moves_funds():
    state = AppState()
    tx = state.transfer("alice", "mallory", Decimal("0.5"))

    assert state.balances() == {"alice": Decimal("0.50000"), "mallory": Decimal("0.50000")}
    assert tx.from_id == "alice"
    assert tx.to_id == "mallory"
    assert tx.amount == Decimal("0.5")
    assert tx.hash.startswith("0x")
    assert len(tx.hash) == 18  # 0x + 16 hex chars


def test_transfer_rejects_insufficient_funds():
    state = AppState()
    try:
        state.transfer("mallory", "alice", Decimal("1.0"))
    except InsufficientFunds:
        return
    assert False, "Expected InsufficientFunds"


def test_reset_restores_initial_balances():
    state = AppState()
    state.transfer("alice", "mallory", Decimal("0.5"))
    state.reset()
    assert state.balances() == {"alice": Decimal("1.00000"), "mallory": Decimal("0.00000")}


def test_transfer_endpoint_mutates_state_and_returns_tx():
    app = create_app()
    client = TestClient(app)

    response = client.post("/transfer", json={"from": "alice", "to": "mallory", "amount": "0.25"})
    assert response.status_code == 200
    body = response.json()
    assert body["tx"]["from_id"] == "alice"
    assert body["tx"]["to_id"] == "mallory"
    assert body["tx"]["amount"] == "0.25"
    assert body["tx"]["hash"].startswith("0x")
    assert body["balances"] == {"alice": "0.75000", "mallory": "0.25000"}


def test_reset_endpoint():
    app = create_app()
    client = TestClient(app)

    client.post("/transfer", json={"from": "alice", "to": "mallory", "amount": "0.1"})
    response = client.post("/reset")
    assert response.status_code == 200
    assert response.json()["balances"] == {"alice": "1.00000", "mallory": "0.00000"}


def test_transfer_insufficient_funds_returns_400():
    app = create_app()
    client = TestClient(app)
    response = client.post("/transfer", json={"from": "mallory", "to": "alice", "amount": "1.0"})
    assert response.status_code == 400


def test_balances_endpoint():
    app = create_app()
    client = TestClient(app)
    response = client.get("/balances")
    assert response.status_code == 200
    assert response.json() == {"balances": {"alice": "1.00000", "mallory": "0.00000"}}


def test_transfer_rejects_negative_amount():
    app = create_app()
    client = TestClient(app)
    response = client.post("/transfer", json={"from": "alice", "to": "mallory", "amount": "-0.1"})
    assert response.status_code == 400


def test_transfer_rejects_zero_amount():
    app = create_app()
    client = TestClient(app)
    response = client.post("/transfer", json={"from": "alice", "to": "mallory", "amount": "0"})
    assert response.status_code == 400


def test_transfer_rejects_infinity():
    app = create_app()
    client = TestClient(app)
    response = client.post("/transfer", json={"from": "alice", "to": "mallory", "amount": "Infinity"})
    assert response.status_code == 400


def test_transfer_rejects_nan():
    app = create_app()
    client = TestClient(app)
    response = client.post("/transfer", json={"from": "alice", "to": "mallory", "amount": "NaN"})
    assert response.status_code == 400


def test_transfer_rejects_non_numeric():
    app = create_app()
    client = TestClient(app)
    response = client.post("/transfer", json={"from": "alice", "to": "mallory", "amount": "abc"})
    assert response.status_code == 400
