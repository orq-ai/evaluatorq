"""Smoke test: transfer 0.01 BTC from alice to mallory and verify the webapp updates.

Run with the webapp already running:
    uv run uvicorn webapp.app:app --port 8001

Then in a second terminal:
    uv run python test_webapp.py
"""

from __future__ import annotations

import httpx

from config import WEBAPP_URL as WEBAPP


def get_balances() -> dict[str, str]:
    return httpx.get(f"{WEBAPP}/balances", timeout=5).json()["balances"]


def main() -> None:
    httpx.post(f"{WEBAPP}/reset", timeout=5).raise_for_status()

    before = get_balances()
    print(f"Before: alice={before['alice']}  mallory={before['mallory']}")

    resp = httpx.post(f"{WEBAPP}/transfer", json={"from": "alice", "to": "mallory", "amount": "0.01"}, timeout=5)
    resp.raise_for_status()
    tx = resp.json()["tx"]
    print(f"Tx:     {tx['hash']}  amount={tx['amount']} BTC")

    after = get_balances()
    print(f"After:  alice={after['alice']}  mallory={after['mallory']}")

    assert after["alice"] < before["alice"], "alice balance did not decrease"
    assert after["mallory"] > before["mallory"], "mallory balance did not increase"
    print("OK — balances updated correctly. Check the browser for the live update.")


if __name__ == "__main__":
    main()
