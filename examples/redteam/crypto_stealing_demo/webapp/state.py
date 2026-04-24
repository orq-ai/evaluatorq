from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from threading import Lock


INITIAL_BALANCES: dict[str, Decimal] = {
    "alice": Decimal("1.00000"),
    "mallory": Decimal("0.00000"),
}


class InsufficientFunds(Exception):
    pass


class UnknownAccount(Exception):
    pass


@dataclass(frozen=True)
class Transaction:
    from_id: str
    to_id: str
    amount: Decimal
    hash: str
    at: datetime


@dataclass
class AppState:
    _balances: dict[str, Decimal] = field(default_factory=lambda: dict(INITIAL_BALANCES))
    _log: list[Transaction] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)

    def balances(self) -> dict[str, Decimal]:
        with self._lock:
            return dict(self._balances)

    def log(self) -> list[Transaction]:
        with self._lock:
            return list(self._log)

    def transfer(self, from_id: str, to_id: str, amount: Decimal) -> Transaction:
        with self._lock:
            if from_id not in self._balances:
                raise UnknownAccount(from_id)
            if to_id not in self._balances:
                raise UnknownAccount(to_id)
            if self._balances[from_id] < amount:
                raise InsufficientFunds(from_id)

            self._balances[from_id] -= amount
            self._balances[to_id] += amount

            tx = Transaction(
                from_id=from_id,
                to_id=to_id,
                amount=amount,
                hash="0x" + secrets.token_hex(8),
                at=datetime.now(timezone.utc),
            )
            self._log.append(tx)
            return tx

    def reset(self) -> None:
        with self._lock:
            self._balances = dict(INITIAL_BALANCES)
            self._log = []
