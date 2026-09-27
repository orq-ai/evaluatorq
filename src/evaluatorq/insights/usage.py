"""Per-stage provider usage for one Insights run, read from each call's own response."""

from __future__ import annotations

from typing import TYPE_CHECKING

from evaluatorq.common.structured_output import sum_structured_usage

if TYPE_CHECKING:
    from evaluatorq.contracts import Usage


class UsageLedger:
    """Collects the usage of every provider call, keyed by Insights stage."""

    def __init__(self) -> None:
        self._calls: dict[str, list[Usage | None]] = {}
        self._warnings: set[str] = set()

    def add(self, stage: str, usage: Usage | None) -> None:
        self._calls.setdefault(stage, []).append(usage)

    def claim_warning(self, key: str) -> bool:
        """Claim a warning slot once per run; return whether this call claimed it."""
        if key in self._warnings:
            return False
        self._warnings.add(key)
        return True

    def totals(self) -> dict[str, Usage | None]:
        return {stage: sum_structured_usage(items) for stage, items in self._calls.items()}
