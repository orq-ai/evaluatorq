"""Per-stage provider usage for one Insights run, read from each call's own response."""

from __future__ import annotations

from typing import TYPE_CHECKING

from evaluatorq.common.structured_output import sum_structured_usage
from evaluatorq.contracts import Usage as UsageModel

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
        totals: dict[str, Usage | None] = {}
        for stage, items in self._calls.items():
            # A recorded None is an attempted provider call with no usage block.
            # Cache hits are never added to the ledger, so retain this call count
            # for partial-cost reporting instead of letting the shared reducer
            # drop it as though no call had happened.
            unknown = sum(usage is None for usage in items)
            recorded = [usage for usage in items if usage is not None]
            if unknown:
                recorded.append(UsageModel(calls=unknown))
            totals[stage] = sum_structured_usage(recorded)
        return totals
