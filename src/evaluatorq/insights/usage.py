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

    def add(self, stage: str, usage: Usage | None) -> None:
        self._calls.setdefault(stage, []).append(usage)

    def totals(self) -> dict[str, Usage | None]:
        return {stage: sum_structured_usage(items) for stage, items in self._calls.items()}
