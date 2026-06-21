"""Shared fixtures for simulation tests."""

from __future__ import annotations

import pytest

from evaluatorq.contracts import TokenUsage
from evaluatorq.simulation.types import SimulationResult, TerminatedBy


@pytest.fixture
def sim_result_factory():
    def _make(
        *,
        goal_achieved: bool = True,
        persona: str = "p",
        scenario: str = "s",
        turn_count: int = 2,
        error: str | None = None,
    ) -> SimulationResult:
        meta: dict[str, object] = {"persona": persona, "scenario": scenario}
        if error is not None:
            meta["error"] = error
        return SimulationResult(
            messages=[],
            terminated_by=TerminatedBy.error if error else TerminatedBy.judge,
            reason="done",
            goal_achieved=goal_achieved,
            goal_completion_score=1.0 if goal_achieved else 0.0,
            rules_broken=[],
            turn_count=turn_count,
            turn_metrics=[],
            token_usage=TokenUsage(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
            ),
            metadata=meta,
        )

    return _make
