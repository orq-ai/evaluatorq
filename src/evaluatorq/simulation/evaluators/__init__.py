"""Built-in evaluators for agent simulation.

These evaluators assess simulation results using a scorer pattern
compatible with evaluatorq integration.
"""

from __future__ import annotations

from collections.abc import Callable

from evaluatorq.simulation.types import SimulationResult

SimulationScorer = Callable[[SimulationResult], float]


# ---------------------------------------------------------------------------
# Individual scorers
# ---------------------------------------------------------------------------


def goal_achieved_scorer(result: SimulationResult) -> float:
    """Evaluate if the simulation goal was achieved. Returns 1 if achieved, 0 otherwise."""
    return 1.0 if result.goal_achieved else 0.0


def criteria_met_scorer(result: SimulationResult) -> float:
    """Evaluate how many criteria were met.

    Returns a value between 0 and 1 based on the ratio of met criteria.
    Uses the criteria_results from metadata if available; otherwise returns 1.0.
    """
    criteria_results = result.criteria_results or {}
    keys = list(criteria_results.keys())

    if not keys:
        return 1.0

    met = sum(1 for v in criteria_results.values() if v)
    total = len(keys)
    return met / total if total > 0 else 1.0


def turn_efficiency_scorer(result: SimulationResult) -> float:
    """Evaluate conversation efficiency (fewer turns = better).

    Returns a value between 0 and 1.
    """
    total_turns = result.turn_count
    goal_achieved = result.goal_achieved

    if not goal_achieved:
        return 0.0

    if total_turns <= 2:
        return 1.0
    if total_turns <= 4:
        return 0.9
    if total_turns <= 6:
        return 0.7

    return max(0.3, 1.0 - (total_turns - 6) * 0.1)


def conversation_quality_scorer(result: SimulationResult) -> float:
    """Evaluate overall conversation quality.

    Composite score based on:
    - Goal achievement (40%)
    - Criteria met (30%)
    - Turn efficiency (30%)
    """
    goal_score = goal_achieved_scorer(result)
    criteria_score = criteria_met_scorer(result)
    efficiency_score = turn_efficiency_scorer(result)

    score = goal_score * 0.4 + criteria_score * 0.3 + efficiency_score * 0.3
    return round(score * 100) / 100


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SIMULATION_EVALUATORS: dict[str, SimulationScorer] = {
    "goal_achieved": goal_achieved_scorer,
    "criteria_met": criteria_met_scorer,
    "turn_efficiency": turn_efficiency_scorer,
    "conversation_quality": conversation_quality_scorer,
}


def get_evaluator(name: str) -> SimulationScorer:
    """Get a built-in simulation evaluator by name.

    Raises:
            ValueError: If evaluator not found.
    """
    evaluator = SIMULATION_EVALUATORS.get(name)
    if not evaluator:
        available = ", ".join(SIMULATION_EVALUATORS.keys())
        raise ValueError(f"Unknown evaluator: {name}. Available: {available}")
    return evaluator


def get_all_evaluators() -> dict[str, SimulationScorer]:
    """Get all built-in simulation evaluators."""
    return dict(SIMULATION_EVALUATORS)


__all__ = [
    "SimulationScorer",
    "SIMULATION_EVALUATORS",
    "get_evaluator",
    "get_all_evaluators",
    "goal_achieved_scorer",
    "criteria_met_scorer",
    "turn_efficiency_scorer",
    "conversation_quality_scorer",
]
