"""Built-in evaluators for agent simulation.

These evaluators assess simulation results using a scorer pattern
compatible with evaluatorq integration.
"""

from __future__ import annotations

from evaluatorq.simulation.evaluators.scorers import (
    DEFAULT_SCORING_CONFIG,
    SIMULATION_EVALUATORS,
    ConversationQualityScore,
    SimulationScorer,
    SimulationScoringConfig,
    conversation_quality_scorer,
    criteria_met_scorer,
    get_all_evaluators,
    get_evaluator,
    goal_achieved_scorer,
    turn_efficiency_scorer,
)

__all__ = [
    'DEFAULT_SCORING_CONFIG',
    'SIMULATION_EVALUATORS',
    'ConversationQualityScore',
    'SimulationScorer',
    'SimulationScoringConfig',
    'conversation_quality_scorer',
    'criteria_met_scorer',
    'get_all_evaluators',
    'get_evaluator',
    'goal_achieved_scorer',
    'turn_efficiency_scorer',
]
