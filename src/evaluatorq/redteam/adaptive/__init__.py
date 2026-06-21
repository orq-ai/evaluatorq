"""Adaptive dynamic red teaming pipeline for evaluatorq.

Key public classes and functions re-exported for convenience:

    from evaluatorq.redteam.adaptive import (
        MultiTurnOrchestrator,
        OWASPEvaluator,
        AgentCapabilities,
        classify_agent_capabilities,
        plan_strategies_for_categories,
    )
"""

from evaluatorq.redteam.adaptive.capability_classifier import AgentCapabilities, classify_agent_capabilities
from evaluatorq.redteam.adaptive.evaluator import OWASPEvaluator, evaluate_attack
from evaluatorq.redteam.adaptive.orchestrator import MultiTurnOrchestrator
from evaluatorq.redteam.adaptive.strategy_planner import plan_strategies_for_categories
from evaluatorq.redteam.adaptive.strategy_registry import (
    get_strategies_for_category,
    list_available_categories,
    select_applicable_strategies,
)

__all__ = [
    'AgentCapabilities',
    'MultiTurnOrchestrator',
    'OWASPEvaluator',
    'classify_agent_capabilities',
    'evaluate_attack',
    'get_strategies_for_category',
    'list_available_categories',
    'plan_strategies_for_categories',
    'select_applicable_strategies',
]
