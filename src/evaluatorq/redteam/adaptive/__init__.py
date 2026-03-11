"""Adaptive dynamic red teaming pipeline for evaluatorq.

Key public classes and functions re-exported for convenience:

    from evaluatorq.redteam.adaptive import (
        MultiTurnOrchestrator,
        OWASPEvaluator,
        AgentCapabilities,
        classify_agent_capabilities,
        plan_strategies_for_categories,
        generate_dynamic_datapoints,
        create_dynamic_redteam_job,
        create_dynamic_evaluator,
    )
"""

from evaluatorq.redteam.adaptive.capability_classifier import AgentCapabilities, classify_agent_capabilities
from evaluatorq.redteam.adaptive.evaluator import OWASPEvaluator, evaluate_attack
from evaluatorq.redteam.adaptive.orchestrator import MultiTurnOrchestrator
from evaluatorq.redteam.adaptive.pipeline import (
    cleanup_memory_entities,
    create_dynamic_evaluator,
    create_dynamic_redteam_job,
    generate_dynamic_datapoints,
)
from evaluatorq.redteam.adaptive.strategy_planner import plan_strategies_for_categories
from evaluatorq.redteam.adaptive.strategy_registry import (
    STRATEGY_REGISTRY,
    get_strategies_for_category,
    list_available_categories,
    select_applicable_strategies,
)

__all__ = [
    'AgentCapabilities',
    'classify_agent_capabilities',
    'OWASPEvaluator',
    'evaluate_attack',
    'MultiTurnOrchestrator',
    'cleanup_memory_entities',
    'create_dynamic_evaluator',
    'create_dynamic_redteam_job',
    'generate_dynamic_datapoints',
    'plan_strategies_for_categories',
    'STRATEGY_REGISTRY',
    'get_strategies_for_category',
    'list_available_categories',
    'select_applicable_strategies',
]
