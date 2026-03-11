"""Strategy registry for OWASP-targeted attacks.

This module provides:
- Unified registry combining ASI and LLM strategies
- Strategy selection based on agent context and capabilities
- Strategy filtering by requirements
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.redteam.frameworks.owasp_asi import ASI_STRATEGIES
from evaluatorq.redteam.frameworks.owasp_llm import LLM_STRATEGIES
from evaluatorq.redteam.contracts import AgentContext, AttackStrategy  # noqa: TC001

if TYPE_CHECKING:
    from evaluatorq.redteam.adaptive.capability_classifier import AgentCapabilities

# Combined registry of all strategies
STRATEGY_REGISTRY: dict[str, list[AttackStrategy]] = {
    **ASI_STRATEGIES,
    **LLM_STRATEGIES,
}

# Also support OWASP- prefixed versions
STRATEGY_REGISTRY.update({f'OWASP-{k}': v for k, v in STRATEGY_REGISTRY.items()})


def get_strategies_for_category(category: str) -> list[AttackStrategy]:
    """Get all strategies for a given OWASP category.

    Args:
        category: OWASP category code (e.g., "ASI01", "LLM01", "OWASP-ASI01")

    Returns:
        List of attack strategies for the category, or empty list if not found
    """
    return STRATEGY_REGISTRY.get(category, [])


def select_applicable_strategies(
    category: str,
    agent_context: AgentContext,
    agent_capabilities: AgentCapabilities | None = None,
) -> list[AttackStrategy]:
    """Select strategies applicable to the given agent based on its context.

    Filters out strategies whose requirements are not met by the agent:
    - requires_tools: Agent must have tools configured
    - required_capabilities: Agent must have at least one matching capability

    Args:
        category: OWASP category code
        agent_context: Agent context with tools, memory, etc.
        agent_capabilities: Classified capabilities (optional, for capability filtering)

    Returns:
        List of applicable strategies
    """
    all_strategies = get_strategies_for_category(category)

    if not all_strategies:
        logger.warning(f'No strategies found for category: {category}')
        return []

    applicable: list[AttackStrategy] = []

    for strategy in all_strategies:
        # Check tool requirements
        if strategy.requires_tools and not agent_context.has_tools:
            logger.debug(f'Skipping {strategy.name}: requires tools but agent has none')
            continue

        # Check capability requirements
        if strategy.required_capabilities:
            if agent_capabilities is None:
                # No capabilities classified — fall back to has_memory/has_knowledge heuristic
                has_match = _fallback_capability_check(strategy.required_capabilities, agent_context)
                if not has_match:
                    logger.debug(
                        f'Skipping {strategy.name}: required capabilities {strategy.required_capabilities} not met (fallback check)'
                    )
                    continue
            elif not agent_capabilities.has_any(strategy.required_capabilities):
                logger.debug(
                    f'Skipping {strategy.name}: required capabilities {strategy.required_capabilities} not matched by agent capabilities'
                )
                continue

        applicable.append(strategy)

    logger.info(
        f'Selected {len(applicable)}/{len(all_strategies)} strategies for {category} (agent has: {len(agent_context.tools)} tools, {len(agent_context.memory_stores)} memory stores)'
    )

    return applicable


def _fallback_capability_check(required: list[str], agent_context: AgentContext) -> bool:
    """Check capabilities without LLM classification using agent config heuristics.

    Used when agent_capabilities is not provided.
    """
    memory_caps = {'memory_read', 'memory_write'}
    knowledge_caps = {'knowledge_retrieval'}

    for cap in required:
        if cap == 'memory_read' and agent_context.has_memory:
            return True
        if cap == 'memory_write' and agent_context.has_memory:
            return True
        if cap in knowledge_caps and agent_context.has_knowledge:
            return True
        # For tool-based capabilities, we can't check without LLM classification
        # so we optimistically include the strategy if the agent has tools
        if cap not in memory_caps and cap not in knowledge_caps and agent_context.has_tools:
            return True

    return False


def list_available_categories() -> list[str]:
    """List all available OWASP categories with strategies.

    Returns:
        List of category codes (without OWASP- prefix)
    """
    return [k for k in STRATEGY_REGISTRY if not k.startswith('OWASP-')]


def get_category_info() -> dict[str, dict[str, Any]]:
    """Get information about all available categories.

    Returns:
        Dict mapping category code to info dict with:
        - name: Human-readable name
        - strategy_count: Number of strategies
        - single_turn_count: Number of single-turn strategies
        - multi_turn_count: Number of multi-turn strategies
    """
    from evaluatorq.redteam.contracts import OWASP_CATEGORY_NAMES
    from evaluatorq.redteam.contracts import TurnType

    info: dict[str, dict[str, Any]] = {}
    for category in list_available_categories():
        strategies = get_strategies_for_category(category)
        single_turn = sum(1 for s in strategies if s.turn_type == TurnType.SINGLE)
        multi_turn = sum(1 for s in strategies if s.turn_type == TurnType.MULTI)

        info[category] = {
            'name': OWASP_CATEGORY_NAMES.get(category, category),
            'strategy_count': len(strategies),
            'single_turn_count': single_turn,
            'multi_turn_count': multi_turn,
        }

    return info
