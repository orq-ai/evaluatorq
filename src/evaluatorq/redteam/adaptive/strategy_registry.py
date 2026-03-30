"""Strategy registry for OWASP-targeted attacks.

This module provides:
- Unified registry combining ASI and LLM strategies
- Strategy selection based on agent context and capabilities
- Strategy filtering by requirements
"""

from __future__ import annotations

import types
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.redteam.frameworks.owasp_asi import ASI_STRATEGIES
from evaluatorq.redteam.frameworks.owasp_llm import LLM_STRATEGIES
from evaluatorq.redteam.frameworks.owasp.evaluators import OWASP_EVALUATOR_REGISTRY
from evaluatorq.redteam.contracts import OWASP_CATEGORY_NAMES, AgentCapability, AgentContext, AttackStrategy, TurnType, Vulnerability  # noqa: TC001
from evaluatorq.redteam.vulnerability_registry import CATEGORY_TO_VULNERABILITY, VULNERABILITY_DEFS

if TYPE_CHECKING:
    from evaluatorq.redteam.adaptive.capability_classifier import AgentCapabilities

# Combined registry of all strategies
STRATEGY_REGISTRY: Mapping[str, list[AttackStrategy]]
VULNERABILITY_STRATEGY_REGISTRY: Mapping[Vulnerability, list[AttackStrategy]]

_strategy_registry: dict[str, list[AttackStrategy]] = {
    **ASI_STRATEGIES,
    **LLM_STRATEGIES,
}

# Also support OWASP- prefixed versions
_strategy_registry.update({f'OWASP-{k}': v for k, v in _strategy_registry.items()})

# Primary registry keyed by vulnerability
_vulnerability_strategy_registry: dict[Vulnerability, list[AttackStrategy]] = {}
for _cat, _strategies in _strategy_registry.items():
    if _cat.startswith('OWASP-'):
        continue  # skip prefixed duplicates
    _vuln = CATEGORY_TO_VULNERABILITY.get(_cat)
    if _vuln is not None:
        _vulnerability_strategy_registry[_vuln] = _strategies

STRATEGY_REGISTRY = types.MappingProxyType(_strategy_registry)  # pyright: ignore[reportConstantRedefinition]
VULNERABILITY_STRATEGY_REGISTRY = types.MappingProxyType(_vulnerability_strategy_registry)  # pyright: ignore[reportConstantRedefinition]


def get_strategies_for_vulnerability(vuln: Vulnerability) -> list[AttackStrategy]:
    """Get all strategies for a given vulnerability.

    Args:
        vuln: Vulnerability enum value

    Returns:
        List of attack strategies for the vulnerability, or empty list if not found
    """
    return VULNERABILITY_STRATEGY_REGISTRY.get(vuln, [])


def get_strategies_for_category(category: str) -> list[AttackStrategy]:
    """Get all strategies for a given OWASP category.

    Args:
        category: OWASP category code (e.g., "ASI01", "LLM01", "OWASP-ASI01")

    Returns:
        List of attack strategies for the category, or empty list if not found
    """
    return STRATEGY_REGISTRY.get(category, [])


def _filter_applicable_strategies(
    all_strategies: list[AttackStrategy],
    label: str,
    agent_context: AgentContext,
    agent_capabilities: AgentCapabilities | None,
) -> list[AttackStrategy]:
    """Filter strategies to those applicable for the given agent context.

    Shared implementation used by both select_applicable_strategies() and
    select_applicable_strategies_for_vulnerability().

    Args:
        all_strategies: Full list of candidate strategies to filter.
        label: Human-readable label for logging (category code or vulnerability value).
        agent_context: Agent context with tools, memory, etc.
        agent_capabilities: Classified capabilities (optional, for capability filtering).

    Returns:
        List of applicable strategies.
    """
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
            elif agent_capabilities.classification_failed:
                logger.debug(
                    f'Including {strategy.name} optimistically: tool classification failed'
                )
            elif not agent_capabilities.has_any(strategy.required_capabilities):
                logger.debug(
                    f'Skipping {strategy.name}: required capabilities {strategy.required_capabilities} not matched by agent capabilities'
                )
                continue

        applicable.append(strategy)

    logger.debug(
        f'Selected {len(applicable)}/{len(all_strategies)} strategies for {label} '
        f'(agent has: {len(agent_context.tools)} tools, {len(agent_context.memory_stores)} memory stores)'
    )

    return applicable


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
        logger.debug(f'No hardcoded strategies for category: {category} (will use generated strategies only)')
        return []

    return _filter_applicable_strategies(all_strategies, category, agent_context, agent_capabilities)


def select_applicable_strategies_for_vulnerability(
    vuln: Vulnerability,
    agent_context: AgentContext,
    agent_capabilities: AgentCapabilities | None = None,
) -> list[AttackStrategy]:
    """Select strategies applicable to the given agent for a vulnerability.

    Mirrors select_applicable_strategies() but operates on a Vulnerability enum
    rather than an OWASP category string.

    Filters out strategies whose requirements are not met by the agent:
    - requires_tools: Agent must have tools configured
    - required_capabilities: Agent must have at least one matching capability

    Args:
        vuln: Vulnerability enum value
        agent_context: Agent context with tools, memory, etc.
        agent_capabilities: Classified capabilities (optional, for capability filtering)

    Returns:
        List of applicable strategies
    """
    all_strategies = get_strategies_for_vulnerability(vuln)

    if not all_strategies:
        logger.debug(f'No hardcoded strategies for vulnerability: {vuln.value} (will use generated strategies only)')
        return []

    return _filter_applicable_strategies(all_strategies, vuln.value, agent_context, agent_capabilities)


def _fallback_capability_check(required: list[AgentCapability], agent_context: AgentContext) -> bool:
    """Check capabilities without LLM classification using agent config heuristics.

    Used when agent_capabilities is not provided.
    """
    memory_caps = {AgentCapability.MEMORY_READ, AgentCapability.MEMORY_WRITE}
    knowledge_caps = {AgentCapability.KNOWLEDGE_RETRIEVAL}

    for cap in required:
        if cap == AgentCapability.MEMORY_READ and agent_context.has_memory:
            return True
        if cap == AgentCapability.MEMORY_WRITE and agent_context.has_memory:
            return True
        if cap in knowledge_caps and agent_context.has_knowledge:
            return True
        # For tool-based capabilities, we can't check without LLM classification
        # so we optimistically include the strategy if the agent has tools
        if cap not in memory_caps and cap not in knowledge_caps and agent_context.has_tools:
            return True

    return False


def list_available_categories() -> list[str]:
    """List all OWASP categories that can be tested.

    Includes categories with hardcoded strategies AND categories that only
    have an evaluator (these can still be tested via LLM-generated strategies
    in dynamic mode).

    Returns:
        List of category codes (without OWASP- prefix)
    """
    strategy_cats = {k for k in STRATEGY_REGISTRY if not k.startswith('OWASP-')}
    evaluator_cats = {k for k in OWASP_EVALUATOR_REGISTRY if not k.startswith('OWASP-')}
    return sorted(strategy_cats | evaluator_cats)


def get_category_info() -> dict[str, dict[str, Any]]:
    """Get information about all available categories.

    Returns:
        Dict mapping category code to info dict with:
        - name: Human-readable name
        - strategy_count: Number of strategies
        - single_turn_count: Number of single-turn strategies
        - multi_turn_count: Number of multi-turn strategies
        - vulnerability: Vulnerability enum value (or None if unmapped)
        - vulnerability_name: Human-readable vulnerability name (or None if unmapped)
    """
    info: dict[str, dict[str, Any]] = {}
    for category in list_available_categories():
        strategies = get_strategies_for_category(category)
        single_turn = sum(1 for s in strategies if s.turn_type == TurnType.SINGLE)
        multi_turn = sum(1 for s in strategies if s.turn_type == TurnType.MULTI)

        vuln = CATEGORY_TO_VULNERABILITY.get(category)
        vuln_name: str | None = None
        if vuln is not None:
            vdef = VULNERABILITY_DEFS.get(vuln)
            vuln_name = vdef.name if vdef is not None else vuln.value

        info[category] = {
            'name': OWASP_CATEGORY_NAMES.get(category, category),
            'strategy_count': len(strategies),
            'single_turn_count': single_turn,
            'multi_turn_count': multi_turn,
            'vulnerability': vuln,
            'vulnerability_name': vuln_name,
        }

    return info
