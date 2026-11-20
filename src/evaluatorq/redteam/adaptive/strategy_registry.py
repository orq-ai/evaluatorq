"""Strategy registry for OWASP-targeted attacks.

This module provides:
- Unified registry combining ASI and LLM strategies
- Strategy selection based on agent context and capabilities
- Strategy filtering by requirements
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any, TypedDict

from loguru import logger

from evaluatorq.redteam.contracts import (
    OWASP_CATEGORY_NAMES,
    AgentCapability,
    AgentContext,
    AnyAgentCapability,
    AnyVulnerability,
    AttackStrategy,
    AttackTechnique,
    DeliveryMethod,
    TurnType,
)
from evaluatorq.redteam.frameworks.owasp.evaluators import OWASP_EVALUATOR_REGISTRY
from evaluatorq.redteam.frameworks.owasp_asi import ASI_STRATEGIES
from evaluatorq.redteam.frameworks.owasp_llm import LLM_STRATEGIES
from evaluatorq.redteam.vulnerability_registry import (
    CATEGORY_TO_VULNERABILITY,
    VULNERABILITY_DEFS,
    get_primary_category,
    resolve_vulnerabilities,
)

if TYPE_CHECKING:
    from evaluatorq.redteam.adaptive.capability_classifier import AgentCapabilities

_BUILTIN_STRATEGIES: dict[str, list[AttackStrategy]] = {**ASI_STRATEGIES, **LLM_STRATEGIES}
_custom_strategies: dict[AnyVulnerability, list[AttackStrategy]] = {}

_strategy_registry: dict[str, list[AttackStrategy]] = {}
_vulnerability_strategy_registry: dict[AnyVulnerability, list[AttackStrategy]] = {}


def _rebuild_strategy_registries() -> None:
    """Refill both registries in place from built-ins plus registered custom strategies.

    The public names are read-only proxies over these dicts and are bound by name in
    other modules, so the dicts are mutated, never replaced.
    """
    _strategy_registry.clear()
    _strategy_registry.update(_BUILTIN_STRATEGIES)
    _vulnerability_strategy_registry.clear()
    for cat, strategies in _BUILTIN_STRATEGIES.items():
        vuln = CATEGORY_TO_VULNERABILITY.get(cat)
        if vuln is not None:
            _vulnerability_strategy_registry[vuln] = strategies
    for vuln, strategies in _custom_strategies.items():
        merged = [*_vulnerability_strategy_registry.get(vuln, []), *strategies]
        _vulnerability_strategy_registry[vuln] = merged
        _strategy_registry[get_primary_category(vuln)] = merged
    _strategy_registry.update({f'OWASP-{k}': v for k, v in list(_strategy_registry.items())})


_rebuild_strategy_registries()

STRATEGY_REGISTRY = types.MappingProxyType(_strategy_registry)

VULNERABILITY_STRATEGY_REGISTRY = types.MappingProxyType(_vulnerability_strategy_registry)


def register_strategy(strategy: AttackStrategy) -> None:
    """Add a custom attack strategy to its vulnerability's strategies (RES-1922).

    The strategy's vulnerability must be built in or registered, and its name must be
    unique within that vulnerability. Registering an identical strategy again is a no-op.
    """
    if strategy.vulnerability is None:
        msg = f'Strategy {strategy.name!r} needs a vulnerability to be registered.'
        raise ValueError(msg)
    vuln = strategy.vulnerability
    for existing in _vulnerability_strategy_registry.get(vuln, []):
        if existing.name == strategy.name:
            if existing.model_dump(mode='json') == strategy.model_dump(mode='json'):
                return
            msg = f'Vulnerability {vuln.value!r} already has a strategy named {strategy.name!r}.'
            raise ValueError(msg)
    _custom_strategies.setdefault(vuln, []).append(strategy)
    _rebuild_strategy_registries()


def clear_custom_strategies() -> None:
    """Drop every registered custom strategy (test isolation)."""
    _custom_strategies.clear()
    _rebuild_strategy_registries()


def refresh_strategy_registries() -> None:
    """Re-derive the category keys after vulnerability registrations changed."""
    _rebuild_strategy_registries()


def get_strategies_for_vulnerability(vuln: AnyVulnerability) -> list[AttackStrategy]:
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
                logger.debug(f'Including {strategy.name} optimistically: tool classification failed')
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
    vuln: AnyVulnerability,
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


def _filter_by_method(
    strategies: list[AttackStrategy],
    names: set[str] | None,
    delivery_methods: set[DeliveryMethod | str] | None,
    attack_techniques: set[AttackTechnique] | None = None,
) -> list[AttackStrategy]:
    """Filter strategies by name and/or delivery method.

    Pure predicate — no logging, no I/O. Reusable from both the planner (for
    hardcoded and generated strategies) and from future config-file loaders.
    Selection semantics are AND when both filters are supplied.

    Args:
        strategies: Candidate strategies to filter.
        names: Set of accepted `AttackStrategy.name` values. ``None`` disables
            the name filter. Empty set filters out everything.
        delivery_methods: Set of accepted `DeliveryMethod` values. ``None``
            disables the delivery-method filter. A strategy passes if any of
            its `delivery_methods` overlaps the selection. Empty set filters
            out everything.
        attack_techniques: Set of accepted `AttackTechnique` values, matched
            against the strategy's single `attack_technique`. ``None`` disables
            the filter. Empty set filters out everything.

    Returns:
        Strategies that pass every supplied filter (AND semantics).
    """
    if names is None and delivery_methods is None and attack_techniques is None:
        return list(strategies)

    kept: list[AttackStrategy] = []
    for strategy in strategies:
        if names is not None and strategy.name not in names:
            continue
        if delivery_methods is not None and not (set(strategy.delivery_methods) & delivery_methods):
            continue
        if attack_techniques is not None and strategy.attack_technique not in attack_techniques:
            continue
        kept.append(strategy)
    return kept


def known_strategy_names() -> set[str]:
    """Return all strategy names registered in the hardcoded registry.

    Used to validate user-supplied `--strategy` filters. Excludes runtime
    LLM-generated strategy names, which carry a ``generated_`` prefix and
    cannot be enumerated ahead of a run.
    """
    return {strategy.name for strategy in curated_strategies()}


def curated_strategies() -> list[AttackStrategy]:
    """Every hand-written strategy once, in registry order, without the ``OWASP-`` aliases."""
    return [s for category, group in _strategy_registry.items() if not category.startswith('OWASP-') for s in group]


def strategy_needs(strategy: AttackStrategy) -> str:
    """What a target needs for ``strategy`` to run, e.g. ``tools + web_request or user_data``; ``''`` for nothing.

    Mirrors the skip rules in `_filter_applicable_strategies`: tools when ``requires_tools``, and at
    least one of ``required_capabilities``.
    """
    capabilities = ' or '.join(capability.value for capability in strategy.required_capabilities)
    return ' + '.join(part for part in ('tools' if strategy.requires_tools else '', capabilities) if part)


class StrategyRow(TypedDict):
    """One curated strategy as `eq redteam strategies` lists it and the red teaming guide tabulates it."""

    name: str
    category: str
    vulnerability: str
    description: str
    turn_type: str
    needs: str
    requires_tools: bool
    required_capabilities: list[str]
    attack_technique: str
    delivery_methods: list[str]


def strategy_catalog() -> list[StrategyRow]:
    """`curated_strategies` as display rows."""
    return [
        StrategyRow(
            name=s.name,
            category=s.category,
            vulnerability=CATEGORY_TO_VULNERABILITY[s.category].value,
            description=s.description,
            turn_type=s.turn_type.value,
            needs=strategy_needs(s),
            requires_tools=s.requires_tools,
            required_capabilities=[c.value for c in s.required_capabilities],
            attack_technique=s.attack_technique.value,
            delivery_methods=[d.value for d in s.delivery_methods],
        )
        for s in curated_strategies()
    ]


def _fallback_capability_check(required: list[AnyAgentCapability], agent_context: AgentContext) -> bool:
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


def default_vulnerabilities() -> list[AnyVulnerability]:
    """The vulnerabilities a run covers when it selects no categories or vulnerabilities."""
    return resolve_vulnerabilities(list_available_categories())


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
