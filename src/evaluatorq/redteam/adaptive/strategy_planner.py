"""Shared strategy planning for dynamic red teaming.

This module centralizes strategy selection + optional generation so both
standalone and evaluatorq flows stay aligned.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.redteam.adaptive.capability_classifier import AgentCapabilities, classify_agent_capabilities
from evaluatorq.redteam.adaptive.objective_generator import generate_strategies_for_vulnerability
from evaluatorq.redteam.adaptive.strategy_registry import (
    get_strategies_for_vulnerability,
    select_applicable_strategies,
    select_applicable_strategies_for_vulnerability,
)
from evaluatorq.redteam.contracts import PIPELINE_CONFIG, LLMConfig, TurnType, Vulnerability
from evaluatorq.redteam.tracing import set_span_attrs, with_redteam_span
from evaluatorq.redteam.vulnerability_registry import resolve_category

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from evaluatorq.redteam.contracts import AgentContext, AttackStrategy


async def plan_strategies_for_vulnerabilities(
    *,
    agent_context: AgentContext,
    vulnerabilities: list[Vulnerability],
    llm_client: AsyncOpenAI | None,
    attack_model: str,
    max_turns: int,
    max_per_category: int | None,
    generate_additional_strategies: bool,
    generated_strategy_count: int,
    generation_parallelism: int | None = None,
    attacker_instructions: str | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    pipeline_config: LLMConfig | None = None,
) -> tuple[dict[Vulnerability, list[AttackStrategy]], dict[Vulnerability, dict[str, Any]], AgentCapabilities]:
    """Build per-vulnerability strategy plans for dynamic red teaming.

    This is the primary planning function keyed by Vulnerability enum values.
    Use plan_strategies_for_categories() for backwards-compatible category-string-keyed output.
    """
    cfg = pipeline_config or PIPELINE_CONFIG
    if llm_client is not None:
        async with with_redteam_span("orq.redteam.capability_classification", {
            "orq.redteam.num_tools": len(agent_context.tools) if agent_context.tools else 0,
            "orq.redteam.model": attack_model,
        }) as cap_span:
            agent_capabilities = await classify_agent_capabilities(
                agent_context=agent_context,
                llm_client=llm_client,
                model=attack_model,
                llm_kwargs=llm_kwargs,
                pipeline_config=cfg,
            )
            set_span_attrs(cap_span, {
                "orq.redteam.num_capabilities": len(agent_capabilities.all_capabilities()),
            })
    else:
        agent_capabilities = AgentCapabilities()
        logger.debug('Skipping capability classification: no llm_client provided')

    all_hardcoded_by_vuln: dict[Vulnerability, list[AttackStrategy]] = {}
    applicable_hardcoded_by_vuln: dict[Vulnerability, list[AttackStrategy]] = {}

    for vuln in vulnerabilities:
        all_hardcoded = get_strategies_for_vulnerability(vuln)
        applicable_hardcoded = select_applicable_strategies_for_vulnerability(
            vuln=vuln,
            agent_context=agent_context,
            agent_capabilities=agent_capabilities if llm_client is not None else None,
        )
        all_hardcoded_by_vuln[vuln] = all_hardcoded
        applicable_hardcoded_by_vuln[vuln] = applicable_hardcoded

    generated_by_vuln: dict[Vulnerability, list[AttackStrategy]] = {vuln: [] for vuln in vulnerabilities}
    generated_single_by_vuln: dict[Vulnerability, list[AttackStrategy]] = {vuln: [] for vuln in vulnerabilities}
    generated_multi_by_vuln: dict[Vulnerability, list[AttackStrategy]] = {vuln: [] for vuln in vulnerabilities}
    generation_errors: dict[Vulnerability, str | None] = {vuln: None for vuln in vulnerabilities}

    if generate_additional_strategies and llm_client is not None and generated_strategy_count > 0:
        async with with_redteam_span("orq.redteam.strategy_planning", {
            "orq.redteam.num_vulnerabilities": len(vulnerabilities),
        }) as strat_span:
            effective_parallelism = max(1, generation_parallelism or len(vulnerabilities) or 1)
            semaphore = asyncio.Semaphore(effective_parallelism)

            async def _generate_for_vulnerability(vuln: Vulnerability) -> tuple[Vulnerability, list[AttackStrategy], str | None]:
                try:
                    async with semaphore:
                        generated = await generate_strategies_for_vulnerability(
                            vuln=vuln,
                            agent_context=agent_context,
                            llm_client=llm_client,
                            model=attack_model,
                            count=generated_strategy_count,
                            turn_type=None,
                            max_turns=max_turns,
                            attacker_instructions=attacker_instructions,
                            llm_kwargs=llm_kwargs,
                            pipeline_config=cfg,
                        )
                    return vuln, generated, None
                except Exception as e:
                    logger.error(
                        f'Strategy generation failed for {vuln.value}, no strategies will be tested for this vulnerability: {e}'
                    )
                    return vuln, [], str(e)

            generation_results = await asyncio.gather(*(_generate_for_vulnerability(vuln) for vuln in vulnerabilities))
            for vuln, generated, generation_error in generation_results:
                generated_by_vuln[vuln] = generated
                generation_errors[vuln] = generation_error
                generated_single = [s for s in generated if s.turn_type == TurnType.SINGLE]
                generated_multi = [s for s in generated if s.turn_type == TurnType.MULTI]
                generated_single_by_vuln[vuln] = generated_single
                generated_multi_by_vuln[vuln] = generated_multi
                logger.debug(
                    f'Added {len(generated)} generated strategies for {vuln.value} '
                    f'({len(generated_single)} single-turn, {len(generated_multi)} multi-turn)'
                )
            total_generated = sum(len(v) for v in generated_by_vuln.values())
            set_span_attrs(strat_span, {"orq.redteam.generated_count": total_generated})

    all_vuln_strategies: dict[Vulnerability, list[AttackStrategy]] = {}
    strategy_selection: dict[Vulnerability, dict[str, Any]] = {}

    for vuln in vulnerabilities:
        all_hardcoded = all_hardcoded_by_vuln[vuln]
        applicable_hardcoded = applicable_hardcoded_by_vuln[vuln]
        generated = generated_by_vuln[vuln]
        generated_single = generated_single_by_vuln[vuln]
        generated_multi = generated_multi_by_vuln[vuln]

        applicable = [*applicable_hardcoded, *generated]
        if max_per_category:
            applicable = applicable[:max_per_category]

        all_vuln_strategies[vuln] = applicable
        strategy_selection[vuln] = {
            'all_hardcoded': [s.model_dump(mode='json') for s in all_hardcoded],
            'applicable': [s.model_dump(mode='json') for s in applicable],
            'all_hardcoded_count': len(all_hardcoded),
            'applicable_count': len(applicable),
            'generated_count': len(generated),
            'generated_single_count': len(generated_single),
            'generated_multi_count': len(generated_multi),
            'filtered_count': len(all_hardcoded) - len(applicable_hardcoded),
            'total_selected': len(applicable),
            'generation_error': generation_errors.get(vuln),
        }

    return all_vuln_strategies, strategy_selection, agent_capabilities


async def plan_strategies_for_categories(
    *,
    agent_context: AgentContext,
    categories: list[str],
    llm_client: AsyncOpenAI | None,
    attack_model: str,
    max_turns: int,
    max_per_category: int | None,
    generate_additional_strategies: bool,
    generated_strategy_count: int,
    generation_parallelism: int | None = None,
    attacker_instructions: str | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    pipeline_config: LLMConfig | None = None,
) -> tuple[dict[str, list[AttackStrategy]], dict[str, dict[str, Any]], AgentCapabilities]:
    """Build per-category strategy plans for dynamic red teaming.

    Resolves each category string to a Vulnerability enum and delegates to
    plan_strategies_for_vulnerabilities(), then remaps the results back to
    category-string keys for backwards compatibility.

    Categories that cannot be resolved (unknown codes) fall back to direct
    category-based lookup so legacy behaviour is preserved.
    """
    # Resolve each category to a Vulnerability, preserving order and mapping back
    # to original category strings for the return value.
    category_to_vuln: dict[str, Vulnerability] = {}
    unresolved_categories: list[str] = []

    for category in categories:
        try:
            category_to_vuln[category] = resolve_category(category)
        except KeyError:  # noqa: PERF203
            logger.warning(f'Could not resolve category {category!r} to a Vulnerability — using direct category lookup')
            unresolved_categories.append(category)

    vulnerabilities = list(dict.fromkeys(category_to_vuln.values()))  # deduplicated, order-preserving

    # Run the vulnerability-first plan for all resolved categories
    vuln_strategies, vuln_selection, agent_capabilities = await plan_strategies_for_vulnerabilities(
        agent_context=agent_context,
        vulnerabilities=vulnerabilities,
        llm_client=llm_client,
        attack_model=attack_model,
        max_turns=max_turns,
        max_per_category=max_per_category,
        generate_additional_strategies=generate_additional_strategies,
        generated_strategy_count=generated_strategy_count,
        generation_parallelism=generation_parallelism,
        attacker_instructions=attacker_instructions,
        llm_kwargs=llm_kwargs,
        pipeline_config=pipeline_config,
    )

    # Remap results back to original category strings
    all_category_strategies: dict[str, list[AttackStrategy]] = {}
    strategy_selection: dict[str, dict[str, Any]] = {}

    for category in categories:
        if category in category_to_vuln:
            vuln = category_to_vuln[category]
            all_category_strategies[category] = vuln_strategies.get(vuln, [])
            strategy_selection[category] = vuln_selection.get(vuln, {})
        else:
            # Fallback for unresolved categories: use filtered category lookup
            all_category_strategies[category] = select_applicable_strategies(
                category=category,
                agent_context=agent_context,
                agent_capabilities=agent_capabilities,
            )
            strategy_selection[category] = {
                'all_hardcoded': [s.model_dump(mode='json') for s in all_category_strategies[category]],
                'applicable': [s.model_dump(mode='json') for s in all_category_strategies[category]],
                'all_hardcoded_count': len(all_category_strategies[category]),
                'applicable_count': len(all_category_strategies[category]),
                'generated_count': 0,
                'generated_single_count': 0,
                'generated_multi_count': 0,
                'filtered_count': 0,
                'total_selected': len(all_category_strategies[category]),
            }

    return all_category_strategies, strategy_selection, agent_capabilities
