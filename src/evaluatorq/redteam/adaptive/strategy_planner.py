"""Shared strategy planning for dynamic red teaming.

This module centralizes strategy selection + optional generation so both
standalone and evaluatorq flows stay aligned.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.redteam.adaptive.capability_classifier import AgentCapabilities, classify_agent_capabilities
from evaluatorq.redteam.adaptive.objective_generator import generate_strategies_for_category
from evaluatorq.redteam.adaptive.strategy_registry import get_strategies_for_category, select_applicable_strategies
from evaluatorq.redteam.contracts import TurnType

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from evaluatorq.redteam.contracts import AgentContext, AttackStrategy


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
) -> tuple[dict[str, list[AttackStrategy]], dict[str, dict[str, Any]], AgentCapabilities]:
    """Build per-category strategy plans for dynamic red teaming."""
    if llm_client is not None:
        agent_capabilities = await classify_agent_capabilities(
            agent_context=agent_context,
            llm_client=llm_client,
            model=attack_model,
        )
    else:
        agent_capabilities = AgentCapabilities()
        logger.info('Skipping capability classification: no llm_client provided')

    all_category_strategies: dict[str, list[AttackStrategy]] = {}
    strategy_selection: dict[str, dict[str, Any]] = {}
    all_hardcoded_by_category: dict[str, list[AttackStrategy]] = {}
    applicable_hardcoded_by_category: dict[str, list[AttackStrategy]] = {}

    for category in categories:
        all_hardcoded = get_strategies_for_category(category)
        applicable_hardcoded = select_applicable_strategies(
            category=category,
            agent_context=agent_context,
            agent_capabilities=agent_capabilities if llm_client is not None else None,
        )
        all_hardcoded_by_category[category] = all_hardcoded
        applicable_hardcoded_by_category[category] = applicable_hardcoded

    generated_by_category: dict[str, list[AttackStrategy]] = {category: [] for category in categories}
    generated_single_by_category: dict[str, list[AttackStrategy]] = {category: [] for category in categories}
    generated_multi_by_category: dict[str, list[AttackStrategy]] = {category: [] for category in categories}

    if generate_additional_strategies and llm_client is not None and generated_strategy_count > 0:
        effective_parallelism = max(1, generation_parallelism or len(categories) or 1)
        semaphore = asyncio.Semaphore(effective_parallelism)

        async def _generate_for_category(category: str) -> tuple[str, list[AttackStrategy]]:
            try:
                async with semaphore:
                    generated = await generate_strategies_for_category(
                        category=category,
                        agent_context=agent_context,
                        llm_client=llm_client,
                        model=attack_model,
                        count=generated_strategy_count,
                        turn_type=None,
                        max_turns=max_turns,
                    )
                return category, generated
            except Exception as e:
                logger.warning(f'Failed to generate strategies for {category}: {e}')
                return category, []

        generation_results = await asyncio.gather(*(_generate_for_category(category) for category in categories))
        for category, generated in generation_results:
            generated_by_category[category] = generated
            generated_single = [s for s in generated if s.turn_type == TurnType.SINGLE]
            generated_multi = [s for s in generated if s.turn_type == TurnType.MULTI]
            generated_single_by_category[category] = generated_single
            generated_multi_by_category[category] = generated_multi
            logger.info(
                f'Added {len(generated)} generated strategies for {category} ({len(generated_single)} single-turn, {len(generated_multi)} multi-turn)'
            )

    for category in categories:
        all_hardcoded = all_hardcoded_by_category[category]
        applicable_hardcoded = applicable_hardcoded_by_category[category]
        generated = generated_by_category[category]
        generated_single = generated_single_by_category[category]
        generated_multi = generated_multi_by_category[category]

        applicable = [*applicable_hardcoded, *generated]
        if max_per_category:
            applicable = applicable[:max_per_category]

        all_category_strategies[category] = applicable
        strategy_selection[category] = {
            'all_hardcoded': [s.model_dump(mode='json') for s in all_hardcoded],
            'applicable': [s.model_dump(mode='json') for s in applicable],
            'all_hardcoded_count': len(all_hardcoded),
            'applicable_count': len(applicable_hardcoded),
            'generated_count': len(generated),
            'generated_single_count': len(generated_single),
            'generated_multi_count': len(generated_multi),
            'filtered_count': len(all_hardcoded) - len(applicable_hardcoded),
            'total_selected': len(applicable),
        }

    return all_category_strategies, strategy_selection, agent_capabilities
