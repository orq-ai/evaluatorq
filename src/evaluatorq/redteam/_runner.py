"""Unified red teaming runner that dispatches to dynamic/static/hybrid pipelines."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.redteam.adaptive.strategy_registry import list_available_categories
from evaluatorq.redteam.contracts import AgentContext, RedTeamReport

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from evaluatorq.redteam.backends.base import AgentTargetFactory, ErrorMapper, MemoryCleanup


async def red_team(
    target: str,
    *,
    mode: str = 'dynamic',
    categories: list[str] | None = None,
    max_turns: int = 5,
    max_per_category: int | None = None,
    attack_model: str = 'azure/gpt-5-mini',
    evaluator_model: str = 'azure/gpt-5-mini',
    parallelism: int = 5,
    generate_strategies: bool = True,
    generated_strategy_count: int = 2,
    max_datapoints: int | None = None,
    cleanup_memory: bool = True,
    backend: str = 'orq',
    target_factory: AgentTargetFactory | None = None,
    error_mapper: ErrorMapper | None = None,
    memory_cleanup: MemoryCleanup | None = None,
    llm_client: AsyncOpenAI | None = None,
    description: str | None = None,
    dataset_path: Any = None,
) -> RedTeamReport:
    """Unified entry point for red teaming.

    Args:
        target: Target identifier — ``"agent:<key>"`` for ORQ agents, or
            ``"openai:<model>"`` for direct OpenAI-compatible endpoints.
        mode: Execution mode — ``"dynamic"``, ``"static"``, or ``"hybrid"``.
        categories: OWASP categories to test (e.g., ``["ASI01", "ASI03"]``).
            Defaults to all available categories.
        max_turns: Maximum conversation turns for multi-turn attacks.
        max_per_category: Cap strategies per category (None = no cap).
        attack_model: Model for adversarial prompt generation.
        evaluator_model: Model for OWASP evaluation scoring.
        parallelism: Maximum concurrent evaluatorq jobs.
        generate_strategies: Whether to generate additional LLM-based strategies.
        generated_strategy_count: Number of strategies to generate per category.
        max_datapoints: Cap total datapoints (None = no cap).
        cleanup_memory: Whether to clean up memory entities after dynamic runs.
        backend: Backend name (``"orq"`` or ``"openai"``).
        target_factory: Custom target factory (overrides backend default).
        error_mapper: Custom error mapper (overrides backend default).
        memory_cleanup: Custom memory cleanup (overrides backend default).
        llm_client: Pre-configured AsyncOpenAI client for attack/strategy generation.
        description: Optional description for the report.
        dataset_path: Path to static dataset (required for static/hybrid modes).

    Returns:
        RedTeamReport with results and summary statistics.

    Raises:
        ValueError: If mode is invalid or required arguments are missing.
        NotImplementedError: For static/hybrid modes (Phase 4).
    """
    if mode == 'dynamic':
        return await _run_dynamic(
            target=target,
            categories=categories,
            max_turns=max_turns,
            max_per_category=max_per_category,
            attack_model=attack_model,
            evaluator_model=evaluator_model,
            parallelism=parallelism,
            generate_strategies=generate_strategies,
            generated_strategy_count=generated_strategy_count,
            max_datapoints=max_datapoints,
            cleanup_memory=cleanup_memory,
            backend=backend,
            target_factory=target_factory,
            error_mapper=error_mapper,
            memory_cleanup=memory_cleanup,
            llm_client=llm_client,
            description=description,
        )
    if mode == 'static':
        return await _run_static(
            target=target,
            categories=categories,
            evaluator_model=evaluator_model,
            parallelism=parallelism,
            max_datapoints=max_datapoints,
            _backend=backend,
            dataset_path=dataset_path,
            description=description,
            llm_client=llm_client,
        )
    if mode == 'hybrid':
        raise NotImplementedError(
            'mode="hybrid" is not yet available. '
            + 'Hybrid mode will compose dynamic + static in a single evaluatorq run.'
        )
    msg = f'Invalid mode {mode!r}. Must be "dynamic", "static", or "hybrid".'
    raise ValueError(msg)


def _parse_target(target: str) -> tuple[str, str]:
    """Parse ``"kind:value"`` target string.

    Returns:
        Tuple of (kind, value), e.g. (``"agent"``, ``"my-agent-key"``).
    """
    if ':' not in target:
        # Default to agent kind
        return 'agent', target
    kind, _, value = target.partition(':')
    if not value:
        msg = f'Target {target!r} is missing a value after the colon.'
        raise ValueError(msg)
    return kind.lower(), value


async def _run_dynamic(
    *,
    target: str,
    categories: list[str] | None,
    max_turns: int,
    max_per_category: int | None,
    attack_model: str,
    evaluator_model: str,
    parallelism: int,
    generate_strategies: bool,
    generated_strategy_count: int,
    max_datapoints: int | None,
    cleanup_memory: bool,
    backend: str,
    target_factory: AgentTargetFactory | None,
    error_mapper: ErrorMapper | None,
    memory_cleanup: MemoryCleanup | None,
    llm_client: AsyncOpenAI | None,
    description: str | None,
) -> RedTeamReport:
    """Run dynamic red teaming via evaluatorq."""
    from evaluatorq import evaluatorq

    from evaluatorq.redteam.backends.base import DefaultErrorMapper
    from evaluatorq.redteam.backends.registry import create_async_llm_client, resolve_backend
    from evaluatorq.redteam.adaptive.pipeline import (
        cleanup_memory_entities,
        create_dynamic_evaluator,
        create_dynamic_redteam_job,
        generate_dynamic_datapoints,
    )
    from evaluatorq.redteam.reports.converters import dynamic_evaluatorq_results_to_report

    _target_kind, target_value = _parse_target(target)
    pipeline_start = datetime.now(tz=timezone.utc).astimezone()

    # Resolve backend
    backend_bundle = resolve_backend(backend, llm_client=llm_client)
    resolved_factory = target_factory or backend_bundle.target_factory
    resolved_error_mapper = error_mapper or DefaultErrorMapper()
    resolved_memory_cleanup = memory_cleanup or backend_bundle.memory_cleanup

    # Get agent context
    agent_context: AgentContext = await backend_bundle.context_provider.get_agent_context(target_value)
    resolved_categories = categories or list_available_categories()

    # Create LLM client for strategy generation
    resolved_llm_client = llm_client
    if resolved_llm_client is None and generate_strategies:
        resolved_llm_client = create_async_llm_client()

    # Stage 1: Generate datapoints
    logger.info(
        f'Generating attack datapoints for {target} '
        + f'({len(resolved_categories)} categories, max_turns={max_turns})'
    )
    datapoints, _filtering_metadata = await generate_dynamic_datapoints(
        agent_context=agent_context,
        categories=resolved_categories,
        max_per_category=max_per_category,
        max_turns=max_turns,
        generate_additional_strategies=generate_strategies,
        generated_strategy_count=generated_strategy_count,
        llm_client=resolved_llm_client,
        attack_model=attack_model,
        parallelism=parallelism,
    )
    if max_datapoints is not None and max_datapoints > 0:
        datapoints = datapoints[:max_datapoints]
    if not datapoints:
        msg = 'No datapoints generated. Check categories and agent capabilities.'
        raise ValueError(msg)

    # Stage 2: Create job and evaluator
    dynamic_job = create_dynamic_redteam_job(
        agent_key=target_value,
        agent_context=agent_context,
        red_team_model=attack_model,
        max_turns=max_turns,
        target_factory=resolved_factory,
        error_mapper=resolved_error_mapper,
        attack_llm_client=resolved_llm_client,
    )
    evaluator = create_dynamic_evaluator(evaluator_model=evaluator_model, llm_client=resolved_llm_client)

    # Stage 3: Run evaluatorq
    logger.info(f'Running {len(datapoints)} attacks against {target} (parallelism={parallelism})')
    try:
        results = await evaluatorq(
            'dynamic-red-team',
            data=datapoints,
            jobs=[dynamic_job],
            evaluators=[evaluator],
            parallelism=parallelism,
            print_results=False,
            description=description or f'Dynamic red teaming for {target}',
        )
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.warning('Run cancelled — attempting memory cleanup')
        if cleanup_memory and agent_context.has_memory:
            try:
                await cleanup_memory_entities(agent_context, [
                    dp.inputs['memory_entity_id']
                    for dp in datapoints
                    if 'memory_entity_id' in dp.inputs
                ], memory_cleanup=resolved_memory_cleanup)
            except Exception as e:
                logger.warning(f'Memory cleanup failed during cancellation: {type(e).__name__}: {e}')
        raise

    pipeline_duration = (datetime.now(tz=timezone.utc).astimezone() - pipeline_start).total_seconds()

    # Stage 4: Normalize results
    logger.info(f'Normalizing results ({pipeline_duration:.1f}s elapsed)')
    report = dynamic_evaluatorq_results_to_report(
        agent_context=agent_context,
        categories_tested=resolved_categories,
        results=results,
        duration_seconds=pipeline_duration,
        description=description or f'Dynamic red teaming for {target}',
    )

    # Stage 5: Cleanup memory entities
    if cleanup_memory and agent_context.has_memory:
        entity_ids = [
            dp.inputs['memory_entity_id']
            for dp in datapoints
            if 'memory_entity_id' in dp.inputs
        ]
        if entity_ids:
            logger.info(f'Cleaning up {len(entity_ids)} memory entities')
            await cleanup_memory_entities(agent_context, entity_ids, memory_cleanup=resolved_memory_cleanup)

    return report


async def _run_static(
    *,
    target: str,
    categories: list[str] | None,
    evaluator_model: str,
    parallelism: int,
    max_datapoints: int | None,
    _backend: str,
    dataset_path: Any,
    description: str | None,
    llm_client: AsyncOpenAI | None = None,
) -> RedTeamReport:
    """Run static red teaming via evaluatorq."""
    from pathlib import Path

    from evaluatorq import evaluatorq

    from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import (
        create_owasp_evaluator,
        load_owasp_agentic_dataset,
    )
    from evaluatorq.redteam.reports.converters import static_evaluatorq_results_to_reports
    from evaluatorq.redteam.runtime.jobs import create_model_job

    target_kind, target_value = _parse_target(target)
    pipeline_start = datetime.now(tz=timezone.utc).astimezone()

    # Load dataset (may use synchronous ORQ SDK internally, so run in thread)
    path = Path(dataset_path) if dataset_path is not None else None
    data = await asyncio.to_thread(
        load_owasp_agentic_dataset,
        num_samples=max_datapoints,
        categories=categories,
        path=path,
    )

    # Create job based on target kind
    if target_kind == 'agent':
        model_job = create_model_job(agent_key=target_value)
    elif target_kind == 'openai':
        model_job = create_model_job(model=target_value, llm_client=llm_client)
    elif target_kind == 'deployment':
        model_job = create_model_job(deployment_key=target_value, llm_client=llm_client)
    else:
        model_job = create_model_job(model=target_value, llm_client=llm_client)

    evaluator = create_owasp_evaluator(evaluator_model=evaluator_model, llm_client=llm_client)

    # Run evaluatorq
    logger.info(f'Running static red teaming against {target} (parallelism={parallelism})')
    try:
        results = await evaluatorq(
            'static-red-team',
            data=data,
            jobs=[model_job],
            evaluators=[evaluator],
            parallelism=parallelism,
            print_results=False,
            description=description or f'Static red teaming for {target}',
        )
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.warning('Static red teaming run cancelled')
        raise

    pipeline_duration = (datetime.now(tz=timezone.utc).astimezone() - pipeline_start).total_seconds()

    # Normalize results — static runs may produce multiple job reports
    reports = static_evaluatorq_results_to_reports(
        results=results,
        agent_model=target_value if target_kind != 'agent' else None,
        agent_key=target_value if target_kind == 'agent' else None,
        description=description or f'Static red teaming for {target}',
    )

    # Return the first (primary) report, or merge if needed
    if not reports:
        from evaluatorq.redteam.reports.converters import static_results_to_report

        return static_results_to_report(
            [],
            agent_model=target_value if target_kind != 'agent' else None,
            agent_key=target_value if target_kind == 'agent' else None,
            description=description,
        )

    primary_report = next(iter(reports.values()))
    primary_report.duration_seconds = pipeline_duration
    return primary_report
