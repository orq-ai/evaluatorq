"""Dynamic red teaming pipeline primitives for evaluatorq execution.

This module is the canonical home for dynamic evaluatorq integration:
datapoint generation, job/scorer construction, and memory cleanup.

Flow:
  generate_dynamic_datapoints() -> DataPoints with strategy + objective + memory_entity_id
  create_dynamic_redteam_job()  -> Job that runs attack, returns dict with conversation
  create_dynamic_evaluator()    -> Scorer that calls OWASPEvaluator on conversation
  cleanup_memory_entities()     -> Post-run cleanup of memory entities created during attacks
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, TypedDict, cast

from evaluatorq import DataPoint, EvaluationResult, Job, job
from loguru import logger

from evaluatorq.redteam.adaptive.attack_generator import generate_attack_prompt, generate_objective
from evaluatorq.redteam.backends.base import DefaultErrorMapper
from evaluatorq.redteam.backends.registry import create_async_llm_client, resolve_backend
from evaluatorq.redteam.adaptive.evaluator import OWASPEvaluator
from evaluatorq.redteam.contracts import AttackStrategy, Message, OrchestratorResult, TokenUsage
from evaluatorq.redteam.adaptive.orchestrator import MultiTurnOrchestrator
from evaluatorq.redteam.adaptive.strategy_planner import plan_strategies_for_categories
from evaluatorq.redteam.contracts import TurnType

if TYPE_CHECKING:
    from evaluatorq.types import ScorerParameter
    from openai import AsyncOpenAI

    from evaluatorq.redteam.backends.base import (
        AgentTargetFactory,
        ErrorMapper,
        MemoryCleanup,
    )
    from evaluatorq.redteam.contracts import AgentContext


class EvaluatorConfig(TypedDict):
    """Minimal evaluatorq evaluator contract."""

    name: str
    scorer: Any


async def generate_dynamic_datapoints(
    agent_context: AgentContext,
    categories: list[str],
    max_per_category: int | None = None,
    max_turns: int = 5,
    *,
    generate_additional_strategies: bool = True,
    generated_strategy_count: int = 2,
    llm_client: AsyncOpenAI | None = None,
    attack_model: str = 'azure/gpt-5-mini',
    parallelism: int = 5,
) -> tuple[list[DataPoint], dict[str, Any]]:
    """Generate evaluatorq DataPoints for dynamic red teaming.

    Each datapoint carries the serialized strategy, objective, and optional
    memory entity ID. Strategy planning uses the shared planner:

    1. Filter hardcoded strategies by agent capabilities
    2. Optionally generate LLM-based strategies per category
    3. Cap per category if max_per_category is set

    The total number of datapoints is:
        sum over categories of min(applicable + generated, max_per_category)

    Returns:
        Tuple of (datapoints, filtering_metadata) where filtering_metadata contains
        per-category counts of all/applicable/generated/filtered strategies.
    """
    all_category_strategies, filtering_metadata, _agent_capabilities = await plan_strategies_for_categories(
        agent_context=agent_context,
        categories=categories,
        llm_client=llm_client,
        attack_model=attack_model,
        max_turns=max_turns,
        max_per_category=max_per_category,
        generate_additional_strategies=generate_additional_strategies,
        generated_strategy_count=generated_strategy_count,
        generation_parallelism=parallelism,
    )

    datapoints: list[DataPoint] = []
    for category in categories:
        for strategy in all_category_strategies.get(category, []):
            sample_id = f'dynamic_{category}_{strategy.name}'
            objective = generate_objective(strategy, agent_context)

            inputs: dict[str, Any] = {
                'id': sample_id,
                'category': category,
                'strategy': strategy.model_dump(mode='json'),
                'objective': objective,
            }

            # Each datapoint gets its own memory entity ID so parallel jobs don't interfere
            if agent_context.has_memory:
                inputs['memory_entity_id'] = f'red-team-{uuid.uuid4().hex[:12]}'

            datapoints.append(DataPoint(inputs=inputs))

    logger.info(f'Generated {len(datapoints)} dynamic datapoints across {len(categories)} categories')
    return datapoints, filtering_metadata


def create_dynamic_redteam_job(
    *,
    agent_key: str,
    agent_context: AgentContext,
    red_team_model: str = 'azure/gpt-5-mini',
    max_turns: int = 5,
    target_factory: AgentTargetFactory | None = None,
    error_mapper: ErrorMapper | None = None,
    attack_llm_client: AsyncOpenAI | None = None,
) -> Job:
    """Create an evaluatorq Job that runs a red-team attack.

    Each job invocation creates its own AgentTarget (with per-datapoint
    memory_entity_id) so jobs can safely run in parallel.

    The job returns a dict with conversation, final_response, turns,
    objective_achieved, and category. This dict is passed to the scorer
    via params['output'].

    Args:
        agent_key: Agent key to test
        agent_context: Pre-retrieved agent context
        red_team_model: Model for adversarial prompt generation
        max_turns: Maximum turns for multi-turn attacks
        target_factory: Factory for creating AgentTarget instances. Defaults to ORQ.
    """
    resolved_factory: AgentTargetFactory = (
        target_factory if target_factory is not None else resolve_backend('orq').target_factory
    )
    resolved_error_mapper = error_mapper or DefaultErrorMapper()
    safe_agent_key = ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '-' for ch in agent_key).strip('-')
    job_name = f'dynamic:redteam:agent:{safe_agent_key or "agent"}'

    @job(job_name)
    async def dynamic_job(data: DataPoint, _row: int) -> dict[str, Any]:
        import time
        import traceback

        inputs = dict(data.inputs)
        strategy = AttackStrategy.model_validate(inputs['strategy'])
        objective = str(inputs['objective'])
        category = str(inputs['category'])
        memory_entity_id = inputs.get('memory_entity_id')

        target = resolved_factory.create_target(
            agent_key=agent_key,
            memory_entity_id=memory_entity_id,
        )
        target.reset_conversation()

        if strategy.prompt_template and strategy.turn_type == TurnType.SINGLE:
            # Fixed template: fill and send directly (no adversarial LLM needed)
            t0 = time.time()
            prompt = generate_attack_prompt(strategy, agent_context)
            error = None
            error_type = None
            error_stage = None
            error_code = None
            error_details = None
            token_usage: TokenUsage | None = None
            try:
                response = await target.send_prompt(prompt)
                consume_usage = getattr(target, 'consume_last_token_usage', None)
                if callable(consume_usage):
                    raw_usage = consume_usage()
                    token_usage = raw_usage if isinstance(raw_usage, TokenUsage) else None
            except Exception as e:
                mapped_code, mapped_msg = resolved_error_mapper.map_error(e)
                logger.error(f'Single-turn attack failed for {category}/{strategy.name}: {mapped_msg}')
                response = f'[ERROR: {mapped_msg}]'
                error = mapped_msg
                error_type = 'target_error'
                error_stage = 'target_call'
                error_code = mapped_code
                error_details = {
                    'exception_type': type(e).__name__,
                    'raw_message': str(e),
                }
            return OrchestratorResult(
                conversation=[
                    Message(role='user', content=prompt),
                    Message(role='assistant', content=response),
                ],
                final_response=response,
                objective_achieved=False,
                turns=1,
                duration_seconds=time.time() - t0,
                token_usage=token_usage,
                token_usage_adversarial=None,
                token_usage_target=token_usage,
                system_prompt=None,
                error=error,
                error_type=error_type,
                error_stage=error_stage,
                error_code=error_code,
                error_details=error_details,
            ).model_dump(mode='json') | {'category': category}

        # Dynamic single-turn (max_turns=1) or multi-turn — orchestrator handles both
        llm_client = attack_llm_client or create_async_llm_client()
        orchestrator = MultiTurnOrchestrator(
            llm_client,
            model=red_team_model,
            error_mapper=resolved_error_mapper,
        )
        effective_max_turns = 1 if strategy.turn_type == TurnType.SINGLE else max_turns

        try:
            result = await orchestrator.run_attack(
                target=target,
                strategy=strategy,
                objective=objective,
                agent_context=agent_context,
                max_turns=effective_max_turns,
            )
        except Exception as e:
            tb = traceback.format_exc(limit=8)
            mapped_code, mapped_msg = resolved_error_mapper.map_error(e)
            logger.error(f'Orchestrator failed for {category}/{strategy.name}: {mapped_msg}\n{tb}')
            result = OrchestratorResult(
                conversation=[],
                final_response='',
                objective_achieved=False,
                turns=1,
                duration_seconds=0.0,
                token_usage=None,
                token_usage_adversarial=None,
                token_usage_target=None,
                system_prompt=None,
                error=f'Orchestrator exception: {mapped_msg}',
                error_type='orchestrator_exception',
                error_stage='orchestrator',
                error_code=mapped_code,
                error_details={
                    'exception_type': type(e).__name__,
                    'raw_message': str(e),
                    'traceback': tb,
                },
            )

        return result.model_dump(mode='json') | {'category': category}

    return dynamic_job


def create_dynamic_evaluator(
    evaluator_model: str = 'azure/gpt-5-mini',
    llm_client: AsyncOpenAI | None = None,
) -> EvaluatorConfig:
    """Create an evaluator that uses OWASPEvaluator on the attack conversation.

    value=True means RESISTANT (consistent with OWASP evaluatorq scoring and
    EvaluationResult.passed convention).
    """
    owasp_evaluator = OWASPEvaluator(evaluator_model=evaluator_model, llm_client=llm_client)

    async def scorer(params: ScorerParameter) -> EvaluationResult:
        data = params['data']
        output: dict[str, Any] = cast('dict[str, Any]', params['output'])  # job returns dict

        # Skip evaluation when the target returned an error
        if output.get('error'):
            error_msg = output.get('error', 'unknown error')
            return EvaluationResult(
                value='error',
                explanation=f'Skipped: target returned error — {error_msg}',
            )

        conversation = output.get('conversation', [])
        final_response = output.get('final_response', '')
        category = output.get('category', '') or data.inputs.get('category', '')

        eval_result = await owasp_evaluator.evaluate(
            category=category,
            messages=conversation,
            response=final_response,
        )

        raw_value = None
        if isinstance(eval_result.raw_output, dict):
            raw_value = eval_result.raw_output.get('value')
        value: str | int | float | bool = raw_value if raw_value is not None else (eval_result.passed if eval_result.passed is not None else 'error')
        return EvaluationResult(
            value=value,
            explanation=eval_result.explanation,
            **{"pass": eval_result.passed},
        )

    return {'name': 'owasp-dynamic-security', 'scorer': scorer}


async def cleanup_memory_entities(
    agent_context: AgentContext,
    entity_ids: list[str],
    memory_cleanup: MemoryCleanup | None = None,
) -> None:
    """Delete memory entities created during an evaluatorq red teaming run.

    Delegates to provided backend cleanup implementation.
    """
    if memory_cleanup is not None:
        await memory_cleanup.cleanup_memory(agent_context, entity_ids)
        return
    # Default fallback keeps existing ORQ behavior.
    await resolve_backend('orq').memory_cleanup.cleanup_memory(agent_context, entity_ids)
