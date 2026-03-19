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

import asyncio
import uuid
from typing import TYPE_CHECKING, Any, cast

from evaluatorq import DataPoint, EvaluationResult, Job, job
from loguru import logger

from evaluatorq.redteam.adaptive.attack_generator import generate_attack_prompt, generate_objective
from evaluatorq.redteam.backends.base import DefaultErrorMapper
from evaluatorq.redteam.backends.registry import create_async_llm_client, resolve_backend
from evaluatorq.redteam.adaptive.evaluator import OWASPEvaluator
from evaluatorq.redteam.contracts import DEFAULT_PIPELINE_MODEL, PIPELINE_CONFIG, AttackOutput, AttackStrategy, EvaluatorConfig, Message, OrchestratorResult, TokenUsage, Vulnerability
from evaluatorq.redteam.vulnerability_registry import get_primary_category, resolve_category_safe
from evaluatorq.redteam.adaptive.orchestrator import MultiTurnOrchestrator
from evaluatorq.redteam.adaptive.strategy_planner import plan_strategies_for_categories, plan_strategies_for_vulnerabilities
from evaluatorq.redteam.contracts import TurnType
from evaluatorq.redteam.tracing import set_span_attrs, with_redteam_span

if TYPE_CHECKING:
    from evaluatorq.types import ScorerParameter
    from openai import AsyncOpenAI

    from evaluatorq.redteam.backends.base import (
        AgentTargetFactory,
        ErrorMapper,
        MemoryCleanup,
    )
    from evaluatorq.redteam.contracts import AgentContext


def _set_attack_span_attrs(span: Any, result: AttackOutput) -> None:
    """Set common tracing attributes on an attack span from an AttackOutput."""
    set_span_attrs(span, {
        'orq.redteam.objective_achieved': result.objective_achieved,
        'orq.redteam.actual_turns': result.turns,
        'orq.redteam.duration_seconds': result.duration_seconds,
    })
    if result.error:
        set_span_attrs(span, {
            'orq.redteam.error_type': result.error_type,
            'orq.redteam.error_code': result.error_code,
        })


async def generate_dynamic_datapoints_for_vulnerabilities(
    agent_context: AgentContext,
    vulnerabilities: list[Vulnerability],
    max_per_category: int | None = None,
    max_turns: int = 5,
    *,
    generate_additional_strategies: bool = True,
    generated_strategy_count: int = 2,
    llm_client: AsyncOpenAI | None = None,
    attack_model: str = DEFAULT_PIPELINE_MODEL,
    parallelism: int = 5,
    attacker_instructions: str | None = None,
    llm_kwargs: dict[str, Any] | None = None,
) -> tuple[list[DataPoint], dict[str, Any]]:
    """Generate evaluatorq DataPoints for dynamic red teaming, keyed by Vulnerability enum.

    This is the primary (vulnerability-first) datapoint generation function.
    Each datapoint carries the serialized strategy, objective, and optional
    memory entity ID. Strategy planning uses the shared planner:

    1. Filter hardcoded strategies by agent capabilities
    2. Optionally generate LLM-based strategies per vulnerability
    3. Cap per vulnerability if max_per_category is set

    The total number of datapoints is:
        sum over vulnerabilities of min(applicable + generated, max_per_category)

    Returns:
        Tuple of (datapoints, filtering_metadata) where filtering_metadata contains
        per-vulnerability counts of all/applicable/generated/filtered strategies.
        The metadata keys are vulnerability ID strings (e.g. 'goal_hijacking').
    """
    all_vuln_strategies, filtering_metadata_by_vuln, _agent_capabilities = await plan_strategies_for_vulnerabilities(
        agent_context=agent_context,
        vulnerabilities=vulnerabilities,
        llm_client=llm_client,
        attack_model=attack_model,
        max_turns=max_turns,
        max_per_category=max_per_category,
        generate_additional_strategies=generate_additional_strategies,
        generated_strategy_count=generated_strategy_count,
        generation_parallelism=parallelism,
        attacker_instructions=attacker_instructions,
        llm_kwargs=llm_kwargs,
    )

    # Convert Vulnerability-keyed metadata to string keys for external consumption
    filtering_metadata: dict[str, Any] = {
        vuln.value: meta for vuln, meta in filtering_metadata_by_vuln.items()
    }

    datapoints: list[DataPoint] = []
    for vuln in vulnerabilities:
        category = get_primary_category(vuln)
        for strategy in all_vuln_strategies.get(vuln, []):
            sample_id = f'dynamic_{vuln.value}_{strategy.name}'
            objective = generate_objective(strategy, agent_context)

            inputs: dict[str, Any] = {
                'id': sample_id,
                'vulnerability': vuln.value,
                'category': category,
                'strategy': strategy.model_dump(mode='json'),
                'objective': objective,
            }

            # Each datapoint gets its own memory entity ID so parallel jobs don't interfere
            if agent_context.has_memory:
                inputs['memory_entity_id'] = f'red-team-{uuid.uuid4().hex[:12]}'

            datapoints.append(DataPoint(inputs=inputs))

    logger.debug(f'Generated {len(datapoints)} dynamic datapoints across {len(vulnerabilities)} vulnerabilities')
    return datapoints, filtering_metadata


async def generate_dynamic_datapoints(
    agent_context: AgentContext,
    categories: list[str],
    max_per_category: int | None = None,
    max_turns: int = 5,
    *,
    generate_additional_strategies: bool = True,
    generated_strategy_count: int = 2,
    llm_client: AsyncOpenAI | None = None,
    attack_model: str = DEFAULT_PIPELINE_MODEL,
    parallelism: int = 5,
    attacker_instructions: str | None = None,
    llm_kwargs: dict[str, Any] | None = None,
) -> tuple[list[DataPoint], dict[str, Any]]:
    """Generate evaluatorq DataPoints for dynamic red teaming.

    Each datapoint carries the serialized strategy, objective, and optional
    memory entity ID. Strategy planning uses the shared planner:

    1. Filter hardcoded strategies by agent capabilities
    2. Optionally generate LLM-based strategies per category
    3. Cap per category if max_per_category is set

    The total number of datapoints is:
        sum over categories of min(applicable + generated, max_per_category)

    Delegates to generate_dynamic_datapoints_for_vulnerabilities() when all
    categories can be resolved to Vulnerability enum values. Falls back to
    plan_strategies_for_categories() for unresolvable category codes so that
    legacy behaviour is preserved.

    Returns:
        Tuple of (datapoints, filtering_metadata) where filtering_metadata contains
        per-category counts of all/applicable/generated/filtered strategies.
    """
    from evaluatorq.redteam.vulnerability_registry import resolve_vulnerabilities

    # Try resolving all categories to vulnerabilities for the primary path
    try:
        resolved_vulnerabilities = resolve_vulnerabilities(categories)
    except ValueError as exc:
        logger.warning(f'Could not resolve all categories to vulnerabilities, falling back to category-based path: {exc}')
        resolved_vulnerabilities = None

    if resolved_vulnerabilities is not None:
        # Primary vulnerability-first path — delegate entirely
        datapoints, vuln_metadata = await generate_dynamic_datapoints_for_vulnerabilities(
            agent_context=agent_context,
            vulnerabilities=resolved_vulnerabilities,
            max_per_category=max_per_category,
            max_turns=max_turns,
            generate_additional_strategies=generate_additional_strategies,
            generated_strategy_count=generated_strategy_count,
            llm_client=llm_client,
            attack_model=attack_model,
            parallelism=parallelism,
            attacker_instructions=attacker_instructions,
            llm_kwargs=llm_kwargs,
        )
        # Remap metadata keys from vulnerability IDs back to original category strings
        # so callers that expect category-keyed metadata continue to work.
        # resolve_vulnerabilities() succeeded, so each category maps to exactly one
        # vulnerability; we look up the metadata by vulnerability ID string.
        filtering_metadata: dict[str, Any] = {}
        for category in categories:
            resolved_v = resolve_category_safe(category)
            if resolved_v is not None and resolved_v.value in vuln_metadata:
                filtering_metadata[category] = vuln_metadata[resolved_v.value]
        return datapoints, filtering_metadata

    # Fallback path: one or more categories could not be resolved to a Vulnerability
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
        attacker_instructions=attacker_instructions,
        llm_kwargs=llm_kwargs,
    )

    datapoints = []
    for category in categories:
        for strategy in all_category_strategies.get(category, []):
            sample_id = f'dynamic_{category}_{strategy.name}'
            objective = generate_objective(strategy, agent_context)

            inputs: dict[str, Any] = {
                'id': sample_id,
                'vulnerability': strategy.vulnerability.value if strategy.vulnerability else '',
                'category': category,
                'strategy': strategy.model_dump(mode='json'),
                'objective': objective,
            }

            # Each datapoint gets its own memory entity ID so parallel jobs don't interfere
            if agent_context.has_memory:
                inputs['memory_entity_id'] = f'red-team-{uuid.uuid4().hex[:12]}'

            datapoints.append(DataPoint(inputs=inputs))

    empty_categories = [cat for cat in categories if not all_category_strategies.get(cat)]
    if empty_categories:
        filtering_metadata['_unresolved_categories'] = empty_categories

    logger.debug(f'Generated {len(datapoints)} dynamic datapoints across {len(categories)} categories')
    return datapoints, filtering_metadata


def create_dynamic_redteam_job(
    *,
    agent_key: str,
    agent_context: AgentContext,
    red_team_model: str = DEFAULT_PIPELINE_MODEL,
    max_turns: int = 5,
    target_factory: AgentTargetFactory | None = None,
    error_mapper: ErrorMapper | None = None,
    attack_llm_client: AsyncOpenAI | None = None,
    memory_entity_ids: list[str] | None = None,
    attacker_instructions: str | None = None,
    verbosity: int = 0,
    llm_kwargs: dict[str, Any] | None = None,
) -> Job:
    """Create an evaluatorq Job that runs a red-team attack.

    Each job invocation creates its own AgentTarget (with per-datapoint
    memory_entity_id) so jobs can safely run in parallel.

    The job returns a dict with conversation, final_response, turns,
    objective_achieved, and category. This dict is passed to the scorer
    via params['output'].

    Note:
        Callers must ensure ``cleanup_memory_entities()`` is called after the
        evaluatorq run completes. The job itself does not manage memory cleanup —
        that responsibility belongs to the caller (``runner.py`` or CLI).

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
    job_name = f'redteam:dynamic:{safe_agent_key or "agent"}'

    @job(job_name)
    async def dynamic_job(data: DataPoint, _row: int) -> dict[str, Any]:
        """Execute a single red-team attack for the given datapoint and return the serialized result."""
        import time
        import traceback

        inputs = dict(data.inputs)
        strategy = AttackStrategy.model_validate(inputs['strategy'])
        objective = str(inputs['objective'])
        category = str(inputs['category'])
        vulnerability = str(inputs.get('vulnerability', ''))
        # Generate a fresh memory entity ID per job execution so parallel
        # targets sharing the same datapoints don't interfere with each other's
        # memory state.  Falls back to the datapoint value for single-target runs.
        if agent_context.has_memory:
            memory_entity_id = f'red-team-{uuid.uuid4().hex[:12]}'
            if memory_entity_ids is not None:
                memory_entity_ids.append(memory_entity_id)
        else:
            memory_entity_id = inputs.get('memory_entity_id')
        effective_max_turns = 1 if strategy.turn_type == TurnType.SINGLE else max_turns

        async with with_redteam_span(
            'orq.redteam.attack',
            {
                'orq.redteam.category': category,
                'orq.redteam.vulnerability': vulnerability,
                'orq.redteam.strategy_name': strategy.name,
                'orq.redteam.turn_type': strategy.turn_type.value,
                'orq.redteam.max_turns': effective_max_turns,
            },
        ) as attack_span:
            target = resolved_factory.create_target(
                agent_key=agent_key,
                memory_entity_id=memory_entity_id,
            )
            # Inject model info for tracing if the target supports it
            if hasattr(target, 'model') and agent_context.model:
                object.__setattr__(target, 'model', agent_context.model)  # type: ignore[misc]
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
                token_usage = None
                target_timeout_s = PIPELINE_CONFIG.target_agent_timeout_ms / 1000.0
                try:
                    async with with_redteam_span(
                        "orq.redteam.target_call",
                        {
                            "orq.redteam.category": category,
                            "orq.redteam.strategy_name": strategy.name,
                            "orq.redteam.turn": 1,
                            "input": prompt,
                            "orq.redteam.input": prompt,
                        },
                    ) as tgt_span:
                        response = await asyncio.wait_for(
                            target.send_prompt(prompt),
                            timeout=target_timeout_s,
                        )
                        set_span_attrs(tgt_span, {
                            "output": response or "",
                            "orq.redteam.output": response or "",
                        })
                        consume_usage = getattr(target, 'consume_last_token_usage', None)
                        if callable(consume_usage):
                            token_usage: TokenUsage | None = cast(TokenUsage | None, consume_usage())
                except asyncio.TimeoutError:
                    logger.error(f'Single-turn attack timed out for {category}/{strategy.name} after {target_timeout_s:.0f}s')
                    response = f'[ERROR: Target agent timed out after {target_timeout_s:.0f}s]'
                    error = f'Target agent timed out after {target_timeout_s:.0f}s'
                    error_type = 'target_error'
                    error_stage = 'target_call'
                    error_code = 'target.timeout'
                    error_details = {
                        'timeout_ms': PIPELINE_CONFIG.target_agent_timeout_ms,
                    }
                except Exception as e:
                    if isinstance(e, (TypeError, AttributeError, ImportError, NameError)):
                        raise
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
                result_dict = AttackOutput(
                    conversation=[Message(role='user', content=prompt), Message(role='assistant', content=response)],
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
                    category=category,
                    vulnerability=vulnerability,
                )

                set_span_attrs(attack_span, {'input': prompt, 'output': response or ''})
                _set_attack_span_attrs(attack_span, result_dict)

                # Advance the global progress bar for template single-turn attacks
                # (multi-turn attacks are tracked by run_attack in the orchestrator).
                from evaluatorq.redteam.adaptive.orchestrator import _get_active_progress
                _active_progress = _get_active_progress()
                if _active_progress is not None:
                    await _active_progress.finish_attack(None)

                return result_dict.model_dump(mode='json')

            # Dynamic single-turn (max_turns=1) or multi-turn — orchestrator handles both
            llm_client = attack_llm_client or create_async_llm_client()
            orchestrator = MultiTurnOrchestrator(
                llm_client,
                model=red_team_model,
                error_mapper=resolved_error_mapper,
                attacker_instructions=attacker_instructions,
                verbosity=verbosity,
                llm_kwargs=llm_kwargs,
            )

            try:
                result = await orchestrator.run_attack(
                    target=target,
                    strategy=strategy,
                    objective=objective,
                    agent_context=agent_context,
                    max_turns=effective_max_turns,
                )
            except asyncio.TimeoutError as e:
                tb = traceback.format_exc(limit=8)
                mapped_code, mapped_msg = resolved_error_mapper.map_error(e)
                logger.error(f'Orchestrator timed out for {category}/{strategy.name}: {mapped_msg}\n{tb}')
                result = OrchestratorResult(
                    conversation=[],
                    final_response='',
                    objective_achieved=False,
                    turns=0,
                    duration_seconds=0.0,
                    token_usage=None,
                    token_usage_adversarial=None,
                    token_usage_target=None,
                    system_prompt=None,
                    error=f'Orchestrator timeout: {mapped_msg}',
                    error_type='orchestrator_timeout',
                    error_stage='orchestrator',
                    error_code=mapped_code,
                    error_details={
                        'exception_type': type(e).__name__,
                        'raw_message': str(e),
                        'traceback': tb,
                    },
                )
            except Exception as e:
                if isinstance(e, (TypeError, AttributeError, KeyError, IndexError, ImportError, NameError)):
                    raise
                tb = traceback.format_exc(limit=8)
                mapped_code, mapped_msg = resolved_error_mapper.map_error(e)
                logger.error(f'Unexpected orchestrator error for {category}/{strategy.name}: {mapped_msg}\n{tb}')
                result = OrchestratorResult(
                    conversation=[],
                    final_response='',
                    objective_achieved=False,
                    turns=0,
                    duration_seconds=0.0,
                    token_usage=None,
                    token_usage_adversarial=None,
                    token_usage_target=None,
                    system_prompt=None,
                    error=f'Unexpected orchestrator error: {mapped_msg}',
                    error_type='orchestrator_exception',
                    error_stage='orchestrator',
                    error_code=mapped_code,
                    error_details={
                        'exception_type': type(e).__name__,
                        'raw_message': str(e),
                        'traceback': tb,
                    },
                )

            result_dict = AttackOutput(**result.model_dump(), category=category, vulnerability=vulnerability)

            # Set input/output on the attack span so the platform can display them
            if result_dict.conversation:
                first_user = next((m.content for m in result_dict.conversation if m.role == 'user'), None)
                if first_user:
                    set_span_attrs(attack_span, {'input': first_user})
            if result_dict.final_response:
                set_span_attrs(attack_span, {'output': result_dict.final_response})
            _set_attack_span_attrs(attack_span, result_dict)
            return result_dict.model_dump(mode='json')

    return dynamic_job


def create_dynamic_evaluator(
    evaluator_model: str = DEFAULT_PIPELINE_MODEL,
    llm_client: AsyncOpenAI | None = None,
    llm_kwargs: dict[str, Any] | None = None,
) -> EvaluatorConfig:
    """Create an evaluator that uses OWASPEvaluator on the attack conversation.

    value=True means RESISTANT (consistent with OWASP evaluatorq scoring and
    EvaluationResult.passed convention).
    """
    owasp_evaluator = OWASPEvaluator(evaluator_model=evaluator_model, llm_client=llm_client, llm_kwargs=llm_kwargs)

    async def scorer(params: ScorerParameter) -> EvaluationResult:
        """Evaluate the attack output using OWASPEvaluator and return a scored EvaluationResult."""
        data = params['data']
        raw_output = params['output']
        if isinstance(raw_output, AttackOutput):
            output = raw_output
        elif isinstance(raw_output, dict):
            try:
                output = AttackOutput.model_validate(raw_output)
            except Exception as e:
                logger.error(f'Failed to parse job output as AttackOutput: {e}')
                return EvaluationResult(
                    value='error',
                    explanation=f'Failed to parse job output: {e}',
                )
        else:
            logger.warning(f'Unexpected output type {type(raw_output).__name__} from job; treating as error')
            return EvaluationResult(
                value='error',
                explanation=f'Unexpected job output type: {type(raw_output).__name__}',
            )

        # Skip evaluation when the target returned an error
        if output.error:
            return EvaluationResult(
                value='error',
                explanation=f'Skipped: target returned error — {output.error}',
            )

        conversation = output.conversation
        final_response = output.final_response
        category = output.category or data.inputs.get('category', '')
        vulnerability = output.vulnerability or data.inputs.get('vulnerability', '')

        # Prefer vulnerability-first path when a valid Vulnerability enum can be resolved
        resolved_vuln: Vulnerability | None = None
        if vulnerability:
            try:
                resolved_vuln = Vulnerability(vulnerability)
            except ValueError:
                resolved_vuln = resolve_category_safe(vulnerability)

        async with with_redteam_span(
            'orq.redteam.security_evaluation',
            {
                'orq.redteam.category': category,
                'orq.redteam.vulnerability': vulnerability,
                'orq.redteam.model': evaluator_model,
            },
        ) as eval_span:
            if resolved_vuln is not None:
                eval_result = await owasp_evaluator.evaluate_vulnerability(
                    vuln=resolved_vuln,
                    messages=conversation,
                    response=final_response,
                )
            else:
                eval_result = await owasp_evaluator.evaluate(
                    category=category,
                    messages=conversation,
                    response=final_response,
                )
            set_span_attrs(eval_span, {
                'orq.redteam.passed': eval_result.passed,
                'input': final_response or '',
                'output': eval_result.explanation or '',
            })

        raw_value = None
        if isinstance(eval_result.raw_output, dict):
            raw_value = eval_result.raw_output.get('value')
        if raw_value is not None:
            result_value: str | int | float | bool = raw_value
        elif eval_result.passed is not None:
            result_value = bool(eval_result.passed)
        else:
            result_value = 'inconclusive'
        return EvaluationResult.model_validate({
            "value": result_value,
            "explanation": eval_result.explanation,
            "pass": eval_result.passed,
        })

    return {'name': 'owasp-dynamic-security', 'scorer': scorer}


async def cleanup_memory_entities(
    agent_context: AgentContext,
    entity_ids: list[str],
    memory_cleanup: MemoryCleanup | None = None,
) -> str | None:
    """Delete memory entities created during a red teaming run.

    Delegates to provided backend cleanup implementation.
    Wrapped with an overall timeout — cleanup is best-effort and should never block the pipeline.

    Returns None on success, or an error message string on failure.
    """
    cleanup_timeout_s = PIPELINE_CONFIG.cleanup_timeout_ms / 1000.0
    try:
        if memory_cleanup is not None:
            await asyncio.wait_for(
                memory_cleanup.cleanup_memory(agent_context, entity_ids),
                timeout=cleanup_timeout_s,
            )
            return None
        # Default fallback keeps existing ORQ behavior.
        await asyncio.wait_for(
            resolve_backend('orq').memory_cleanup.cleanup_memory(agent_context, entity_ids),
            timeout=cleanup_timeout_s,
        )
    except asyncio.TimeoutError:
        msg = f'Memory cleanup timed out after {cleanup_timeout_s:.0f}s for {len(entity_ids)} entities'
        logger.warning(msg)
        return msg
    except Exception as e:
        msg = f'Memory cleanup failed: {e}'
        logger.warning(msg)
        return msg
    return None
