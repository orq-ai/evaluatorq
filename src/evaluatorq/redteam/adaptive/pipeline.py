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
import contextlib
import time
import traceback
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq import DataPoint, EvaluationResult, Job, job
from evaluatorq.common.jury import append_jury_summary, attach_jury_raw_output
from evaluatorq.common.messages import coerce_content_text
from evaluatorq.common.target_call import call_target_with_retry, close_target
from evaluatorq.common.thread_context import build_thread_id, conversation_thread
from evaluatorq.common.tracing import set_span_attrs, truncate_for_span
from evaluatorq.contracts import AgentResponse, Message
from evaluatorq.redteam.adaptive.attack_generator import generate_attack_prompt, generate_objective
from evaluatorq.redteam.adaptive.evaluator import OWASPEvaluator
from evaluatorq.redteam.adaptive.orchestrator import (
    MultiTurnOrchestrator,
    _get_active_progress,
    _merge_usage,
    replay_seed_context,
)
from evaluatorq.redteam.adaptive.strategy_planner import (
    plan_strategies_for_categories,
    plan_strategies_for_vulnerabilities,
)
from evaluatorq.redteam.backends.registry import create_async_llm_client, resolve_backend
from evaluatorq.redteam.contracts import (
    DEFAULT_PIPELINE_MODEL,
    PIPELINE_CONFIG,
    AttackOutput,
    AttackStrategy,
    AttackTechnique,
    DeliveryMethod,
    EvaluatorConfig,
    EvaluatorqEvaluatorConfig,
    JuryResult,
    LLMCallConfig,
    LLMConfig,
    OrchestratorResult,
    TextOutputItem,
    ToolCallOutputItem,
    Turn,
    TurnType,
    Vulnerability,
)
from evaluatorq.redteam.traces import TRACE_SEED_MESSAGES_KEY, TRACE_START_FROM_KEY, TraceStart, parse_trace_seed
from evaluatorq.redteam.tracing import annotate_current_span, set_jury_span_attrs, with_redteam_span
from evaluatorq.redteam.vulnerability_registry import (
    get_primary_category,
    resolve_category_safe,
    resolve_vulnerabilities,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from evaluatorq.redteam.adaptive.capability_classifier import AgentCapabilities
    from evaluatorq.redteam.backends.base import Backend
    from evaluatorq.redteam.contracts import AgentContext
    from evaluatorq.types import ScorerParameter


def _set_attack_span_attrs(span: Any, result: AttackOutput) -> None:
    """Set common tracing attributes on an attack span from an AttackOutput."""
    set_span_attrs(
        span,
        {
            'orq.redteam.objective_achieved': result.objective_achieved,
            'orq.redteam.actual_turns': result.n_turns,
            'orq.redteam.duration_seconds': result.duration_seconds,
        },
    )
    if result.objective_rationale:
        # Attacker self-report (signal, not evidence) — recorded for auditing only.
        set_span_attrs(span, {'orq.redteam.objective_rationale': result.objective_rationale[:500]})
    if result.error:
        set_span_attrs(
            span,
            {
                'orq.redteam.error_type': result.error_type,
                'orq.redteam.error_code': result.error_code,
            },
        )


def _turn_has_content(turn: Turn) -> bool:
    """True when the target reply has real content to judge: a tool call (often the
    vulnerability itself for agent targets) or non-blank text. Reasoning-only or
    empty-body replies (e.g. a silently-blocked 200) return False — feeding them to the
    judge would score a non-response as RESISTANT. Mirrors ``turns_to_messages``, which
    drops reasoning items when building a transcript.
    """
    return any(
        isinstance(item, ToolCallOutputItem) or (isinstance(item, TextOutputItem) and bool(item.text.strip()))
        for item in turn.target.output
    )


def _error_codes(turns: list[Turn]) -> str:
    """Comma-separated sorted set of target error codes across ``turns``, or ``'none'``."""
    return ', '.join(sorted({t.target.error.code for t in turns if t.target.error and t.target.error.code})) or 'none'


# The only trace-derived keys anything in `redteam/` ever reads back off an
# expanded seed x strategy row. `Trace.to_datapoint()` puts far more than this on
# a seed (the full production transcript, `recorded_output`, `retrievals`,
# `trace_metadata`, ...) — carrying all of it into every cross-product row
# multiplies a handful of traces by dozens of strategies into hundreds of
# in-memory copies of data nothing downstream reads.
_TRACE_SEED_PROJECTED_KEYS = (TRACE_SEED_MESSAGES_KEY, TRACE_START_FROM_KEY, 'source_trace_id')


def expand_trace_seed_datapoints(seeds: list[DataPoint] | None, dynamic_datapoints: list[DataPoint]) -> list[DataPoint]:
    """Cross each trace seed with the already-selected attack strategies.

    Only the three keys anything downstream reads are carried from the seed
    (see `_TRACE_SEED_PROJECTED_KEYS`); the seed's other fields (the full
    imported transcript, recorded output, retrievals, ...) are dropped rather
    than copied into every row. The seed slice wins the merge — caller-supplied
    values win merges, and the seed is what identifies which trace this row
    replays, not the strategy.
    """
    if seeds is None:
        return dynamic_datapoints
    expanded: list[DataPoint] = []
    for seed_index, seed in enumerate(seeds):
        seed_slice = {key: seed.inputs[key] for key in _TRACE_SEED_PROJECTED_KEYS if key in seed.inputs}
        for attack in dynamic_datapoints:
            inputs = {**attack.inputs, **seed_slice}
            seed_id = seed.inputs.get('source_trace_id') or seed.inputs.get('id') or 'seed'
            attack_id = attack.inputs.get('id', 'attack')
            inputs['id'] = f'trace_{seed_id}_{seed_index}_{attack_id}'
            expanded.append(DataPoint(inputs=inputs, expected_output=attack.expected_output))
    return expanded


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
    datapoint_parallelism: int = 5,
    attacker_instructions: str | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    pipeline_config: LLMConfig | None = None,
    agent_capabilities: AgentCapabilities | None = None,
    strategy_names: set[str] | None = None,
    delivery_methods: set[DeliveryMethod | str] | None = None,
    attack_techniques: set[AttackTechnique] | None = None,
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
    cfg = pipeline_config or PIPELINE_CONFIG
    all_vuln_strategies, filtering_metadata_by_vuln, _agent_capabilities = await plan_strategies_for_vulnerabilities(
        agent_context=agent_context,
        vulnerabilities=vulnerabilities,
        llm_client=llm_client,
        attack_model=attack_model,
        max_turns=max_turns,
        max_per_category=max_per_category,
        generate_additional_strategies=generate_additional_strategies,
        generated_strategy_count=generated_strategy_count,
        generation_parallelism=datapoint_parallelism,
        attacker_instructions=attacker_instructions,
        llm_kwargs=llm_kwargs,
        pipeline_config=cfg,
        agent_capabilities=agent_capabilities,
        strategy_names=strategy_names,
        delivery_methods=delivery_methods,
        attack_techniques=attack_techniques,
    )

    # Convert Vulnerability-keyed metadata to string keys for external consumption
    filtering_metadata: dict[str, Any] = {vuln.value: meta for vuln, meta in filtering_metadata_by_vuln.items()}

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
    datapoint_parallelism: int = 5,
    attacker_instructions: str | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    pipeline_config: LLMConfig | None = None,
    agent_capabilities: AgentCapabilities | None = None,
    strategy_names: set[str] | None = None,
    delivery_methods: set[DeliveryMethod | str] | None = None,
    attack_techniques: set[AttackTechnique] | None = None,
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
    cfg = pipeline_config or PIPELINE_CONFIG
    # Try resolving all categories to vulnerabilities for the primary path
    try:
        resolved_vulnerabilities = resolve_vulnerabilities(categories)
    except ValueError as exc:
        logger.warning(
            f'Could not resolve all categories to vulnerabilities, falling back to category-based path: {exc}'
        )
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
            datapoint_parallelism=datapoint_parallelism,
            attacker_instructions=attacker_instructions,
            llm_kwargs=llm_kwargs,
            pipeline_config=cfg,
            agent_capabilities=agent_capabilities,
            strategy_names=strategy_names,
            delivery_methods=delivery_methods,
            attack_techniques=attack_techniques,
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
        generation_parallelism=datapoint_parallelism,
        attacker_instructions=attacker_instructions,
        llm_kwargs=llm_kwargs,
        pipeline_config=cfg,
        agent_capabilities=agent_capabilities,
        strategy_names=strategy_names,
        delivery_methods=delivery_methods,
        attack_techniques=attack_techniques,
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

            datapoints.append(DataPoint(inputs=inputs))

    empty_categories = [cat for cat in categories if not all_category_strategies.get(cat)]
    if empty_categories:
        filtering_metadata['_unresolved_categories'] = empty_categories

    logger.debug(f'Generated {len(datapoints)} dynamic datapoints across {len(categories)} categories')
    return datapoints, filtering_metadata


def _trace_seed_from_inputs(inputs: dict[str, Any]) -> tuple[list[Message] | None, TraceStart | None]:
    """Validate and deserialize optional trace replay state from a datapoint.

    Delegates to the shared `parse_trace_seed` (`redteam/traces.py`) so this
    per-attack read can never disagree with the `red_team()` API-boundary
    precheck on what a valid trace seed looks like. `required=False`: most
    dynamic rows are not trace seeds at all, so an absent `trace_seed_messages`
    is `(None, None)` here rather than a raise. Once it IS present,
    `trace_start_from` is required — never defaulted (see finding 5: a
    silent `first_user` default here used to replay an imported
    `last_assistant` transcript as a single opening turn with no error).
    """
    return parse_trace_seed(inputs, label='datapoint', required=False)


def _register_job_target(
    target: Any,
    cleanup_stack: contextlib.AsyncExitStack,
    *,
    agent_context: AgentContext,
    memory_entity_ids: list[str] | None,
) -> None:
    """Register target cleanup, memory tracking, and optional trace model metadata."""
    cleanup_stack.push_async_callback(close_target, target)
    target_memory_id = getattr(target, 'memory_entity_id', None)
    if target_memory_id is not None and memory_entity_ids is not None:
        memory_entity_ids.append(target_memory_id)
    if hasattr(target, 'model') and agent_context.model:
        object.__setattr__(target, 'model', agent_context.model)  # type: ignore[misc]


def _effective_max_turns(strategy: AttackStrategy, max_turns: int) -> int:
    return 1 if strategy.turn_type == TurnType.SINGLE else max_turns


def _is_fixed_template_attack(strategy: AttackStrategy) -> bool:
    return bool(strategy.prompt_template) and strategy.turn_type == TurnType.SINGLE


def _serialize_dynamic_result(
    result: AttackOutput,
    *,
    trace_seeded: bool,
    effective_max_turns: int,
    thread_id: str | None,
) -> dict[str, Any]:
    """Serialize an attack while preserving a zero-turn trace bootstrap failure."""
    payload = result.model_dump(mode='json')
    if trace_seeded and result.error is not None and not result.turns:
        payload['turns'] = 0
    return {**payload, 'max_turns': effective_max_turns, 'thread_id': thread_id}


def _safe_agent_key(agent_key: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '-' for ch in agent_key).strip('-')


def create_dynamic_redteam_job(
    *,
    agent_key: str,
    agent_context: AgentContext,
    red_team_model: str = DEFAULT_PIPELINE_MODEL,
    max_turns: int = 5,
    backend: Backend | None = None,
    attack_llm_client: AsyncOpenAI | None = None,
    memory_entity_ids: list[str] | None = None,
    attacker_instructions: str | None = None,
    verbosity: int = 0,
    llm_kwargs: dict[str, Any] | None = None,
    pipeline_config: LLMConfig | None = None,
    run_id: str | None = None,
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
        backend: Backend for creating targets and mapping errors. Defaults to ORQ.
    """
    cfg = pipeline_config or PIPELINE_CONFIG
    resolved_backend: Backend = backend if backend is not None else resolve_backend('orq', pipeline_config=cfg)
    safe_agent_key = _safe_agent_key(agent_key)
    job_name = f'redteam:dynamic:{safe_agent_key or "agent"}'

    @job(job_name)
    async def dynamic_job(data: DataPoint, _row: int) -> dict[str, Any]:
        """Execute a single red-team attack for the given datapoint and return the serialized result."""
        inputs = dict(data.inputs)
        strategy = AttackStrategy.model_validate(inputs['strategy'])
        objective = str(inputs['objective'])
        category = str(inputs['category'])
        vulnerability = str(inputs.get('vulnerability', ''))
        effective_max_turns = _effective_max_turns(strategy, max_turns)
        thread_id = build_thread_id(run_id, safe_agent_key, _row)

        async with (
            with_redteam_span(
                'orq.redteam.attack',
                {
                    'orq.redteam.category': category,
                    'orq.redteam.vulnerability': vulnerability,
                    'orq.redteam.strategy_name': strategy.name,
                    'orq.redteam.turn_type': strategy.turn_type.value,
                    'orq.redteam.max_turns': effective_max_turns,
                },
            ) as attack_span,
            contextlib.AsyncExitStack() as cleanup_stack,
        ):
            target = resolved_backend.create_target(agent_key=agent_key)
            _register_job_target(
                target,
                cleanup_stack,
                agent_context=agent_context,
                memory_entity_ids=memory_entity_ids,
            )
            seed_messages, trace_start_from = _trace_seed_from_inputs(inputs)

            if _is_fixed_template_attack(strategy):
                # Fixed template: fill and send directly (no adversarial LLM needed)
                t0 = time.time()
                prompt = generate_attack_prompt(strategy, agent_context)
                token_usage = None
                seed_context: list[Message] = []
                bootstrap_usage = None

                @asynccontextmanager
                async def _attempt_span(i: int):
                    async with with_redteam_span(
                        'orq.redteam.target_call',
                        {
                            'orq.redteam.category': category,
                            'orq.redteam.strategy_name': strategy.name,
                            'orq.redteam.turn': 1,
                            'orq.redteam.target_attempt': i + 1,
                            'input': truncate_for_span(prompt),
                            'orq.redteam.input': truncate_for_span(prompt),
                        },
                    ) as span:
                        yield span

                def _record_attempt_response(span: Any, response: AgentResponse) -> None:
                    response_text = truncate_for_span(response.text or '')
                    set_span_attrs(
                        span,
                        {
                            'output': response_text,
                            'orq.redteam.output': response_text,
                        },
                    )

                # One Orq thread per attack, opened before the seed replay: a bootstrap
                # sent outside this scope reaches the target without the attack's thread
                # id, so the replay and the attack land in different conversations.
                with conversation_thread(thread_id) as thread_id:
                    if seed_messages is not None:
                        seed_context, bootstrap_usage, bootstrap_error = await replay_seed_context(
                            target,
                            seed_messages,
                            trace_start_from or TraceStart.FIRST_USER,
                            target_agent_timeout_ms=cfg.target_agent_timeout_ms,
                            max_target_retries=cfg.max_target_retries,
                            map_error=resolved_backend.map_error,
                        )
                        if bootstrap_error is not None:
                            result_dict = AttackOutput(
                                turns=[],
                                objective_achieved=False,
                                duration_seconds=time.time() - t0,
                                token_usage=bootstrap_usage,
                                token_usage_adversarial=None,
                                token_usage_target=None,
                                token_usage_bootstrap=bootstrap_usage,
                                seed_context=seed_context,
                                system_prompt=None,
                                max_turns=effective_max_turns,
                                **bootstrap_error,
                                category=category,
                                vulnerability=vulnerability,
                            )
                            _set_attack_span_attrs(attack_span, result_dict)
                            output_payload = result_dict.model_dump(mode='json')
                            output_payload['turns'] = 0
                            active_progress = _get_active_progress()
                            if active_progress is not None:
                                await active_progress.finish_attack(None)
                            return {
                                **output_payload,
                                'max_turns': effective_max_turns,
                                'thread_id': thread_id,
                            }
                    result = await call_target_with_retry(
                        target,
                        [*seed_context, Message(role='user', content=prompt)],
                        target_agent_timeout_ms=cfg.target_agent_timeout_ms,
                        max_target_retries=cfg.max_target_retries,
                        map_error=resolved_backend.map_error,
                        on_attempt=_attempt_span,
                        on_attempt_response=_record_attempt_response,
                    )

                agent_resp = result.response
                response = agent_resp.text
                token_usage = agent_resp.usage if result.succeeded else None
                error_fields = result.error_payload()
                result_dict = AttackOutput(
                    turns=[
                        Turn(
                            attacker=AgentResponse(text=prompt),
                            target=agent_resp
                            if agent_resp.output
                            else AgentResponse(
                                output=[TextOutputItem(text=response or '', annotations=[])],
                            ),
                        )
                    ],
                    objective_achieved=False,
                    duration_seconds=time.time() - t0,
                    token_usage=_merge_usage(token_usage, bootstrap_usage),
                    token_usage_adversarial=None,
                    token_usage_target=token_usage,
                    token_usage_bootstrap=bootstrap_usage,
                    seed_context=seed_context,
                    system_prompt=None,
                    **error_fields,
                    category=category,
                    vulnerability=vulnerability,
                )

                set_span_attrs(
                    attack_span,
                    {
                        'input': truncate_for_span(prompt),
                        'output': truncate_for_span(response or ''),
                    },
                )
                _set_attack_span_attrs(attack_span, result_dict)

                # Advance the global progress bar for template single-turn attacks
                # (multi-turn attacks are tracked by run_attack in the orchestrator).
                active_progress = _get_active_progress()
                if active_progress is not None:
                    await active_progress.finish_attack(None)

                return {**result_dict.model_dump(mode='json'), 'max_turns': effective_max_turns, 'thread_id': thread_id}

            # Dynamic single-turn (max_turns=1) or multi-turn — orchestrator handles both
            llm_client = attack_llm_client or create_async_llm_client()
            orchestrator = MultiTurnOrchestrator(
                llm_client,
                model=red_team_model,
                backend=resolved_backend,
                attacker_instructions=attacker_instructions,
                verbosity=verbosity,
                llm_kwargs=llm_kwargs,
                pipeline_config=cfg,
            )

            try:
                # One Orq thread per attack groups all its turns in observability.
                with conversation_thread(thread_id) as thread_id:
                    result = await orchestrator.run_attack(
                        target=target,
                        strategy=strategy,
                        objective=objective,
                        agent_context=agent_context,
                        max_turns=effective_max_turns,
                        seed_messages=seed_messages,
                        trace_start_from=trace_start_from,
                    )
            except asyncio.TimeoutError as e:
                tb = traceback.format_exc(limit=8)
                mapped_code, mapped_msg = resolved_backend.map_error(e)
                logger.error(f'Orchestrator timed out for {category}/{strategy.name}: {mapped_msg}\n{tb}')
                result = OrchestratorResult(
                    objective_achieved=False,
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
                mapped_code, mapped_msg = resolved_backend.map_error(e)
                logger.error(f'Unexpected orchestrator error for {category}/{strategy.name}: {mapped_msg}\n{tb}')
                result = OrchestratorResult(
                    objective_achieved=False,
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
            if result_dict.chat_completions:
                first_user = next((m.content for m in result_dict.chat_completions if m.role == 'user'), None)
                if first_user:
                    set_span_attrs(attack_span, {'input': truncate_for_span(first_user)})
            if result_dict.final_response:
                set_span_attrs(attack_span, {'output': truncate_for_span(result_dict.final_response)})
            _set_attack_span_attrs(attack_span, result_dict)
            return _serialize_dynamic_result(
                result_dict,
                trace_seeded=seed_messages is not None,
                effective_max_turns=effective_max_turns,
                thread_id=thread_id,
            )

    return dynamic_job


def _append_jury_summary(explanation: str, jury: JuryResult | None) -> str:
    """Append a compact jury reliability summary to the evaluator explanation.

    Surfaces inter-judge agreement in the results table when a jury ran (RES-739).
    Returns the explanation unchanged for single-judge runs.
    """
    return append_jury_summary(explanation, jury)


def create_dynamic_evaluator(
    evaluator_model: str = DEFAULT_PIPELINE_MODEL,
    llm_client: AsyncOpenAI | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    cfg: LLMCallConfig | EvaluatorConfig | None = None,
    judges: list[str] | None = None,
    judge_repetitions: int = 1,
    replacement_judges: list[str] | None = None,
    min_successful_judges: int = 1,
    target_models: list[str] | None = None,
    *,
    strict_panel: bool = False,
) -> EvaluatorqEvaluatorConfig:
    """Create an evaluator that uses OWASPEvaluator on the attack conversation.

    value=True means RESISTANT (consistent with OWASP evaluatorq scoring and
    EvaluationResult.passed convention).

    ``judges`` adds panel models alongside ``evaluator_model``; ``judge_repetitions``
    runs each judge N times with majority voting; ``replacement_judges`` stand in for
    failed judges; ``min_successful_judges`` is the decisive-verdict floor (RES-739).
    ``target_models`` (known direct-model targets only) drives the self-judge/family
    warning; ``strict_panel`` upgrades composition warnings to hard errors.
    """
    owasp_evaluator = OWASPEvaluator(
        evaluator_model=evaluator_model,
        llm_client=llm_client,
        llm_kwargs=llm_kwargs,
        cfg=cfg,
        judges=judges,
        repetitions=judge_repetitions,
        replacement_judges=replacement_judges,
        min_successful_judges=min_successful_judges,
        target_models=target_models,
        strict_panel=strict_panel,
    )

    async def scorer(params: ScorerParameter) -> EvaluationResult:
        """Evaluate the attack output using OWASPEvaluator and return a scored EvaluationResult."""
        data = params['data']
        raw_output = params['output']
        if isinstance(raw_output, AttackOutput):
            output = raw_output
        elif isinstance(raw_output, dict):
            try:
                output = AttackOutput.model_validate(raw_output)
            except Exception as e:  # noqa: BLE001
                inputs = getattr(data, 'inputs', {}) or {}
                logger.error(
                    'Failed to parse job output as AttackOutput '
                    f'(datapoint_id={inputs.get("id", "<unknown>")!r}, '
                    f'category={inputs.get("category", "<unknown>")!r}, '
                    f'strategy_name={inputs.get("strategy_name", "<unknown>")!r}): {e}'
                )
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

        # An error means the attempted conversation did not complete. Preserve the
        # transcript for diagnosis, but do not score a partial exchange as either
        # RESISTANT or VULNERABLE.
        if output.error:
            return EvaluationResult(
                value='error',
                explanation=f'Skipped: incomplete attack run — {output.error}',
            )

        category = output.category or data.inputs.get('category', '')
        vulnerability = output.vulnerability or data.inputs.get('vulnerability', '')

        # A red-team score requires at least one complete, judgeable target response.
        # ``inconclusive`` distinguishes a non-response with no explicit transport error
        # from a failed run, while an errored turn remains an explicit run error even for
        # historical outputs that predate the orchestrator's same-exchange retry policy.
        if not output.turns:
            return EvaluationResult(
                value='inconclusive',
                explanation='Skipped: target produced no responses, so there is no evidence to evaluate.',
            )
        scorable_turns = [t for t in output.turns if not t.errored and _turn_has_content(t)]
        if len(scorable_turns) != len(output.turns):
            value = 'error' if any(t.errored for t in output.turns) else 'inconclusive'
            return EvaluationResult(
                value=value,
                explanation=(
                    f'Skipped: incomplete target transcript (errored or empty; '
                    f'codes: {_error_codes(output.turns)}) — nothing to evaluate'
                ),
            )
        # For TraceStart.LAST_ASSISTANT, output.seed_context is the imported
        # conversation the attack continues — turn 1 of scorable_turns often
        # refers back into it. Without it in front, the judge has no way to
        # resolve that reference, and `passed=True` means RESISTANT, so the
        # judge's failure mode defaults to the optimistic verdict. Marked
        # distinctly from the attacker's own turns so the rubric can tell
        # recorded/imported content from attacker-generated content.
        # to_chat_completion, not a {role, content} flatten: an imported turn can
        # carry tool_calls / tool_call_id / name, and dropping those presents a
        # tool-using assistant turn to the judge as an empty message.
        seed_context_messages: list[dict[str, Any]] = []
        for message in output.seed_context:
            rendered = message.to_chat_completion()
            rendered['content'] = f'[imported trace context] {coerce_content_text(message.content)}'
            seed_context_messages.append(rendered)
        input_messages = [
            *seed_context_messages,
            *({'role': 'user', 'content': t.attacker.text} for t in scorable_turns),
        ]
        output_messages = [item for t in scorable_turns for item in t.target.output]

        # Prefer vulnerability-first path when a valid Vulnerability enum can be resolved
        resolved_vuln: Vulnerability | None = None
        if vulnerability:
            try:
                resolved_vuln = Vulnerability(vulnerability)
            except ValueError:
                resolved_vuln = resolve_category_safe(vulnerability)

        async with annotate_current_span(
            {
                'orq.redteam.category': category,
                'orq.redteam.vulnerability': vulnerability,
                'orq.redteam.model': evaluator_model,
            },
        ) as eval_span:
            if resolved_vuln is not None:
                eval_result = await owasp_evaluator.evaluate_vulnerability(
                    vuln=resolved_vuln,
                    messages=input_messages,
                    output_messages=output_messages,
                )
            else:
                eval_result = await owasp_evaluator.evaluate(
                    category=category,
                    messages=input_messages,
                    output_messages=output_messages,
                )
            set_span_attrs(
                eval_span,
                {
                    'orq.redteam.passed': eval_result.passed,
                    'input': output.final_response or '',
                    'output': eval_result.explanation or '',
                },
            )
            set_jury_span_attrs(eval_span, eval_result.jury)

        # ``passed`` is the judge's boolean verdict (RESISTANT/VULNERABLE); surface it
        # as the score value, or 'inconclusive' when the judge errored.
        result_value: str | int | float | bool = (
            bool(eval_result.passed) if eval_result.passed is not None else 'inconclusive'
        )
        # The generic EvaluationResult is framework-agnostic, so the structured jury
        # breakdown rides through raw_output (the judge's-raw-response passthrough)
        # rather than as a typed field; the red-team converters lift it back onto
        # UnifiedEvaluationResult.jury so per-judge votes + agreement reach the report
        # (RES-739 DoD), not just the OTel span + inline explanation text.
        # attach_jury_raw_output passes a jury-less raw_output through untouched, so a
        # pre-existing ``None`` stays ``null`` in serialized reports (consumers
        # distinguish null from {}).
        scored_raw_output = attach_jury_raw_output(eval_result.raw_output, eval_result.jury)
        return EvaluationResult.model_validate({
            'value': result_value,
            'explanation': _append_jury_summary(eval_result.explanation, eval_result.jury),
            'pass': eval_result.passed,
            # Carry the judge's cost + raw response so the report layer can surface
            # and aggregate evaluator token usage. They are kept in local result
            # dumps but stripped from the Orq platform upload at the send boundary
            # (see evaluatorq.send_results).
            'token_usage': eval_result.token_usage,
            'raw_output': scored_raw_output,
        })

    # evaluator_type marks these LLM-judge scorers so the tracing layer emits the
    # gen_ai.evaluation.* evaluator-span attributes (opt-in; see set_evaluation_attributes).
    return {'name': 'owasp-dynamic-security', 'scorer': scorer, 'evaluator_type': 'llm_eval'}


async def cleanup_memory_entities(
    agent_context: AgentContext,
    entity_ids: list[str],
    memory_cleanup: Backend | None = None,
    pipeline_config: LLMConfig | None = None,
) -> str | None:
    """Delete memory entities created during a red teaming run.

    Delegates to provided backend cleanup implementation.
    Wrapped with an overall timeout — cleanup is best-effort and should never block the pipeline.

    Returns None on success, or an error message string on failure.
    """
    cfg = pipeline_config or PIPELINE_CONFIG
    cleanup_timeout_s = cfg.cleanup_timeout_ms / 1000.0
    try:
        if memory_cleanup is not None:
            await asyncio.wait_for(
                memory_cleanup.cleanup_memory(agent_context, entity_ids),
                timeout=cleanup_timeout_s,
            )
            return None
        # Default fallback keeps existing ORQ behavior.
        await asyncio.wait_for(
            resolve_backend('orq').cleanup_memory(agent_context, entity_ids),
            timeout=cleanup_timeout_s,
        )
    except asyncio.TimeoutError:
        msg = f'Memory cleanup timed out after {cleanup_timeout_s:.0f}s for {len(entity_ids)} entities'
        logger.warning(msg)
        return msg
    except Exception as e:  # noqa: BLE001
        msg = f'Memory cleanup failed: {e}'
        logger.warning(msg)
        return msg
    return None
