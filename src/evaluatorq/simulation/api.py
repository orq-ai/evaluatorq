"""High-level simulation API functions.

`simulate()` and `generate_and_simulate()` route execution through the
`evaluatorq()` framework so they inherit auto-upload, OTel tracing, results
display, CI gating, and dataset-id support (RES-594 / RES-598). They run
inside an ``orq.simulation.pipeline`` span and accept the unified
``AgentTarget`` target shape introduced by RES-808.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import uuid
from typing import TYPE_CHECKING, Any, cast

from evaluatorq.simulation.types import DEFAULT_MODEL

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from opentelemetry.trace import Span

    from evaluatorq.contracts import AgentTarget
    from evaluatorq.simulation.agents.base import BaseAgent
    from evaluatorq.simulation.evaluators.scorers import SimulationScorer
    from evaluatorq.simulation.generators import FirstMessageGenerator
    from evaluatorq.simulation.hooks import SimulationHooks
    from evaluatorq.simulation.types import (
        Datapoint,
        Message,
        Persona,
        Scenario,
        SimulationResult,
    )
    from evaluatorq.types import DataPoint, DataPointResult, Evaluator

logger = logging.getLogger(__name__)


class SimulationDroppedError(RuntimeError):
    """Raised when simulation job(s) produced no result and were dropped.

    Distinct subclass of RuntimeError so callers can grep/catch the
    cache-miss path specifically rather than matching on the message.
    """


async def simulate(
    *,
    evaluation_name: str = '',
    agent_key: str | None = None,
    target_callback: Callable[[list[Message]], str | Awaitable[str]] | None = None,
    target: Callable[[list[Message]], str | Awaitable[str]] | AgentTarget | None = None,
    personas: list[Persona] | None = None,
    scenarios: list[Scenario] | None = None,
    datapoints: list[Datapoint] | None = None,
    dataset_id: str | None = None,
    max_turns: int = 10,
    model: str = DEFAULT_MODEL,
    evaluator_names: list[str] | None = None,
    parallelism: int = 5,
    user_simulator: BaseAgent | None = None,
    judge: BaseAgent | None = None,
    hooks: SimulationHooks | None = None,
    upload_results: bool = True,
    evaluation_description: str | None = None,
    path: str | None = None,
    exit_on_failure: bool = True,
) -> list[SimulationResult]:
    """Run agent simulations through the evaluatorq() framework.

    Builds simulation Datapoints (cartesian persona × scenario when needed),
    wraps them as evaluatorq ``DataPoint``s, and delegates execution,
    parallelism, tracing, results display, and (by default) upload to
    ``evaluatorq()``. Returns the raw ``SimulationResult`` list so existing
    callers continue to work.

    Args:
        target_callback: Callable that receives the conversation history and
            returns the agent's response. Kept for backwards compatibility.
        target: Alias for ``target_callback``. Takes precedence when both are
            supplied. May also be an ``AgentTarget`` instance, which is routed
            to the runner's ``target_agent`` path (it speaks ``respond(messages)``).
        user_simulator: Pre-constructed ``BaseAgent`` to drive the user side.
            When omitted a default ``UserSimulatorAgent`` is built from ``model``.
        judge: Pre-constructed ``BaseAgent`` used to evaluate each turn.
        hooks: Optional ``SimulationHooks`` for run/datapoint/turn lifecycle
            events. Sync or async; ``async def`` is preferred (a sync hook
            works but emits a one-time ``DeprecationWarning``). Defaults to
            ``DefaultHooks`` (loguru baseline).
        dataset_id: When set, fetch simulation datapoints from the named Orq
            dataset instead of taking them inline. Mutually exclusive with
            ``datapoints``, ``personas``, ``scenarios``. Each dataset row's
            ``inputs`` must already match one of the simulation input shapes
            (``datapoint`` / ``persona`` + ``scenario`` / etc.).
        upload_results: When ``True`` (the default) and ``ORQ_API_KEY`` is set,
            results are uploaded to the Orq platform as an experiment. Pass
            ``False`` to suppress the upload (e.g. for local-only runs).
        evaluation_description: Optional description attached to the
            experiment. Mirrors ``evaluatorq()``.
        path: Optional Orq folder path (e.g. ``"MyProject/MyFolder"``).
        exit_on_failure: When ``True`` (the default), exit non-zero if any
            datapoint or evaluator produced a failure — this is the "CI
            gating for free" benefit of routing through ``evaluatorq()``.
            Two paths: score-based failures (``pass_=False`` on a scorer
            result) call ``sys.exit(1)`` via evaluatorq's own gate; dropped
            jobs (job raised, no result cached) raise ``SimulationDroppedError``
            from ``simulate()`` itself. Both exit non-zero when uncaught. Pass
            ``False`` for interactive / exploratory runs where failures
            should surface as warnings + error metadata instead.
    """
    from evaluatorq.simulation.tracing import with_simulation_span
    from evaluatorq.tracing.setup import flush_tracing, init_tracing_if_needed

    await init_tracing_if_needed()

    try:
        async with with_simulation_span(
            'orq.simulation.pipeline',
            {
                'orq.simulation.evaluation_name': evaluation_name,
                'orq.simulation.max_turns': max_turns,
                'orq.simulation.parallelism': parallelism,
            },
        ) as pipeline_span:
            return await _simulate_core(
                caller='simulate',
                evaluation_name=evaluation_name,
                agent_key=agent_key,
                target_callback=target_callback,
                target=target,
                personas=personas,
                scenarios=scenarios,
                datapoints=datapoints,
                dataset_id=dataset_id,
                max_turns=max_turns,
                model=model,
                evaluator_names=evaluator_names,
                parallelism=parallelism,
                user_simulator=user_simulator,
                judge=judge,
                upload_results=upload_results,
                evaluation_description=evaluation_description,
                path=path,
                exit_on_failure=exit_on_failure,
                pipeline_span=pipeline_span,
                hooks=hooks,
            )
    finally:
        await flush_tracing()


async def generate_and_simulate(
    *,
    evaluation_name: str = '',
    agent_description: str,
    agent_key: str | None = None,
    target_callback: Callable[[list[Message]], str | Awaitable[str]] | None = None,
    target: Callable[[list[Message]], str | Awaitable[str]] | AgentTarget | None = None,
    num_personas: int = 5,
    num_scenarios: int = 5,
    max_turns: int = 10,
    model: str = DEFAULT_MODEL,
    evaluator_names: list[str] | None = None,
    parallelism: int = 5,
    user_simulator: BaseAgent | None = None,
    judge: BaseAgent | None = None,
    hooks: SimulationHooks | None = None,
    upload_results: bool = True,
    evaluation_description: str | None = None,
    path: str | None = None,
    exit_on_failure: bool = True,
) -> list[SimulationResult]:
    """Generate personas/scenarios, then run simulations via evaluatorq().

    Accepts the same ``target`` shapes as :func:`simulate` — a plain callable,
    an ``AgentTarget`` instance, or an ``agent_key`` for the Orq deployment
    bridge. ``ORQ_API_KEY`` is required because persona/scenario/first-message
    generation calls the Orq router. ``upload_results`` defaults to ``True``;
    set it to ``False`` to skip uploading the final experiment.
    ``exit_on_failure`` defaults to ``True``; see :func:`simulate` for the
    full semantics of the CI-gate behaviour and how to opt out. ``hooks``
    mirrors :func:`simulate`; note the ``on_confirm`` gate fires AFTER
    persona/scenario/first-message generation, so those generation tokens are
    already spent when the gate is consulted.
    """
    from openai import AsyncOpenAI

    from evaluatorq.simulation.generators import PersonaGenerator, ScenarioGenerator
    from evaluatorq.simulation.tracing import with_simulation_span
    from evaluatorq.tracing.setup import flush_tracing, init_tracing_if_needed

    await init_tracing_if_needed()

    api_key = _require_orq_api_key('generate_and_simulate')

    try:
        async with with_simulation_span(
            'orq.simulation.pipeline',
            {
                'orq.simulation.evaluation_name': evaluation_name,
                'orq.simulation.mode': 'generate_and_simulate',
                'orq.simulation.num_personas': num_personas,
                'orq.simulation.num_scenarios': num_scenarios,
                'orq.simulation.max_turns': max_turns,
                'orq.simulation.parallelism': parallelism,
            },
        ) as pipeline_span:
            shared_client = AsyncOpenAI(
                api_key=api_key,
                base_url=f'{os.environ.get("ORQ_BASE_URL", "https://api.orq.ai")}/v2/router',
            )
            try:
                persona_gen = PersonaGenerator(model=model, client=shared_client)
                scenario_gen = ScenarioGenerator(model=model, client=shared_client)
                gen_personas, gen_scenarios = await asyncio.gather(
                    persona_gen.generate(
                        agent_description=agent_description,
                        num_personas=num_personas,
                    ),
                    scenario_gen.generate(
                        agent_description=agent_description,
                        num_scenarios=num_scenarios,
                    ),
                )
            finally:
                await shared_client.close()

            return await _simulate_core(
                caller='generate_and_simulate',
                evaluation_name=evaluation_name,
                agent_key=agent_key,
                target_callback=target_callback,
                target=target,
                personas=gen_personas,
                scenarios=gen_scenarios,
                datapoints=None,
                dataset_id=None,
                max_turns=max_turns,
                model=model,
                evaluator_names=evaluator_names,
                parallelism=parallelism,
                user_simulator=user_simulator,
                judge=judge,
                upload_results=upload_results,
                evaluation_description=evaluation_description,
                path=path,
                exit_on_failure=exit_on_failure,
                pipeline_span=pipeline_span,
                hooks=hooks,
            )
    finally:
        await flush_tracing()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


async def _simulate_core(
    *,
    caller: str,
    evaluation_name: str,
    agent_key: str | None,
    target_callback: Callable[[list[Message]], str | Awaitable[str]] | None,
    target: Callable[[list[Message]], str | Awaitable[str]] | AgentTarget | None,
    personas: list[Persona] | None,
    scenarios: list[Scenario] | None,
    datapoints: list[Datapoint] | None,
    dataset_id: str | None,
    max_turns: int,
    model: str,
    evaluator_names: list[str] | None,
    parallelism: int,
    user_simulator: BaseAgent | None,
    judge: BaseAgent | None,
    upload_results: bool,
    evaluation_description: str | None,
    path: str | None,
    exit_on_failure: bool,
    pipeline_span: Span | None,
    hooks: SimulationHooks | None = None,
) -> list[SimulationResult]:
    """Core simulation logic (runs inside the orq.simulation.pipeline span).

    Resolves the target (callable, ``AgentTarget``, or ``agent_key`` bridge),
    resolves/generates the datapoints, then drives the run-level lifecycle
    hooks (``on_confirm`` gate → ``on_run_start`` → ``on_run_complete``) around
    ``_simulate_via_evaluatorq``, which routes execution through evaluatorq's
    upload, CI gating, and results display. Per-datapoint / per-turn hooks fire
    inside ``SimulationRunner``.
    """
    from evaluatorq.common.async_utils import await_maybe
    from evaluatorq.simulation.exceptions import SimulationCancelledError
    from evaluatorq.simulation.hooks import DefaultHooks, SimulationRunMeta
    from evaluatorq.simulation.tracing import set_span_attrs

    target_callback_resolved, target_agent = _resolve_target(target, target_callback, agent_key)

    sim_datapoints = await _resolve_or_generate_datapoints(
        caller=caller,
        datapoints=datapoints,
        personas=personas,
        scenarios=scenarios,
        dataset_id=dataset_id,
        model=model,
    )

    set_span_attrs(
        pipeline_span,
        {'orq.simulation.datapoints_count': len(sim_datapoints)},
    )

    resolved_hooks = hooks or DefaultHooks()
    # The sync-hook deprecation nudge fires once in SimulationRunner.__init__
    # (the single choke point both this path and direct runner use share).
    resolved_evaluator_names = evaluator_names if evaluator_names is not None else ['goal_achieved', 'criteria_met']
    run_meta: SimulationRunMeta = {
        'num_datapoints': len(sim_datapoints),
        'model': model,
        'max_turns': max_turns,
        'parallelism': parallelism,
        'evaluation_name': evaluation_name,
        'evaluator_names': resolved_evaluator_names,
    }
    # Gate first — before the evaluatorq run is built. A decline is a clean abort.
    if not await await_maybe(resolved_hooks.on_confirm(run_meta)):
        raise SimulationCancelledError('Simulation run declined by on_confirm hook')

    # Terminal hook always pairs with on_run_start; results is [] on early
    # failure. on_run_complete is unguarded (a raising hook propagates, per the
    # hook exception policy); runner/target cleanup lives inside
    # _simulate_via_evaluatorq's own finally, so it runs regardless.
    results: list[SimulationResult] = []
    await await_maybe(resolved_hooks.on_run_start(run_meta))
    try:
        results = await _simulate_via_evaluatorq(
            caller=caller,
            evaluation_name=evaluation_name,
            target_callback=target_callback_resolved,
            target_agent=target_agent,
            sim_datapoints=sim_datapoints,
            max_turns=max_turns,
            model=model,
            evaluator_names=evaluator_names,
            parallelism=parallelism,
            user_simulator=user_simulator,
            judge=judge,
            upload_results=upload_results,
            evaluation_description=evaluation_description,
            path=path,
            exit_on_failure=exit_on_failure,
            pipeline_span=pipeline_span,
            hooks=resolved_hooks,
        )
    finally:
        await await_maybe(resolved_hooks.on_run_complete(results))
    return results


def _resolve_target(
    target: Callable[..., Any] | AgentTarget | None,
    target_callback: Callable[..., Any] | None,
    agent_key: str | None,
) -> tuple[Callable[[list[Message]], str | Awaitable[str]] | None, AgentTarget | None]:
    """Resolve the simulation target into (callback, agent) for the runner.

    ``target`` takes precedence over ``target_callback`` when both are given.
    An ``AgentTarget`` instance is routed to the runner's ``target_agent`` path
    (it speaks ``respond(messages)``, not the callable shape); plain callables
    and the ``agent_key`` deployment bridge stay on the callback path.
    """
    from evaluatorq.contracts import AgentTarget
    from evaluatorq.simulation.adapters import from_orq_deployment

    resolved = target or target_callback
    if isinstance(resolved, AgentTarget):
        return None, resolved
    if not resolved and agent_key:
        resolved = from_orq_deployment(agent_key)
    if not resolved:
        raise ValueError('Either target_callback (or target) or agent_key is required')
    return resolved, None


def _require_orq_api_key(caller: str) -> str:
    api_key = os.environ.get('ORQ_API_KEY')
    if not api_key:
        raise ValueError(f'ORQ_API_KEY environment variable is not set. Set it before calling {caller}().')
    return api_key


async def _resolve_or_generate_datapoints(
    *,
    caller: str,
    datapoints: list[Datapoint] | None,
    personas: list[Persona] | None,
    scenarios: list[Scenario] | None,
    dataset_id: str | None,
    model: str,
) -> list[Datapoint]:
    """Return ready-to-run Datapoints.

    Resolution precedence: ``dataset_id`` → ``datapoints`` → persona × scenario
    cartesian product (with first-message generation). The three sources are
    mutually exclusive. ``caller`` names the public entry point so any
    API-key error message points the user at the right function.
    """
    sources = [
        ('dataset_id', dataset_id is not None),
        ('datapoints', datapoints is not None),
        ('personas/scenarios', personas is not None or scenarios is not None),
    ]
    chosen = [name for name, present in sources if present]
    if len(chosen) > 1:
        raise ValueError(f'Pass exactly one of dataset_id, datapoints, or personas+scenarios; got: {", ".join(chosen)}')

    if dataset_id is not None:
        api_key = _require_orq_api_key(caller)
        return await _fetch_simulation_datapoints_from_orq(api_key, dataset_id)

    if datapoints is not None:
        if not datapoints:
            raise ValueError("'datapoints' must be non-empty")
        return datapoints

    if personas is None or scenarios is None:
        raise ValueError("Either provide 'dataset_id', 'datapoints', or both 'personas' and 'scenarios'")
    if not personas or not scenarios:
        raise ValueError("'personas' and 'scenarios' arrays must both be non-empty")

    from openai import AsyncOpenAI

    from evaluatorq.simulation.generators import FirstMessageGenerator

    api_key = _require_orq_api_key(caller)
    shared_client = AsyncOpenAI(
        api_key=api_key,
        base_url=f'{os.environ.get("ORQ_BASE_URL", "https://api.orq.ai")}/v2/router',
    )
    try:
        first_msg_gen = FirstMessageGenerator(model=model, client=shared_client)
        pairs = [(p, s) for p in personas for s in scenarios]

        generated: list[Datapoint] = []
        batch_size = 5
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            batch_results = await asyncio.gather(
                *[_generate_single_datapoint(first_msg_gen, p, s) for p, s in batch],
                return_exceptions=True,
            )
            for (p, s), outcome in zip(batch, batch_results, strict=True):
                if isinstance(outcome, BaseException):
                    logger.warning(
                        'first-message generation failed for persona=%r scenario=%r: %r — skipping',
                        p.name,
                        s.name,
                        outcome,
                    )
                    continue
                generated.append(outcome)
        if not generated:
            raise RuntimeError('first-message generation produced no datapoints — every persona×scenario pair failed')
        return generated
    finally:
        await shared_client.close()


async def _fetch_simulation_datapoints_from_orq(api_key: str, dataset_id: str) -> list[Datapoint]:
    """Stream the named Orq dataset and parse each row into a simulation
    Datapoint via the same shape-tolerant extractor used by the inline path.
    """
    from pydantic import ValidationError

    from evaluatorq.fetch_data import fetch_dataset_batches, setup_orq_client
    from evaluatorq.simulation._datapoint_io import _extract_single_datapoint

    orq_client = setup_orq_client(api_key)
    out: list[Datapoint] = []
    row = 0
    async for batch in fetch_dataset_batches(orq_client, dataset_id):
        for eq_dp in batch.datapoints:
            try:
                out.append(_extract_single_datapoint(eq_dp))
            except (ValueError, ValidationError) as e:
                raise ValueError(f'dataset {dataset_id!r} row {row}: {e}') from e
            row += 1
    if not out:
        raise ValueError(f'Dataset {dataset_id!r} returned zero simulation-compatible datapoints')
    return out


async def _generate_single_datapoint(gen: FirstMessageGenerator, persona: Persona, scenario: Scenario) -> Datapoint:
    from evaluatorq.simulation.tracing import with_simulation_span
    from evaluatorq.simulation.utils.prompt_builders import generate_datapoint

    async with with_simulation_span(
        'orq.simulation.first_message_generation',
        {
            'orq.simulation.persona': persona.name,
            'orq.simulation.scenario': scenario.name,
            'orq.simulation.model': getattr(gen, '_model', None),
        },
    ):
        first_message = await gen.generate(persona, scenario)
    return generate_datapoint(persona, scenario, first_message)


def _build_simulation_job_and_cache(
    *,
    job_name: str,
    sim_dp_by_id: dict[int, Datapoint],
    target_callback: Callable[[list[Message]], str | Awaitable[str]] | None,
    target_agent: AgentTarget | None,
    model: str,
    max_turns: int,
    user_simulator: BaseAgent | None,
    judge: BaseAgent | None,
    hooks: SimulationHooks | None,
) -> tuple[
    Callable[[DataPoint, int], Awaitable[dict[str, Any]]],
    dict[int, SimulationResult],
    Any,
]:
    """Build the simulation job_fn, its result cache, and the shared runner.

    Both the cache and ``sim_dp_by_id`` are keyed by ``id(data)`` — the
    identity of the evaluatorq DataPoint instance. evaluatorq passes the same
    instance to the job and every scorer within a single run (it does not
    serialize/rehydrate DataPoints mid-run), so identity is stable for the
    call lifetime. This avoids polluting the UI inputs column with a synthetic
    index key and removes a pydantic re-parse per datapoint.
    """
    from evaluatorq.common.async_utils import await_maybe
    from evaluatorq.simulation.convert import to_open_responses
    from evaluatorq.simulation.hooks import DefaultHooks
    from evaluatorq.simulation.runner.simulation import SimulationRunner, _error_result
    from evaluatorq.simulation.types import TerminatedBy

    runner = SimulationRunner(
        target_callback=target_callback,
        target_agent=target_agent,
        model=model,
        max_turns=max_turns,
        user_simulator=user_simulator,
        judge=judge,
        hooks=hooks,
    )
    # _simulate_core always passes a resolved hooks; the fallback only guards
    # the (internal-only) direct-call path. A user-supplied hooks object is the
    # same instance the runner uses, so per-datapoint and per-turn hooks share
    # one object. The deprecation nudge still fires once, in the runner ctor.
    resolved_hooks: SimulationHooks = hooks or DefaultHooks()
    result_cache: dict[int, SimulationResult] = {}

    async def job_fn(data: DataPoint, _row: int) -> dict[str, Any]:
        sim_dp = sim_dp_by_id.get(id(data))
        if sim_dp is None:
            raise RuntimeError(
                'DataPoint instance not found in sim_dp_by_id — '
                "simulate()'s scorer/result wiring requires the same "
                'DataPoint instance evaluatorq was given'
            )
        # evaluatorq parallelises datapoints and calls runner.run() per row, so
        # the per-datapoint lifecycle hooks (which live in run_batch, bypassed
        # here) are fired from the job instead. run() fires on_turn_complete
        # internally. A raising per-datapoint hook surfaces as this job's
        # failure (evaluatorq captures it per-row), matching run_batch's
        # gather(return_exceptions=True) isolation.
        #
        # on_datapoint_start is wrapped so a raise still surfaces via
        # on_datapoint_error + on_datapoint_complete for this datapoint (the
        # guarantee in SimulationHooks' docstring), mirroring run_batch — then
        # re-raised so evaluatorq records the job failure. Without this a
        # RichHooks task created in on_datapoint_start would never advance.
        try:
            await await_maybe(resolved_hooks.on_datapoint_start(sim_dp))
        except Exception as start_err:
            err = _error_result(str(start_err), sim_dp.persona, sim_dp.scenario)
            err.metadata['datapoint_id'] = sim_dp.id
            result_cache[id(data)] = err
            await await_maybe(resolved_hooks.on_datapoint_error(sim_dp, start_err))
            await await_maybe(resolved_hooks.on_datapoint_complete(err))
            raise
        result = await runner.run(datapoint=sim_dp, max_turns=max_turns)
        result.metadata['datapoint_id'] = sim_dp.id
        result_cache[id(data)] = result
        if result.terminated_by in (TerminatedBy.error, TerminatedBy.timeout):
            reason = result.metadata.get('error') or result.reason
            await await_maybe(resolved_hooks.on_datapoint_error(sim_dp, RuntimeError(reason)))
        await await_maybe(resolved_hooks.on_datapoint_complete(result))
        return {'name': job_name, 'output': to_open_responses(result, model)}

    return job_fn, result_cache, runner


def _adapt_simulation_scorer(
    name: str,
    sim_scorer: SimulationScorer,
    result_cache: dict[int, SimulationResult],
) -> Evaluator:
    """Wrap a SimulationScorer as an evaluatorq Evaluator.

    The evaluatorq scorer receives ``{data: DataPoint, output: Output}``;
    it recovers the raw ``SimulationResult`` from ``result_cache`` keyed by
    ``id(data)`` — the same DataPoint instance the upstream job cached against.

    Note: ``on_evaluator_complete`` is NOT fired here. evaluatorq's
    ``process_evaluator`` wraps this scorer in a try/except, so a hook raising
    inside it would be swallowed (recorded as a scorer error) rather than
    propagating. The hook is fired from :func:`_stamp_evaluator_scores`, which
    runs outside evaluatorq's guard, preserving the unguarded-propagation
    contract.
    """
    from evaluatorq.types import EvaluationResult, ScorerParameter

    async def scorer(params: ScorerParameter) -> EvaluationResult:
        # Both failure paths raise rather than returning a sentinel `0.0` —
        # evaluatorq's process_evaluator catches and records the error on
        # the EvaluatorScore, so callers (and exit_on_failure) see a real
        # failure instead of mistaking a degenerate score for a low result.
        data = params['data']
        sim_result = result_cache.get(id(data))
        if sim_result is None:
            logger.error(
                'scorer %r: no SimulationResult cached for DataPoint id=%r — '
                'upstream simulation job failed or did not run',
                name,
                id(data),
            )
            raise RuntimeError(f'missing simulation result for DataPoint id={id(data)} (scorer={name!r})')
        try:
            value = sim_scorer(sim_result)
        except Exception as e:
            logger.exception('scorer %r raised on DataPoint id=%s', name, id(data))
            sim_result.metadata.setdefault('scorer_errors', {})[name] = repr(e)
            raise
        return EvaluationResult(value=value)

    return {'name': name, 'scorer': scorer}


async def _simulate_via_evaluatorq(
    *,
    caller: str,
    evaluation_name: str,
    target_callback: Callable[[list[Message]], str | Awaitable[str]] | None,
    target_agent: AgentTarget | None,
    sim_datapoints: list[Datapoint],
    max_turns: int,
    model: str,
    evaluator_names: list[str] | None,
    parallelism: int,
    user_simulator: BaseAgent | None,
    judge: BaseAgent | None,
    upload_results: bool,
    evaluation_description: str | None,
    path: str | None,
    exit_on_failure: bool,
    pipeline_span: Span | None,
    hooks: SimulationHooks,
) -> list[SimulationResult]:
    """Wrap simulation Datapoints as evaluatorq DataPoints and run."""
    from datetime import datetime, timezone

    from evaluatorq.evaluatorq import evaluatorq
    from evaluatorq.simulation.evaluators import get_evaluator
    from evaluatorq.simulation.tracing import record_token_usage, set_span_attrs
    from evaluatorq.types import DataPoint

    resolved_evaluator_names = evaluator_names if evaluator_names is not None else ['goal_achieved', 'criteria_met']
    scorers = [(name, get_evaluator(name)) for name in resolved_evaluator_names]

    eq_datapoints = [DataPoint(inputs={'datapoint': dp.model_dump(mode='json')}) for dp in sim_datapoints]
    # Map the evaluatorq DataPoint instance back to its source simulation
    # Datapoint by identity, so the job can run the source directly without
    # re-parsing inputs["datapoint"] on every run.
    sim_dp_by_id = {id(eq): sim for eq, sim in zip(eq_datapoints, sim_datapoints, strict=True)}

    job_fn, result_cache, runner = _build_simulation_job_and_cache(
        job_name='simulation',
        sim_dp_by_id=sim_dp_by_id,
        target_callback=target_callback,
        target_agent=target_agent,
        model=model,
        max_turns=max_turns,
        user_simulator=user_simulator,
        judge=judge,
        hooks=hooks,
    )

    evaluators = [_adapt_simulation_scorer(name, fn, result_cache) for name, fn in scorers]

    start = datetime.now(tz=timezone.utc)
    run_name = evaluation_name or f'simulation-{start.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:8]}'

    try:
        eq_results = await evaluatorq(
            run_name,
            data=eq_datapoints,
            jobs=[job_fn],
            evaluators=evaluators,
            parallelism=parallelism,
            description=evaluation_description,
            path=path,
            _send_results=upload_results,
            _exit_on_failure=exit_on_failure,
        )
    finally:
        # Close the runner, then any target that owns resources (e.g.
        # OrqResponsesTarget built its own AsyncOpenAI client). Plain
        # callables have no close(); duck-type to avoid coupling to the type.
        try:
            await runner.close()
        except Exception:
            # Don't let runner-cleanup errors mask the primary exception
            # from evaluatorq() if there was one.
            logger.exception('SimulationRunner.close() raised during cleanup')
        target_close = getattr(target_agent or target_callback, 'close', None)
        if callable(target_close):
            try:
                maybe = target_close()
                if inspect.isawaitable(maybe):
                    await maybe
            except Exception:
                # Same guard as runner.close(): a target raising on close
                # must not mask the primary exception from evaluatorq().
                logger.exception('target close() raised during cleanup')

    # Mirror evaluator scores onto SimulationResult.metadata once, from the
    # final evaluatorq result, so the scorer stays pure (no side-effects mid-run)
    # and callers inspecting SimulationResult.metadata still find evaluator_scores.
    # Also fires on_evaluator_complete per (datapoint, evaluator) — done here,
    # outside evaluatorq's per-scorer try/except, so an unguarded hook raise
    # propagates (caught by _simulate_core's finally, which still fires
    # on_run_complete before re-raising).
    await _stamp_evaluator_scores(eq_results, result_cache, evaluation_name, hooks)

    expected = len(eq_datapoints)
    missing = [i for i, eq in enumerate(eq_datapoints) if id(eq) not in result_cache]
    if missing:
        msg = (
            f'{caller}(): {len(missing)} of {expected} simulation job(s) failed '
            f'and produced no result (missing rows: {missing})'
        )
        if exit_on_failure:
            raise SimulationDroppedError(msg)
        logger.warning(msg)

    results = [result_cache[id(eq)] for eq in eq_datapoints if id(eq) in result_cache]

    # Aggregate token usage onto the pipeline span.
    total_prompt = sum((r.token_usage.prompt_tokens or 0) for r in results)
    total_completion = sum((r.token_usage.completion_tokens or 0) for r in results)
    total_total = sum((r.token_usage.total_tokens or 0) for r in results)
    record_token_usage(
        pipeline_span,
        prompt_tokens=total_prompt,
        completion_tokens=total_completion,
        total_tokens=total_total,
    )
    set_span_attrs(
        pipeline_span,
        {
            'orq.simulation.results_count': len(results),
            'orq.simulation.goal_achieved_count': sum(1 for r in results if r.goal_achieved),
        },
    )

    return results


async def _stamp_evaluator_scores(
    eq_results: list[DataPointResult],
    result_cache: dict[int, SimulationResult],
    evaluation_name: str,
    hooks: SimulationHooks,
) -> None:
    """Walk the evaluatorq result, stamp each evaluator score onto the matching
    SimulationResult.metadata["evaluator_scores"], and fire on_evaluator_complete.

    Matches by ``id(data_point)`` — on success evaluatorq returns the same
    DataPoint instance the job cached against. Error rows carry a placeholder
    DataPoint whose id won't match, so they're skipped.

    ``on_evaluator_complete`` is fired here (not in the scorer) because this runs
    outside evaluatorq's per-scorer try/except, so an unguarded hook raise
    propagates instead of being swallowed as a scorer error. Driven via
    ``await_maybe`` for sync-or-async hooks.
    """
    from evaluatorq.common.async_utils import await_maybe

    for dp_result in eq_results:
        sim_result = result_cache.get(id(dp_result.data_point))
        if sim_result is None or not dp_result.job_results:
            continue
        datapoint_id = sim_result.metadata.get('datapoint_id', '')
        # Stamp evaluation_name before firing on_evaluator_complete so a hook
        # reading result.metadata['evaluation_name'] sees it (the hook receives
        # sim_result as its 4th arg).
        if evaluation_name:
            sim_result.metadata['evaluation_name'] = evaluation_name
        scores_dict = sim_result.metadata.setdefault('evaluator_scores', {})
        for job_result in dp_result.job_results:
            for score in job_result.evaluator_scores or []:
                # Simulation scorers always produce a numeric value (see
                # _adapt_simulation_scorer -> EvaluationResult(value=float)).
                value = cast('float', score.score.value)
                if isinstance(scores_dict, dict):
                    scores_dict[score.evaluator_name] = value
                await await_maybe(hooks.on_evaluator_complete(datapoint_id, score.evaluator_name, value, sim_result))
