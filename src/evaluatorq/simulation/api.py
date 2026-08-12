"""High-level simulation API functions.

`simulate()` and `generate_and_simulate()` route execution through the
`evaluatorq()` framework so they inherit auto-upload, OTel tracing, results
display, CI gating, and dataset-id support (RES-594 / RES-598). They run
inside an ``Evaluatorq - Agent Simulation`` span and accept the unified
``AgentTarget`` target shape introduced by RES-808.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evaluatorq.common.llm_client import resolve_results_base_url
from evaluatorq.common.thread_context import _evaluatorq_run_scope, build_thread_id, evaluatorq_pipeline
from evaluatorq.simulation._config import SimulationConfig
from evaluatorq.simulation.types import DEFAULT_EVALUATOR_NAMES, DEFAULT_MAX_TURNS, DEFAULT_MODEL
from evaluatorq.simulation.utils.run_store import auto_save_run, build_simulation_run, fetch_agent_info, write_report

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from openai import AsyncOpenAI
    from opentelemetry.trace import Span

    from evaluatorq.common.run_manifest import ManifestWriter
    from evaluatorq.contracts import AgentResponse, AgentTarget
    from evaluatorq.simulation.agents.base import BaseAgent
    from evaluatorq.simulation.evaluators.scorers import SimulationScorer
    from evaluatorq.simulation.generators import FirstMessageGenerator
    from evaluatorq.simulation.hooks import SimulationHooks
    from evaluatorq.simulation.replay import SimulationReplay
    from evaluatorq.simulation.types import (
        Message,
        Persona,
        Scenario,
        SimulationDatapoint,
        SimulationResult,
        SimulationRun,
    )
    from evaluatorq.types import DataPoint, DataPointResult, Evaluator

    EmitDatapoints = Callable[[list[SimulationDatapoint]], None]
    RunPostProcessor = Callable[[SimulationRun], Awaitable[None]]

logger = logging.getLogger(__name__)

# Re-exported from here for back-compat (``simulation.__init__`` imports it from
# api); the canonical definition lives in ``simulation.exceptions``.
from evaluatorq.simulation.exceptions import SimulationDroppedError


def _agent_key_of(target: object) -> str | None:
    """Best-effort bare agent key from an AgentTarget, or None if not an agent.

    ``ORQAgentTarget`` exposes ``.agent_key`` directly. The Responses-router
    ``OrqResponsesTarget`` (what ``agent:<key>`` actually resolves to) has no
    such attribute — it carries the key as ``config.model == 'agent/<key>'``.
    Both paths must yield the bare ``<key>`` so the run label and the
    ``fetch_agent_info`` snapshot use the real key instead of a literal 'agent'.
    """
    key = getattr(target, 'agent_key', None)
    if key:
        return key
    model = getattr(getattr(target, 'config', None), 'model', None)
    if isinstance(model, str) and model.startswith('agent/'):
        return model.removeprefix('agent/') or None
    return None


# Method names inspected for the sync-hook deprecation nudge. Fired per user
# child BEFORE composing — once composed, the async CompositeSimulationHooks
# would mask a sync child from the runner's own (now-redundant) check.
_SIM_HOOK_METHOD_NAMES: tuple[str, ...] = (
    'on_confirm',
    'on_run_start',
    'on_stage_start',
    'on_stage_end',
    'on_generate_inputs_ready',
    'on_datapoint_start',
    'on_turn_complete',
    'on_datapoint_complete',
    'on_evaluator_complete',
    'on_datapoint_error',
    'on_run_complete',
)


def _compose_sim_hooks(
    hooks: SimulationHooks | Sequence[SimulationHooks] | None,
    *,
    save: bool,
    run_id: str,
    run_name: str,
    run_output: str | Path | None,
) -> tuple[SimulationHooks, ManifestWriter | None]:
    """Compose user hooks (+ manifest) once, at the outer entry point.

    Normalises ``hooks`` to a list, warns per sync child, then builds one
    ``CompositeSimulationHooks``. When ``save`` is set, a fresh
    ``ManifestWriter`` is minted and its ``ManifestStageHooks`` registered
    FIRST so stage status is durable before any user hook runs; the raw writer
    is returned alongside for the runner's terminal ``complete``/``cancel``/
    ``fail`` calls (§4.5). When not saving there is no manifest (``None``).

    Preserves today's default: with no user hooks a ``DefaultHooks()`` is still
    composed in, so the confirm-plan log + completion summary aren't lost.
    """
    from evaluatorq.common.hook_compose import compose_run_hooks
    from evaluatorq.simulation.hooks import CompositeSimulationHooks, DefaultHooks, ManifestStageHooks

    def _make_manifest() -> ManifestWriter:
        from evaluatorq.common.run_manifest import start_manifest
        from evaluatorq.simulation.utils.run_store import get_sim_runs_dir

        manifest_runs_dir = Path(run_output).parent if run_output is not None else get_sim_runs_dir()
        return start_manifest(
            run_id=run_id,
            surface='sim',
            run_name=run_name,
            runs_dir=manifest_runs_dir,
        )

    composed, manifest_writer = compose_run_hooks(
        hooks,
        method_names=_SIM_HOOK_METHOD_NAMES,
        composite_cls=CompositeSimulationHooks,
        default_hooks_factory=DefaultHooks,
        manifest_factory=_make_manifest if save else None,
        manifest_hook_factory=ManifestStageHooks,
    )
    return composed, manifest_writer


async def simulate(
    *,
    evaluation_name: str = '',
    target: str | Callable[[list[Message]], str | Awaitable[str]] | AgentTarget | None = None,
    personas: list[Persona] | None = None,
    scenarios: list[Scenario] | None = None,
    datapoints: list[SimulationDatapoint] | None = None,
    dataset_id: str | None = None,
    experiment_id: str | None = None,
    experiment_run_id: str | None = None,
    memory_entity_id: str | None = None,
    previous_run: str | None = None,
    max_turns: int | None = None,
    sim_model: str = DEFAULT_MODEL,
    evaluator_names: list[str] | None = None,
    parallelism: int = 5,
    user_simulator: BaseAgent | None = None,
    judge: BaseAgent | None = None,
    hooks: SimulationHooks | Sequence[SimulationHooks] | None = None,
    generation_client: AsyncOpenAI | None = None,
    upload_results: bool = True,
    evaluation_description: str | None = None,
    orq_results_path: str | None = None,
    exit_on_failure: bool = True,
    save: bool = False,
    report: str | Path | None = None,
    executive_summary: bool = True,
) -> list[SimulationResult]:
    """Run agent simulations through the evaluatorq() framework.

    Builds simulation Datapoints (cartesian persona x scenario when needed),
    wraps them as evaluatorq ``DataPoint``s, and delegates execution,
    parallelism, tracing, results display, and (by default) upload to
    ``evaluatorq()``. Returns the raw ``SimulationResult`` list so existing
    callers continue to work.

    Args:
        target: The agent under test. Accepts a ``str`` (``"agent:<key>"`` or bare
            ``"<key>"`` → hosted Orq agent via the Responses router;
            ``"deployment:<key>"`` → Orq deployment), an ``AgentTarget``
            instance (routed to the runner's ``target_agent`` path; speaks
            ``respond(messages)``), or a callable that receives the conversation
            history and returns the agent's response.
        user_simulator: Pre-constructed ``BaseAgent`` to drive the user side.
            When omitted a default ``UserSimulatorAgent`` is built from
            ``sim_model``. ``sim_model`` drives the user-simulator, the judge,
            and datapoint generators.
        judge: Pre-constructed ``BaseAgent`` used to evaluate each turn.
        hooks: Optional ``SimulationHooks`` for run/datapoint/turn lifecycle
            events. Sync or async; ``async def`` is preferred (a sync hook
            works but emits a one-time ``DeprecationWarning``). Defaults to
            ``DefaultHooks`` (loguru baseline).
        generation_client: Optional pre-built ``AsyncOpenAI`` used for datapoint
            generation (first-message). When omitted, generation falls back to
            the same env-based provider resolution as
            `generate_and_simulate` (``ORQ_API_KEY`` → ``OPENAI_API_KEY``).
        dataset_id: When set, fetch simulation datapoints from the named Orq
            dataset instead of taking them inline. Mutually exclusive with
            ``datapoints``, ``personas``, ``scenarios``. Each dataset row's
            ``inputs`` must already match one of the simulation input shapes
            (``datapoint`` / ``persona`` + ``scenario`` / etc.).
        experiment_id: When set, fetch simulation datapoints from the named Orq
            experiment's rows instead of taking them inline (direct mode of
            "experiments as input"). Same mutual exclusivity and row-shape
            rules as ``dataset_id``; rows uploaded by a previous simulation
            run round-trip as-is. Requires ``ORQ_API_KEY``. To generate *new*
            datapoints seeded by an experiment instead, see
            `evaluatorq.simulation.extend_from_experiment`.
        experiment_run_id: A specific run (manifest) of ``experiment_id`` to
            load. Latest run when omitted. Only valid with ``experiment_id``.
        memory_entity_id: Memory ``entity_id`` sent with every call to an
            ``agent:<key>`` (or bare ``<key>``) target. The Responses router
            requires a memory scope when the target agent has a memory store
            attached; pass this to run against a specific (e.g. seeded) entity.
            When omitted, a fresh id is minted per conversation so memory-backed
            agents work out of the box and parallel conversations never share
            memory. One explicit id is shared across the run's conversations.
            Only valid for string agent targets.
        max_turns: Cap on conversation turns. Defaults to 10, or — when
            replaying via ``previous_run`` — to the cap the replayed run used.
            An explicit value always wins.
        previous_run: Replay a saved run instead of building new cases. Accepts
            a run's file name, its run id (full or an unambiguous 8+ character
            prefix), a path to a saved run JSON, or ``"latest"``, resolved
            against ``.evaluatorq/sim-runs/``. The stored personas, scenarios,
            and first messages are re-used verbatim — no generation, no dataset
            fetch — so only the target and evaluators change between runs.
            Mutually exclusive with ``datapoints``, ``dataset_id``,
            ``experiment_id``, and ``personas``/``scenarios``.
        upload_results: When ``True`` (the default) and ``ORQ_API_KEY`` is set,
            results are uploaded to the Orq platform as an experiment. Pass
            ``False`` to suppress the upload (e.g. for local-only runs).
        evaluation_description: Optional human-readable note passed straight
            through to ``evaluatorq(description=...)`` and shown as the
            experiment's description in the Orq UI. Pure metadata — nothing
            branches on it, and it only matters when results are uploaded
            (``upload_results=True``). Leave ``None`` for local-only runs.
        orq_results_path: Optional Orq folder path (e.g. ``"MyProject/MyFolder"``).
        exit_on_failure: When ``True`` (the default), exit non-zero if any
            datapoint was *dropped* — a job raised with no result cached —
            by raising ``SimulationDroppedError`` from ``simulate()`` itself
            (the "CI gating for free" benefit). Scorer verdicts (``pass_``,
            e.g. goal not achieved) are reporting only and never exit the
            process: an underperforming but otherwise healthy run still
            returns its results. Pass ``False`` for interactive / exploratory
            runs where even dropped rows should surface as warnings + error
            metadata instead.
        save: When ``True``, persist the completed run to the local run store
            (``.evaluatorq/sim-runs/`` unless ``report`` is set). Unlike the CLI
            (which auto-saves to ``.evaluatorq/sim-runs/`` by default), the SDK
            defaults to ``save=False`` — the caller opts in.
        report: Optional path to write the full SimulationRun report JSON
            (results + scorer averages + metadata). When omitted and ``save``
            is ``True``, the run is auto-saved under ``.evaluatorq/sim-runs/``.
        executive_summary: When ``True`` (the default), generate the LLM
            narrative summary and store it on the returned run — and in any
            saved file — so the dashboard shows saved prose instead of the
            computed fallback sentence. Best-effort: no-op without LLM creds.
            Set ``False`` to skip the extra LLM call.

    Usage:

    ```python
    import asyncio

    from evaluatorq.simulation import CommunicationStyle, Criterion, Persona, Scenario, simulate

    persona = Persona(
        name='Impatient Customer',
        patience=0.2,
        assertiveness=0.8,
        politeness=0.4,
        technical_level=0.3,
        communication_style=CommunicationStyle.terse,
        background='Wants a refund urgently',
    )
    scenario = Scenario(
        name='Refund',
        goal='Get a full refund',
        criteria=[Criterion(description='Agent asks for order details', type='must_happen')],
    )


    async def support_agent(messages: list) -> str:
        return 'Thanks for reaching out. How can I assist you today?'


    async def main() -> None:
        results = await simulate(
            evaluation_name='basic-simulation-example',
            target=support_agent,
            personas=[persona],
            scenarios=[scenario],
            max_turns=6,
            evaluator_names=['goal_achieved', 'criteria_met'],
        )
        print(f'{sum(r.goal_achieved for r in results)}/{len(results)} reached the goal')


    asyncio.run(main())
    ```
    """
    run = await _simulate_run(
        evaluation_name=evaluation_name,
        target=target,
        personas=personas,
        scenarios=scenarios,
        datapoints=datapoints,
        dataset_id=dataset_id,
        experiment_id=experiment_id,
        experiment_run_id=experiment_run_id,
        memory_entity_id=memory_entity_id,
        previous_run=previous_run,
        max_turns=max_turns,
        sim_model=sim_model,
        evaluator_names=evaluator_names,
        parallelism=parallelism,
        user_simulator=user_simulator,
        judge=judge,
        hooks=hooks,
        generation_client=generation_client,
        upload_results=upload_results,
        evaluation_description=evaluation_description,
        orq_results_path=orq_results_path,
        exit_on_failure=exit_on_failure,
        save=save,
        report=report,
        executive_summary=executive_summary,
    )
    return run.results


async def _simulate_run(
    *,
    evaluation_name: str = '',
    target: str | Callable[[list[Message]], str | Awaitable[str]] | AgentTarget | None = None,
    personas: list[Persona] | None = None,
    scenarios: list[Scenario] | None = None,
    datapoints: list[SimulationDatapoint] | None = None,
    dataset_id: str | None = None,
    experiment_id: str | None = None,
    experiment_run_id: str | None = None,
    memory_entity_id: str | None = None,
    previous_run: str | None = None,
    max_turns: int | None = None,
    sim_model: str = DEFAULT_MODEL,
    evaluator_names: list[str] | None = None,
    parallelism: int = 5,
    user_simulator: BaseAgent | None = None,
    judge: BaseAgent | None = None,
    hooks: SimulationHooks | Sequence[SimulationHooks] | None = None,
    generation_client: AsyncOpenAI | None = None,
    upload_results: bool = True,
    evaluation_description: str | None = None,
    orq_results_path: str | None = None,
    exit_on_failure: bool = True,
    save: bool = False,
    report: str | Path | None = None,
    executive_summary: bool = False,
    post_run: RunPostProcessor | None = None,
) -> SimulationRun:
    """Internal counterpart of `simulate` that returns the full ``SimulationRun``.

    Same keyword-only signature as `simulate` (which is a thin
    ``.results`` unwrapper around this). Exists so callers that need the full
    run (e.g. the CLI, for its experiment URL / executive summary / save
    plumbing) don't have to rebuild it from the results list.
    """
    from evaluatorq.simulation.exceptions import SimulationCancelledError
    from evaluatorq.simulation.tracing import with_simulation_span
    from evaluatorq.tracing.setup import flush_tracing, init_tracing_if_needed

    await init_tracing_if_needed()

    # Mint the run id + manifest at the outer entry so the same id flows to the
    # manifest and to _simulate_core's experiment linking, and so the manifest
    # brackets the whole run (bare simulate has only the SIMULATE stage — no
    # GENERATE phase). Minted before the span opens so it can be stamped as a
    # span attribute rather than set after the fact.
    run_id = uuid.uuid4().hex

    try:
        async with with_simulation_span(
            'Evaluatorq - Agent Simulation',
            {
                'orq.simulation.evaluation_name': evaluation_name,
                # max_turns is None until _simulate_core resolves it (an explicit
                # value, else a replayed run's cap, else the default), and
                # set_span_attrs drops None — so the resolved value is stamped
                # there rather than guessed here.
                'orq.simulation.parallelism': parallelism,
            },
        ) as pipeline_span:
            # Compose once; the manifest hook is registered first, and the raw
            # writer is retained for terminal complete/cancel/fail calls. The
            # run-id bind is what reaches every LLM call — a ContextVar set here
            # is visible to the nested evaluatorq() and copied into child tasks.
            with _evaluatorq_run_scope(run_id, pipeline_span), evaluatorq_pipeline('agent_simulation'):
                composed_hooks, manifest_writer = _compose_sim_hooks(
                    hooks,
                    save=save,
                    run_id=run_id,
                    run_name=evaluation_name or 'sim',
                    run_output=report,
                )
                # Outer manifest guard (FIX 1): _simulate_core owns the terminal
                # transition, but any failure between manifest creation and
                # _simulate_core (e.g. building SimulationConfig) would otherwise
                # leave the manifest stuck 'running'. Terminal calls are idempotent,
                # so this composes safely with _simulate_core's own finalization.
                try:
                    config = SimulationConfig(
                        evaluation_name=evaluation_name,
                        target=target,
                        personas=personas,
                        scenarios=scenarios,
                        datapoints=datapoints,
                        dataset_id=dataset_id,
                        experiment_id=experiment_id,
                        experiment_run_id=experiment_run_id,
                        memory_entity_id=memory_entity_id,
                        previous_run=previous_run,
                        max_turns=max_turns,
                        model=sim_model,
                        evaluator_names=evaluator_names,
                        parallelism=parallelism,
                        user_simulator=user_simulator,
                        judge=judge,
                        generation_client=generation_client,
                        upload_results=upload_results,
                        evaluation_description=evaluation_description,
                        orq_results_path=orq_results_path,
                        exit_on_failure=exit_on_failure,
                        save=save,
                        run_output=report,
                        executive_summary=executive_summary,
                        hooks=composed_hooks,
                    )
                    return await _simulate_core(
                        config=config,
                        caller='simulate',
                        pipeline_span=pipeline_span,
                        run_id=run_id,
                        manifest_writer=manifest_writer,
                        post_run=post_run,
                    )
                except SimulationCancelledError:
                    if manifest_writer is not None:
                        manifest_writer.cancel()
                    raise
                except BaseException as exc:
                    if manifest_writer is not None:
                        manifest_writer.fail(str(exc) or type(exc).__name__)
                    raise
    finally:
        await flush_tracing()


async def generate_and_simulate(
    *,
    evaluation_name: str = '',
    agent_description: str | None = None,
    target: str | Callable[[list[Message]], str | Awaitable[str]] | AgentTarget | None = None,
    memory_entity_id: str | None = None,
    num_personas: int = 5,
    num_scenarios: int = 5,
    max_turns: int | None = None,
    sim_model: str = DEFAULT_MODEL,
    evaluator_names: list[str] | None = None,
    parallelism: int = 5,
    user_simulator: BaseAgent | None = None,
    judge: BaseAgent | None = None,
    hooks: SimulationHooks | Sequence[SimulationHooks] | None = None,
    generation_client: AsyncOpenAI | None = None,
    upload_results: bool = True,
    evaluation_description: str | None = None,
    orq_results_path: str | None = None,
    exit_on_failure: bool = True,
    emit_datapoints: EmitDatapoints | None = None,
    save: bool = False,
    report: str | Path | None = None,
    executive_summary: bool = True,
) -> list[SimulationResult]:
    """Generate personas/scenarios, then run simulations via evaluatorq().

    Accepts the same ``target`` shapes as `simulate` — a plain callable,
    an ``AgentTarget`` instance, or a string (``"agent:<key>"`` / bare ``"<key>"``
    for a hosted Orq agent, ``"deployment:<key>"`` for the Orq deployment bridge).
    ``memory_entity_id`` mirrors `simulate` too: it is sent as the memory
    scope with every call to a string agent target (``agent:<key>`` or bare
    ``<key>``), for agents with a memory store attached. When omitted, a fresh
    per-target id is minted so memory-backed agents still work out of the box.
    Persona/scenario/first-message generation resolves its provider via
    the shared factory: an injected ``generation_client`` → ``ORQ_API_KEY``
    (Orq router) → ``OPENAI_API_KEY`` (with optional ``OPENAI_BASE_URL`` for an
    OpenAI-compatible endpoint). No API key is needed when a client is injected.
    ``sim_model`` drives persona/scenario/first-message generation, the
    user-simulator, and the judge. ``upload_results`` defaults to ``True``; set
    it to ``False`` to skip uploading the final experiment. ``exit_on_failure``
    defaults to ``True``; see `simulate` for the full semantics of the
    CI-gate behaviour and how to opt out. ``hooks`` mirrors `simulate`;
    note the ``on_confirm`` gate fires AFTER persona/scenario/first-message
    generation, so those generation tokens are already spent when the gate is
    consulted. ``agent_description`` takes precedence when supplied. Otherwise,
    an ``agent:<key>`` target (or bare agent key) resolves its description from
    Orq; other targets must provide ``agent_description``. A missing or blank
    description from both sources raises ``ValueError`` before generation begins.

    ``evaluation_description``: Optional human-readable note passed straight
    through to the uploaded experiment (shown as its description in the Orq UI).
    Pure metadata — nothing branches on it, and it only matters when
    ``upload_results`` is ``True``. Leave ``None`` for local-only runs.

    ``emit_datapoints``: Optional callback invoked with the generated datapoints
    before simulation — used by the CLI's ``--datapoints`` to persist the
    exact inputs.

    ``save``: When ``True``, persist the completed run to the local run store
    (``.evaluatorq/sim-runs/`` unless ``report`` is set). Unlike the CLI (which
    auto-saves to ``.evaluatorq/sim-runs/`` by default), the SDK defaults to
    ``save=False`` — the caller opts in.

    ``report``: Optional path to write the full SimulationRun report JSON
    (results + scorer averages + metadata). When omitted and ``save`` is
    ``True``, the run is auto-saved under ``.evaluatorq/sim-runs/``.

    ``executive_summary``: When ``True`` (the default), generate the LLM
    narrative summary and store it on the returned run — and in any saved file.
    Best-effort: no-op without LLM creds. Set ``False`` to skip the LLM call.

    Usage:

    ```python
    import asyncio

    from evaluatorq.simulation import generate_and_simulate


    async def main() -> None:
        results = await generate_and_simulate(
            evaluation_name='support-agent-sim',
            target='agent:my-support-agent',  # hosted Orq agent, routed via ORQ_API_KEY
            agent_description=(
                'Customer support agent for an e-commerce store; handles refunds, orders, and product questions.'
            ),
            num_personas=3,
            num_scenarios=4,  # -> 12 persona x scenario simulations
            max_turns=6,
            evaluator_names=['goal_achieved', 'criteria_met'],
        )
        passed = sum(r.goal_achieved for r in results)
        print(f'Pass rate: {passed}/{len(results)}')


    asyncio.run(main())
    ```
    """
    run = await _generate_and_simulate_run(
        evaluation_name=evaluation_name,
        agent_description=agent_description,
        target=target,
        memory_entity_id=memory_entity_id,
        num_personas=num_personas,
        num_scenarios=num_scenarios,
        max_turns=max_turns,
        sim_model=sim_model,
        evaluator_names=evaluator_names,
        parallelism=parallelism,
        user_simulator=user_simulator,
        judge=judge,
        hooks=hooks,
        generation_client=generation_client,
        upload_results=upload_results,
        evaluation_description=evaluation_description,
        orq_results_path=orq_results_path,
        exit_on_failure=exit_on_failure,
        emit_datapoints=emit_datapoints,
        save=save,
        report=report,
        executive_summary=executive_summary,
    )
    return run.results


async def _generate_datapoints_inner(
    *,
    caller: str,
    agent_description: str,
    num_personas: int,
    num_scenarios: int,
    model: str,
    generation_client: AsyncOpenAI | None,
    hooks: SimulationHooks | None = None,
    persona_seeds: list[str] | None = None,
    scenario_seeds: list[str] | None = None,
) -> tuple[list[SimulationDatapoint], AsyncOpenAI, bool]:
    """Shared generation trio: personas/scenarios + first-message datapoints.

    Builds one shared generation client (mirrors the pattern both callers
    used inline) and runs the two-step pipeline shared by `generate`
    and `_generate_and_simulate_run`: ``_generate_personas_scenarios``
    followed by ``_resolve_or_generate_datapoints``, with the
    ``on_generate_inputs_ready`` hook fired in between exactly as both
    callers previously did inline. Must be called from inside the caller's
    own tracing span so child spans nest correctly (span nesting is
    implicit via OTel contextvars).

    Returns ``(datapoints, gen_client, gen_owned)``. Callers own closing the
    client: `generate` has no further use for it and closes it right
    away, while `_generate_and_simulate_run` keeps it open to pass
    into ``SimulationConfig.generation_client`` for the simulate stage and
    closes it only after ``_simulate_core`` returns. On any exception raised
    from within this helper (before the client is handed back to the
    caller), the client is closed here so it can't leak.
    """
    from evaluatorq.common.async_utils import await_maybe
    from evaluatorq.openresponses.client import build_simulation_client
    from evaluatorq.simulation.hooks import DefaultHooks

    gen_client, gen_owned = build_simulation_client(generation_client)
    try:
        gen_hooks = hooks or DefaultHooks()
        gen_personas, gen_scenarios = await _generate_personas_scenarios(
            agent_description=agent_description,
            num_personas=num_personas,
            num_scenarios=num_scenarios,
            model=model,
            generation_client=gen_client,
            persona_seeds=persona_seeds,
            scenario_seeds=scenario_seeds,
        )
        await await_maybe(gen_hooks.on_generate_inputs_ready(len(gen_personas), len(gen_scenarios)))

        datapoints = await _resolve_or_generate_datapoints(
            caller=caller,
            datapoints=None,
            personas=gen_personas,
            scenarios=gen_scenarios,
            dataset_id=None,
            experiment_id=None,
            experiment_run_id=None,
            previous_run=None,
            model=model,
            generation_client=gen_client,
        )
    except BaseException:
        if gen_owned:
            await gen_client.close()
        raise
    return datapoints, gen_client, gen_owned


async def _generate_and_simulate_run(
    *,
    evaluation_name: str = '',
    agent_description: str | None = None,
    target: str | Callable[[list[Message]], str | Awaitable[str]] | AgentTarget | None = None,
    memory_entity_id: str | None = None,
    num_personas: int = 5,
    num_scenarios: int = 5,
    max_turns: int | None = None,
    sim_model: str = DEFAULT_MODEL,
    evaluator_names: list[str] | None = None,
    parallelism: int = 5,
    user_simulator: BaseAgent | None = None,
    judge: BaseAgent | None = None,
    hooks: SimulationHooks | Sequence[SimulationHooks] | None = None,
    generation_client: AsyncOpenAI | None = None,
    upload_results: bool = True,
    evaluation_description: str | None = None,
    orq_results_path: str | None = None,
    exit_on_failure: bool = True,
    emit_datapoints: EmitDatapoints | None = None,
    save: bool = False,
    report: str | Path | None = None,
    executive_summary: bool = False,
    post_run: RunPostProcessor | None = None,
) -> SimulationRun:
    """Internal counterpart of `generate_and_simulate` returning the full ``SimulationRun``.

    Same keyword-only signature as `generate_and_simulate` (which is a
    thin ``.results`` unwrapper around this). See `_simulate_run` for
    why this split exists.
    """
    from evaluatorq.common.async_utils import await_maybe
    from evaluatorq.simulation.exceptions import SimulationCancelledError
    from evaluatorq.simulation.hooks import SimStage
    from evaluatorq.simulation.tracing import with_simulation_span
    from evaluatorq.tracing.setup import flush_tracing, init_tracing_if_needed

    await init_tracing_if_needed()

    # Mint the run id + manifest at the outer entry, BEFORE the generate phase,
    # gated on save (D3: the GENERATE stage must reach the manifest too). The SAME
    # composed hooks thread through both the generate phase and _simulate_core, so
    # both stages are recorded. Minted before the span opens so it can be stamped
    # as a span attribute rather than set after the fact.
    run_id = uuid.uuid4().hex

    try:
        async with with_simulation_span(
            'Evaluatorq - Agent Simulation',
            {
                'orq.simulation.evaluation_name': evaluation_name,
                'orq.simulation.mode': 'generate_and_simulate',
                'orq.simulation.num_personas': num_personas,
                'orq.simulation.num_scenarios': num_scenarios,
                # max_turns is None until _simulate_core resolves it (an explicit
                # value, else a replayed run's cap, else the default), and
                # set_span_attrs drops None — so the resolved value is stamped
                # there rather than guessed here.
                'orq.simulation.parallelism': parallelism,
            },
        ) as pipeline_span:
            # The raw writer is retained for terminal complete/cancel/fail calls.
            with _evaluatorq_run_scope(run_id, pipeline_span), evaluatorq_pipeline('agent_simulation'):
                composed_hooks, manifest_writer = _compose_sim_hooks(
                    hooks,
                    save=save,
                    run_id=run_id,
                    run_name=evaluation_name or 'sim',
                    run_output=report,
                )
                # Outer manifest guard (FIX 1): the terminal manifest calls live in
                # _simulate_core, but a failure in the GENERATE phase (before
                # _simulate_core is ever reached) would otherwise leave the manifest
                # stuck 'running'. Guard the whole generate→simulate span: a declined
                # confirm cancels, any other failure fails. Terminal calls are
                # idempotent (guarded by status != RUNNING), so this composes safely
                # with _simulate_core's own finalization — whichever fires first wins.
                try:
                    resolved_agent_description = await _resolve_generation_agent_description(
                        agent_description=agent_description,
                        target=target,
                    )
                    datapoints: list[SimulationDatapoint] = []
                    gen_client: AsyncOpenAI | None = None
                    gen_owned = False
                    # One try/finally owns the client close so it fires even if the
                    # emit_datapoints callback raises between generation and simulate.
                    try:
                        await await_maybe(composed_hooks.on_stage_start(SimStage.GENERATE, {}))
                        try:
                            datapoints, gen_client, gen_owned = await _generate_datapoints_inner(
                                caller='generate_and_simulate',
                                agent_description=resolved_agent_description,
                                num_personas=num_personas,
                                num_scenarios=num_scenarios,
                                model=sim_model,
                                generation_client=generation_client,
                                hooks=composed_hooks,
                            )
                            if emit_datapoints is not None:
                                emit_datapoints(datapoints)
                        finally:
                            # Thread the in-flight exception into the GENERATE
                            # stage-end meta so a failed generate records the stage as
                            # 'error', mirroring SIMULATE (§4.4). None on the success
                            # path → 'completed'.
                            meta: dict[str, Any] = {'num_datapoints': len(datapoints)} if datapoints else {}
                            meta['error'] = sys.exc_info()[1]
                            await await_maybe(composed_hooks.on_stage_end(SimStage.GENERATE, meta))

                        config = SimulationConfig(
                            evaluation_name=evaluation_name,
                            target=target,
                            personas=None,
                            scenarios=None,
                            datapoints=datapoints,
                            dataset_id=None,
                            memory_entity_id=memory_entity_id,
                            max_turns=max_turns,
                            model=sim_model,
                            evaluator_names=evaluator_names,
                            parallelism=parallelism,
                            user_simulator=user_simulator,
                            judge=judge,
                            generation_client=gen_client,
                            upload_results=upload_results,
                            evaluation_description=evaluation_description,
                            orq_results_path=orq_results_path,
                            exit_on_failure=exit_on_failure,
                            save=save,
                            run_output=report,
                            executive_summary=executive_summary,
                            hooks=composed_hooks,
                        )
                        return await _simulate_core(
                            config=config,
                            caller='generate_and_simulate',
                            pipeline_span=pipeline_span,
                            run_id=run_id,
                            manifest_writer=manifest_writer,
                            post_run=post_run,
                        )
                    finally:
                        if gen_owned and gen_client is not None:
                            await gen_client.close()
                except SimulationCancelledError:
                    if manifest_writer is not None:
                        manifest_writer.cancel()
                    raise
                except BaseException as exc:
                    if manifest_writer is not None:
                        manifest_writer.fail(str(exc) or type(exc).__name__)
                    raise
    finally:
        await flush_tracing()


async def generate(
    *,
    agent_description: str,
    num_personas: int = 5,
    num_scenarios: int = 5,
    sim_model: str = DEFAULT_MODEL,
    hooks: SimulationHooks | Sequence[SimulationHooks] | None = None,
    generation_client: AsyncOpenAI | None = None,
    persona_seeds: list[str] | None = None,
    scenario_seeds: list[str] | None = None,
) -> list[SimulationDatapoint]:
    """Generate ready-to-run simulation ``SimulationDatapoint``s from an agent description.

    Produces personas and scenarios, then builds one ``SimulationDatapoint`` per
    persona x scenario pair (each with a generated first message). Returns the
    datapoints without running any simulation — feed them to `simulate`
    via ``datapoints=...``, or persist them (e.g. JSONL) and reuse.

    Pass ``persona_seeds`` / ``scenario_seeds`` to steer a dimension: each seed
    is an archetype (e.g. ``"angry retiree"``) the LLM fleshes out into one full
    object, overriding ``num_personas`` / ``num_scenarios`` for that dimension.
    The other dimension still auto-generates. Seeded x auto still crosses into
    the full persona x scenario grid.

    This freezes the simulation *inputs* (personas, scenarios, first messages)
    so every `simulate` run scores the same fixed dataset — useful for
    apples-to-apples comparison across agent versions. It does **not** make
    simulation deterministic: the agent under test, the user-simulator, and the
    judge remain stochastic at simulate time, so scores still vary run to run.

    Provider resolution matches `generate_and_simulate`: an injected
    ``generation_client`` → ``ORQ_API_KEY`` (Orq router) → ``OPENAI_API_KEY``
    (with optional ``OPENAI_BASE_URL`` for an OpenAI-compatible endpoint).
    ``sim_model`` drives persona/scenario/first-message generation.
    """
    from evaluatorq.common.async_utils import await_maybe
    from evaluatorq.simulation.hooks import SimStage
    from evaluatorq.simulation.tracing import with_simulation_span
    from evaluatorq.tracing.setup import flush_tracing, init_tracing_if_needed

    await init_tracing_if_needed()

    # Minted before the span opens so it can be stamped as a span attribute.
    run_id = uuid.uuid4().hex

    try:
        async with with_simulation_span(
            'orq.simulation.generate',
            {
                'orq.simulation.mode': 'generate',
                'orq.simulation.num_personas': num_personas,
                'orq.simulation.num_scenarios': num_scenarios,
            },
        ) as pipeline_span:
            with _evaluatorq_run_scope(run_id, pipeline_span), evaluatorq_pipeline('agent_simulation'):
                # Bracket generation with the same GENERATE stage hooks the
                # generate_and_simulate path uses, so the standalone command
                # isn't silent. Empty on_stage_end meta: the CLI prints its own
                # "✓ Generated N datapoint(s)" line, so the count isn't doubled.
                gen_hooks, _ = _compose_sim_hooks(
                    hooks,
                    save=False,
                    run_id='',
                    run_name='generate',
                    run_output=None,
                )
                await await_maybe(gen_hooks.on_stage_start(SimStage.GENERATE, {}))
                try:
                    datapoints, gen_client, gen_owned = await _generate_datapoints_inner(
                        caller='generate',
                        agent_description=agent_description,
                        num_personas=num_personas,
                        num_scenarios=num_scenarios,
                        model=sim_model,
                        generation_client=generation_client,
                        hooks=gen_hooks,
                        persona_seeds=persona_seeds,
                        scenario_seeds=scenario_seeds,
                    )
                finally:
                    await await_maybe(gen_hooks.on_stage_end(SimStage.GENERATE, {}))
                # generate() has no further use for the client (unlike
                # _generate_and_simulate_run, which keeps it open for the
                # simulate stage), so close it here once generation is done.
                if gen_owned:
                    await gen_client.close()
                return datapoints
    finally:
        await flush_tracing()


# ---------------------------------------------------------------------------
# Seeded generation — the intermediate tier (archetype → full object)
# ---------------------------------------------------------------------------


async def generate_personas(
    seeds: list[str],
    *,
    agent_description: str = '',
    context: str = '',
    sim_model: str = DEFAULT_MODEL,
    generation_client: AsyncOpenAI | None = None,
) -> list[Persona]:
    """Generate one ``Persona`` per archetype seed (e.g. ``"angry customer"``).

    The intermediate tier between fully-auto generation
    (`generate_and_simulate`) and hand-built ``Persona`` objects: you name
    each archetype, the LLM fills every trait. Provider resolves via the shared
    factory (``ORQ_API_KEY`` → ``OPENAI_API_KEY``) unless ``generation_client``
    is injected.
    """
    from evaluatorq.simulation.exceptions import SimulationError
    from evaluatorq.simulation.generators import PersonaGenerator

    if not seeds:
        raise ValueError('generate_personas requires at least one seed')
    description = agent_description or 'a general-purpose conversational assistant'
    # No run id is bound here: this entrypoint owns no root span, so an id would
    # be undiscoverable. The calls are still traced — PersonaGenerator.generate
    # opens `orq.simulation.persona_generation` with a child LLM span, and on the
    # sim paths those already nest under the run's root span.
    gen = PersonaGenerator(model=sim_model, client=generation_client)
    try:
        batches = await asyncio.gather(*[
            gen.generate(
                agent_description=description,
                context=context,
                num_personas=1,
                edge_case_percentage=0.0,
                seed=seed,
            )
            for seed in seeds
        ])
    finally:
        await gen.close()
    personas: list[Persona] = []
    for seed, batch in zip(seeds, batches, strict=True):
        if not batch:
            raise SimulationError(f'persona generation returned nothing for seed: {seed!r}')
        personas.append(batch[0])
    return personas


async def generate_persona(
    seed: str,
    *,
    agent_description: str = '',
    context: str = '',
    sim_model: str = DEFAULT_MODEL,
    generation_client: AsyncOpenAI | None = None,
) -> Persona:
    """Generate one ``Persona`` from a short archetype seed (e.g. ``"angry customer"``).

    See `generate_personas` for the batch form and provider resolution.
    """
    personas = await generate_personas(
        [seed],
        agent_description=agent_description,
        context=context,
        sim_model=sim_model,
        generation_client=generation_client,
    )
    return personas[0]


async def generate_scenarios(
    seeds: list[str],
    *,
    agent_description: str = '',
    context: str = '',
    sim_model: str = DEFAULT_MODEL,
    generation_client: AsyncOpenAI | None = None,
) -> list[Scenario]:
    """Generate one ``Scenario`` per situation seed (e.g. ``"disputes a refund denial"``).

    The scenario counterpart to `generate_personas`: you name each
    situation, the LLM fills the goal, context, and success/failure criteria.
    """
    from evaluatorq.simulation.exceptions import SimulationError
    from evaluatorq.simulation.generators import ScenarioGenerator

    if not seeds:
        raise ValueError('generate_scenarios requires at least one seed')
    description = agent_description or 'a general-purpose conversational assistant'
    # No run id bound here — see the note in generate_personas.
    gen = ScenarioGenerator(model=sim_model, client=generation_client)
    try:
        batches = await asyncio.gather(*[
            gen.generate(
                agent_description=description,
                context=context,
                num_scenarios=1,
                edge_case_percentage=0.0,
                seed=seed,
            )
            for seed in seeds
        ])
    finally:
        await gen.close()
    scenarios: list[Scenario] = []
    for seed, batch in zip(seeds, batches, strict=True):
        if not batch:
            raise SimulationError(f'scenario generation returned nothing for seed: {seed!r}')
        scenarios.append(batch[0])
    return scenarios


async def generate_scenario(
    seed: str,
    *,
    agent_description: str = '',
    context: str = '',
    sim_model: str = DEFAULT_MODEL,
    generation_client: AsyncOpenAI | None = None,
) -> Scenario:
    """Generate one ``Scenario`` from a short situation seed.

    See `generate_scenarios` for the batch form and provider resolution.
    """
    scenarios = await generate_scenarios(
        [seed],
        agent_description=agent_description,
        context=context,
        sim_model=sim_model,
        generation_client=generation_client,
    )
    return scenarios[0]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


async def _resolve_generation_agent_description(
    *,
    agent_description: str | None,
    target: str | Callable[[list[Message]], str | Awaitable[str]] | AgentTarget | None,
) -> str:
    """Return an explicit description or resolve one from an Orq agent target.

    Generation needs a reliable, user-facing account of the target's role. An
    ``agent:<key>`` target (or a bare agent key) can supply that through the
    Orq agent context; a deployment, callable, and direct model cannot. Keep
    this separate from
    `_resolve_target`: the execution target is a stateless Responses
    target, while the composite red-team backend resolves platform metadata.
    """
    if agent_description and agent_description.strip():
        return agent_description

    if isinstance(target, str):
        from evaluatorq.redteam.backends.registry import make_agent_backend
        from evaluatorq.redteam.contracts import LLMConfig, TargetConfig, TargetKind, parse_target

        kind, agent_key = parse_target(target)
        if kind is TargetKind.AGENT:
            backend = make_agent_backend(
                target_config=TargetConfig(system_prompt=None),
                pipeline_config=LLMConfig(),
            )
            context = await backend.resolve_context(agent_key)
        else:
            context = None
    else:
        context = None

    description = context.description if context is not None else None
    if description and description.strip():
        return description

    raise ValueError(
        'generate_and_simulate requires agent_description or an agent target with a non-empty description.'
    )


async def _generate_personas_scenarios(
    *,
    agent_description: str,
    num_personas: int,
    num_scenarios: int,
    model: str,
    generation_client: AsyncOpenAI | None,
    persona_seeds: list[str] | None = None,
    scenario_seeds: list[str] | None = None,
) -> tuple[list[Persona], list[Scenario]]:
    """Generate personas and scenarios concurrently from an agent description.

    Shared by `generate` and `generate_and_simulate`. When
    ``persona_seeds`` / ``scenario_seeds`` are given, that dimension is built one
    object per seed — each seed is an archetype the LLM fleshes out (e.g.
    ``"angry retiree"``) — instead of auto-generating ``num_*``; the other
    dimension still auto-generates. Provider is resolved via the shared factory;
    the client is closed only when owned here (i.e. not injected).
    """
    from evaluatorq.openresponses.client import build_simulation_client
    from evaluatorq.simulation.exceptions import SimulationError
    from evaluatorq.simulation.generators import PersonaGenerator, ScenarioGenerator

    gen_client, gen_owned = build_simulation_client(generation_client)
    try:
        persona_gen = PersonaGenerator(model=model, client=gen_client)
        scenario_gen = ScenarioGenerator(model=model, client=gen_client)

        async def _seeded(gen: Any, seeds: list[str], kind: str, kwarg: str) -> list[Any]:
            # One object per seed (seed=archetype, LLM fills the rest); no
            # edge-case padding so the output maps 1:1 to the seeds given.
            batches = await asyncio.gather(*[
                gen.generate(agent_description=agent_description, edge_case_percentage=0.0, seed=s, **{kwarg: 1})
                for s in seeds
            ])
            out: list[Any] = []
            for seed, batch in zip(seeds, batches, strict=True):
                if not batch:
                    raise SimulationError(f'{kind} generation returned nothing for seed: {seed!r}')
                out.append(batch[0])
            return out

        personas_coro = (
            _seeded(persona_gen, persona_seeds, 'persona', 'num_personas')
            if persona_seeds
            else persona_gen.generate(agent_description=agent_description, num_personas=num_personas)
        )
        scenarios_coro = (
            _seeded(scenario_gen, scenario_seeds, 'scenario', 'num_scenarios')
            if scenario_seeds
            else scenario_gen.generate(agent_description=agent_description, num_scenarios=num_scenarios)
        )
        gen_personas, gen_scenarios = await asyncio.gather(personas_coro, scenarios_coro)
    finally:
        if gen_owned:
            await gen_client.close()
    return gen_personas, gen_scenarios


def _log_saved_run(path: Path) -> None:
    """Log a saved run with a cwd-relative path + the dashboard-load hint."""
    try:
        display = path.relative_to(Path.cwd())
    except ValueError:
        display = path
    logger.info(f'Run saved: {display} — explore with: eq dashboard {display.parent}')


async def _simulate_core(
    *,
    config: SimulationConfig,
    caller: str,
    pipeline_span: Span | None,
    run_id: str,
    manifest_writer: ManifestWriter | None = None,
    post_run: RunPostProcessor | None = None,
) -> SimulationRun:
    """Core simulation logic (runs inside the Evaluatorq - Agent Simulation span).

    Resolves the target (callable, ``AgentTarget``, or ``"agent:"``/``"deployment:"`` string),
    resolves/generates the datapoints, then drives the run-level lifecycle
    hooks (``on_confirm`` gate → ``on_run_start`` → ``on_run_complete``) around
    ``_simulate_via_evaluatorq``, which routes execution through evaluatorq's
    upload, CI gating, and results display. Per-datapoint / per-turn hooks fire
    inside ``SimulationRunner``. Always builds and returns the full
    ``SimulationRun`` on the success path; only persists it to disk when
    ``save`` is ``True``. The ``SimulationDroppedError`` abort path re-raises
    with partial results instead of returning a run.
    """
    from evaluatorq.common.async_utils import await_maybe
    from evaluatorq.common.tracing import set_span_attrs
    from evaluatorq.simulation.exceptions import SimulationCancelledError
    from evaluatorq.simulation.hooks import DefaultHooks, SimStage, SimulationRunMeta
    from evaluatorq.simulation.replay import load_simulation_replay

    evaluation_name = config.evaluation_name
    model = config.model
    parallelism = config.parallelism
    save = config.save
    run_output = config.run_output

    target_callable, target_agent, target_kind_hint = _resolve_target(
        config.target, memory_entity_id=config.memory_entity_id
    )

    # ``run_id`` is minted at the outer entry (_simulate_run /
    # _generate_and_simulate_run) so the same id brackets the manifest and this
    # core's experiment linking. Each conversation's Orq thread id is
    # f"{run_id}:{index}"; persisted on SimulationRun / SimulationResult so the
    # dashboard can deep-link to the run's traces in Orq observability.

    replayed = load_simulation_replay(config.previous_run) if config.previous_run is not None else None
    sim_datapoints = await _resolve_or_generate_datapoints(
        caller=caller,
        datapoints=config.datapoints,
        personas=config.personas,
        scenarios=config.scenarios,
        dataset_id=config.dataset_id,
        experiment_id=config.experiment_id,
        experiment_run_id=config.experiment_run_id,
        previous_run=config.previous_run,
        model=model,
        generation_client=config.generation_client,
        replayed=replayed,
    )
    # An explicit max_turns wins; otherwise a replay restores the cap its cases
    # ran under, and only then do we fall back to the default. Rebind config so
    # every downstream reader (run meta, the runner, the saved run) agrees.
    max_turns = (
        config.max_turns
        if config.max_turns is not None
        else ((replayed.max_turns if replayed is not None else None) or DEFAULT_MAX_TURNS)
    )
    if config.max_turns != max_turns:
        config = config.model_copy(update={'max_turns': max_turns})

    set_span_attrs(
        pipeline_span,
        {'orq.simulation.datapoints_count': len(sim_datapoints), 'orq.simulation.max_turns': max_turns},
    )

    resolved_hooks = config.hooks or DefaultHooks()
    # The sync-hook deprecation nudge fires once in SimulationRunner.__init__
    # (the single choke point both this path and direct runner use share).
    resolved_evaluator_names = config.evaluator_names if config.evaluator_names is not None else DEFAULT_EVALUATOR_NAMES
    # Derive a human-readable target label mirroring the save block's precedence:
    # AgentTarget instances → "agent:<key>" (or, for the self-describing
    # openai_model/vercel targets, their own `.name`), deployment strings →
    # "deployment:<key>", plain callables → "callback".
    if target_agent is not None and target_kind_hint in ('openai_model', 'vercel'):
        target_name = getattr(target_agent, 'name', None) or target_kind_hint
        target_label = target_name
    elif target_agent is not None:
        agent_key_attr = _agent_key_of(target_agent)
        target_label = f'agent:{agent_key_attr}' if agent_key_attr else 'agent'
        target_name = agent_key_attr or 'agent'
    elif target_kind_hint == 'orq_deployment':
        dep_key = getattr(target_callable, 'deployment_key', None)
        target_label = f'deployment:{dep_key}' if dep_key else 'deployment'
        target_name = dep_key or 'deployment'
    else:
        target_label = 'callback'
        target_name = 'callback'
    # Model the target ran, only when the client knows it (e.g. an OpenAI-model
    # target exposes ``.model``); orq agents/deployments resolve it server-side.
    target_model = getattr(target_agent, 'model', None) or getattr(target_callable, 'model', None)
    run_meta: SimulationRunMeta = {
        'num_datapoints': len(sim_datapoints),
        'model': model,
        'max_turns': max_turns,
        'parallelism': parallelism,
        'evaluation_name': evaluation_name,
        'evaluator_names': resolved_evaluator_names,
        'target': target_label,
    }
    # The manifest was minted at the outer entry and composed in as the first
    # hook (ManifestStageHooks), so stage transitions reach disk automatically.
    # The raw ``manifest_writer`` reference is retained here purely for the
    # terminal complete/cancel/fail transitions (§4.5) — those cannot be a hook
    # because a hook can't guarantee a terminal write on crash.
    #
    # Everything from the confirm gate through save is guarded by the outer
    # try/except: any failure — including a hook raising before/inside the inner
    # try or in its finally — marks the manifest 'error' (or 'cancelled' on a
    # declined confirm), so a run can never stay 'running' once it has finished.
    # Terminal transitions are idempotent, so the single one that lands first wins.
    try:
        # Gate first — before the evaluatorq run is built. A decline is a clean
        # cancel (SimulationCancelledError → writer.cancel below), never an error.
        if not await await_maybe(resolved_hooks.on_confirm(run_meta)):
            raise SimulationCancelledError('Simulation run declined by on_confirm hook')

        # SIMULATE stage brackets the run: start fires after on_confirm passes and
        # before on_run_start (which opens the live Progress region); end fires in
        # the finally after on_run_complete (which closes it). Never emit between
        # on_run_start and on_run_complete — that would tear the live render.
        await await_maybe(resolved_hooks.on_stage_start(SimStage.SIMULATE, {}))
        # Terminal hook always pairs with on_run_start; results is [] on early
        # failure. on_run_complete is unguarded (a raising hook propagates, per the
        # hook exception policy); runner/target cleanup lives inside
        # _simulate_via_evaluatorq's own finally, so it runs regardless.
        results: list[SimulationResult] = []
        # Filled by evaluatorq() with the uploaded experiment's URL (if any) so we
        # can persist it on the SimulationRun and surface it in terminal/dashboard.
        experiment_url_out: list[str] = []
        await await_maybe(resolved_hooks.on_run_start(run_meta))
        try:
            results = await _simulate_via_evaluatorq(
                config=config,
                caller=caller,
                target=target_callable,
                target_agent=target_agent,
                sim_datapoints=sim_datapoints,
                pipeline_span=pipeline_span,
                hooks=resolved_hooks,
                run_id=run_id,
                experiment_url_out=experiment_url_out,
            )
            # Fire on_evaluator_complete here — AFTER results is assigned and
            # OUTSIDE evaluatorq's per-scorer try/except — so an unguarded hook
            # raise propagates (per the contract) while on_run_complete in the
            # finally still receives the real results, not []. Scores were stamped
            # onto each result's metadata by _stamp_evaluator_scores.
            for r in results:
                dp_id = r.metadata.get('datapoint_id', '')
                evaluator_scores = r.metadata.get('evaluator_scores') or {}
                for evaluator_name, score in evaluator_scores.items():
                    await await_maybe(resolved_hooks.on_evaluator_complete(dp_id, evaluator_name, score, r))
        except SimulationDroppedError as dropped:
            # exit_on_failure aborted the run, but the rows that succeeded are real
            # results — hand them to on_run_complete (via the finally) instead of [].
            results = dropped.partial_results
            raise
        finally:
            # Truthful stage status (§4.4): on_stage_end always fires (RichHooks
            # render pairing) but reports the in-flight exception via
            # sys.exc_info()[1] — None on the success path (stage → 'completed'),
            # the live exception on failure (stage → 'error' in the manifest).
            await await_maybe(resolved_hooks.on_run_complete(results))
            await await_maybe(
                resolved_hooks.on_stage_end(
                    SimStage.SIMULATE,
                    {'num_results': len(results), 'error': sys.exc_info()[1]},
                )
            )

        # Always build the run on the success path (an aborted run re-raised
        # above) so every caller — SDK and CLI alike — gets the full
        # SimulationRun (target/agent metadata, run_id, experiment_url) without
        # having to rebuild it from the bare results list.
        # _resolve_target's kind hint resolves the target_kind the dashboard reads;
        # hint is 'orq_agent' for AgentTarget / "agent:" strings, 'orq_deployment'
        # for "deployment:" strings, and None for plain callables.
        target_kind = target_kind_hint or 'callback'
        agent_info = None
        if target_kind == 'orq_agent':
            agent_key = _agent_key_of(target_agent) or target_name.removeprefix('agent:')
            if agent_key and agent_key != 'agent':
                agent_info = await fetch_agent_info(agent_key)
        run = build_simulation_run(
            run_name=evaluation_name or 'sim',
            mode='simulate' if caller == 'simulate' else 'run',
            target_kind=target_kind,
            target=target_name,
            target_model=target_model if isinstance(target_model, str) else None,
            max_turns=max_turns,
            agent_info=agent_info,
            evaluator_names=resolved_evaluator_names,
            results=results,
            run_id=run_id,
            experiment_url=experiment_url_out[0] if experiment_url_out else None,
            # Persist the cases themselves so this run can be replayed later.
            datapoints=sim_datapoints,
        )

        # Generate the LLM narrative before persistence so a saved report carries
        # it. This is best-effort: simulations remain useful without LLM creds or
        # if summary generation itself fails.
        if config.executive_summary:
            from evaluatorq.simulation.reports.executive_summary import populate_run_executive_summary

            try:
                await populate_run_executive_summary(run, enabled=True, model=model)
            except Exception:
                logger.warning('Failed to generate executive summary (results still returned)', exc_info=True)

        # CLI-only report enrichments run before persistence while the pipeline
        # span and evaluatorq run context are still active. This keeps their LLM
        # spans under the main simulation root and binds the same run metadata.
        if post_run is not None:
            await post_run(run)

        # Persist only when the caller opts in (save=True).
        # TODO(RES-963): inline because on_run_complete carries no run metadata and
        # hooks aren't yet composable; move to a save hook once that lands.
        saved_path = None
        if save:
            # A persistence failure (disk full, read-only .evaluatorq/, perms, or the
            # collision-exhaustion RuntimeError) must NOT discard a completed, paid-for
            # run. Log and still return the run — the saved file is a convenience.
            try:
                if run_output is not None:
                    write_report(run, Path(run_output))
                    saved_path = Path(run_output)
                else:
                    saved_path = auto_save_run(run=run, run_name=run.run_name)
                _log_saved_run(saved_path)
            except Exception as exc:
                # Broad by design: this guards a disk-write side-effect that must
                # never discard already-completed work. Covers OSError, the
                # collision RuntimeError, and pydantic serialization errors
                # (PydanticSerializationError <: ValueError) from model_dump_json.
                logger.exception(f'Failed to save simulation run (results still returned): {exc}')

        # Fully finished — mark the manifest completed (with the report path when
        # the save succeeded) even if the report write itself failed. Embed a
        # compact summary (the exact fields the `runs` table + dashboard sim card
        # render) so a list row needs zero full-report reads.
        if manifest_writer is not None:
            manifest_writer.complete(
                report_path=saved_path,
                summary={
                    'mode': run.mode,
                    'target_kind': run.target_kind,
                    'total_results': run.total_results,
                    'scorer_averages': dict(run.scorer_averages),
                },
            )
    except SimulationCancelledError:
        # A declined on_confirm is a clean cancel, not a failure (Dec1): the run
        # is 'cancelled' and any already-completed stage (e.g. GENERATE in the
        # generate_and_simulate path) stays 'completed' — nothing is relabeled.
        if manifest_writer is not None:
            manifest_writer.cancel()
        raise
    except BaseException as exc:
        # Any other failure marks the manifest errored before propagating —
        # otherwise it would read as 'running' forever. fail() closes only
        # still-open stages 'error' (a stage that already finished stays
        # truthful). BaseException so KeyboardInterrupt / CancelledError are
        # captured too; always re-raised.
        if manifest_writer is not None:
            manifest_writer.fail(str(exc) or type(exc).__name__, stage=SimStage.SIMULATE)
        raise
    return run


def _resolve_target(
    target: str | Callable[..., Any] | AgentTarget | None,
    *,
    memory_entity_id: str | None = None,
) -> tuple[
    Callable[[list[Message]], str | Awaitable[str] | Awaitable[AgentResponse]] | None, AgentTarget | None, str | None
]:
    """Resolve the simulation target into ``(callback, agent, kind_hint)`` for the runner.

    Accepts:

    * an ``AgentTarget`` instance → routed to the runner's ``target_agent`` path
      (it speaks ``respond(messages)``); kind hint is ``"orq_agent"`` for the
      generic/Orq case, but the two concrete non-Orq subclasses self-describe
      via a distinct hint (``"openai_model"`` for ``OpenAIModelTarget``,
      ``"vercel"`` for ``VercelAISdkTarget``) so CLI ``--openai-model`` /
      ``--vercel-url`` runs don't get mislabeled as Orq agents (and don't
      trigger a spurious ``fetch_agent_info`` round-trip);
    * a ``str`` → ``"agent:<key>"`` or bare ``"<key>"`` builds a hosted Orq agent
      target via the Responses router (hint ``"orq_agent"``) — the same backend
      the ``eq sim`` CLI uses for ``--target agent:<key>``; ``"deployment:<key>"``
      bridges to an Orq deployment callback (hint ``"orq_deployment"``);
    * a plain callable → callback path (hint ``None`` → ``"callback"``).
    """
    from evaluatorq.contracts import AgentTarget
    from evaluatorq.integrations.vercel_ai_sdk_integration import VercelAISdkTarget
    from evaluatorq.redteam.backends.openai import OpenAIModelTarget
    from evaluatorq.simulation.adapters import from_orq_deployment

    resolved = target

    if memory_entity_id is not None and not memory_entity_id.strip():
        # A blank id would be silently dropped downstream (falsy check in the
        # target) and reproduce the exact 400 this parameter exists to prevent.
        raise ValueError('memory_entity_id cannot be blank; pass a real entity id or omit it.')
    if memory_entity_id is not None and not isinstance(resolved, str):
        raise ValueError(
            "memory_entity_id is only supported with string 'agent:<key>' (or bare '<key>') targets; "
            'for an AgentTarget instance, set memory_entity_id on the instance directly.'
        )

    if isinstance(resolved, str):
        from evaluatorq.redteam.contracts import TargetKind, parse_target

        if not resolved.strip():
            raise ValueError("Target string is empty; use 'agent:<key>', 'deployment:<key>', or a bare '<key>'.")
        kind, value = parse_target(resolved)
        if kind is TargetKind.AGENT:
            # Same composite backend the CLI / red-team runner use for
            # ``agent:<key>``: exec via the Responses router (``agent/<key>``),
            # NOT the ORQ SDK agents endpoint. Keeps SDK and CLI on one path.
            from evaluatorq.redteam.backends.registry import make_agent_backend
            from evaluatorq.redteam.contracts import LLMConfig, TargetConfig

            backend = make_agent_backend(
                target_config=TargetConfig(system_prompt=None),
                pipeline_config=LLMConfig(),
            )
            agent_target = backend.create_target(value)
            if memory_entity_id is not None:
                agent_target.memory_entity_id = memory_entity_id
            return None, agent_target, 'orq_agent'
        if kind is TargetKind.DEPLOYMENT:
            if memory_entity_id is not None:
                raise ValueError("memory_entity_id is only supported with 'agent:<key>' (or bare '<key>') targets.")
            return from_orq_deployment(value), None, 'orq_deployment'
        raise ValueError(
            f'Unsupported target kind {kind.value!r} for simulation; '
            "use 'agent:<key>', 'deployment:<key>', an AgentTarget, or a callable."
        )

    if isinstance(resolved, OpenAIModelTarget):
        return None, resolved, 'openai_model'
    if isinstance(resolved, VercelAISdkTarget):
        return None, resolved, 'vercel'
    if isinstance(resolved, AgentTarget):
        return None, resolved, 'orq_agent'
    if resolved is None:
        raise ValueError(
            "A target is required: pass target= (an AgentTarget, a callable, 'agent:<key>', or 'deployment:<key>')."
        )
    if not callable(resolved):
        raise TypeError(
            f'Unsupported target type {type(resolved).__name__!r}; pass a str '
            "('agent:<key>' / 'deployment:<key>'), an AgentTarget, or a callable."
        )
    return resolved, None, None


def _require_orq_api_key(caller: str) -> str:
    api_key = os.environ.get('ORQ_API_KEY')
    if not api_key:
        raise ValueError(f'ORQ_API_KEY environment variable is not set. Set it before calling {caller}().')
    return api_key


async def _resolve_or_generate_datapoints(
    *,
    caller: str,
    datapoints: list[SimulationDatapoint] | None,
    personas: list[Persona] | None,
    scenarios: list[Scenario] | None,
    dataset_id: str | None,
    experiment_id: str | None = None,
    experiment_run_id: str | None = None,
    previous_run: str | None = None,
    model: str,
    generation_client: AsyncOpenAI | None,
    replayed: SimulationReplay | None = None,
) -> list[SimulationDatapoint]:
    """Return ready-to-run Datapoints.

    Resolution precedence: ``previous_run`` -> ``dataset_id`` -> ``experiment_id``
    -> ``datapoints`` -> persona x scenario cartesian product (with first-message
    generation). The five sources are mutually exclusive. ``caller`` names the
    public entry point so any API-key error message points the user at the right
    function.
    """
    sources = [
        ('previous_run', previous_run is not None),
        ('dataset_id', dataset_id is not None),
        ('experiment_id', experiment_id is not None),
        ('datapoints', datapoints is not None),
        ('personas/scenarios', personas is not None or scenarios is not None),
    ]
    chosen = [name for name, present in sources if present]
    if len(chosen) > 1:
        raise ValueError(
            f'Pass exactly one of previous_run, dataset_id, experiment_id, datapoints, '
            f'or personas+scenarios; got: {", ".join(chosen)}'
        )
    if experiment_run_id is not None and experiment_id is None:
        raise ValueError("'experiment_run_id' requires 'experiment_id'")

    if previous_run is not None:
        # The caller loads the replay (it also needs the run's turn cap) and
        # passes it in, so the file is read once per run.
        from evaluatorq.simulation.replay import load_simulation_replay

        resolved = replayed if replayed is not None else load_simulation_replay(previous_run)
        logger.info(f'Replaying {len(resolved.datapoints)} datapoints from previous run {previous_run!r}.')
        return resolved.datapoints

    if dataset_id is not None:
        api_key = _require_orq_api_key(caller)
        return await _fetch_simulation_datapoints_from_orq(api_key, dataset_id)

    if experiment_id is not None:
        from evaluatorq.simulation.experiments import datapoints_from_experiment

        api_key = _require_orq_api_key(caller)
        return await datapoints_from_experiment(experiment_id, run_id=experiment_run_id, api_key=api_key)

    if datapoints is not None:
        if not datapoints:
            raise ValueError("'datapoints' must be non-empty")
        return datapoints

    if personas is None or scenarios is None:
        raise ValueError(
            "Either provide 'previous_run', 'dataset_id', 'experiment_id', 'datapoints', "
            "or both 'personas' and 'scenarios'"
        )
    if not personas or not scenarios:
        raise ValueError("'personas' and 'scenarios' arrays must both be non-empty")

    from evaluatorq.openresponses.client import build_simulation_client
    from evaluatorq.simulation.generators import FirstMessageGenerator
    from evaluatorq.simulation.tracing import with_simulation_span

    gen_client, gen_owned = build_simulation_client(generation_client)
    try:
        first_msg_gen = FirstMessageGenerator(model=model, client=gen_client)
        pairs = [(p, s) for p in personas for s in scenarios]

        generated: list[SimulationDatapoint] = []
        batch_size = 5
        # One span for the whole generation phase; the per-pair LLM calls nest
        # under it as the client's own spans.
        async with with_simulation_span(
            'orq.simulation.first_message_generation',
            {
                'orq.simulation.model': model,
                'orq.simulation.pair_count': len(pairs),
                'orq.simulation.persona_count': len(personas),
                'orq.simulation.scenario_count': len(scenarios),
            },
        ) as gen_span:
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
                        # The per-pair span is gone, so the phase span carries
                        # the failure identity instead — otherwise a partial
                        # failure is only visible in the logs.
                        if gen_span is not None:
                            gen_span.add_event(
                                'orq.simulation.first_message_generation_failed',
                                {
                                    'orq.simulation.persona': p.name,
                                    'orq.simulation.scenario': s.name,
                                    'error.type': type(outcome).__name__,
                                    'error.message': str(outcome),
                                },
                            )
                        continue
                    generated.append(outcome)
            if gen_span is not None:
                gen_span.set_attribute('orq.simulation.generated_count', len(generated))
                gen_span.set_attribute('orq.simulation.failed_count', len(pairs) - len(generated))
            # Raise inside the span so a total failure marks it ERROR.
            if not generated:
                raise RuntimeError(
                    'first-message generation produced no datapoints - every persona x scenario pair failed'
                )
        return generated
    finally:
        if gen_owned:
            await gen_client.close()


async def _fetch_simulation_datapoints_from_orq(api_key: str, dataset_id: str) -> list[SimulationDatapoint]:
    """Stream the named Orq dataset and parse each row into a simulation
    SimulationDatapoint via the same shape-tolerant extractor used by the inline path.
    """
    from pydantic import ValidationError

    from evaluatorq.fetch_data import fetch_dataset_batches, setup_orq_client
    from evaluatorq.simulation._datapoint_io import _extract_single_datapoint

    orq_client = setup_orq_client(api_key)
    out: list[SimulationDatapoint] = []
    row = 0
    async for batch in fetch_dataset_batches(orq_client, dataset_id):
        for eq_dp in batch.datapoints:
            try:
                out.append(_extract_single_datapoint(eq_dp, source='row'))
            except (ValueError, ValidationError) as e:
                raise ValueError(f'dataset {dataset_id!r} row {row}: {e}') from e
            row += 1
    if not out:
        raise ValueError(f'Dataset {dataset_id!r} returned zero simulation-compatible datapoints')
    return out


async def _generate_single_datapoint(
    gen: FirstMessageGenerator, persona: Persona, scenario: Scenario
) -> SimulationDatapoint:
    from evaluatorq.simulation.utils.prompt_builders import generate_datapoint

    # ponytail: no per-pair span — the caller wraps the whole generation phase
    # in a single orq.simulation.first_message_generation span.
    first_message = await gen.generate(persona, scenario)
    return generate_datapoint(persona, scenario, first_message)


def _build_simulation_job_and_cache(
    *,
    job_name: str,
    sim_dp_by_id: dict[int, SimulationDatapoint],
    target: Callable[[list[Message]], str | Awaitable[str] | Awaitable[AgentResponse]] | None,
    target_agent: AgentTarget | None,
    model: str,
    max_turns: int,
    user_simulator: BaseAgent | None,
    judge: BaseAgent | None,
    generation_client: AsyncOpenAI | None,
    hooks: SimulationHooks | None,
    run_id: str | None = None,
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
        target=target,
        target_agent=target_agent,
        model=model,
        max_turns=max_turns,
        user_simulator=user_simulator,
        judge=judge,
        llm_client=generation_client,
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
        # Deterministic, run-scoped thread id groups this conversation's turns in
        # Orq observability and powers the dashboard's per-conversation deep link.
        # _row is the datapoint's index (stable across the run).
        thread_id = build_thread_id(run_id, _row)
        result = await runner.run(datapoint=sim_dp, max_turns=max_turns, thread_id=thread_id)
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
    propagating. The hook is fired from `_stamp_evaluator_scores`, which
    runs outside evaluatorq's guard, preserving the unguarded-propagation
    contract.
    """
    from evaluatorq.types import EvaluationResult, ScorerParameter

    async def scorer(params: ScorerParameter) -> EvaluationResult:  # noqa: RUF029  # scorer signature is async by evaluatorq contract
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
        # Carry the judge's reasoning through as explanation + pass_ so it lands on
        # the evaluator trace span and the uploaded experiment. `value` is left
        # untouched — experiment averages depend on the exact numeric score.
        explanation, pass_ = _sim_evaluation_details(name, sim_result)
        return EvaluationResult.model_validate({'value': value, 'explanation': explanation, 'pass': pass_})

    # evaluator_type marks these as code scorers so the tracing layer emits the
    # gen_ai.evaluation.* evaluator-span attributes (opt-in; see set_evaluation_attributes).
    # 'python_eval' is Orq's EvaluatorType value for code scorers.
    return {'name': name, 'scorer': scorer, 'evaluator_type': 'python_eval'}


def _sim_evaluation_details(name: str, result: SimulationResult) -> tuple[str | None, bool | None]:
    """Derive (explanation, pass_) for a built-in sim evaluator from the judge's reasoning.

    Only the default evaluators (``goal_achieved`` / ``criteria_met``) carry a
    reasoning surface today; others return ``(None, None)`` so their span/experiment
    detail is unchanged.
    """
    if name == 'goal_achieved':
        return (result.reason or None), result.goal_achieved
    if name == 'criteria_met':
        meta = result.metadata.get('criteria_meta') or []
        if isinstance(meta, list) and meta:
            # Tag each line with the criterion polarity. Without it, a passed
            # 'must_not_happen' rule renders as e.g. "PASS: Agent blames the
            # customer", which reads as if the agent passed *by* misbehaving.
            # "[prohibited]" makes clear PASS means the behavior was avoided.
            def _line(c: dict[str, object]) -> str:
                verdict = 'PASS' if c.get('passed') else 'FAIL'
                polarity = 'prohibited' if c.get('type') == 'must_not_happen' else 'required'
                return f'{verdict} [{polarity}]: {c.get("description", c.get("id", "?"))}'

            lines = [_line(c) for c in meta]
            all_met = all(c.get('passed') for c in meta)
            return '\n'.join(lines), all_met
        # Fallback to the lossy criteria_results dict when criteria_meta is absent.
        criteria_results = result.criteria_results or {}
        if criteria_results:
            lines = [f'{"PASS" if ok else "FAIL"}: {desc}' for desc, ok in criteria_results.items()]
            return '\n'.join(lines), all(criteria_results.values())
        return 'No criteria defined for this scenario.', True
    return None, None


async def _simulate_via_evaluatorq(
    *,
    config: SimulationConfig,
    caller: str,
    target: Callable[[list[Message]], str | Awaitable[str] | Awaitable[AgentResponse]] | None,
    target_agent: AgentTarget | None,
    sim_datapoints: list[SimulationDatapoint],
    pipeline_span: Span | None,
    hooks: SimulationHooks,
    run_id: str | None = None,
    experiment_url_out: list[str] | None = None,
) -> list[SimulationResult]:
    """Wrap simulation Datapoints as evaluatorq DataPoints and run.

    ``experiment_url_out``, when provided, is populated with the uploaded Orq
    experiment's URL (empty when the upload is skipped or fails).
    """
    from datetime import datetime, timezone

    from evaluatorq.common.tracing import record_token_usage, set_span_attrs
    from evaluatorq.evaluatorq import evaluatorq
    from evaluatorq.simulation.evaluators import get_evaluator
    from evaluatorq.types import DataPoint

    evaluation_name = config.evaluation_name
    model = config.model
    # _simulate_core resolves max_turns before calling here; the fallback only
    # covers a direct call that skipped it.
    max_turns = config.max_turns if config.max_turns is not None else DEFAULT_MAX_TURNS
    parallelism = config.parallelism
    user_simulator = config.user_simulator
    judge = config.judge
    generation_client = config.generation_client
    upload_results = config.upload_results
    evaluation_description = config.evaluation_description
    orq_results_path = config.orq_results_path
    exit_on_failure = config.exit_on_failure

    resolved_evaluator_names = config.evaluator_names if config.evaluator_names is not None else DEFAULT_EVALUATOR_NAMES
    scorers = [(name, get_evaluator(name)) for name in resolved_evaluator_names]

    eq_datapoints = [DataPoint(inputs={'datapoint': dp.model_dump(mode='json')}) for dp in sim_datapoints]
    # Map the evaluatorq DataPoint instance back to its source simulation
    # SimulationDatapoint by identity, so the job can run the source directly without
    # re-parsing inputs["datapoint"] on every run.
    sim_dp_by_id = {id(eq): sim for eq, sim in zip(eq_datapoints, sim_datapoints, strict=True)}

    job_fn, result_cache, runner = _build_simulation_job_and_cache(
        job_name='simulation',
        sim_dp_by_id=sim_dp_by_id,
        target=target,
        target_agent=target_agent,
        model=model,
        max_turns=max_turns,
        user_simulator=user_simulator,
        judge=judge,
        generation_client=generation_client,
        hooks=hooks,
        run_id=run_id,
    )

    evaluators = [_adapt_simulation_scorer(name, fn, result_cache) for name, fn in scorers]

    start = datetime.now(tz=timezone.utc)
    run_name = evaluation_name or f'simulation-{start.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:8]}'

    # Resolve the Orq host once from the inference client so results upload to
    # the same server used for inference (RES-912); falls back to env.
    upload_base_url = resolve_results_base_url(generation_client)

    try:
        eq_results = await evaluatorq(
            run_name,
            data=eq_datapoints,
            jobs=[job_fn],
            evaluators=evaluators,
            parallelism=parallelism,
            description=evaluation_description,
            path=orq_results_path,
            # Suppress the inner ProgressService spinner + results table: the
            # simulation drives its own RichHooks progress bar and summary, and
            # two concurrent rich.Live regions flicker against each other.
            print_results=False,
            _send_results=upload_results,
            # Never let evaluatorq's score-based gate (check_pass_failures →
            # sys.exit) fire for simulation. Now that scorers report pass_ (a
            # judge verdict — e.g. goal not achieved), an underperforming but
            # otherwise healthy run must NOT exit the process. exit_on_failure
            # for sim means dropped rows only (SimulationDroppedError below).
            _exit_on_failure=False,
            _base_url=upload_base_url,
            _experiment_url_out=experiment_url_out,
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
        target_close = getattr(target_agent or target, 'close', None)
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
    # on_evaluator_complete is fired by the caller (_simulate_core) from these
    # stamped scores, after results is assembled.
    _stamp_evaluator_scores(eq_results, result_cache, evaluation_name)

    # Build the results from the successful rows BEFORE the missing-rows check,
    # so a SimulationDroppedError can still carry the partial results through to
    # on_run_complete instead of leaving the caller with an empty list.
    results = [result_cache[id(eq)] for eq in eq_datapoints if id(eq) in result_cache]

    expected = len(eq_datapoints)
    missing = [i for i, eq in enumerate(eq_datapoints) if id(eq) not in result_cache]
    if missing:
        msg = (
            f'{caller}(): {len(missing)} of {expected} simulation job(s) failed '
            f'and produced no result (missing rows: {missing})'
        )
        if exit_on_failure:
            raise SimulationDroppedError(msg, partial_results=results)
        logger.warning(msg)

    # Aggregate token usage (including cost breakdown) onto the pipeline span.
    # This also sets `gen_ai.usage.calls` to the summed per-datapoint call count —
    # intended: a call count on this pipeline-level aggregate span is genuine
    # observability richness, not an accidental leak.
    from evaluatorq.contracts import Usage

    record_token_usage(pipeline_span, usage=sum((r.token_usage for r in results), Usage()))
    set_span_attrs(
        pipeline_span,
        {
            'orq.simulation.results_count': len(results),
            'orq.simulation.goal_achieved_count': sum(1 for r in results if r.goal_achieved),
        },
    )

    return results


def _stamp_evaluator_scores(
    eq_results: list[DataPointResult],
    result_cache: dict[int, SimulationResult],
    evaluation_name: str,
) -> None:
    """Walk the evaluatorq result and stamp each evaluator score onto the
    matching SimulationResult.metadata["evaluator_scores"].

    Matches by ``id(data_point)`` — on success evaluatorq returns the same
    DataPoint instance the job cached against. Error rows carry a placeholder
    DataPoint whose id won't match, so they're skipped.

    ``on_evaluator_complete`` is NOT fired here — the caller (_simulate_core)
    fires it from the stamped scores after ``results`` is assembled, so an
    unguarded hook raise still leaves the real results for ``on_run_complete``.
    """
    for dp_result in eq_results:
        sim_result = result_cache.get(id(dp_result.data_point))
        if sim_result is None or not dp_result.job_results:
            continue
        scores_dict = sim_result.metadata.setdefault('evaluator_scores', {})
        for job_result in dp_result.job_results:
            for score in job_result.evaluator_scores or []:
                if isinstance(scores_dict, dict):
                    scores_dict[score.evaluator_name] = score.score.value
        if evaluation_name:
            sim_result.metadata['evaluation_name'] = evaluation_name
