"""High-level simulation API functions.

`simulate()` and `generate_and_simulate()` route execution through the
`evaluatorq()` framework so they inherit auto-upload, OTel tracing, results
display, CI gating, and dataset-id support (RES-594).
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import TYPE_CHECKING, Any

from evaluatorq.simulation.types import DEFAULT_MODEL

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from evaluatorq.simulation.agents.base import BaseAgent
    from evaluatorq.simulation.evaluators.scorers import SimulationScorer
    from evaluatorq.simulation.generators import FirstMessageGenerator
    from evaluatorq.simulation.types import (
        ChatMessage,
        Datapoint,
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
    evaluation_name: str = "",
    agent_key: str | None = None,
    target_callback: Callable[[list[ChatMessage]], str | Awaitable[str]] | None = None,
    target: Callable[[list[ChatMessage]], str | Awaitable[str]] | None = None,
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
            supplied.
        user_simulator: Pre-constructed ``BaseAgent`` to drive the user side.
            When omitted a default ``UserSimulatorAgent`` is built from ``model``.
        judge: Pre-constructed ``BaseAgent`` used to evaluate each turn.
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
    resolved_callback = _resolve_callback(target, target_callback, agent_key)

    sim_datapoints = await _resolve_or_generate_datapoints(
        caller="simulate",
        datapoints=datapoints,
        personas=personas,
        scenarios=scenarios,
        dataset_id=dataset_id,
        model=model,
    )

    return await _simulate_via_evaluatorq(
        caller="simulate",
        evaluation_name=evaluation_name,
        target_callback=resolved_callback,
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
    )


async def generate_and_simulate(
    *,
    evaluation_name: str = "",
    agent_description: str,
    agent_key: str | None = None,
    target_callback: Callable[[list[ChatMessage]], str | Awaitable[str]] | None = None,
    num_personas: int = 5,
    num_scenarios: int = 5,
    max_turns: int = 10,
    model: str = DEFAULT_MODEL,
    evaluator_names: list[str] | None = None,
    parallelism: int = 5,
    user_simulator: BaseAgent | None = None,
    judge: BaseAgent | None = None,
    upload_results: bool = True,
    evaluation_description: str | None = None,
    path: str | None = None,
    exit_on_failure: bool = True,
) -> list[SimulationResult]:
    """Generate personas/scenarios, then run simulations via evaluatorq().

    ``ORQ_API_KEY`` is required because persona/scenario/first-message
    generation calls the Orq router. ``upload_results`` defaults to ``True``;
    set it to ``False`` to skip uploading the final experiment.
    ``exit_on_failure`` defaults to ``True``; see ``simulate()`` for the
    full semantics of the CI-gate behaviour and how to opt out.
    """
    from openai import AsyncOpenAI

    from evaluatorq.simulation.generators import PersonaGenerator, ScenarioGenerator

    api_key = _require_orq_api_key("generate_and_simulate")
    resolved_callback = _resolve_callback(None, target_callback, agent_key)

    shared_client = AsyncOpenAI(
        api_key=api_key,
        base_url=f"{os.environ.get('ORQ_BASE_URL', 'https://api.orq.ai')}/v2/router",
    )
    try:
        persona_gen = PersonaGenerator(model=model, client=shared_client)
        scenario_gen = ScenarioGenerator(model=model, client=shared_client)
        gen_personas, gen_scenarios = await asyncio.gather(
            persona_gen.generate(
                agent_description=agent_description, num_personas=num_personas
            ),
            scenario_gen.generate(
                agent_description=agent_description, num_scenarios=num_scenarios
            ),
        )
    finally:
        await shared_client.close()

    sim_datapoints = await _resolve_or_generate_datapoints(
        caller="generate_and_simulate",
        datapoints=None,
        personas=gen_personas,
        scenarios=gen_scenarios,
        dataset_id=None,
        model=model,
    )

    return await _simulate_via_evaluatorq(
        caller="generate_and_simulate",
        evaluation_name=evaluation_name,
        target_callback=resolved_callback,
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
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolve_callback(
    target: Callable[..., Any] | None,
    target_callback: Callable[..., Any] | None,
    agent_key: str | None,
) -> Callable[[list[ChatMessage]], str | Awaitable[str]]:
    from evaluatorq.simulation.adapters import from_orq_deployment

    resolved = target or target_callback
    if not resolved and agent_key:
        resolved = from_orq_deployment(agent_key)
    if not resolved:
        raise ValueError(
            "Either target_callback (or target) or agent_key is required"
        )
    return resolved


def _require_orq_api_key(caller: str) -> str:
    api_key = os.environ.get("ORQ_API_KEY")
    if not api_key:
        raise ValueError(
            f"ORQ_API_KEY environment variable is not set. Set it before calling {caller}()."
        )
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
        ("dataset_id", dataset_id is not None),
        ("datapoints", datapoints is not None),
        ("personas/scenarios", personas is not None or scenarios is not None),
    ]
    chosen = [name for name, present in sources if present]
    if len(chosen) > 1:
        raise ValueError(
            f"Pass exactly one of dataset_id, datapoints, or personas+scenarios; got: {', '.join(chosen)}"
        )

    if dataset_id is not None:
        api_key = _require_orq_api_key(caller)
        return await _fetch_simulation_datapoints_from_orq(api_key, dataset_id)

    if datapoints is not None:
        if not datapoints:
            raise ValueError("'datapoints' must be non-empty")
        return datapoints

    if personas is None or scenarios is None:
        raise ValueError(
            "Either provide 'dataset_id', 'datapoints', or both 'personas' and 'scenarios'"
        )
    if not personas or not scenarios:
        raise ValueError("'personas' and 'scenarios' arrays must both be non-empty")

    from openai import AsyncOpenAI

    from evaluatorq.simulation.generators import FirstMessageGenerator

    api_key = _require_orq_api_key(caller)
    shared_client = AsyncOpenAI(
        api_key=api_key,
        base_url=f"{os.environ.get('ORQ_BASE_URL', 'https://api.orq.ai')}/v2/router",
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
                        "first-message generation failed for persona=%r scenario=%r: %r — skipping",
                        p.name,
                        s.name,
                        outcome,
                    )
                    continue
                generated.append(outcome)
        if not generated:
            raise RuntimeError(
                "first-message generation produced no datapoints — every persona×scenario pair failed"
            )
        return generated
    finally:
        await shared_client.close()


async def _fetch_simulation_datapoints_from_orq(
    api_key: str, dataset_id: str
) -> list[Datapoint]:
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
                raise ValueError(
                    f"dataset {dataset_id!r} row {row}: {e}"
                ) from e
            row += 1
    if not out:
        raise ValueError(
            f"Dataset {dataset_id!r} returned zero simulation-compatible datapoints"
        )
    return out


async def _generate_single_datapoint(
    gen: FirstMessageGenerator, persona: Persona, scenario: Scenario
) -> Datapoint:
    from evaluatorq.simulation.tracing import with_simulation_span
    from evaluatorq.simulation.utils.prompt_builders import generate_datapoint

    async with with_simulation_span(
        "orq.simulation.first_message_generation",
        {
            "orq.simulation.persona": persona.name,
            "orq.simulation.scenario": scenario.name,
            "orq.simulation.model": getattr(gen, "_model", None),
        },
    ):
        first_message = await gen.generate(persona, scenario)
    return generate_datapoint(persona, scenario, first_message)


def _build_simulation_job_and_cache(
    *,
    job_name: str,
    sim_dp_by_id: dict[int, Datapoint],
    target_callback: Callable[[list[ChatMessage]], str | Awaitable[str]],
    model: str,
    max_turns: int,
    user_simulator: BaseAgent | None,
    judge: BaseAgent | None,
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
    from evaluatorq.simulation.convert import to_open_responses
    from evaluatorq.simulation.runner.simulation import SimulationRunner

    runner = SimulationRunner(
        target_callback=target_callback,
        model=model,
        max_turns=max_turns,
        user_simulator=user_simulator,
        judge=judge,
    )
    result_cache: dict[int, SimulationResult] = {}

    async def job_fn(data: DataPoint, _row: int) -> dict[str, Any]:
        sim_dp = sim_dp_by_id.get(id(data))
        if sim_dp is None:
            raise RuntimeError(
                "DataPoint instance not found in sim_dp_by_id — "
                "simulate()'s scorer/result wiring requires the same "
                "DataPoint instance evaluatorq was given"
            )
        result = await runner.run(datapoint=sim_dp, max_turns=max_turns)
        result_cache[id(data)] = result
        return {"name": job_name, "output": to_open_responses(result, model)}

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
    """
    from evaluatorq.types import EvaluationResult, ScorerParameter

    async def scorer(params: ScorerParameter) -> EvaluationResult:
        # Both failure paths raise rather than returning a sentinel `0.0` —
        # evaluatorq's process_evaluator catches and records the error on
        # the EvaluatorScore, so callers (and exit_on_failure) see a real
        # failure instead of mistaking a degenerate score for a low result.
        data = params["data"]
        sim_result = result_cache.get(id(data))
        if sim_result is None:
            logger.error(
                "scorer %r: no SimulationResult cached for DataPoint id=%r — "
                "upstream simulation job failed or did not run",
                name,
                id(data),
            )
            raise RuntimeError(
                f"missing simulation result for DataPoint id={id(data)} "
                f"(scorer={name!r})"
            )
        try:
            value = sim_scorer(sim_result)
        except Exception as e:
            logger.exception("scorer %r raised on DataPoint id=%s", name, id(data))
            sim_result.metadata.setdefault("scorer_errors", {})[name] = repr(e)
            raise
        return EvaluationResult(value=value)

    return {"name": name, "scorer": scorer}


async def _simulate_via_evaluatorq(
    *,
    caller: str,
    evaluation_name: str,
    target_callback: Callable[[list[ChatMessage]], str | Awaitable[str]],
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
) -> list[SimulationResult]:
    """Wrap simulation Datapoints as evaluatorq DataPoints and run."""
    from datetime import datetime, timezone

    from evaluatorq.evaluatorq import evaluatorq
    from evaluatorq.simulation.evaluators import get_evaluator
    from evaluatorq.types import DataPoint

    resolved_evaluator_names = (
        evaluator_names if evaluator_names is not None else ["goal_achieved", "criteria_met"]
    )
    scorers = [(name, get_evaluator(name)) for name in resolved_evaluator_names]

    eq_datapoints = [
        DataPoint(inputs={"datapoint": dp.model_dump(mode="json")})
        for dp in sim_datapoints
    ]
    # Map the evaluatorq DataPoint instance back to its source simulation
    # Datapoint by identity, so the job can run the source directly without
    # re-parsing inputs["datapoint"] on every run.
    sim_dp_by_id = {
        id(eq): sim for eq, sim in zip(eq_datapoints, sim_datapoints, strict=True)
    }

    job_fn, result_cache, runner = _build_simulation_job_and_cache(
        job_name="simulation",
        sim_dp_by_id=sim_dp_by_id,
        target_callback=target_callback,
        model=model,
        max_turns=max_turns,
        user_simulator=user_simulator,
        judge=judge,
    )

    evaluators = [
        _adapt_simulation_scorer(name, fn, result_cache) for name, fn in scorers
    ]

    start = datetime.now(tz=timezone.utc)
    run_name = (
        evaluation_name
        or f"simulation-{start.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    )

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
        try:
            await runner.close()
        except Exception:
            # Don't let runner-cleanup errors mask the primary exception
            # from evaluatorq() if there was one.
            logger.exception("SimulationRunner.close() raised during cleanup")

    # Mirror evaluator scores onto SimulationResult.metadata once, from the
    # final evaluatorq result, so the scorer stays pure (no side-effects mid-run)
    # and callers inspecting SimulationResult.metadata still find evaluator_scores.
    _stamp_evaluator_scores(eq_results, result_cache)

    expected = len(eq_datapoints)
    missing = [
        i for i, eq in enumerate(eq_datapoints) if id(eq) not in result_cache
    ]
    if missing:
        msg = (
            f"{caller}(): {len(missing)} of {expected} simulation job(s) failed "
            f"and produced no result (missing rows: {missing})"
        )
        if exit_on_failure:
            raise SimulationDroppedError(msg)
        logger.warning(msg)

    return [
        result_cache[id(eq)] for eq in eq_datapoints if id(eq) in result_cache
    ]


def _stamp_evaluator_scores(
    eq_results: list[DataPointResult],
    result_cache: dict[int, SimulationResult],
) -> None:
    """Walk the evaluatorq result and stamp each evaluator score onto the
    matching SimulationResult.metadata["evaluator_scores"].

    Matches by ``id(data_point)`` — on success evaluatorq returns the same
    DataPoint instance the job cached against. Error rows carry a placeholder
    DataPoint whose id won't match, so they're skipped.
    """
    for dp_result in eq_results:
        sim_result = result_cache.get(id(dp_result.data_point))
        if sim_result is None or not dp_result.job_results:
            continue
        scores_dict = sim_result.metadata.setdefault("evaluator_scores", {})
        if not isinstance(scores_dict, dict):
            continue
        for job_result in dp_result.job_results:
            for score in job_result.evaluator_scores or []:
                scores_dict[score.evaluator_name] = score.score.value
