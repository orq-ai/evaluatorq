"""Lifecycle hooks for the agent-simulation runner.

Mirrors ``evaluatorq/redteam/hooks.py``: a ``@runtime_checkable`` Protocol
contract plus concrete ``DefaultHooks`` (loguru no-op baseline) and
``RichHooks`` (live ``rich.progress``) implementations injected via
``simulate(hooks=...)`` / ``SimulationRunner(hooks=...)``.

Public API:
    SimulationHooks   — runtime-checkable Protocol (8 lifecycle methods)
    SimulationRunMeta — TypedDict payload for on_confirm / on_run_start
    DefaultHooks      — loguru baseline (the default when no hooks passed)
    RichHooks         — Rich per-datapoint progress (CLI / interactive)

All hooks are SYNCHRONOUS (parity with redteam). They run on the async event
loop; keep them fast and non-blocking — a slow sync hook stalls every
concurrent simulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

from loguru import logger

if TYPE_CHECKING:
    from rich.console import Console as RichConsole  # noqa: F401  (used by RichHooks in Task 2)

    from evaluatorq.simulation.types import (
        Datapoint,
        SimulationResult,
        TurnMetrics,
    )


class SimulationRunMeta(TypedDict):
    """Payload passed to ``on_confirm`` (gate) and ``on_run_start`` (notify).

    One payload, double duty: built once in ``_simulate_core`` after datapoints
    are resolved, before the runner/target are allocated.
    """

    num_datapoints: int
    model: str
    max_turns: int
    parallelism: int
    evaluation_name: str
    evaluator_names: list[str]


@runtime_checkable
class SimulationHooks(Protocol):
    """Protocol for simulation lifecycle hooks.

    Implementations are injected via ``simulate(hooks=...)`` or
    ``SimulationRunner(hooks=...)``. All methods are synchronous. Hooks fired
    outside the runner's blanket ``try`` (run-level events) propagate; the
    ``on_turn_complete`` call is exception-guarded by the runner because it
    fires inside the per-turn loop under ``run()``'s catch-all.

    ``on_confirm`` is the single pre-run gate (reuses the ``SimulationRunMeta``
    payload). It fires before the runner/target exist; returning ``False``
    aborts the run via ``asyncio.CancelledError``. ``on_run_start`` fires only
    after ``on_confirm`` returns truthy — gate first, then render-init.
    """

    def on_confirm(self, meta: SimulationRunMeta) -> bool: ...
    def on_run_start(self, meta: SimulationRunMeta) -> None: ...
    def on_datapoint_start(self, datapoint: Datapoint) -> None: ...
    def on_turn_complete(self, datapoint_id: str, metrics: TurnMetrics) -> None: ...
    def on_datapoint_complete(self, result: SimulationResult) -> None: ...
    def on_evaluator_complete(
        self, datapoint_id: str, name: str, score: float, result: SimulationResult
    ) -> None: ...
    def on_datapoint_error(
        self, datapoint: Datapoint, exception: BaseException
    ) -> None: ...
    def on_run_complete(self, results: list[SimulationResult]) -> None: ...


class DefaultHooks:
    """Loguru no-op/info baseline — the default when no ``hooks`` is supplied.

    Subclass this to override a single event (e.g. the CLI). Silent at INFO-
    by design so ``hooks=None`` is behaviour-identical to having no hooks.
    """

    def on_confirm(self, meta: SimulationRunMeta) -> bool:
        # Library default: never block. Interactive prompt lives in the CLI
        # (RES-845) which overrides this.
        return True

    def on_run_start(self, meta: SimulationRunMeta) -> None:
        logger.info(
            f"[simulation] Run start: {meta['num_datapoints']} datapoints | "
            f"model={meta['model']!r} | max_turns={meta['max_turns']} | "
            f"parallelism={meta['parallelism']} | "
            f"evaluators={meta['evaluator_names']}"
        )

    def on_datapoint_start(self, datapoint: Datapoint) -> None:
        logger.debug(f"[simulation] Datapoint start: {datapoint.id}")

    def on_turn_complete(self, datapoint_id: str, metrics: TurnMetrics) -> None:
        logger.debug(
            f"[simulation] {datapoint_id} turn {metrics.turn_number} complete"
        )

    def on_datapoint_complete(self, result: SimulationResult) -> None:
        dp_id = result.metadata.get("datapoint_id", "?")
        logger.debug(
            f"[simulation] Datapoint complete: {dp_id} "
            f"terminated_by={result.terminated_by} goal={result.goal_achieved}"
        )

    def on_evaluator_complete(
        self, datapoint_id: str, name: str, score: float, result: SimulationResult
    ) -> None:
        logger.debug(
            f"[simulation] {datapoint_id} evaluator {name!r} -> {score}"
        )

    def on_datapoint_error(
        self, datapoint: Datapoint, exception: BaseException
    ) -> None:
        logger.warning(
            f"[simulation] Datapoint error: {datapoint.id} "
            f"{type(exception).__name__}: {exception}"
        )

    def on_run_complete(self, results: list[SimulationResult]) -> None:
        logger.info(f"[simulation] Run complete: {len(results)} results")
