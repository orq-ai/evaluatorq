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

from typing import TYPE_CHECKING, Any, Protocol, TypedDict, runtime_checkable

from loguru import logger

if TYPE_CHECKING:
    from rich.console import Console as RichConsole

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
    ``SimulationRunner(hooks=...)``. All methods are synchronous; they run on
    the async event loop, so keep them fast — a slow sync hook stalls every
    concurrent simulation.

    Exception policy: only ``on_turn_complete`` is exception-guarded by the
    runner — it fires inside ``run()``'s catch-all, which would otherwise
    mis-attribute a hook bug as a simulation error.

    ``on_datapoint_start`` fires inside the per-datapoint ``run_single`` task
    (gathered with ``return_exceptions=True``), so a raise is captured as that
    datapoint's error result and surfaces via ``on_datapoint_error`` +
    ``on_datapoint_complete`` for that datapoint only — it does NOT abort the
    batch. This is the same isolation behaviour as a failing target or ``run()``
    call.

    ``on_confirm`` and ``on_run_start`` run before the gather and are NOT
    guarded: a raise there propagates and aborts the entire run. All other
    per-datapoint hooks (``on_datapoint_complete``/``_error``,
    ``on_evaluator_complete``) are also unguarded.

    ``on_confirm`` is the single pre-run gate (reuses the ``SimulationRunMeta``
    payload). It fires before the runner/target exist; returning ``False``
    aborts the run via ``SimulationCancelledError``. ``on_run_start`` fires only
    after ``on_confirm`` returns truthy and target resolution + runner
    construction succeed — gate first, then render-init. Note: for
    ``generate_and_simulate`` the gate fires AFTER persona/scenario/
    first-message generation, so those generation tokens are already spent when
    ``on_confirm`` is called — the gate protects the simulation batch, not
    generation.

    ``on_run_complete`` is the terminal hook. It always fires exactly once after
    a successful ``on_confirm`` + ``on_run_start`` pair, even on failure (e.g.
    a batch error or an exception during scoring). ``results`` may be an empty
    list if failure occurred before any results were collected.
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
    """Loguru baseline — the default when no ``hooks`` is supplied.

    Subclass this to override a single event (e.g. the CLI). Emits run-level
    start/complete at INFO and per-datapoint detail at DEBUG; datapoint errors
    at WARNING. ``on_confirm`` never blocks, so ``hooks=None`` is
    control-flow-identical to supplying no hooks (it does still log)."""

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
        """Terminal hook. Always fires exactly once after ``on_run_start``, even
        on failure. ``results`` may be an empty list if failure occurred before
        any results were collected."""
        logger.info(f"[simulation] Run complete: {len(results)} results")


class RichHooks:
    """Rich terminal hooks: one progress task per ``datapoint_id``.

    Lifecycle-tolerant: ``on_run_start`` is optional. If the runner is driven
    directly (no ``_simulate_core``), the ``Progress`` starts lazily on the
    first ``on_datapoint_start`` and ``max_turns`` is unknown (turn fields are
    advisory). Per-item events ``.get()`` their task defensively.

    Args:
        console: a ``rich.console.Console``; a new one is created when ``None``.
    """

    def __init__(self, *, console: RichConsole | None = None) -> None:
        if console is None:
            from rich.console import Console

            console = Console()
        self._console = console
        self._progress: Any = None  # rich.progress.Progress
        self._overall_task_id: Any = None  # rich.progress.TaskID | None
        self._tasks: dict[str, int] = {}  # datapoint_id -> rich TaskID
        self._max_turns: int | None = None
        self._completed = 0

    def on_confirm(self, meta: SimulationRunMeta) -> bool:
        # Core RichHooks does not prompt (no typer dep). The interactive
        # confirm table + typer.confirm lands in the CLI override (RES-845).
        return True

    def _ensure_started(self) -> None:
        if self._progress is not None:
            return
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=self._console,
        )
        self._progress.start()

    def on_run_start(self, meta: SimulationRunMeta) -> None:
        self._ensure_started()
        self._max_turns = meta["max_turns"]
        self._overall_task_id = self._progress.add_task(
            "[bold cyan]Simulations", total=meta["num_datapoints"]
        )

    def on_datapoint_start(self, datapoint: Datapoint) -> None:
        from rich.markup import escape

        self._ensure_started()
        if datapoint.id in self._tasks:
            return
        self._tasks[datapoint.id] = self._progress.add_task(
            f"  {escape(datapoint.scenario.name)}", total=self._max_turns
        )

    def on_turn_complete(self, datapoint_id: str, metrics: TurnMetrics) -> None:
        if self._progress is None:
            return
        task_id = self._tasks.get(datapoint_id)
        if task_id is None:
            return
        self._progress.update(task_id, completed=metrics.turn_number)

    def on_datapoint_complete(self, result: SimulationResult) -> None:
        if self._progress is None:
            return
        from rich.markup import escape

        dp_id = result.metadata.get("datapoint_id")
        task_id = self._tasks.get(dp_id) if dp_id else None
        if dp_id and task_id is not None:
            self._progress.update(
                task_id,
                description=f"  [green]{escape(dp_id)}[/green] {result.terminated_by}",
            )
        self._completed += 1
        if self._overall_task_id is not None:
            self._progress.update(self._overall_task_id, completed=self._completed)

    def on_evaluator_complete(
        self, datapoint_id: str, name: str, score: float, result: SimulationResult
    ) -> None:
        # No per-event render.
        return

    def on_datapoint_error(
        self, datapoint: Datapoint, exception: BaseException
    ) -> None:
        if self._progress is None:
            return
        from rich.markup import escape

        task_id = self._tasks.get(datapoint.id)
        if task_id is not None:
            self._progress.update(
                task_id,
                description=f"  [red]{escape(datapoint.id)} ERROR[/red]",
            )

    def on_run_complete(self, results: list[SimulationResult]) -> None:
        """Terminal hook. Always fires exactly once after ``on_run_start``, even
        on failure. ``results`` may be an empty list if failure occurred before
        any results were collected. Safe to call more than once (idempotent via
        the ``_progress is None`` guard)."""
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            achieved = sum(1 for r in results if r.goal_achieved)
            self._console.print(
                f"[bold green]Done[/bold green]: {len(results)} simulations, "
                f"{achieved} goal-achieved"
            )
