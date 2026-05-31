from __future__ import annotations

import asyncio
import io

import pytest

from evaluatorq.simulation.hooks import (
    DefaultHooks,
    SimulationHooks,
    SimulationRunMeta,
)
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Datapoint,
    Persona,
    Scenario,
    SimulationResult,
    TerminatedBy,
    TokenUsage,
    TurnMetrics,
)


@pytest.fixture
def datapoint_factory():
    def _make(dp_id: str) -> Datapoint:
        persona = Persona(
            name=f"p-{dp_id}",
            patience=0.5,
            assertiveness=0.5,
            politeness=0.5,
            technical_level=0.5,
            communication_style=CommunicationStyle.casual,
            background="d",
        )
        scenario = Scenario(name=f"s-{dp_id}", goal="g")
        return Datapoint(
            id=dp_id,
            persona=persona,
            scenario=scenario,
            user_system_prompt="",
            first_message="hi",
        )

    return _make


def _meta() -> SimulationRunMeta:
    return SimulationRunMeta(
        num_datapoints=1,
        model="m",
        max_turns=3,
        parallelism=1,
        evaluation_name="e",
        evaluator_names=["goal_achieved"],
    )


def _result() -> SimulationResult:
    return SimulationResult(
        messages=[],
        terminated_by=TerminatedBy.judge,
        reason="ok",
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        turn_metrics=[],
        metadata={"datapoint_id": "dp1"},
    )


def _turn_metrics() -> TurnMetrics:
    return TurnMetrics(
        turn_number=1,
        token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        judge_reason="r",
    )


from evaluatorq.simulation.runner.simulation import SimulationRunner
from evaluatorq.simulation.types import Judgment, Message
from evaluatorq.simulation.types import TokenUsage as _TU


class _StubUserSim:
    """Minimal SimulationUserSimulator: no LLM, deterministic replies."""

    def update_context(self, *, persona_context=None, scenario_context=None) -> None:
        pass

    async def generate_first_message(self) -> str:
        return "hello"

    async def respond_async(self, messages, *, llm_purpose=None) -> str:
        return "more"

    def reset_usage(self) -> None:
        pass

    def get_usage(self) -> _TU:
        return _TU(prompt_tokens=0, completion_tokens=0, total_tokens=0)


class _StubJudge:
    """Minimal SimulationJudge returning a fixed Judgment."""

    def __init__(self, *, terminate: bool) -> None:
        self._terminate = terminate

    async def evaluate(self, messages) -> Judgment:
        return Judgment(
            should_terminate=self._terminate,
            reason="stub",
            goal_achieved=self._terminate,
            rules_broken=[],
            goal_completion_score=1.0 if self._terminate else 0.0,
        )

    def reset_usage(self) -> None:
        pass

    def get_usage(self) -> _TU:
        return _TU(prompt_tokens=0, completion_tokens=0, total_tokens=0)


class RecordingHooks(DefaultHooks):
    def __init__(self) -> None:
        self.turn_events: list[tuple[str, int]] = []
        self.completed: list[str] = []
        self.errors: list[tuple[str, str]] = []

    def on_turn_complete(self, datapoint_id, metrics):
        self.turn_events.append((datapoint_id, metrics.turn_number))

    def on_datapoint_complete(self, result):
        self.completed.append(result.metadata.get("datapoint_id", "?"))

    def on_datapoint_error(self, datapoint, exception):
        self.errors.append((datapoint.id, type(exception).__name__))


async def _ok_target(messages: list[Message]) -> str:
    return "fine"


def test_default_hooks_satisfies_protocol():
    assert isinstance(DefaultHooks(), SimulationHooks)


def test_default_hooks_confirm_returns_true():
    # Library callers + hooks=None must never be blocked (redteam parity).
    assert DefaultHooks().on_confirm(_meta()) is True


def test_default_hooks_all_methods_silent(datapoint_factory):
    hooks = DefaultHooks()
    dp = datapoint_factory("dp1")
    # None of these raise; all return None (except on_confirm which returns bool).
    assert hooks.on_run_start(_meta()) is None
    assert hooks.on_datapoint_start(dp) is None
    assert hooks.on_turn_complete("dp1", _turn_metrics()) is None
    assert hooks.on_datapoint_complete(_result()) is None
    assert hooks.on_evaluator_complete("dp1", "goal_achieved", 1.0, _result()) is None
    assert hooks.on_datapoint_error(dp, RuntimeError("boom")) is None
    assert hooks.on_run_complete([_result()]) is None


def test_rich_hooks_satisfies_protocol():
    from rich.console import Console

    from evaluatorq.simulation.hooks import RichHooks

    assert isinstance(RichHooks(console=Console()), SimulationHooks)


def test_rich_hooks_tolerates_runner_only_lifecycle(datapoint_factory):
    """No on_run_start: Progress must start lazily on first datapoint_start
    and per-item events must not KeyError on an unknown datapoint_id."""
    from rich.console import Console

    from evaluatorq.simulation.hooks import RichHooks

    hooks = RichHooks(console=Console())
    dp = datapoint_factory("dp1")

    hooks.on_datapoint_start(dp)  # lazily starts Progress, creates task
    hooks.on_turn_complete("dp1", _turn_metrics())
    hooks.on_turn_complete("unknown", _turn_metrics())  # must not raise
    hooks.on_datapoint_complete(_result())
    hooks.on_run_complete([_result()])  # stops Progress; no prior on_run_start


def test_rich_hooks_full_lifecycle(datapoint_factory):
    from rich.console import Console

    from evaluatorq.simulation.hooks import RichHooks

    hooks = RichHooks(console=Console())
    hooks.on_run_start(_meta())  # creates overall bar
    dp = datapoint_factory("dp1")
    hooks.on_datapoint_start(dp)
    hooks.on_datapoint_start(dp)  # idempotent: no 2nd task
    assert len(hooks._tasks) == 1
    hooks.on_turn_complete("dp1", _turn_metrics())
    hooks.on_datapoint_error(dp, RuntimeError("boom"))  # exercises error render
    hooks.on_datapoint_complete(_result())  # increments overall bar
    assert hooks._completed == 1
    hooks.on_run_complete([_result()])
    hooks.on_run_complete([_result()])  # double-call safe, no raise


def test_rich_hooks_escapes_markup_in_names(datapoint_factory):
    """A scenario name / id containing rich markup must not raise."""
    from rich.console import Console

    from evaluatorq.simulation.hooks import RichHooks

    hooks = RichHooks(console=Console())
    dp = datapoint_factory("[bold]evil[/bold]")  # id + scenario name carry markup
    hooks.on_datapoint_start(dp)  # must not raise MarkupError
    hooks.on_datapoint_error(dp, RuntimeError("x"))


@pytest.mark.asyncio
async def test_on_turn_complete_guard_does_not_corrupt_result(datapoint_factory):
    """A raising on_turn_complete logs a warning but must NOT turn the sim
    into an error result."""

    class RaisingHooks(DefaultHooks):
        def on_turn_complete(self, datapoint_id, metrics):
            raise RuntimeError("hook boom")

    runner = SimulationRunner(
        target_callback=_ok_target,
        model="gpt-4o-mini",
        max_turns=1,
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
        hooks=RaisingHooks(),
    )
    result = await runner.run(datapoint=datapoint_factory("dp1"))
    assert result.terminated_by != TerminatedBy.error


@pytest.mark.asyncio
async def test_on_datapoint_error_fires_for_raising_target(datapoint_factory):
    """A raising target is swallowed into an error SimulationResult; the error
    hook must still fire, and run_batch must still return an error result."""

    async def _boom_target(messages: list[Message]) -> str:
        raise ValueError("target down")

    hooks = RecordingHooks()
    runner = SimulationRunner(
        target_callback=_boom_target,
        model="gpt-4o-mini",
        max_turns=2,
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=False),  # pyright: ignore[reportArgumentType]
        hooks=hooks,
    )
    results = await runner.run_batch(
        [datapoint_factory("dp1")], max_turns=2, max_concurrency=1
    )
    assert len(results) == 1
    assert results[0].terminated_by == TerminatedBy.error
    assert hooks.errors == [("dp1", "RuntimeError")]  # reconstructed from metadata
    assert hooks.completed == ["dp1"]


@pytest.mark.asyncio
async def test_on_turn_complete_fires_per_turn(datapoint_factory):
    """on_turn_complete must fire once per turn across a multi-turn run."""
    hooks = RecordingHooks()
    runner = SimulationRunner(
        target_callback=_ok_target,
        model="gpt-4o-mini",
        max_turns=3,
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=False),  # pyright: ignore[reportArgumentType] never terminates early -> runs all turns
        hooks=hooks,
    )
    await runner.run(datapoint=datapoint_factory("dp1"))
    assert hooks.turn_events == [("dp1", 1), ("dp1", 2), ("dp1", 3)]


@pytest.mark.asyncio
async def test_on_datapoint_error_fires_on_timeout(datapoint_factory):
    """A target slower than the per-sim timeout yields a timeout result AND
    fires on_datapoint_error."""

    async def _slow_target(messages):
        await asyncio.sleep(0.2)
        return "too slow"

    hooks = RecordingHooks()
    runner = SimulationRunner(
        target_callback=_slow_target,
        model="gpt-4o-mini",
        max_turns=2,
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=False),  # pyright: ignore[reportArgumentType]
        hooks=hooks,
    )
    results = await runner.run_batch(
        [datapoint_factory("dp1")],
        max_turns=2,
        timeout_per_simulation=0.01,
        max_concurrency=1,
    )
    assert len(results) == 1
    assert results[0].terminated_by == TerminatedBy.timeout
    assert hooks.errors == [("dp1", "RuntimeError")]
    assert hooks.completed == ["dp1"]


@pytest.mark.asyncio
async def test_concurrency_attribution(datapoint_factory):
    hooks = RecordingHooks()
    runner = SimulationRunner(
        target_callback=_ok_target,
        model="gpt-4o-mini",
        max_turns=2,
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
        hooks=hooks,
    )
    dps = [datapoint_factory(f"dp{i}") for i in range(3)]
    results = await runner.run_batch(dps, max_turns=2, max_concurrency=3)
    assert len(results) == 3
    assert len(hooks.completed) == 3                 # one complete per datapoint
    valid_ids = {"dp0", "dp1", "dp2"}
    assert all(dp_id in valid_ids for dp_id, _ in hooks.turn_events)
    assert all(r.metadata.get("datapoint_id") in valid_ids for r in results)


from evaluatorq.simulation.api import simulate


class RunLevelRecorder(DefaultHooks):
    def __init__(self) -> None:
        self.confirmed: list[int] = []
        self.run_started: list[int] = []
        self.evaluator_events: list[tuple[str, str, float]] = []
        self.run_completed = 0

    def on_confirm(self, meta):
        self.confirmed.append(meta["num_datapoints"])
        return True

    def on_run_start(self, meta):
        self.run_started.append(meta["num_datapoints"])

    def on_evaluator_complete(self, datapoint_id, name, score, result):
        self.evaluator_events.append((datapoint_id, name, score))

    def on_run_complete(self, results):
        self.run_completed += 1


@pytest.mark.asyncio
async def test_simulate_fires_run_level_hooks(datapoint_factory):
    hooks = RunLevelRecorder()

    async def _ok_target(messages):
        return "fine"

    results = await simulate(
        target=_ok_target,
        datapoints=[datapoint_factory("dp1"), datapoint_factory("dp2")],
        max_turns=1,
        evaluator_names=["goal_achieved"],
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
        hooks=hooks,
    )
    assert hooks.confirmed == [2]      # gate fired with the plan size
    assert hooks.run_started == [2]    # notify fired after confirm passed
    assert hooks.run_completed == 1
    # one evaluator event per (datapoint, evaluator); datapoint_id threaded
    assert len(hooks.evaluator_events) == 2
    assert {e[0] for e in hooks.evaluator_events} == {"dp1", "dp2"}
    assert all(e[1] == "goal_achieved" for e in hooks.evaluator_events)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_on_confirm_false_aborts_before_running(datapoint_factory):
    """A declining on_confirm aborts via SimulationCancelledError; on_run_start
    never fires and no simulation runs."""
    from evaluatorq.simulation.exceptions import SimulationCancelledError

    class DecliningHooks(DefaultHooks):
        def __init__(self) -> None:
            self.run_started = 0

        def on_confirm(self, meta):
            return False

        def on_run_start(self, meta):
            self.run_started += 1

    calls = []

    async def _spy_target(messages):
        calls.append(1)
        return "fine"

    hooks = DecliningHooks()
    with pytest.raises(SimulationCancelledError):
        await simulate(
            target=_spy_target,
            datapoints=[datapoint_factory("dp1")],
            max_turns=1,
            evaluator_names=["goal_achieved"],
            user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
            judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
            hooks=hooks,
        )
    assert hooks.run_started == 0
    assert calls == []          # confirm gate aborted before any target call


@pytest.mark.asyncio
async def test_hooks_none_is_behaviour_identical(datapoint_factory):
    """hooks=None must produce the same results + token usage as today."""

    async def _ok_target(messages):
        return "fine"

    # hooks defaults to None -> DefaultHooks
    baseline = await simulate(
        target=_ok_target,
        datapoints=[datapoint_factory("dp1")],
        max_turns=1,
        evaluator_names=["goal_achieved"],
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
    )
    assert len(baseline) == 1
    r = baseline[0]
    assert r.terminated_by == TerminatedBy.judge
    assert r.metadata["evaluator_scores"]["goal_achieved"] in (0.0, 1.0)
    # DefaultHooks is silent at info-: result shape unchanged, datapoint_id set
    assert r.metadata["datapoint_id"] == "dp1"


def test_hooks_exported_from_package():
    import evaluatorq.simulation as sim

    assert sim.SimulationHooks is not None
    assert sim.DefaultHooks is not None
    assert sim.RichHooks is not None
    assert sim.SimulationRunMeta is not None
    assert sim.SimulationError is not None
    assert sim.SimulationCancelledError is not None
    assert issubclass(sim.SimulationCancelledError, sim.SimulationError)
    for name in (
        "SimulationHooks",
        "DefaultHooks",
        "RichHooks",
        "SimulationRunMeta",
        "SimulationError",
        "SimulationCancelledError",
    ):
        assert name in sim.__all__


class _RunLevelRecorder(DefaultHooks):
    def __init__(self) -> None:
        self.started = False
        self.completed_with: list[SimulationResult] | None = None

    def on_run_start(self, meta) -> None:
        self.started = True

    def on_run_complete(self, results) -> None:
        self.completed_with = results


@pytest.mark.asyncio
async def test_on_run_complete_fires_when_a_datapoint_errors(datapoint_factory):
    hooks = _RunLevelRecorder()

    async def boom(messages):
        raise RuntimeError("target down")

    dp = datapoint_factory("dp-err")
    results = await simulate(
        datapoints=[dp],
        target=boom,
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
        max_turns=1,
        evaluator_names=["goal_achieved"],
        hooks=hooks,
    )
    assert hooks.started is True
    assert hooks.completed_with is not None  # terminal fired
    assert len(results) == 1  # errored datapoint still returned


@pytest.mark.asyncio
async def test_on_run_complete_fires_when_scoring_raises(datapoint_factory):
    """on_run_complete must still fire via the finally block even when the scoring
    stage raises (on_evaluator_complete is unguarded), and the original exception
    must propagate out of simulate()."""

    class _ScoringBoom(_RunLevelRecorder):
        def on_evaluator_complete(self, datapoint_id, name, score, result) -> None:
            raise RuntimeError("scoring blew up")

    hooks = _ScoringBoom()
    with pytest.raises(RuntimeError, match="scoring blew up"):
        await simulate(
            datapoints=[datapoint_factory("dp-1")],
            target=_ok_target,
            user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
            judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
            max_turns=1,
            evaluator_names=["goal_achieved"],
            hooks=hooks,
        )
    assert hooks.started is True
    assert hooks.completed_with is not None  # on_run_complete fired via finally even though scoring raised


@pytest.mark.asyncio
async def test_on_datapoint_start_respects_concurrency(datapoint_factory):
    gate = asyncio.Event()
    live = 0
    peak = 0

    class _PeakHooks(DefaultHooks):
        def on_datapoint_start(self, datapoint) -> None:
            nonlocal live, peak
            live += 1
            peak = max(peak, live)

        def on_datapoint_complete(self, result) -> None:
            nonlocal live
            live -= 1

    async def slow_target(messages):
        await gate.wait()
        return "fine"

    runner = SimulationRunner(
        target_callback=slow_target,
        model="gpt-4o-mini",
        max_turns=1,
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
        hooks=_PeakHooks(),
    )
    dps = [datapoint_factory(f"dp-{i}") for i in range(6)]

    async def run():
        task = asyncio.create_task(
            runner.run_batch(dps, max_concurrency=2, timeout_per_simulation=0)  # timeout_per_simulation=0 disables the per-sim timeout so the gate controls timing
        )
        await asyncio.sleep(0.1)
        assert peak == 2  # exactly concurrency-many live before gate opens (6 parked, cap=2)
        gate.set()
        return await task

    results = await asyncio.wait_for(run(), timeout=5)
    await runner.close()
    assert len(results) == 6


@pytest.mark.asyncio
async def test_missing_target_raises_before_on_run_start(datapoint_factory):
    hooks = _RunLevelRecorder()
    with pytest.raises(ValueError):
        await simulate(
            datapoints=[datapoint_factory("dp-1")],
            user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
            judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
            max_turns=1,
            evaluator_names=["goal_achieved"],
            hooks=hooks,
        )
    assert hooks.started is False  # no unpaired on_run_start


def test_rich_hooks_reusable_across_runs():
    from rich.console import Console

    from evaluatorq.simulation.hooks import RichHooks

    hooks = RichHooks(console=Console(file=io.StringIO(), force_terminal=False))
    meta: SimulationRunMeta = SimulationRunMeta(
        num_datapoints=2,
        model="m",
        max_turns=3,
        parallelism=2,
        evaluation_name="",
        evaluator_names=["goal_achieved"],
    )

    hooks.on_run_start(meta)
    assert hooks._completed == 0
    # simulate one completion in run 1
    hooks._completed = 2
    hooks.on_run_complete([])

    # Run 2: state must reset
    hooks.on_run_start(meta)
    # white-box: assert per-run reset contract directly
    assert hooks._completed == 0
    assert hooks._tasks == {}
    assert hooks._overall_task_id is not None  # fresh overall task created
