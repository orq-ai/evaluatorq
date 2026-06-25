# This suite deliberately defines sync hook classes (subclassing the now-async
# DefaultHooks/RichHooks) to exercise the sync-hook compatibility path. The
# sync-vs-async override mismatch is the point, not a bug.
# pyright: reportIncompatibleMethodOverride=false
from __future__ import annotations

import asyncio
import io
import warnings
from typing import Any

import pytest

from evaluatorq.simulation.hooks import (
    DefaultHooks,
    RichHooks,
    SimStage,
    SimulationHooks,
    SimulationRunMeta,
)
from evaluatorq.contracts import TokenUsage
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Datapoint,
    Persona,
    Scenario,
    SimulationResult,
    TerminatedBy,
    TurnMetrics,
)


@pytest.fixture
def datapoint_factory():
    def _make(dp_id: str) -> Datapoint:
        persona = Persona(
            name=f'p-{dp_id}',
            patience=0.5,
            assertiveness=0.5,
            politeness=0.5,
            technical_level=0.5,
            communication_style=CommunicationStyle.casual,
            background='d',
        )
        scenario = Scenario(name=f's-{dp_id}', goal='g')
        return Datapoint(
            id=dp_id,
            persona=persona,
            scenario=scenario,
            user_system_prompt='',
            first_message='hi',
        )

    return _make


def _meta() -> SimulationRunMeta:
    return SimulationRunMeta(
        num_datapoints=1,
        model='m',
        max_turns=3,
        parallelism=1,
        evaluation_name='e',
        evaluator_names=['goal_achieved'],
        target='callback',
    )


def _result() -> SimulationResult:
    return SimulationResult(
        messages=[],
        terminated_by=TerminatedBy.judge,
        reason='ok',
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        turn_metrics=[],
        metadata={'datapoint_id': 'dp1'},
    )


def _turn_metrics() -> TurnMetrics:
    return TurnMetrics(
        turn_number=1,
        token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        judge_reason='r',
    )


from evaluatorq.simulation.runner.simulation import SimulationRunner
from evaluatorq.simulation.types import Judgment, Message
from evaluatorq.contracts import TokenUsage as _TU


class _StubUserSim:
    """Minimal SimulationUserSimulator: no LLM, deterministic replies."""

    def update_context(self, *, persona_context=None, scenario_context=None) -> None:
        pass

    async def generate_first_message(self) -> str:
        return 'hello'

    async def respond_async(self, messages, *, llm_purpose=None) -> str:
        return 'more'

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
            reason='stub',
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
        self.completed.append(result.metadata.get('datapoint_id', '?'))

    def on_datapoint_error(self, datapoint, exception):
        self.errors.append((datapoint.id, type(exception).__name__))


async def _ok_target(messages: list[Message]) -> str:
    return 'fine'


def test_default_hooks_satisfies_protocol():
    assert isinstance(DefaultHooks(), SimulationHooks)


def test_default_hooks_confirm_returns_true():
    # Library callers + hooks=None must never be blocked (redteam parity).
    assert asyncio.run(DefaultHooks().on_confirm(_meta())) is True


def test_default_hooks_all_methods_silent(datapoint_factory):
    hooks = DefaultHooks()
    dp = datapoint_factory('dp1')
    # None of these raise; all return None (except on_confirm which returns bool).
    assert asyncio.run(hooks.on_run_start(_meta())) is None
    assert asyncio.run(hooks.on_datapoint_start(dp)) is None
    assert asyncio.run(hooks.on_turn_complete('dp1', _turn_metrics())) is None
    assert asyncio.run(hooks.on_datapoint_complete(_result())) is None
    assert asyncio.run(hooks.on_evaluator_complete('dp1', 'goal_achieved', 1.0, _result())) is None
    assert asyncio.run(hooks.on_datapoint_error(dp, RuntimeError('boom'))) is None
    assert asyncio.run(hooks.on_run_complete([_result()])) is None


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
    dp = datapoint_factory('dp1')

    asyncio.run(hooks.on_datapoint_start(dp))  # lazily starts Progress, creates task
    asyncio.run(hooks.on_turn_complete('dp1', _turn_metrics()))
    asyncio.run(hooks.on_turn_complete('unknown', _turn_metrics()))  # must not raise
    asyncio.run(hooks.on_datapoint_complete(_result()))
    asyncio.run(hooks.on_run_complete([_result()]))  # stops Progress; no prior on_run_start


def test_rich_hooks_full_lifecycle(datapoint_factory):
    from rich.console import Console

    from evaluatorq.simulation.hooks import RichHooks

    hooks = RichHooks(console=Console())
    asyncio.run(hooks.on_run_start(_meta()))  # creates overall bar
    dp = datapoint_factory('dp1')
    asyncio.run(hooks.on_datapoint_start(dp))
    asyncio.run(hooks.on_datapoint_start(dp))  # idempotent: no 2nd task
    assert len(hooks._tasks) == 1
    asyncio.run(hooks.on_turn_complete('dp1', _turn_metrics()))
    asyncio.run(hooks.on_datapoint_error(dp, RuntimeError('boom')))  # exercises error render
    asyncio.run(hooks.on_datapoint_complete(_result()))  # increments overall bar
    assert hooks._completed == 1
    asyncio.run(hooks.on_run_complete([_result()]))
    asyncio.run(hooks.on_run_complete([_result()]))  # double-call safe, no raise


def test_rich_hooks_error_row_stays_red_after_complete(datapoint_factory):
    """on_datapoint_complete must not overwrite on_datapoint_error's red label
    with green for error/timeout results."""
    from rich.console import Console

    from evaluatorq.simulation.hooks import RichHooks

    hooks = RichHooks(console=Console())
    asyncio.run(hooks.on_run_start(_meta()))
    dp = datapoint_factory('dp1')
    asyncio.run(hooks.on_datapoint_start(dp))
    asyncio.run(hooks.on_datapoint_error(dp, RuntimeError('boom')))

    err = _result()
    err.terminated_by = TerminatedBy.error
    err.metadata['datapoint_id'] = 'dp1'
    asyncio.run(hooks.on_datapoint_complete(err))

    desc = hooks._progress.tasks[hooks._tasks['dp1']].description
    assert '[red]' in desc and '[green]' not in desc


def test_rich_hooks_escapes_markup_in_names(datapoint_factory):
    """A scenario name / id containing rich markup must not raise."""
    from rich.console import Console

    from evaluatorq.simulation.hooks import RichHooks

    hooks = RichHooks(console=Console())
    dp = datapoint_factory('[bold]evil[/bold]')  # id + scenario name carry markup
    asyncio.run(hooks.on_datapoint_start(dp))  # must not raise MarkupError
    asyncio.run(hooks.on_datapoint_error(dp, RuntimeError('x')))


@pytest.mark.asyncio
async def test_on_turn_complete_guard_does_not_corrupt_result(datapoint_factory):
    """A raising on_turn_complete logs a warning but must NOT turn the sim
    into an error result."""

    class RaisingHooks(DefaultHooks):
        def on_turn_complete(self, datapoint_id, metrics):
            raise RuntimeError('hook boom')

    runner = SimulationRunner(
        target_callback=_ok_target,
        model='gpt-4o-mini',
        max_turns=1,
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
        hooks=RaisingHooks(),
    )
    result = await runner.run(datapoint=datapoint_factory('dp1'))
    assert result.terminated_by != TerminatedBy.error


@pytest.mark.asyncio
async def test_on_datapoint_error_fires_for_raising_target(datapoint_factory):
    """A raising target is swallowed into an error SimulationResult; the error
    hook must still fire, and run_batch must still return an error result."""

    async def _boom_target(messages: list[Message]) -> str:
        raise ValueError('target down')

    hooks = RecordingHooks()
    runner = SimulationRunner(
        target_callback=_boom_target,
        model='gpt-4o-mini',
        max_turns=2,
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=False),  # pyright: ignore[reportArgumentType]
        hooks=hooks,
    )
    results = await runner.run_batch([datapoint_factory('dp1')], max_turns=2, max_concurrency=1)
    assert len(results) == 1
    assert results[0].terminated_by == TerminatedBy.error
    assert hooks.errors == [('dp1', 'RuntimeError')]  # reconstructed from metadata
    assert hooks.completed == ['dp1']


@pytest.mark.asyncio
async def test_on_turn_complete_fires_per_turn(datapoint_factory):
    """on_turn_complete must fire once per turn across a multi-turn run."""
    hooks = RecordingHooks()
    runner = SimulationRunner(
        target_callback=_ok_target,
        model='gpt-4o-mini',
        max_turns=3,
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=False),  # pyright: ignore[reportArgumentType] never terminates early -> runs all turns
        hooks=hooks,
    )
    await runner.run(datapoint=datapoint_factory('dp1'))
    assert hooks.turn_events == [('dp1', 1), ('dp1', 2), ('dp1', 3)]


@pytest.mark.asyncio
async def test_on_datapoint_error_fires_on_timeout(datapoint_factory):
    """A target slower than the per-sim timeout yields a timeout result AND
    fires on_datapoint_error."""

    async def _slow_target(messages):
        await asyncio.sleep(0.2)
        return 'too slow'

    hooks = RecordingHooks()
    runner = SimulationRunner(
        target_callback=_slow_target,
        model='gpt-4o-mini',
        max_turns=2,
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=False),  # pyright: ignore[reportArgumentType]
        hooks=hooks,
    )
    results = await runner.run_batch(
        [datapoint_factory('dp1')],
        max_turns=2,
        timeout_per_simulation=0.01,
        max_concurrency=1,
    )
    assert len(results) == 1
    assert results[0].terminated_by == TerminatedBy.timeout
    assert hooks.errors == [('dp1', 'RuntimeError')]
    assert hooks.completed == ['dp1']


@pytest.mark.asyncio
async def test_concurrency_attribution(datapoint_factory):
    hooks = RecordingHooks()
    runner = SimulationRunner(
        target_callback=_ok_target,
        model='gpt-4o-mini',
        max_turns=2,
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
        hooks=hooks,
    )
    dps = [datapoint_factory(f'dp{i}') for i in range(3)]
    results = await runner.run_batch(dps, max_turns=2, max_concurrency=3)
    assert len(results) == 3
    assert len(hooks.completed) == 3  # one complete per datapoint
    valid_ids = {'dp0', 'dp1', 'dp2'}
    assert all(dp_id in valid_ids for dp_id, _ in hooks.turn_events)
    assert all(r.metadata.get('datapoint_id') in valid_ids for r in results)


from evaluatorq.simulation.api import simulate


class RunLevelRecorder(DefaultHooks):
    def __init__(self) -> None:
        self.confirmed: list[int] = []
        self.run_started: list[int] = []
        self.evaluator_events: list[tuple[str, str, float]] = []
        self.run_completed = 0

    def on_confirm(self, meta):
        self.confirmed.append(meta['num_datapoints'])
        return True

    def on_run_start(self, meta):
        self.run_started.append(meta['num_datapoints'])

    def on_evaluator_complete(self, datapoint_id, name, score, result):
        self.evaluator_events.append((datapoint_id, name, score))

    def on_run_complete(self, results):
        self.run_completed += 1


@pytest.mark.asyncio
async def test_simulate_fires_run_level_hooks(datapoint_factory):
    hooks = RunLevelRecorder()

    async def _ok_target(messages):
        return 'fine'

    results = await simulate(
        target=_ok_target,
        datapoints=[datapoint_factory('dp1'), datapoint_factory('dp2')],
        max_turns=1,
        evaluator_names=['goal_achieved'],
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
        hooks=hooks,
    )
    assert hooks.confirmed == [2]  # gate fired with the plan size
    assert hooks.run_started == [2]  # notify fired after confirm passed
    assert hooks.run_completed == 1
    # one evaluator event per (datapoint, evaluator); datapoint_id threaded
    assert len(hooks.evaluator_events) == 2
    assert {e[0] for e in hooks.evaluator_events} == {'dp1', 'dp2'}
    assert all(e[1] == 'goal_achieved' for e in hooks.evaluator_events)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_on_datapoint_start_raise_surfaces_error_and_complete(datapoint_factory):
    """A raising on_datapoint_start in the simulate() path still fires
    on_datapoint_error + on_datapoint_complete for that datapoint (the guarantee
    in SimulationHooks' docstring), mirroring run_batch — the job fails but the
    error result is cached and the batch is not aborted."""

    class StartBoom(DefaultHooks):
        def __init__(self) -> None:
            self.errors: list[tuple[str, str]] = []
            self.completed: list[str | None] = []

        def on_datapoint_start(self, datapoint):
            raise RuntimeError(f'boom-{datapoint.id}')

        def on_datapoint_error(self, datapoint, exception):
            self.errors.append((datapoint.id, str(exception)))

        def on_datapoint_complete(self, result):
            self.completed.append(result.metadata.get('datapoint_id'))

    async def _ok_target(messages):
        return 'fine'

    hooks = StartBoom()
    results = await simulate(
        target=_ok_target,
        datapoints=[datapoint_factory('dp1')],
        max_turns=1,
        evaluator_names=['goal_achieved'],
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
        hooks=hooks,
        exit_on_failure=False,
    )
    # Both per-datapoint hooks fired despite on_datapoint_start raising.
    assert hooks.errors == [('dp1', 'boom-dp1')]
    assert hooks.completed == ['dp1']
    # The cached error result is returned; the target was never invoked.
    assert len(results) == 1
    assert results[0].terminated_by == TerminatedBy.error


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
        return 'fine'

    hooks = DecliningHooks()
    with pytest.raises(SimulationCancelledError):
        await simulate(
            target=_spy_target,
            datapoints=[datapoint_factory('dp1')],
            max_turns=1,
            evaluator_names=['goal_achieved'],
            user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
            judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
            hooks=hooks,
        )
    assert hooks.run_started == 0
    assert calls == []  # confirm gate aborted before any target call


@pytest.mark.asyncio
async def test_hooks_none_is_behaviour_identical(datapoint_factory):
    """hooks=None must produce the same results + token usage as today."""

    async def _ok_target(messages):
        return 'fine'

    # hooks defaults to None -> DefaultHooks
    baseline = await simulate(
        target=_ok_target,
        datapoints=[datapoint_factory('dp1')],
        max_turns=1,
        evaluator_names=['goal_achieved'],
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
    )
    assert len(baseline) == 1
    r = baseline[0]
    assert r.terminated_by == TerminatedBy.judge
    assert r.metadata['evaluator_scores']['goal_achieved'] in (0.0, 1.0)
    # DefaultHooks is silent at info-: result shape unchanged, datapoint_id set
    assert r.metadata['datapoint_id'] == 'dp1'


@pytest.mark.asyncio
async def test_simulate_emits_simulate_stage(datapoint_factory):
    """simulate() must bracket execution with SIMULATE stage hooks."""
    seen: list[str] = []

    class Rec(DefaultHooks):
        async def on_stage_start(self, stage, meta):
            seen.append(f'start:{stage}')

        async def on_stage_end(self, stage, meta):
            seen.append(f'end:{stage}')

    async def _ok_target(messages):
        return 'fine'

    await simulate(
        datapoints=[datapoint_factory('dp1')],
        target=_ok_target,
        max_turns=1,
        hooks=Rec(),
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
        upload_results=False,
    )
    assert f'start:{SimStage.SIMULATE}' in seen
    assert f'end:{SimStage.SIMULATE}' in seen


@pytest.mark.asyncio
async def test_meta_carries_target(datapoint_factory):
    """SimulationRunMeta passed to on_confirm must carry 'target' field."""
    captured: dict[str, Any] = {}

    class Cap(DefaultHooks):
        async def on_confirm(self, meta):
            captured.update(meta)
            return True

    async def _ok_target(messages):
        return 'fine'

    await simulate(
        datapoints=[datapoint_factory('dp1')],
        target=_ok_target,
        max_turns=1,
        hooks=Cap(),
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
        upload_results=False,
    )
    assert 'target' in captured
    assert captured['target'] == 'callback'


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
        'SimulationHooks',
        'DefaultHooks',
        'RichHooks',
        'SimulationRunMeta',
        'SimulationError',
        'SimulationCancelledError',
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
        raise RuntimeError('target down')

    dp = datapoint_factory('dp-err')
    results = await simulate(
        datapoints=[dp],
        target=boom,
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
        max_turns=1,
        evaluator_names=['goal_achieved'],
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
            raise RuntimeError('scoring blew up')

    hooks = _ScoringBoom()
    with pytest.raises(RuntimeError, match='scoring blew up'):
        await simulate(
            datapoints=[datapoint_factory('dp-1')],
            target=_ok_target,
            user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
            judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
            max_turns=1,
            evaluator_names=['goal_achieved'],
            hooks=hooks,
        )
    assert hooks.started is True
    assert hooks.completed_with is not None  # on_run_complete fired via finally even though scoring raised
    assert len(hooks.completed_with) == 1  # real results (not []) passed through despite scoring raise


@pytest.mark.asyncio
async def test_on_run_complete_gets_partial_results_when_a_row_is_dropped(datapoint_factory, monkeypatch):
    """exit_on_failure aborts via SimulationDroppedError when a row produces no
    result, but the rows that *did* succeed must still reach on_run_complete —
    not an empty list (the partial_results path)."""
    from evaluatorq.simulation.api import SimulationDroppedError, simulate
    from evaluatorq.simulation.runner.simulation import SimulationRunner

    def _good_result() -> SimulationResult:
        return SimulationResult(
            messages=[],
            terminated_by=TerminatedBy.max_turns,
            reason='',
            goal_achieved=True,
            goal_completion_score=1.0,
            rules_broken=[],
            turn_count=1,
            token_usage=TokenUsage(),
            turn_metrics=[],
        )

    async def fake_run(self, *, datapoint, max_turns):
        # 'dp-bad' raises uncaught so its job is recorded with no cached result
        # (a dropped row); 'dp-good' returns a real result.
        if datapoint.id == 'dp-bad':
            raise RuntimeError('runner exploded')
        return _good_result()

    monkeypatch.setattr(SimulationRunner, 'run', fake_run)

    hooks = _RunLevelRecorder()
    with pytest.raises(SimulationDroppedError):
        await simulate(
            datapoints=[datapoint_factory('dp-good'), datapoint_factory('dp-bad')],
            target=_ok_target,
            user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
            judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
            max_turns=1,
            evaluator_names=['goal_achieved'],
            hooks=hooks,
        )
    assert hooks.completed_with is not None  # terminal fired despite the drop
    assert len(hooks.completed_with) == 1  # the successful row, not []
    assert hooks.completed_with[0].goal_achieved is True


@pytest.mark.asyncio
async def test_simulation_dropped_error_is_simulation_error():
    """SimulationDroppedError must be catchable as SimulationError and carry the
    partial results passed to it."""
    from evaluatorq.simulation.api import SimulationDroppedError
    from evaluatorq.simulation.exceptions import SimulationError

    partial = [_sim_result_for_drop_test()]
    err = SimulationDroppedError('2 of 3 dropped', partial_results=partial)
    assert isinstance(err, SimulationError)
    assert err.partial_results == partial
    # Default keeps it an empty list, never None.
    assert SimulationDroppedError('boom').partial_results == []


def _sim_result_for_drop_test() -> SimulationResult:
    return SimulationResult(
        messages=[],
        terminated_by=TerminatedBy.max_turns,
        reason='',
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        token_usage=TokenUsage(),
        turn_metrics=[],
    )


@pytest.mark.asyncio
async def test_target_closed_when_on_run_complete_raises(datapoint_factory):
    """A raising on_run_complete must not leak a resource-owning target: the
    nested finally still closes the runner/target even though the unguarded
    terminal hook propagates."""

    class _ClosableTarget:
        def __init__(self) -> None:
            self.closed = False

        async def __call__(self, messages):
            return 'fine'

        async def close(self) -> None:
            self.closed = True

    class _CompleteBoom(_RunLevelRecorder):
        def on_run_complete(self, results) -> None:
            super().on_run_complete(results)
            raise RuntimeError('complete blew up')

    target = _ClosableTarget()
    hooks = _CompleteBoom()
    with pytest.raises(RuntimeError, match='complete blew up'):
        await simulate(
            datapoints=[datapoint_factory('dp-1')],
            target=target,  # pyright: ignore[reportArgumentType]
            user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
            judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
            max_turns=1,
            evaluator_names=['goal_achieved'],
            hooks=hooks,
        )
    assert hooks.completed_with is not None  # terminal still fired
    assert target.closed is True  # cleanup ran despite the raising hook


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
        return 'fine'

    runner = SimulationRunner(
        target_callback=slow_target,
        model='gpt-4o-mini',
        max_turns=1,
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
        hooks=_PeakHooks(),
    )
    dps = [datapoint_factory(f'dp-{i}') for i in range(6)]

    async def run():
        task = asyncio.create_task(
            runner.run_batch(
                dps, max_concurrency=2, timeout_per_simulation=0
            )  # timeout_per_simulation=0 disables the per-sim timeout so the gate controls timing
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
            datapoints=[datapoint_factory('dp-1')],
            user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
            judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
            max_turns=1,
            evaluator_names=['goal_achieved'],
            hooks=hooks,
        )
    assert hooks.started is False  # no unpaired on_run_start


def test_rich_hooks_reusable_across_runs():
    from rich.console import Console

    from evaluatorq.simulation.hooks import RichHooks

    hooks = RichHooks(console=Console(file=io.StringIO(), force_terminal=False))
    meta: SimulationRunMeta = SimulationRunMeta(
        num_datapoints=2,
        model='m',
        max_turns=3,
        parallelism=2,
        evaluation_name='',
        evaluator_names=['goal_achieved'],
        target='callback',
    )

    asyncio.run(hooks.on_run_start(meta))
    assert hooks._completed == 0
    # simulate one completion in run 1
    hooks._completed = 2
    asyncio.run(hooks.on_run_complete([]))

    # Run 2: state must reset
    asyncio.run(hooks.on_run_start(meta))
    # white-box: assert per-run reset contract directly
    assert hooks._completed == 0
    assert hooks._tasks == {}
    assert hooks._overall_task_id is not None  # fresh overall task created


# ---------------------------------------------------------------------------
# Async hook implementations driving full runs (support-both contract).
# These are the async twins of the sync integration tests above: they prove an
# `async def` hook flows through the real `await_maybe` call sites, that an
# async declining `on_confirm` aborts, and that async-raising hooks keep their
# guard/cleanup semantics. A dropped `await` would make these fail (a coroutine
# would be created and silently never run) where the sync twins cannot catch it.
# ---------------------------------------------------------------------------


def _sync_hook_deprecation_warnings(records) -> list[object]:
    """Filter a warning record list down to our sync-hook nudge."""
    return [w for w in records if issubclass(w.category, DeprecationWarning) and 'sync hook(s)' in str(w.message)]


@pytest.mark.asyncio
async def test_simulate_fires_run_level_hooks_async(datapoint_factory):
    """An async custom hook drives the full simulate() run and fires no
    sync-hook DeprecationWarning."""

    class AsyncRunLevelRecorder(DefaultHooks):
        def __init__(self) -> None:
            self.confirmed: list[int] = []
            self.run_started: list[int] = []
            self.evaluator_events: list[tuple[str, str, float]] = []
            self.run_completed = 0

        async def on_confirm(self, meta):
            self.confirmed.append(meta['num_datapoints'])
            return True

        async def on_run_start(self, meta):
            self.run_started.append(meta['num_datapoints'])

        async def on_evaluator_complete(self, datapoint_id, name, score, result):
            self.evaluator_events.append((datapoint_id, name, score))

        async def on_run_complete(self, results):
            self.run_completed += 1

    async def _ok_target(messages):
        return 'fine'

    hooks = AsyncRunLevelRecorder()
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        results = await simulate(
            target=_ok_target,
            datapoints=[datapoint_factory('dp1'), datapoint_factory('dp2')],
            max_turns=1,
            evaluator_names=['goal_achieved'],
            user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
            judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
            hooks=hooks,
        )
    # Side effects observed -> the async hook actually ran (awaited), not dropped.
    assert hooks.confirmed == [2]
    assert hooks.run_started == [2]
    assert hooks.run_completed == 1
    assert {e[0] for e in hooks.evaluator_events} == {'dp1', 'dp2'}
    assert len(results) == 2
    # Async hook -> no deprecation nudge.
    assert _sync_hook_deprecation_warnings(records) == []


@pytest.mark.asyncio
async def test_async_on_confirm_false_aborts(datapoint_factory):
    """An async on_confirm returning False aborts the real runner before any
    target call (the awaited bool gate works, not just sync)."""
    from evaluatorq.simulation.exceptions import SimulationCancelledError

    class AsyncDecliningHooks(DefaultHooks):
        def __init__(self) -> None:
            self.run_started = 0

        async def on_confirm(self, meta):
            return False

        async def on_run_start(self, meta):
            self.run_started += 1

    calls = []

    async def _spy_target(messages):
        calls.append(1)
        return 'fine'

    hooks = AsyncDecliningHooks()
    with pytest.raises(SimulationCancelledError):
        await simulate(
            target=_spy_target,
            datapoints=[datapoint_factory('dp1')],
            max_turns=1,
            evaluator_names=['goal_achieved'],
            user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
            judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
            hooks=hooks,
        )
    assert hooks.run_started == 0
    assert calls == []


@pytest.mark.asyncio
async def test_async_on_turn_complete_guard_does_not_corrupt_result(datapoint_factory):
    """An async-raising on_turn_complete (raises mid-await, inside await_maybe)
    is still swallowed by the runner guard and must not error the result."""

    class AsyncRaisingHooks(DefaultHooks):
        async def on_turn_complete(self, datapoint_id, metrics):
            raise RuntimeError('async hook boom')

    runner = SimulationRunner(
        target_callback=_ok_target,
        model='gpt-4o-mini',
        max_turns=1,
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
        hooks=AsyncRaisingHooks(),
    )
    result = await runner.run(datapoint=datapoint_factory('dp1'))
    assert result.terminated_by != TerminatedBy.error


@pytest.mark.asyncio
async def test_async_target_closed_when_on_run_complete_raises(datapoint_factory):
    """An async on_run_complete that raises mid-await still propagates, and the
    nested finally still closes the resource-owning target."""

    class _ClosableTarget:
        def __init__(self) -> None:
            self.closed = False

        async def __call__(self, messages):
            return 'fine'

        async def close(self) -> None:
            self.closed = True

    class AsyncCompleteBoom(DefaultHooks):
        def __init__(self) -> None:
            self.completed = False

        async def on_run_complete(self, results) -> None:
            self.completed = True
            raise RuntimeError('async complete blew up')

    target = _ClosableTarget()
    hooks = AsyncCompleteBoom()
    with pytest.raises(RuntimeError, match='async complete blew up'):
        await simulate(
            datapoints=[datapoint_factory('dp-1')],
            target=target,  # pyright: ignore[reportArgumentType]
            user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
            judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
            max_turns=1,
            evaluator_names=['goal_achieved'],
            hooks=hooks,
        )
    assert hooks.completed is True  # terminal still fired
    assert target.closed is True  # cleanup ran despite the raising async hook


# ---------------------------------------------------------------------------
# SimStage + on_stage_start/end hook tests (Task 4)
# ---------------------------------------------------------------------------


def test_stage_hooks_keep_protocol_satisfied():
    """Adding on_stage_start/end to the Protocol must not break isinstance checks
    as long as BOTH DefaultHooks and RichHooks implement the methods."""
    from rich.console import Console

    assert isinstance(DefaultHooks(), SimulationHooks)
    assert isinstance(RichHooks(console=Console()), SimulationHooks)


def test_rich_stage_start_renders_rule():
    """RichHooks.on_stage_start must emit a console.rule containing the stage label."""
    buf = io.StringIO()
    h = RichHooks(console=__import__('rich.console', fromlist=['Console']).Console(file=buf, width=80, force_terminal=False))
    asyncio.run(h.on_stage_start(SimStage.SIMULATE, {}))
    assert 'Running Simulations' in buf.getvalue()


def test_rich_stage_end_generate_prints_count():
    """RichHooks.on_stage_end with GENERATE and num_datapoints in meta prints count."""
    buf = io.StringIO()
    from rich.console import Console

    h = RichHooks(console=Console(file=buf, width=80, force_terminal=False))
    asyncio.run(h.on_stage_end(SimStage.GENERATE, {'num_datapoints': 42}))
    assert '42' in buf.getvalue()
    assert 'datapoints' in buf.getvalue()


def test_rich_stage_end_simulate_prints_nothing():
    """RichHooks.on_stage_end with SIMULATE stage must not print."""
    buf = io.StringIO()
    from rich.console import Console

    h = RichHooks(console=Console(file=buf, width=80, force_terminal=False))
    asyncio.run(h.on_stage_end(SimStage.SIMULATE, {}))
    assert buf.getvalue() == ''


def test_default_hooks_stage_methods_are_silent():
    """DefaultHooks.on_stage_start and on_stage_end run without raising."""
    hooks = DefaultHooks()
    asyncio.run(hooks.on_stage_start(SimStage.GENERATE, {}))
    asyncio.run(hooks.on_stage_end(SimStage.GENERATE, {'num_datapoints': 5}))
    asyncio.run(hooks.on_stage_start(SimStage.SIMULATE, {}))
    asyncio.run(hooks.on_stage_end(SimStage.SIMULATE, {}))


def test_simstage_exported_from_package():
    """SimStage must be importable from evaluatorq.simulation."""
    import evaluatorq.simulation as sim

    assert sim.SimStage is not None
    assert 'SimStage' in sim.__all__


# ---------------------------------------------------------------------------
# Task 6: Confirm gate + summary render in RichHooks; enriched DefaultHooks
# ---------------------------------------------------------------------------


def _meta_with_target(n: int = 2) -> SimulationRunMeta:
    """Variant of _meta() with a non-trivial target for Task 6 tests."""
    return SimulationRunMeta(
        num_datapoints=n,
        model='openai/gpt-5.4-mini',
        max_turns=5,
        parallelism=3,
        evaluation_name='sim',
        evaluator_names=['goal_achieved'],
        target='agent:foo',
    )


def test_rich_confirm_skip_renders_plan_with_target():
    """RichHooks.on_confirm with skip_confirm=True must render a Run Plan
    table containing the target and model, then return True without prompting."""
    from rich.console import Console

    buf = io.StringIO()
    h = RichHooks(console=Console(file=buf, width=90, force_terminal=False), skip_confirm=True)
    assert asyncio.run(h.on_confirm(_meta_with_target())) is True
    out = buf.getvalue()
    assert 'Run Plan' in out
    assert 'agent:foo' in out
    assert 'openai/gpt-5.4-mini' in out


def test_rich_run_complete_renders_once(sim_result_factory):
    """RichHooks.on_run_complete must render the summary exactly once even if
    called twice (idempotent guard on _summary_rendered)."""
    from rich.console import Console

    buf = io.StringIO()
    h = RichHooks(console=Console(file=buf, width=90, force_terminal=False))
    results = [sim_result_factory(goal_achieved=True)]
    asyncio.run(h.on_run_complete(results))
    asyncio.run(h.on_run_complete(results))   # second call must be a no-op
    out = buf.getvalue()
    assert out.count('SIMULATION SUMMARY') == 1
    assert 'ui --latest' in out


def test_rich_run_complete_summary_rendered_resets_on_new_run(sim_result_factory):
    """_reset_run_state must clear _summary_rendered so a reused RichHooks
    instance renders a fresh summary on the next on_run_start / on_run_complete
    cycle."""
    from rich.console import Console

    buf = io.StringIO()
    h = RichHooks(console=Console(file=buf, width=90, force_terminal=False))
    results = [sim_result_factory(goal_achieved=True)]

    # First run — renders once.
    asyncio.run(h.on_run_start(_meta_with_target()))
    asyncio.run(h.on_run_complete(results))
    assert buf.getvalue().count('SIMULATION SUMMARY') == 1

    # Second run — reset must clear the guard; summary must render again.
    asyncio.run(h.on_run_start(_meta_with_target()))
    asyncio.run(h.on_run_complete(results))
    assert buf.getvalue().count('SIMULATION SUMMARY') == 2


def test_default_hooks_on_confirm_logs_plan_and_returns_true(caplog):
    """DefaultHooks.on_confirm must log the run plan at INFO and return True."""
    import logging

    hooks = DefaultHooks()
    with caplog.at_level(logging.INFO, logger='evaluatorq.simulation.hooks'):
        result = asyncio.run(hooks.on_confirm(_meta_with_target()))
    assert result is True
    assert 'Run plan' in caplog.text
    assert 'agent:foo' in caplog.text
    assert 'openai/gpt-5.4-mini' in caplog.text
    assert any(r.levelname == 'INFO' for r in caplog.records)


def test_default_hooks_on_run_complete_logs_structured(caplog, sim_result_factory):
    """DefaultHooks.on_run_complete must log goal_achieved count and avg_turns."""
    import logging

    hooks = DefaultHooks()
    results = [
        sim_result_factory(goal_achieved=True),
        sim_result_factory(goal_achieved=False),
    ]
    with caplog.at_level(logging.INFO, logger='evaluatorq.simulation.hooks'):
        asyncio.run(hooks.on_run_complete(results))
    assert 'Run complete' in caplog.text
    assert 'goal_achieved=1/2' in caplog.text
    assert 'avg_turns=' in caplog.text
    assert any(r.levelname == 'INFO' for r in caplog.records)


def test_default_hooks_on_run_complete_warns_on_errors(caplog, sim_result_factory):
    """DefaultHooks.on_run_complete must emit a WARNING when any result errored."""
    import logging

    hooks = DefaultHooks()
    results = [sim_result_factory(goal_achieved=False, error='oops')]
    with caplog.at_level(logging.WARNING, logger='evaluatorq.simulation.hooks'):
        asyncio.run(hooks.on_run_complete(results))
    assert any(r.levelname == 'WARNING' for r in caplog.records)
    assert 'errored' in caplog.text.lower()
