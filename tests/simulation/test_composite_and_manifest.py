# Exercises the Unit-2b wiring: CompositeSimulationHooks fan-out, the
# ManifestStageHooks bridge, and the manifest lifecycle (D3 two-stage recording,
# Dec1 cancel/truthful-error, post-run-failure stage truth). Sync hook classes
# are deliberate (compat path); the override mismatch is the point.
# pyright: reportIncompatibleMethodOverride=false
from __future__ import annotations

import asyncio

import pytest

from evaluatorq.contracts import TokenUsage
from evaluatorq.simulation.hooks import (
    CompositeSimulationHooks,
    DefaultHooks,
    ManifestStageHooks,
    SimulationHooks,
    SimStage,
    SimulationRunMeta,
)
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Judgment,
    Persona,
    Scenario,
    SimulationDatapoint,
)


# ---------------------------------------------------------------------------
# Shared harness (self-contained; mirrors tests/simulation/test_hooks.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def datapoint_factory():
    def _make(dp_id: str) -> SimulationDatapoint:
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
        return SimulationDatapoint(
            id=dp_id,
            persona=persona,
            scenario=scenario,
            user_system_prompt='',
            first_message='hi',
        )

    return _make


class _StubUserSim:
    def update_context(self, *, persona_context=None, scenario_context=None) -> None:
        pass

    async def generate_first_message(self) -> str:
        return 'hello'

    async def respond_async(self, messages, *, llm_purpose=None) -> str:
        return 'more'

    def reset_usage(self) -> None:
        pass

    def get_usage(self) -> TokenUsage:
        return TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)


class _StubJudge:
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

    def get_usage(self) -> TokenUsage:
        return TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)


async def _ok_target(messages) -> str:
    return 'fine'


def _meta() -> SimulationRunMeta:
    return SimulationRunMeta(
        num_datapoints=1,
        model='m',
        max_turns=3,
        datapoint_parallelism=1,
        evaluation_name='e',
        evaluator_names=['goal_achieved'],
        target='callback',
    )


# ---------------------------------------------------------------------------
# CompositeSimulationHooks — void fan-out
# ---------------------------------------------------------------------------


class _VoidRec(DefaultHooks):
    def __init__(self, name: str, calls: list[str], *, raise_on: str | None = None) -> None:
        self.name = name
        self.calls = calls
        self.raise_on = raise_on

    async def on_run_start(self, meta) -> None:
        self.calls.append(self.name)
        if self.raise_on == 'on_run_start':
            raise RuntimeError(f'boom-{self.name}')


def test_composite_void_fans_out_to_all_children():
    calls: list[str] = []
    composite = CompositeSimulationHooks([_VoidRec('a', calls), _VoidRec('b', calls)])
    asyncio.run(composite.on_run_start(_meta()))
    assert calls == ['a', 'b']


def test_composite_void_raising_child_still_runs_later_child_and_reraises_first():
    calls: list[str] = []
    composite = CompositeSimulationHooks([
        _VoidRec('a', calls, raise_on='on_run_start'),
        _VoidRec('b', calls, raise_on='on_run_start'),
    ])
    with pytest.raises(RuntimeError, match='boom-a'):  # first exception wins
        asyncio.run(composite.on_run_start(_meta()))
    assert calls == ['a', 'b']  # later child still ran despite the earlier raise


# ---------------------------------------------------------------------------
# CompositeSimulationHooks — on_confirm combine + exception policy
# ---------------------------------------------------------------------------


class _Confirm(DefaultHooks):
    def __init__(self, name: str, calls: list[str], *, value: bool = True, raise_: bool = False) -> None:
        self.name = name
        self.calls = calls
        self.value = value
        self.raise_ = raise_

    async def on_confirm(self, meta) -> bool:
        self.calls.append(self.name)
        if self.raise_:
            raise RuntimeError(f'boom-{self.name}')
        return self.value


def test_composite_on_confirm_all_true_returns_true():
    calls: list[str] = []
    composite = CompositeSimulationHooks([_Confirm('a', calls, value=True), _Confirm('b', calls, value=True)])
    assert asyncio.run(composite.on_confirm(_meta())) is True
    assert calls == ['a', 'b']


def test_composite_on_confirm_any_false_returns_false_but_runs_all():
    calls: list[str] = []
    composite = CompositeSimulationHooks([_Confirm('a', calls, value=True), _Confirm('b', calls, value=False)])
    assert asyncio.run(composite.on_confirm(_meta())) is False
    assert calls == ['a', 'b']  # not short-circuited


def test_composite_on_confirm_raising_child_still_runs_later_and_reraises():
    calls: list[str] = []
    composite = CompositeSimulationHooks([
        _Confirm('a', calls, raise_=True),
        _Confirm('b', calls, value=True),
    ])
    with pytest.raises(RuntimeError, match='boom-a'):
        asyncio.run(composite.on_confirm(_meta()))
    assert calls == ['a', 'b']  # same run-all policy as void methods


# ---------------------------------------------------------------------------
# ManifestStageHooks — only stage methods touch the writer
# ---------------------------------------------------------------------------


class _FakeWriter:
    def __init__(self) -> None:
        self.started: list[tuple[object, object]] = []
        self.ended: list[tuple[object, object, object]] = []

    def start_stage(self, stage, *, target=None) -> None:
        self.started.append((stage, target))

    def end_stage(self, stage, *, target=None, error=None) -> None:
        self.ended.append((stage, target, error))


def test_manifest_stage_hooks_bridge_start_and_end():
    w = _FakeWriter()
    h: SimulationHooks = ManifestStageHooks(w)  # pyright: ignore[reportArgumentType]
    asyncio.run(h.on_stage_start(SimStage.GENERATE, {'target': None}))
    err = RuntimeError('x')
    asyncio.run(h.on_stage_end(SimStage.SIMULATE, {'error': err}))
    assert w.started == [(SimStage.GENERATE, None)]
    assert w.ended == [(SimStage.SIMULATE, None, err)]


def test_manifest_stage_hooks_other_methods_are_noops():
    w = _FakeWriter()
    h = ManifestStageHooks(w)  # pyright: ignore[reportArgumentType]
    assert asyncio.run(h.on_confirm(_meta())) is True
    assert asyncio.run(h.on_run_start(_meta())) is None
    assert asyncio.run(h.on_run_complete([])) is None
    assert w.started == [] and w.ended == []  # untouched by non-stage events


def test_manifest_stage_hooks_satisfies_protocol():
    assert isinstance(ManifestStageHooks(_FakeWriter()), SimulationHooks)  # pyright: ignore[reportArgumentType]


def test_composite_satisfies_protocol():
    assert isinstance(CompositeSimulationHooks([DefaultHooks()]), SimulationHooks)


# ---------------------------------------------------------------------------
# Manifest lifecycle through the real simulate()/generate_and_simulate() paths
# ---------------------------------------------------------------------------


def _stub_generation(monkeypatch, datapoint):
    """Stub the generation trio so generate_and_simulate does no network I/O."""
    from evaluatorq.simulation import api

    async def _fake_personas_scenarios(**_kwargs):
        return [], []

    async def _fake_resolve(**_kwargs):
        return [datapoint]

    monkeypatch.setattr(api, '_generate_personas_scenarios', _fake_personas_scenarios)
    monkeypatch.setattr(api, '_resolve_or_generate_datapoints', _fake_resolve)
    monkeypatch.setattr(
        'evaluatorq.openresponses.client.build_simulation_client',
        lambda _client=None, **_kwargs: (object(), False),
    )


@pytest.mark.asyncio
async def test_generate_and_simulate_records_both_stages(datapoint_factory, monkeypatch):
    """Sim D3 (headline): generate_and_simulate records BOTH 'generate' and
    'simulate' stages; the manifest is no longer single-stage."""
    from evaluatorq.common.run_manifest import list_manifests
    from evaluatorq.simulation.api import generate_and_simulate
    from evaluatorq.simulation.utils.run_store import get_sim_runs_dir

    _stub_generation(monkeypatch, datapoint_factory('dp1'))

    await generate_and_simulate(
        agent_description='a helpful assistant',
        target=_ok_target,
        max_turns=1,
        evaluator_names=['goal_achieved'],
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
        upload_results=False,
        save=True,
    )

    manifests = list_manifests(get_sim_runs_dir())
    assert len(manifests) == 1
    m = manifests[0]
    stage_names = [s.name for s in m.stages]
    assert stage_names == ['generate', 'simulate']  # D3: both phases recorded
    assert all(s.status == 'completed' for s in m.stages)
    assert m.status == 'completed'


@pytest.mark.asyncio
async def test_bare_simulate_records_only_simulate_stage(datapoint_factory):
    """Counterpart to D3: bare simulate() has no GENERATE phase → one stage."""
    from evaluatorq.common.run_manifest import list_manifests
    from evaluatorq.simulation.api import simulate
    from evaluatorq.simulation.utils.run_store import get_sim_runs_dir

    await simulate(
        target=_ok_target,
        datapoints=[datapoint_factory('dp1')],
        max_turns=1,
        evaluator_names=['goal_achieved'],
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
        upload_results=False,
        save=True,
    )

    manifests = list_manifests(get_sim_runs_dir())
    assert len(manifests) == 1
    m = manifests[0]
    assert [s.name for s in m.stages] == ['simulate']
    assert m.status == 'completed'


@pytest.mark.asyncio
async def test_decline_after_generate_cancels_and_keeps_generate_completed(datapoint_factory, monkeypatch):
    """Dec1 cancel: a declined on_confirm (which fires AFTER generate) → run
    'cancelled', 'generate' stays 'completed', nothing flipped to error."""
    from evaluatorq.common.run_manifest import list_manifests
    from evaluatorq.simulation.api import generate_and_simulate
    from evaluatorq.simulation.exceptions import SimulationCancelledError
    from evaluatorq.simulation.utils.run_store import get_sim_runs_dir

    _stub_generation(monkeypatch, datapoint_factory('dp1'))

    class Decline(DefaultHooks):
        async def on_confirm(self, meta) -> bool:
            return False

    with pytest.raises(SimulationCancelledError):
        await generate_and_simulate(
            agent_description='a helpful assistant',
            target=_ok_target,
            max_turns=1,
            evaluator_names=['goal_achieved'],
            user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
            judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
            hooks=Decline(),
            upload_results=False,
            save=True,
        )

    manifests = list_manifests(get_sim_runs_dir())
    assert len(manifests) == 1
    m = manifests[0]
    assert m.status == 'cancelled'
    gen = next(s for s in m.stages if s.name == 'generate')
    assert gen.status == 'completed'  # finished stage stays truthful
    assert all(s.status != 'error' for s in m.stages)  # nothing manufactured to error


@pytest.mark.asyncio
async def test_simulate_phase_failure_marks_simulate_stage_error(datapoint_factory):
    """Dec1 truthful error: a failure INSIDE the simulate phase marks the
    SIMULATE stage 'error' (via sys.exc_info in the finally) AND the run
    'error' — the two are consistent, not contradictory."""
    from evaluatorq.common.run_manifest import list_manifests
    from evaluatorq.simulation.api import simulate
    from evaluatorq.simulation.utils.run_store import get_sim_runs_dir

    class ScoringBoom(DefaultHooks):
        async def on_evaluator_complete(self, datapoint_id, name, score, result) -> None:
            raise RuntimeError('scoring blew up')

    with pytest.raises(RuntimeError, match='scoring blew up'):
        await simulate(
            target=_ok_target,
            datapoints=[datapoint_factory('dp1')],
            max_turns=1,
            evaluator_names=['goal_achieved'],
            user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
            judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
            hooks=ScoringBoom(),
            upload_results=False,
            save=True,
        )

    manifests = list_manifests(get_sim_runs_dir())
    assert len(manifests) == 1
    m = manifests[0]
    assert m.status == 'error'
    sim_stage = next(s for s in m.stages if s.name == 'simulate')
    assert sim_stage.status == 'error'  # truthful: the stage body raised


@pytest.mark.asyncio
async def test_post_stage_failure_leaves_simulate_stage_completed(datapoint_factory, monkeypatch):
    """Dec1(b): a failure in post-stage glue AFTER the SIMULATE stage closed
    leaves that stage 'completed' (R1: never relabel a finished stage) while
    the run is 'error'."""
    from evaluatorq.common.run_manifest import list_manifests
    from evaluatorq.simulation import api
    from evaluatorq.simulation.api import simulate
    from evaluatorq.simulation.utils.run_store import get_sim_runs_dir

    def _boom(**_kwargs):
        raise RuntimeError('build blew up')

    # build_simulation_run runs after on_stage_end(SIMULATE) has closed the
    # stage 'completed', so its failure must not relabel the closed stage.
    monkeypatch.setattr(api, 'build_simulation_run', _boom)

    with pytest.raises(RuntimeError, match='build blew up'):
        await simulate(
            target=_ok_target,
            datapoints=[datapoint_factory('dp1')],
            max_turns=1,
            evaluator_names=['goal_achieved'],
            user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
            judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
            upload_results=False,
            save=True,
        )

    manifests = list_manifests(get_sim_runs_dir())
    assert len(manifests) == 1
    m = manifests[0]
    assert m.status == 'error'
    sim_stage = next(s for s in m.stages if s.name == 'simulate')
    assert sim_stage.status == 'completed'  # closed before the glue failure → stays truthful


@pytest.mark.asyncio
async def test_multiple_user_hooks_fan_out(datapoint_factory):
    """Public multi-hooks: a list of two user hooks both observe the run."""
    from evaluatorq.simulation.api import simulate

    class Rec(DefaultHooks):
        def __init__(self) -> None:
            self.completed = 0

        async def on_run_complete(self, results) -> None:
            self.completed += 1

    h1, h2 = Rec(), Rec()
    results = await simulate(
        target=_ok_target,
        datapoints=[datapoint_factory('dp1')],
        max_turns=1,
        evaluator_names=['goal_achieved'],
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
        hooks=[h1, h2],
        upload_results=False,
    )
    assert len(results) == 1
    assert h1.completed == 1 and h2.completed == 1  # both children fanned out


@pytest.mark.asyncio
async def test_generate_fans_out_stage_events_to_multiple_user_hooks(monkeypatch):
    """Standalone generate() supports the same public multi-hook contract."""
    from evaluatorq.simulation import api
    from evaluatorq.simulation.api import generate

    calls: list[str] = []

    class Recorder(DefaultHooks):
        async def on_stage_start(self, stage, meta) -> None:
            name = stage.value if isinstance(stage, SimStage) else stage
            calls.append(f'start:{name}')

        async def on_stage_end(self, stage, meta) -> None:
            name = stage.value if isinstance(stage, SimStage) else stage
            calls.append(f'end:{name}')

    async def _fake_generate(**_kwargs):
        return [], None, False

    monkeypatch.setattr(api, '_generate_datapoints_inner', _fake_generate)

    await generate(
        agent_description='a helpful assistant',
        num_personas=1,
        num_scenarios=1,
        hooks=[Recorder(), Recorder()],
    )

    assert calls == ['start:generate', 'start:generate', 'end:generate', 'end:generate']


@pytest.mark.asyncio
async def test_generate_and_simulate_tracks_description_resolution_failure(monkeypatch):
    """The manifest exists before description resolution and records its failure."""
    from evaluatorq.common.run_manifest import list_manifests
    from evaluatorq.simulation import api
    from evaluatorq.simulation.api import generate_and_simulate
    from evaluatorq.simulation.utils.run_store import get_sim_runs_dir

    async def _boom(**_kwargs):
        raise RuntimeError('description lookup exploded')

    monkeypatch.setattr(api, '_resolve_generation_agent_description', _boom)

    with pytest.raises(RuntimeError, match='description lookup exploded'):
        await generate_and_simulate(save=True)

    [manifest] = list_manifests(get_sim_runs_dir())
    assert manifest.status == 'error'
    assert manifest.error == 'description lookup exploded'


@pytest.mark.asyncio
async def test_generate_phase_failure_marks_manifest_error_not_running(datapoint_factory, monkeypatch):
    """FIX 1: a failure DURING the generate phase of generate_and_simulate must
    finalize the manifest as 'error' — not leave it stuck 'running' forever
    (the manifest is minted before generate, but _simulate_core, which owns the
    terminal calls, is never reached when generation raises)."""
    from evaluatorq.common.run_manifest import list_manifests
    from evaluatorq.simulation import api
    from evaluatorq.simulation.api import generate_and_simulate
    from evaluatorq.simulation.utils.run_store import get_sim_runs_dir

    async def _fake_personas_scenarios(**_kwargs):
        return [], []

    async def _boom_resolve(**_kwargs):
        raise RuntimeError('datapoint generation exploded')

    monkeypatch.setattr(api, '_generate_personas_scenarios', _fake_personas_scenarios)
    monkeypatch.setattr(api, '_resolve_or_generate_datapoints', _boom_resolve)
    monkeypatch.setattr(
        'evaluatorq.openresponses.client.build_simulation_client',
        lambda _client=None, **_kwargs: (object(), False),
    )

    with pytest.raises(RuntimeError, match='datapoint generation exploded'):
        await generate_and_simulate(
            agent_description='a helpful assistant',
            target=_ok_target,
            max_turns=1,
            evaluator_names=['goal_achieved'],
            user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
            judge=_StubJudge(terminate=True),  # pyright: ignore[reportArgumentType]
            upload_results=False,
            save=True,
        )

    manifests = list_manifests(get_sim_runs_dir())
    assert len(manifests) == 1
    m = manifests[0]
    assert m.status == 'error'  # NOT stuck 'running'
    assert m.ended_at is not None
    # The GENERATE stage that was in-flight when it raised is truthfully 'error'.
    gen = next(s for s in m.stages if s.name == 'generate')
    assert gen.status == 'error'
