"""Replaying a prior simulation run (`previous_run=` / `--from-run`)."""
# ruff: noqa: S101

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from evaluatorq.common.replay import ReplayError
from evaluatorq.contracts import LLMCallConfig
from evaluatorq.simulation.replay import load_simulation_replay
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Criterion,
    Persona,
    Scenario,
    SimulationDatapoint,
    StartingEmotion,
)
from evaluatorq.simulation.utils.run_store import auto_save_run, build_simulation_run, get_sim_runs_dir

if TYPE_CHECKING:
    from pathlib import Path


def _datapoint(name: str = 'dp-1') -> SimulationDatapoint:
    persona = Persona(
        name='Impatient Ien',
        patience=2,
        assertiveness=8,
        politeness=4,
        technical_level=6,
        communication_style=CommunicationStyle.terse,
        background='Long-time customer with a stalled refund.',
    )
    scenario = Scenario(
        name='Refund chase',
        goal='Get the refund released today',
        context='Third contact about the same order.',
        starting_emotion=StartingEmotion.frustrated,
        criteria=[Criterion(description='Agent states a refund date', type='must_happen')],
    )
    return SimulationDatapoint(
        id=name,
        persona=persona,
        scenario=scenario,
        user_system_prompt='',
        first_message='Where is my refund?',
    )


def test_saved_run_carries_its_datapoints(tmp_path: Path) -> None:
    dp = _datapoint()
    run = build_simulation_run(
        run_name='sim',
        mode='simulate',
        target_kind='callback',
        evaluator_names=[],
        results=[],
        datapoints=[dp],
    )
    path = auto_save_run(run=run, run_name='sim')

    stored = json.loads(path.read_text(encoding='utf-8'))
    assert stored['datapoints'][0]['id'] == 'dp-1'

    replayed = load_simulation_replay('latest', get_sim_runs_dir())
    assert replayed.datapoints == [dp]


def test_replay_resolves_by_run_name(tmp_path: Path) -> None:
    run = build_simulation_run(
        run_name='nightly',
        mode='simulate',
        target_kind='callback',
        evaluator_names=[],
        results=[],
        datapoints=[_datapoint('dp-a'), _datapoint('dp-b')],
    )
    auto_save_run(run=run, run_name='nightly')

    replayed = load_simulation_replay('nightly', get_sim_runs_dir())
    assert [dp.id for dp in replayed.datapoints] == ['dp-a', 'dp-b']


def test_run_saved_without_datapoints_cannot_be_replayed(tmp_path: Path) -> None:
    run = build_simulation_run(
        run_name='legacy',
        mode='simulate',
        target_kind='callback',
        evaluator_names=[],
        results=[],
    )
    auto_save_run(run=run, run_name='legacy')

    with pytest.raises(ReplayError, match='records no simulation datapoints'):
        load_simulation_replay('latest', get_sim_runs_dir())


def test_missing_run_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match='Could not resolve previous simulation run'):
        load_simulation_replay('does-not-exist', get_sim_runs_dir())


@pytest.mark.asyncio
async def test_previous_run_short_circuits_other_sources(tmp_path: Path) -> None:
    from evaluatorq.simulation.api import _resolve_or_generate_datapoints

    dp = _datapoint()
    run = build_simulation_run(
        run_name='sim',
        mode='simulate',
        target_kind='callback',
        evaluator_names=[],
        results=[],
        datapoints=[dp],
    )
    auto_save_run(run=run, run_name='sim')

    resolved = await _resolve_or_generate_datapoints(
        caller='simulate',
        datapoints=None,
        personas=None,
        scenarios=None,
        dataset_id=None,
        previous_run='latest',
        llm_config=LLMCallConfig(model='openai/gpt-5.4-mini'),
        generation_client=None,
    )
    assert resolved == [dp]


@pytest.mark.asyncio
async def test_previous_run_is_mutually_exclusive(tmp_path: Path) -> None:
    from evaluatorq.simulation.api import _resolve_or_generate_datapoints

    with pytest.raises(ValueError, match='Pass exactly one of previous_run'):
        await _resolve_or_generate_datapoints(
            caller='simulate',
            datapoints=[_datapoint()],
            personas=None,
            scenarios=None,
            dataset_id=None,
            previous_run='latest',
            llm_config=LLMCallConfig(model='openai/gpt-5.4-mini'),
            generation_client=None,
        )


def test_a_red_team_run_is_rejected_with_a_pointer_to_the_right_command(tmp_path: Path) -> None:
    """Both surfaces store a top-level `datapoints`, so the discriminator is `pipeline`."""
    runs = tmp_path / 'runs'
    runs.mkdir(parents=True)
    (runs / 'rt_20260101_000000.json').write_text(
        json.dumps({
            'run_name': 'rt',
            'pipeline': 'dynamic',
            # A red team run saved by this same version: non-empty datapoints, so
            # the guard cannot be gated on the list being empty.
            'datapoints': [{'id': 'a', 'category': 'ASI01', 'strategy': {'name': 's'}}],
        }),
        encoding='utf-8',
    )

    with pytest.raises(ReplayError, match='eq redteam run --from-run'):
        load_simulation_replay('latest', runs)


def test_max_turns_is_restored_from_the_replayed_run(tmp_path: Path) -> None:
    run = build_simulation_run(
        run_name='sim',
        mode='simulate',
        target_kind='callback',
        evaluator_names=[],
        results=[],
        max_turns=7,
        datapoints=[_datapoint()],
    )
    auto_save_run(run=run, run_name='sim')

    assert load_simulation_replay('latest', get_sim_runs_dir()).max_turns == 7


@pytest.fixture
def pipeline_spans():
    """Collect simulation spans in memory."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

    class _Collect(SpanExporter):
        def __init__(self) -> None:
            self.spans: list[object] = []

        def export(self, spans):  # noqa: ANN001, ANN202
            self.spans.extend(spans)
            return SpanExportResult.SUCCESS

    exporter = _Collect()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with patch('evaluatorq.simulation.tracing.get_tracer', return_value=provider.get_tracer('test')):
        yield exporter
    provider.shutdown()


def _pipeline_attrs(exporter) -> dict[str, object]:
    for span in exporter.spans:
        if span.name == 'Evaluatorq - Agent Simulation':
            return dict(span.attributes or {})
    raise AssertionError('no Evaluatorq - Agent Simulation span was recorded')


@pytest.mark.asyncio
async def test_pipeline_span_carries_the_default_turn_cap(pipeline_spans) -> None:
    """max_turns is None until resolved, and set_span_attrs drops None — so the
    attribute has to be stamped after resolution or it vanishes from traces."""
    from evaluatorq.simulation.api import simulate

    with (
        patch('evaluatorq.simulation.api._resolve_or_generate_datapoints', new=AsyncMock(return_value=[])),
        patch('evaluatorq.simulation.api._simulate_via_evaluatorq', new=AsyncMock(return_value=[])),
    ):
        await simulate(evaluation_name='x', target=lambda _m: 'hi', datapoints=[])

    assert _pipeline_attrs(pipeline_spans)['orq.simulation.max_turns'] == 10


@pytest.mark.asyncio
async def test_pipeline_span_carries_a_replayed_turn_cap(tmp_path: Path, pipeline_spans) -> None:
    from evaluatorq.simulation.api import simulate

    run = build_simulation_run(
        run_name='sim',
        mode='simulate',
        target_kind='callback',
        evaluator_names=[],
        results=[],
        max_turns=7,
        datapoints=[_datapoint()],
    )
    auto_save_run(run=run, run_name='sim')

    with patch('evaluatorq.simulation.api._simulate_via_evaluatorq', new=AsyncMock(return_value=[])):
        await simulate(evaluation_name='x', target=lambda _m: 'hi', previous_run='latest')

    assert _pipeline_attrs(pipeline_spans)['orq.simulation.max_turns'] == 7


def test_a_saved_run_stamps_the_replay_format_version() -> None:
    """The marker is only meaningful next to stored cases, so it rides with them."""
    from evaluatorq.common.replay import REPLAY_VERSION
    from evaluatorq.simulation.utils.run_store import build_simulation_run

    with_cases = build_simulation_run(
        run_name='sim',
        mode='simulate',
        target_kind='orq_agent',
        evaluator_names=[],
        results=[],
        datapoints=[_datapoint()],
    )
    without_cases = build_simulation_run(
        run_name='sim',
        mode='simulate',
        target_kind='orq_agent',
        evaluator_names=[],
        results=[],
    )

    assert with_cases.replay_version == REPLAY_VERSION
    assert without_cases.replay_version is None


def test_replay_error_is_importable_from_the_simulation_surface() -> None:
    """SDK callers passing previous_run= need to catch it without reaching into
    evaluatorq.common."""
    import evaluatorq.simulation as sim
    from evaluatorq.common.replay import ReplayError as internal

    assert sim.ReplayError is internal
    assert 'ReplayError' in sim.__all__
