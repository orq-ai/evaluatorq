"""Tests for the run->orq-dataset round-trip: dataset-format output, the
`upload-dataset` command, and stringified persona/scenario read tolerance."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evaluatorq.simulation._datapoint_io import _extract_single_datapoint
from evaluatorq.simulation.cli import app
from evaluatorq.simulation.contracts import (
    CommunicationStyle,
    EmotionalArc,
    Persona,
    Scenario,
    SimulationDatapoint,
    StartingEmotion,
)
from evaluatorq.simulation.utils.dataset_export import (
    load_datapoints_from_jsonl,
    to_orq_dataset_rows,
)
from evaluatorq.types import DataPoint

runner = CliRunner()


def _datapoint() -> SimulationDatapoint:
    persona = Persona(
        name='P',
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.terse,
        background='bg',
        emotional_arc=EmotionalArc.stable,
    )
    scenario = Scenario(name='S', goal='g', context='c', starting_emotion=StartingEmotion.neutral, criteria=[])
    return SimulationDatapoint(id='dp1', persona=persona, scenario=scenario, user_system_prompt='', first_message='hi')


def test_to_orq_dataset_rows_uses_scalar_inputs() -> None:
    # The orq datasets API rejects nested objects in `inputs`, so every value
    # must be scalar and expected_output must be a string, not null.
    row = to_orq_dataset_rows([_datapoint()])[0]
    assert all(isinstance(v, str) for v in row['inputs'].values())
    assert row['expected_output'] == ''
    assert json.loads(row['inputs']['persona'])['name'] == 'P'


def test_dataset_envelope_round_trips(tmp_path: Path) -> None:
    path = tmp_path / 'rows.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in to_orq_dataset_rows([_datapoint()])), encoding='utf-8')
    loaded = load_datapoints_from_jsonl(str(path))
    assert loaded[0].persona.name == 'P'
    assert loaded[0].scenario.goal == 'g'


def test_extract_datapoint_accepts_stringified_fields() -> None:
    dp = _datapoint()
    got = _extract_single_datapoint(
        DataPoint(inputs={'persona': dp.persona.model_dump_json(), 'scenario': dp.scenario.model_dump_json()})
    )
    assert got.persona.name == 'P'
    assert got.scenario.goal == 'g'


def _write_raw(path: Path) -> None:
    path.write_text(_datapoint().model_dump_json() + '\n', encoding='utf-8')


class _FakeDatasets:
    def __init__(self) -> None:
        self.created: dict[str, Any] | None = None
        self.appended: tuple[str, list[Any]] | None = None

    def create(self, *, display_name: str, path: str) -> Any:
        # Mirrors the orq-ai-sdk >= 4.16 signature; 4.15 takes one `request=` dict.
        self.created = {'display_name': display_name, 'path': path}
        return type('DS', (), {'id': 'ds_new'})()

    def create_datapoint(self, *, dataset_id: str, request_body: list[Any]) -> None:
        self.appended = (dataset_id, request_body)


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> _FakeDatasets:
    ds = _FakeDatasets()
    client = type('Client', (), {'datasets': ds})()
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    monkeypatch.setattr('evaluatorq.fetch_data.setup_orq_client', lambda _key: client)
    return ds


def test_upload_dataset_creates_and_uploads(tmp_path: Path, fake_client: _FakeDatasets) -> None:
    dp_file = tmp_path / 'dp.jsonl'
    _write_raw(dp_file)
    result = runner.invoke(app, ['upload-dataset', '-i', str(dp_file), '-n', 'My Set'])
    assert result.exit_code == 0, result.output
    assert fake_client.created == {'display_name': 'My Set', 'path': 'Default'}
    assert fake_client.appended is not None
    dataset_id, rows = fake_client.appended
    assert dataset_id == 'ds_new'
    assert len(rows) == 1
    assert 'eq sim simulate --dataset-id ds_new --target agent:<your-agent-key>' in result.output


def test_upload_dataset_extends_existing(tmp_path: Path, fake_client: _FakeDatasets) -> None:
    dp_file = tmp_path / 'dp.jsonl'
    _write_raw(dp_file)
    result = runner.invoke(app, ['upload-dataset', '-i', str(dp_file), '--dataset-id', 'ds_existing'])
    assert result.exit_code == 0, result.output
    assert fake_client.created is None  # extends, does not create
    assert fake_client.appended is not None
    assert fake_client.appended[0] == 'ds_existing'


def test_upload_dataset_requires_name_or_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    dp_file = tmp_path / 'dp.jsonl'
    _write_raw(dp_file)
    result = runner.invoke(app, ['upload-dataset', '-i', str(dp_file)])
    assert result.exit_code != 0


def test_write_datapoints_dataset_format(tmp_path: Path) -> None:
    from evaluatorq.simulation.cli import _write_datapoints

    raw, env = tmp_path / 'raw.jsonl', tmp_path / 'env.jsonl'
    _write_datapoints([_datapoint()], raw)
    _write_datapoints([_datapoint()], env, dataset_format=True)
    assert 'inputs' not in json.loads(raw.read_text().splitlines()[0])
    assert 'inputs' in json.loads(env.read_text().splitlines()[0])


@pytest.mark.parametrize('args', [[], ['-d', 'x.jsonl', '--dataset-id', 'ds']])
def test_simulate_requires_exactly_one_source(args: list[str]) -> None:
    result = runner.invoke(app, ['simulate', *args])
    assert result.exit_code != 0


class _LegacyDatasets(_FakeDatasets):
    def create(self, *, request: dict[str, Any]) -> Any:  # ty: ignore[incompatible-method-override]
        # The orq-ai-sdk 4.15 signature.
        self.created = request
        return type('DS', (), {'id': 'ds_legacy'})()


def test_upload_dataset_still_creates_on_a_pre_4_16_sdk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _LegacyDatasets()
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    monkeypatch.setattr('evaluatorq.fetch_data.setup_orq_client', lambda _key: type('Client', (), {'datasets': ds})())
    dp_file = tmp_path / 'dp.jsonl'
    _write_raw(dp_file)

    with pytest.warns(FutureWarning, match=r'orq-ai-sdk older than 4\.16 is deprecated'):
        result = runner.invoke(app, ['upload-dataset', '-i', str(dp_file), '-n', 'My Set'])

    assert result.exit_code == 0, result.output
    assert ds.created == {'display_name': 'My Set', 'path': 'Default'}
    assert ds.appended is not None and ds.appended[0] == 'ds_legacy'


def test_upload_dataset_on_a_current_sdk_does_not_warn(
    tmp_path: Path, fake_client: _FakeDatasets, recwarn: pytest.WarningsRecorder
) -> None:
    dp_file = tmp_path / 'dp.jsonl'
    _write_raw(dp_file)
    result = runner.invoke(app, ['upload-dataset', '-i', str(dp_file), '-n', 'My Set'])
    assert result.exit_code == 0, result.output
    assert not [w for w in recwarn if issubclass(w.category, FutureWarning)]


def test_installed_sdk_exposes_the_signature_the_upload_reads() -> None:
    """The fakes above stand in for the SDK; this pins that the real method's signature is readable, not hidden behind a wrapper."""
    orq_ai_sdk = pytest.importorskip('orq_ai_sdk')
    datasets = orq_ai_sdk.Orq(api_key='k').datasets
    parameters = inspect.signature(datasets.create).parameters
    assert ('display_name' in parameters) != ('request' in parameters)
