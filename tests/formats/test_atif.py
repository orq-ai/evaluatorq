"""ATIF 1.7/1.8 models: version dispatch, Harbor validators, fixtures."""

# ruff: noqa: S101

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import ValidationError

from evaluatorq.formats.atif import AtifTrajectory

FIXTURES = Path(__file__).parent / 'fixtures' / 'atif'


def _traj(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        'schema_version': 'ATIF-v1.7',
        'session_id': 's1',
        'agent': {'name': 'a', 'version': '1'},
        'steps': [{'step_id': 1, 'source': 'user', 'message': 'hi'}],
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize('name', sorted(p.name for p in FIXTURES.glob('*.json')))
def test_fixtures_load_and_reserialise(name: str) -> None:
    raw = (FIXTURES / name).read_text()
    traj = AtifTrajectory.from_json(raw)
    again = AtifTrajectory.from_json(traj.to_json())
    assert again == traj
    assert json.loads(traj.to_json())['schema_version'] == json.loads(raw)['schema_version']


def test_embedded_subagents_fixture_has_subagent() -> None:
    traj = AtifTrajectory.from_json((FIXTURES / 'phoenix_v17_embedded_subagents.json').read_text())
    assert traj.subagent_trajectories
    assert traj.steps[1].llm_call_count == 0


@pytest.mark.parametrize('version', ['ATIF-v1.6', 'ATIF-v1.9', 'ATIF-v2'])
def test_unsupported_version_names_the_version(version: str) -> None:
    with pytest.raises(ValidationError, match=version):
        AtifTrajectory.model_validate(_traj(schema_version=version))


@pytest.mark.parametrize('as_text', [True, False])
def test_from_json_requires_schema_version_at_the_root(as_text: bool) -> None:
    data = _traj()
    del data['schema_version']
    with pytest.raises(ValueError, match='schema_version is required'):
        AtifTrajectory.from_json(json.dumps(data) if as_text else data)


def test_from_json_rejects_an_unsupported_version() -> None:
    with pytest.raises(ValidationError, match='ATIF-v1.6'):
        AtifTrajectory.from_json(json.dumps(_traj(schema_version='ATIF-v1.6')).encode())


def test_from_json_rejects_a_non_object_document() -> None:
    with pytest.raises(ValueError, match='JSON object'):
        AtifTrajectory.from_json('[1, 2]')


def test_built_in_code_without_schema_version_defaults_to_1_7() -> None:
    data = _traj()
    del data['schema_version']
    sub = _traj(trajectory_id='sub')
    del sub['schema_version']
    traj = AtifTrajectory.model_validate({**data, 'subagent_trajectories': [sub]})
    assert traj.schema_version == 'ATIF-v1.7'
    assert traj.subagent_trajectories is not None
    assert traj.subagent_trajectories[0].schema_version == 'ATIF-v1.7'


def test_embedded_subagent_without_version_reads_from_json() -> None:
    sub = _traj(trajectory_id='sub')
    del sub['schema_version']
    traj = AtifTrajectory.from_json(json.dumps(_traj(subagent_trajectories=[sub])))
    assert json.loads(traj.to_json())['subagent_trajectories'][0]['schema_version'] == 'ATIF-v1.7'


def test_nanosecond_timestamp_is_accepted() -> None:
    steps = [{'step_id': 1, 'source': 'user', 'message': 'a', 'timestamp': '2026-04-20T10:00:00.123456789Z'}]
    traj = AtifTrajectory.model_validate(_traj(steps=steps))
    assert traj.steps[0].timestamp == '2026-04-20T10:00:00.123456789Z'


def test_step_ids_must_be_sequential() -> None:
    steps = [{'step_id': 1, 'source': 'user', 'message': 'a'}, {'step_id': 3, 'source': 'user', 'message': 'b'}]
    with pytest.raises(ValidationError, match='sequential'):
        AtifTrajectory.model_validate(_traj(steps=steps))


@pytest.mark.parametrize('field', ['model_name', 'reasoning_effort', 'reasoning_content', 'metrics'])
def test_agent_only_fields_rejected_on_user_step(field: str) -> None:
    value: Any = {'prompt_tokens': 1} if field == 'metrics' else 'x'
    steps = [{'step_id': 1, 'source': 'user', 'message': 'a', field: value}]
    with pytest.raises(ValidationError, match='only applicable when source is'):
        AtifTrajectory.model_validate(_traj(steps=steps))


def test_tool_calls_rejected_on_user_step() -> None:
    call = {'tool_call_id': 'c', 'function_name': 'f', 'arguments': {}}
    steps = [{'step_id': 1, 'source': 'user', 'message': 'a', 'tool_calls': [call]}]
    with pytest.raises(ValidationError, match='only applicable'):
        AtifTrajectory.model_validate(_traj(steps=steps))


@pytest.mark.parametrize('field', ['metrics', 'reasoning_content'])
def test_llm_call_count_zero_forbids_llm_fields(field: str) -> None:
    value: Any = {'prompt_tokens': 1} if field == 'metrics' else 'x'
    steps = [{'step_id': 1, 'source': 'agent', 'message': '', 'llm_call_count': 0, field: value}]
    with pytest.raises(ValidationError, match='llm_call_count is 0'):
        AtifTrajectory.model_validate(_traj(steps=steps))


def test_source_call_id_must_match_a_call_in_same_step() -> None:
    step = {
        'step_id': 1, 'source': 'agent', 'message': '',
        'tool_calls': [{'tool_call_id': 'c1', 'function_name': 'f', 'arguments': {}}],
        'observation': {'results': [{'source_call_id': 'zzz', 'content': 'r'}]},
    }
    with pytest.raises(ValidationError, match='zzz'):
        AtifTrajectory.model_validate(_traj(steps=[step]))


def test_embedded_subagent_needs_unique_trajectory_id() -> None:
    sub = _traj()
    with pytest.raises(ValidationError, match='trajectory_id is required'):
        AtifTrajectory.model_validate(_traj(subagent_trajectories=[sub]))
    sub_a = _traj(trajectory_id='t')
    with pytest.raises(ValidationError, match='not unique'):
        AtifTrajectory.model_validate(_traj(subagent_trajectories=[sub_a, sub_a]))


def test_subagent_ref_needs_id_or_path() -> None:
    step = {
        'step_id': 1, 'source': 'agent', 'message': '',
        'observation': {'results': [{'subagent_trajectory_ref': [{'session_id': 'only'}]}]},
    }
    with pytest.raises(ValidationError, match='resolvable'):
        AtifTrajectory.model_validate(_traj(steps=[step]))


@pytest.mark.parametrize(
    'part',
    [
        {'type': 'text'},
        {'type': 'text', 'text': 'a', 'source': {'media_type': 'image/png', 'path': 'p'}},
        {'type': 'image', 'text': 'a', 'source': {'media_type': 'image/png', 'path': 'p'}},
        {'type': 'image'},
        {'type': 'image', 'source': {'media_type': 'audio/wav', 'path': 'p'}},
    ],
)
def test_content_part_rules(part: dict[str, Any]) -> None:
    steps = [{'step_id': 1, 'source': 'user', 'message': [part]}]
    with pytest.raises(ValidationError):
        AtifTrajectory.model_validate(_traj(steps=steps))


def test_bad_timestamp_rejected() -> None:
    steps = [{'step_id': 1, 'source': 'user', 'message': 'a', 'timestamp': 'yesterday'}]
    with pytest.raises(ValidationError, match='ISO 8601'):
        AtifTrajectory.model_validate(_traj(steps=steps))


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        AtifTrajectory.model_validate(_traj(surprise=1))


def _audio_traj() -> AtifTrajectory:
    part = {'type': 'audio', 'source': {'media_type': 'audio/wav', 'path': 'a.wav', 'duration_sec': 1.5}}
    steps = [{'step_id': 1, 'source': 'user', 'message': [part]}]
    return AtifTrajectory.model_validate(_traj(schema_version='ATIF-v1.8', steps=steps))


def test_audio_stamps_1_8_and_forced_1_7_raises() -> None:
    traj = _audio_traj()
    assert json.loads(traj.to_json())['schema_version'] == 'ATIF-v1.8'
    with pytest.raises(ValueError, match='audio'):
        traj.to_json(version='1.7')


def test_plain_trajectory_writes_1_7_and_can_force_1_8() -> None:
    traj = AtifTrajectory.model_validate(_traj())
    assert json.loads(traj.to_json())['schema_version'] == 'ATIF-v1.7'
    forced: Literal['1.8'] = '1.8'
    assert json.loads(traj.to_json(version=forced))['schema_version'] == 'ATIF-v1.8'


def test_audio_in_embedded_subagent_stamps_whole_document_1_8() -> None:
    audio = _audio_traj().model_dump(mode='json', exclude_none=True)
    audio['trajectory_id'] = 'sub'
    doc = json.loads(AtifTrajectory.model_validate(_traj(subagent_trajectories=[audio])).to_json())
    assert doc['schema_version'] == 'ATIF-v1.8'
    assert doc['subagent_trajectories'][0]['schema_version'] == 'ATIF-v1.8'


def test_audio_media_type_alias_normalised() -> None:
    part = {'type': 'audio', 'source': {'media_type': 'audio/mp3', 'path': 'a.mp3'}}
    steps = [{'step_id': 1, 'source': 'user', 'message': [part]}]
    traj = AtifTrajectory.model_validate(_traj(schema_version='ATIF-v1.8', steps=steps))
    assert traj.has_audio()
    assert 'audio/mpeg' in traj.to_json()


def test_v1_8_without_audio_keeps_1_8_by_default() -> None:
    traj = AtifTrajectory.model_validate(_traj(schema_version='ATIF-v1.8'))
    assert not traj.has_audio()
    assert json.loads(traj.to_json())['schema_version'] == 'ATIF-v1.8'


def test_v1_8_without_audio_can_downgrade_to_1_7() -> None:
    traj = AtifTrajectory.model_validate(_traj(schema_version='ATIF-v1.8'))
    assert json.loads(traj.to_json(version='1.7'))['schema_version'] == 'ATIF-v1.7'
