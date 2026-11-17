"""`red_team(dataset=)` accepts a `StaticDataset` that is already in memory.

`load_openresponses_dataset()` returns one, and before this nothing public took
it: a dataset written in the OpenResponses shape could be loaded and not run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from evaluatorq.redteam import load_openresponses_dataset, red_team
from evaluatorq.redteam.contracts import RedTeamInput, Severity, StaticDataset, TurnType, VulnerabilityDomain
from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import is_huggingface_source, load_owasp_agentic_dataset
from evaluatorq.redteam.runner import _dataset_label

_FIXTURE = Path(__file__).parent / 'fixtures' / 'static_e2e_dataset.json'


def _fixture_dataset() -> StaticDataset:
    return StaticDataset.model_validate(json.loads(_FIXTURE.read_text(encoding='utf-8')))


def test_an_in_memory_dataset_gives_the_same_datapoints_as_its_file() -> None:
    from_file = load_owasp_agentic_dataset(dataset=_FIXTURE)
    from_memory = load_owasp_agentic_dataset(dataset=_fixture_dataset())

    assert len(from_memory) == 3
    assert [dp.inputs for dp in from_memory] == [dp.inputs for dp in from_file]


def test_filters_and_the_sample_cap_apply_to_an_in_memory_dataset() -> None:
    dataset = _fixture_dataset()
    categories = sorted({s.input.category for s in dataset.samples})

    assert len(load_owasp_agentic_dataset(dataset=dataset, delivery_methods=['direct-request'])) == 3
    assert load_owasp_agentic_dataset(dataset=dataset, delivery_methods=['base64']) == []
    assert len(load_owasp_agentic_dataset(dataset=dataset, num_samples=1)) == 1
    one_category = load_owasp_agentic_dataset(dataset=dataset, categories=[categories[0]])
    assert one_category
    assert {dp.inputs['category'] for dp in one_category} == {categories[0]}
    assert len(dataset.samples) == 3, 'the dataset passed in is not modified'


def test_a_dataset_written_in_the_openresponses_shape_can_be_loaded_and_turned_into_datapoints(tmp_path: Path) -> None:
    meta = RedTeamInput(
        id='or-1',
        vulnerability='prompt_injection',
        severity=Severity.HIGH,
        vulnerability_domain=VulnerabilityDomain.MODEL,
        turn_type=TurnType.MULTI,
        source='unit-test',
    ).model_dump(mode='json')
    rows = [
        {
            'input': meta,
            'openresponses_input': [
                {'role': 'user', 'content': 'first attack turn'},
                {'role': 'assistant', 'content': 'a refusal'},
                {'role': 'user', 'content': [{'type': 'input_text', 'text': 'second attack turn'}]},
            ],
        }
    ]
    path = tmp_path / 'attacks.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')

    datapoints = load_owasp_agentic_dataset(dataset=load_openresponses_dataset(path))

    assert len(datapoints) == 1
    inputs = datapoints[0].inputs
    assert inputs['id'] == 'or-1'
    assert [(m['role'], m['content']) for m in inputs['messages']] == [
        ('user', 'first attack turn'),
        ('assistant', 'a refusal'),
        ('user', 'second attack turn'),
    ]


def test_an_in_memory_dataset_never_needs_huggingface() -> None:
    assert is_huggingface_source(_fixture_dataset()) is False
    assert is_huggingface_source(None) is True
    assert is_huggingface_source('hf:org/repo') is True
    assert is_huggingface_source(Path('local.json')) is False


def test_run_metadata_names_an_in_memory_dataset_without_dumping_it() -> None:
    assert _dataset_label(_fixture_dataset()) == 'StaticDataset (3 samples, in memory)'
    assert _dataset_label('hf:org/repo') == 'hf:org/repo'
    assert _dataset_label(Path('data/local.json')) == 'data/local.json'
    assert _dataset_label(None) is None
    assert _dataset_label('') is None


@pytest.mark.asyncio
async def test_red_team_passes_an_in_memory_dataset_to_the_static_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.redteam.test_runner import _make_report, _run_result

    dataset = _fixture_dataset()
    # With huggingface_hub "missing", a HuggingFace preflight would raise: an in-memory dataset must not trigger it.
    monkeypatch.setitem(sys.modules, 'huggingface_hub', None)
    with patch(
        'evaluatorq.redteam.runner._run_static', new_callable=AsyncMock, return_value=_run_result(_make_report())
    ) as run_static:
        await red_team('agent:test', mode='static', dataset=dataset)

    run_static.assert_awaited_once()
    call = run_static.await_args
    assert call is not None
    passed = [*call.args, *call.kwargs.values()]
    assert any(getattr(arg, 'dataset', None) is dataset or arg is dataset for arg in passed), (
        'the dataset object reaches the static run as it was given'
    )
