"""Unit tests for the shared `parse_json_or_value` helper the datapoint readers share."""

from __future__ import annotations

import json
from datetime import datetime

from evaluatorq.contracts import TokenUsage
from evaluatorq.simulation import _datapoint_io
from evaluatorq.common.messages import parse_json_or_value
from evaluatorq.simulation._datapoint_io import _extract_single_datapoint
from evaluatorq.simulation.contracts import SimulationResult, TerminatedBy
from evaluatorq.simulation.utils import dataset_export
from evaluatorq.types import DataPoint


def test_parse_json_or_value_parses_valid_json_object_string() -> None:
    result = parse_json_or_value('{"name": "Priya", "age": 30}')
    assert result == {"name": "Priya", "age": 30}


def test_parse_json_or_value_parses_valid_json_array_string() -> None:
    result = parse_json_or_value('["a", "b", "c"]')
    assert result == ["a", "b", "c"]


def test_parse_json_or_value_returns_original_string_on_malformed_json() -> None:
    malformed = '{"name": "P"'
    result = parse_json_or_value(malformed)
    assert result == malformed
    assert result is malformed


def test_parse_json_or_value_returns_plain_non_json_string_unchanged() -> None:
    plain = "hello"
    result = parse_json_or_value(plain)
    assert result == plain
    assert result is plain


def test_parse_json_or_value_passes_through_dict_unchanged() -> None:
    value = {"already": "a dict"}
    result = parse_json_or_value(value)
    assert result is value


def test_parse_json_or_value_passes_through_none_and_int_unchanged() -> None:
    assert parse_json_or_value(None) is None
    assert parse_json_or_value(42) == 42


def test_dataset_export_uses_the_shared_parser() -> None:
    assert dataset_export.parse_json_or_value is parse_json_or_value
    assert _datapoint_io.parse_json_or_value is parse_json_or_value


def test_parse_jsonl_returns_dicts_for_non_pydantic_class() -> None:
    content = '{"name": "Priya"}\n{"name": "Sam"}\n'

    assert dataset_export.parse_jsonl(content, cls=dict) == [
        {'name': 'Priya'},
        {'name': 'Sam'},
    ]


def test_parse_jsonl_validates_pydantic_models() -> None:
    content = '{"input_tokens": 3, "output_tokens": 5}\n'

    assert dataset_export.parse_jsonl(content, cls=TokenUsage) == [
        TokenUsage(input_tokens=3, output_tokens=5)
    ]


_DP = {
    "id": "dp1",
    "persona": {
        "name": "Priya",
        "patience": 0.5,
        "assertiveness": 0.5,
        "politeness": 0.5,
        "technical_level": 0.5,
        "communication_style": "casual",
        "background": "customer",
    },
    "scenario": {"name": "refund", "goal": "get a refund"},
    "user_system_prompt": "",
    "first_message": "hi",
}


def test_extract_accepts_json_string_datapoint() -> None:
    # The Orq API can hand back nested inputs as JSON strings; the datapoint
    # field must get the same coercion persona/scenario already have.
    dp = _extract_single_datapoint(DataPoint(inputs={"datapoint": json.dumps(_DP)}))
    assert dp.persona.name == "Priya"
    assert dp.scenario.goal == "get a refund"


def test_extract_accepts_json_string_datapoints_list() -> None:
    dp = _extract_single_datapoint(DataPoint(inputs={"datapoints": json.dumps([_DP])}))
    assert dp.persona.name == "Priya"


def test_extract_accepts_json_string_element_in_datapoints() -> None:
    dp = _extract_single_datapoint(DataPoint(inputs={"datapoints": [json.dumps(_DP)]}))
    assert dp.persona.name == "Priya"


def test_results_to_jsonl_preserves_evaluator_details() -> None:
    datapoint = _extract_single_datapoint(DataPoint(inputs={"datapoint": json.dumps(_DP)}))
    result = SimulationResult(
        messages=[],
        terminated_by=TerminatedBy.max_turns,
        reason='',
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        token_usage=TokenUsage(),
        turn_metrics=[],
        evaluator_details={'criteria_met': {'criteria_verified': True}},
    )

    exported = dataset_export.results_to_jsonl([{'datapoint': datapoint, 'result': result}])

    assert json.loads(exported)['evaluator_details'] == {'criteria_met': {'criteria_verified': True}}


def test_results_to_jsonl_stringifies_non_json_evaluator_detail() -> None:
    datapoint = _extract_single_datapoint(DataPoint(inputs={'datapoint': json.dumps(_DP)}))
    result = SimulationResult(
        messages=[],
        terminated_by=TerminatedBy.max_turns,
        reason='',
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        token_usage=TokenUsage(),
        turn_metrics=[],
        evaluator_details={'custom': {'opaque': datetime(2026, 9, 7, 12, 0, 0)}},
    )

    exported = dataset_export.results_to_jsonl([{'datapoint': datapoint, 'result': result}])

    assert json.loads(exported)['evaluator_details'] == {'custom': {'opaque': '2026-09-07 12:00:00'}}
