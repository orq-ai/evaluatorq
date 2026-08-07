"""Unit tests for the shared `_as_obj` helper in `_datapoint_io.py`."""

from __future__ import annotations

import json

from evaluatorq.simulation import _datapoint_io
from evaluatorq.simulation._datapoint_io import _as_obj, _extract_single_datapoint
from evaluatorq.simulation.utils import dataset_export
from evaluatorq.types import DataPoint


def test_as_obj_parses_valid_json_object_string() -> None:
    result = _as_obj('{"name": "Priya", "age": 30}')
    assert result == {"name": "Priya", "age": 30}


def test_as_obj_parses_valid_json_array_string() -> None:
    result = _as_obj('["a", "b", "c"]')
    assert result == ["a", "b", "c"]


def test_as_obj_returns_original_string_on_malformed_json() -> None:
    malformed = '{"name": "P"'
    result = _as_obj(malformed)
    assert result == malformed
    assert result is malformed


def test_as_obj_returns_plain_non_json_string_unchanged() -> None:
    plain = "hello"
    result = _as_obj(plain)
    assert result == plain
    assert result is plain


def test_as_obj_passes_through_dict_unchanged() -> None:
    value = {"already": "a dict"}
    result = _as_obj(value)
    assert result is value


def test_as_obj_passes_through_none_and_int_unchanged() -> None:
    assert _as_obj(None) is None
    assert _as_obj(42) == 42


def test_dataset_export_as_obj_is_shared_implementation() -> None:
    assert dataset_export._as_obj is _datapoint_io._as_obj


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
