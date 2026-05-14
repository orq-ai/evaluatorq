"""Tests for FunctionCall._serialize_arguments and arguments_dict.

Covers the fallback / failure branches:
- dict args → JSON string
- string args pass through
- non-JSON-serializable values use default=str
- arguments_dict returns {} for malformed JSON
- arguments_dict returns {} when JSON is not an object
"""

from __future__ import annotations

import json
from datetime import datetime

from evaluatorq.openresponses.convert_models import FunctionCall


class TestSerializeArguments:
    def test_dict_args_serialized_to_json_string(self):
        fc = FunctionCall(name="lookup", call_id="c1", arguments={"q": "x", "n": 1})  # pyright: ignore[reportArgumentType]
        assert json.loads(fc.arguments) == {"q": "x", "n": 1}

    def test_string_args_pass_through_unchanged(self):
        fc = FunctionCall(name="lookup", call_id="c1", arguments='{"q": "raw"}')
        assert fc.arguments == '{"q": "raw"}'

    def test_non_serializable_uses_default_str(self):
        dt = datetime(2024, 1, 1, 12, 0, 0)
        fc = FunctionCall(name="lookup", call_id="c1", arguments={"when": dt})  # pyright: ignore[reportArgumentType]
        parsed = json.loads(fc.arguments)
        assert "when" in parsed
        assert "2024-01-01" in parsed["when"]


class TestArgumentsDict:
    def test_valid_object_returns_dict(self):
        fc = FunctionCall(name="lookup", call_id="c1", arguments='{"q": "x"}')
        assert fc.arguments_dict == {"q": "x"}

    def test_malformed_json_returns_empty_dict(self):
        fc = FunctionCall(name="lookup", call_id="c1", arguments="{not json")
        assert fc.arguments_dict == {}

    def test_array_json_returns_empty_dict(self):
        fc = FunctionCall(name="lookup", call_id="c1", arguments="[1, 2, 3]")
        assert fc.arguments_dict == {}

    def test_null_json_returns_empty_dict(self):
        fc = FunctionCall(name="lookup", call_id="c1", arguments="null")
        assert fc.arguments_dict == {}

    def test_scalar_json_returns_empty_dict(self):
        fc = FunctionCall(name="lookup", call_id="c1", arguments="42")
        assert fc.arguments_dict == {}
