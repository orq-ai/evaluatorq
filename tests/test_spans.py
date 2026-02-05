"""Tests for set_evaluation_attributes in tracing/spans."""

import json
from unittest.mock import MagicMock

import pytest

from evaluatorq.tracing.spans import set_evaluation_attributes


@pytest.fixture()
def mock_span():
    """Create a mock span that tracks set_attribute calls."""
    attributes: dict[str, object] = {}
    span = MagicMock()

    def _set_attribute(key: str, value: object):
        attributes[key] = value

    span.set_attribute = MagicMock(side_effect=_set_attribute)
    span._attributes = attributes
    return span


class TestSetEvaluationAttributes:
    """Mirrors TS setEvaluationAttributes tests."""

    def test_sets_number_score_as_string(self, mock_span: MagicMock):
        set_evaluation_attributes(mock_span, 0.85, "good score", True)

        assert mock_span._attributes["orq.score"] == "0.85"
        assert mock_span._attributes["orq.explanation"] == "good score"
        assert mock_span._attributes["orq.pass"] is True

    def test_sets_boolean_score_as_string(self, mock_span: MagicMock):
        set_evaluation_attributes(mock_span, True)

        assert mock_span._attributes["orq.score"] == "True"

    def test_sets_string_score_directly(self, mock_span: MagicMock):
        set_evaluation_attributes(mock_span, "excellent")

        assert mock_span._attributes["orq.score"] == "excellent"

    def test_json_serializes_dict_score(self, mock_span: MagicMock):
        cell = {
            "type": "bert_score",
            "value": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
        }
        set_evaluation_attributes(mock_span, cell)

        assert mock_span._attributes["orq.score"] == json.dumps(cell)

    def test_does_not_set_optional_attributes_when_none(self, mock_span: MagicMock):
        set_evaluation_attributes(mock_span, 1.0)

        assert mock_span.set_attribute.call_count == 1
        assert "orq.explanation" not in mock_span._attributes
        assert "orq.pass" not in mock_span._attributes

    def test_handles_none_span_gracefully(self):
        # Should not throw
        set_evaluation_attributes(None, 1.0, "test", True)
