"""Tests for send_results serialization behavior."""

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel

from evaluatorq.types import (
    DataPoint,
    DataPointResult,
    EvaluationResult,
    EvaluationResultCell,
    EvaluatorScore,
    JobResult,
)


@pytest.fixture()
def build_results():
    """Factory fixture that creates a single DataPointResult with given score and error."""

    def _build(
        score_value: float | bool | str | EvaluationResultCell,
        error: str | None = None,
    ) -> list[DataPointResult]:
        return [
            DataPointResult(
                data_point=DataPoint(inputs={"text": "hello"}),
                job_results=[
                    JobResult(
                        job_name="job1",
                        output="result",
                        evaluator_scores=[
                            EvaluatorScore(
                                evaluator_name="eval1",
                                score=EvaluationResult(
                                    value=score_value, explanation="test"
                                ),
                                error=error,
                            ),
                        ],
                    ),
                ],
            ),
        ]

    return _build


def serialize(results: list[DataPointResult]) -> dict[str, Any]:
    """Serialize results the same way send_results does via Pydantic."""

    class Payload(BaseModel):
        results: list[DataPointResult]

    return Payload(results=results).model_dump(mode="json", by_alias=True)


def extract_score(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the first evaluator score from a serialized payload."""
    return payload["results"][0]["jobResults"][0]["evaluatorScores"][0]["score"]


class TestSendResultsSerialization:
    """Mirrors TS sendResultsToOrqEffect serialization tests."""

    def test_serializes_number_score_value_as_is(self, build_results: Callable[..., list[DataPointResult]]):
        payload = serialize(build_results(0.85))
        score = extract_score(payload)
        assert score["value"] == 0.85

    def test_serializes_boolean_score_value_as_is(self, build_results: Callable[..., list[DataPointResult]]):
        payload = serialize(build_results(True))
        score = extract_score(payload)
        assert score["value"] is True

    def test_serializes_string_score_value_as_is(self, build_results: Callable[..., list[DataPointResult]]):
        payload = serialize(build_results("good"))
        score = extract_score(payload)
        assert score["value"] == "good"

    def test_serializes_evaluation_result_cell_correctly(self, build_results: Callable[..., list[DataPointResult]]):
        cell = EvaluationResultCell(
            type="bert_score",
            value={"precision": 0.9, "recall": 0.8, "f1": 0.85},
        )
        payload = serialize(build_results(cell))
        score = extract_score(payload)
        assert score["value"] == {
            "type": "bert_score",
            "value": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
        }

    def test_serializes_error_strings(self, build_results: Callable[..., list[DataPointResult]]):
        payload = serialize(build_results(0.5, error="eval failed"))
        eval_score = payload["results"][0]["jobResults"][0]["evaluatorScores"][0]
        assert eval_score["error"] == "eval failed"
