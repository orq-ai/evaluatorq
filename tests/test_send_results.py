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


def serialize_with_fixup(results: list[DataPointResult]) -> dict[str, Any]:
    """Serialize results the same way send_results does, including the post-serialization fixup."""

    class Payload(BaseModel):
        results: list[DataPointResult]

    payload_dict = Payload(results=results).model_dump(mode="json", exclude_none=True, by_alias=True)

    for result in payload_dict.get("results", []):
        for jr in result.get("jobResults") or []:
            jr.setdefault("output", None)
            jr.setdefault("error", "")
            for es in jr.get("evaluatorScores") or []:
                if "score" in es:
                    es["score"].setdefault("explanation", "")
                es.setdefault("error", "")

    return payload_dict


class TestSendResultsDefaults:
    """Tests for the post-serialization fixup that ensures required fields are present."""

    def test_null_output_preserved_after_fixup(self):
        # output=None is stripped by exclude_none; the fixup must restore it as None
        results = [
            DataPointResult(
                data_point=DataPoint(inputs={"text": "hello"}),
                job_results=[
                    JobResult(
                        job_name="job1",
                        output=None,
                        evaluator_scores=[
                            EvaluatorScore(
                                evaluator_name="eval1",
                                score=EvaluationResult(value=0.5),
                            ),
                        ],
                    ),
                ],
            ),
        ]
        payload = serialize_with_fixup(results)
        jr = payload["results"][0]["jobResults"][0]
        assert "output" in jr
        assert jr["output"] is None

    def test_null_error_becomes_empty_string(self):
        # error=None on JobResult is stripped by exclude_none; the fixup must restore it as ""
        results = [
            DataPointResult(
                data_point=DataPoint(inputs={"text": "hello"}),
                job_results=[
                    JobResult(
                        job_name="job1",
                        output="result",
                        error=None,
                        evaluator_scores=[
                            EvaluatorScore(
                                evaluator_name="eval1",
                                score=EvaluationResult(value=0.5),
                            ),
                        ],
                    ),
                ],
            ),
        ]
        payload = serialize_with_fixup(results)
        jr = payload["results"][0]["jobResults"][0]
        assert "error" in jr
        assert jr["error"] == ""

    def test_null_explanation_becomes_empty_string(self):
        # explanation=None on EvaluationResult is stripped by exclude_none; fixup must restore it as ""
        results = [
            DataPointResult(
                data_point=DataPoint(inputs={"text": "hello"}),
                job_results=[
                    JobResult(
                        job_name="job1",
                        output="result",
                        evaluator_scores=[
                            EvaluatorScore(
                                evaluator_name="eval1",
                                score=EvaluationResult(value=0.5, explanation=None),
                            ),
                        ],
                    ),
                ],
            ),
        ]
        payload = serialize_with_fixup(results)
        score = payload["results"][0]["jobResults"][0]["evaluatorScores"][0]["score"]
        assert "explanation" in score
        assert score["explanation"] == ""

    def test_null_evaluator_score_error_becomes_empty_string(self):
        # error=None on EvaluatorScore is stripped by exclude_none; fixup must restore it as ""
        results = [
            DataPointResult(
                data_point=DataPoint(inputs={"text": "hello"}),
                job_results=[
                    JobResult(
                        job_name="job1",
                        output="result",
                        evaluator_scores=[
                            EvaluatorScore(
                                evaluator_name="eval1",
                                score=EvaluationResult(value=0.5),
                                error=None,
                            ),
                        ],
                    ),
                ],
            ),
        ]
        payload = serialize_with_fixup(results)
        es = payload["results"][0]["jobResults"][0]["evaluatorScores"][0]
        assert "error" in es
        assert es["error"] == ""
