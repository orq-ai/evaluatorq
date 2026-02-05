"""Tests for calculate_evaluator_averages in table_display."""

from collections.abc import Callable
from typing import Any

import pytest

from evaluatorq.table_display import calculate_evaluator_averages
from evaluatorq.types import (
    DataPoint,
    DataPointResult,
    EvaluationResult,
    EvaluationResultCell,
    EvaluatorScore,
    JobResult,
)


@pytest.fixture()
def make_result():
    """Factory fixture that creates a DataPointResult for a given job and scores."""

    def _make(
        job_name: str,
        scores: list[dict[str, Any]],
    ) -> DataPointResult:
        evaluator_scores = [
            EvaluatorScore(
                evaluator_name=s["evaluator_name"],
                score=EvaluationResult(
                    value=s["value"],
                    explanation=s.get("explanation"),
                    pass_=s.get("pass_"),
                ),
                error=s.get("error"),
            )
            for s in scores
        ]

        return DataPointResult(
            data_point=DataPoint(inputs={"text": "test"}),
            job_results=[
                JobResult(
                    job_name=job_name,
                    output="output",
                    evaluator_scores=evaluator_scores,
                ),
            ],
        )

    return _make


class TestCalculateEvaluatorAverages:
    """Mirrors TS calculateEvaluatorAverages tests."""

    def test_calculates_average_for_number_scores(self, make_result: Callable[..., DataPointResult]):
        results = [
            make_result("job1", [{"evaluator_name": "accuracy", "value": 0.5}]),
            make_result("job1", [{"evaluator_name": "accuracy", "value": 1.0}]),
        ]

        data = calculate_evaluator_averages(results)
        display_value, _ = data["averages"]["accuracy"]["job1"]
        assert display_value == "0.75"

    def test_calculates_pass_rate_for_boolean_scores(self, make_result: Callable[..., DataPointResult]):
        results = [
            make_result("job1", [{"evaluator_name": "pass_check", "value": True}]),
            make_result("job1", [{"evaluator_name": "pass_check", "value": False}]),
        ]

        data = calculate_evaluator_averages(results)
        display_value, _ = data["averages"]["pass_check"]["job1"]
        assert display_value == "50.0%"

    def test_renders_string_scores_as_string(self, make_result: Callable[..., DataPointResult]):
        results = [
            make_result("job1", [{"evaluator_name": "quality", "value": "good"}]),
        ]

        data = calculate_evaluator_averages(results)
        display_value, _ = data["averages"]["quality"]["job1"]
        assert display_value == "[string]"

    def test_renders_evaluation_result_cell_as_structured(self, make_result: Callable[..., DataPointResult]):
        cell = EvaluationResultCell(
            type="bert_score",
            value={"precision": 0.9, "recall": 0.8, "f1": 0.85},
        )
        results = [
            make_result("job1", [{"evaluator_name": "bert_score", "value": cell}]),
        ]

        data = calculate_evaluator_averages(results)
        display_value, style = data["averages"]["bert_score"]["job1"]
        assert display_value == "[structured]"
        assert style == "dim"

    def test_handles_mixed_evaluator_types(self, make_result: Callable[..., DataPointResult]):
        cell_1 = EvaluationResultCell(type="bert_score", value={"f1": 0.9})
        cell_2 = EvaluationResultCell(type="bert_score", value={"f1": 0.7})

        results = [
            make_result(
                "job1",
                [
                    {"evaluator_name": "accuracy", "value": 0.8},
                    {"evaluator_name": "pass_check", "value": True},
                    {"evaluator_name": "quality", "value": "excellent"},
                    {"evaluator_name": "bert", "value": cell_1},
                ],
            ),
            make_result(
                "job1",
                [
                    {"evaluator_name": "accuracy", "value": 0.6},
                    {"evaluator_name": "pass_check", "value": False},
                    {"evaluator_name": "quality", "value": "good"},
                    {"evaluator_name": "bert", "value": cell_2},
                ],
            ),
        ]

        data = calculate_evaluator_averages(results)
        evaluator_names = data["evaluator_names"]
        averages = data["averages"]

        assert "accuracy" in evaluator_names
        assert "pass_check" in evaluator_names
        assert "quality" in evaluator_names
        assert "bert" in evaluator_names

        assert averages["accuracy"]["job1"][0] == "0.70"
        assert averages["pass_check"]["job1"][0] == "50.0%"
        assert averages["quality"]["job1"][0] == "[string]"
        assert averages["bert"]["job1"][0] == "[structured]"

    def test_handles_empty_results(self):
        data = calculate_evaluator_averages([])
        assert data["job_names"] == []
        assert data["evaluator_names"] == []
        assert data["averages"] == {}

    def test_shows_dash_for_evaluator_with_errors(self, make_result: Callable[..., DataPointResult]):
        results = [
            make_result(
                "job1",
                [
                    {
                        "evaluator_name": "failing",
                        "value": 0,
                        "error": "evaluator failed",
                    },
                ],
            ),
        ]

        data = calculate_evaluator_averages(results)
        display_value, _ = data["averages"]["failing"]["job1"]
        assert display_value == "-"

    def test_100_percent_boolean_pass_rate(self, make_result: Callable[..., DataPointResult]):
        results = [
            make_result("job1", [{"evaluator_name": "check", "value": True}]),
            make_result("job1", [{"evaluator_name": "check", "value": True}]),
        ]

        data = calculate_evaluator_averages(results)
        display_value, style = data["averages"]["check"]["job1"]
        assert display_value == "100.0%"
        assert style == "green"
