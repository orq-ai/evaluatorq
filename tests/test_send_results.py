"""Tests for send_results serialization behavior."""

import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from evaluatorq.send_results import SendResultsError, send_results_to_orq
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
        score_value: float | bool | str | EvaluationResultCell | dict[str, Any],
        error: str | None = None,
        output: Any = "result",
    ) -> list[DataPointResult]:
        return [
            DataPointResult(
                data_point=DataPoint(inputs={"text": "hello"}),
                job_results=[
                    JobResult(
                        job_name="job1",
                        output=output,
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

    def test_stringifies_arbitrary_object_score_values(self, build_results: Callable[..., list[DataPointResult]]):
        payload = serialize(build_results({"reason": "too long", "tokens": 120}))
        score = extract_score(payload)
        assert score["value"] == '{"reason": "too long", "tokens": 120}'

    def test_stringifies_object_job_outputs(self, build_results: Callable[..., list[DataPointResult]]):
        payload = serialize(
            build_results(
                0.9,
                output={"answer": "hello", "confidence": 0.9},
            )
        )
        output = payload["results"][0]["jobResults"][0]["output"]
        assert output == '{"answer": "hello", "confidence": 0.9}'

    def test_keeps_evaluation_result_cell_score_value_unchanged_in_serialization(
        self, build_results: Callable[..., list[DataPointResult]]
    ):
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

    def test_stringifies_dict_output_with_nested_pydantic_model(self, build_results: Callable[..., list[DataPointResult]]):
        from evaluatorq.redteam.contracts import TokenUsage

        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, calls=1)
        payload = serialize(
            build_results(0.9, output={"response": "hello", "token_usage": usage})
        )
        output = payload["results"][0]["jobResults"][0]["output"]
        assert isinstance(output, str)
        parsed = json.loads(output)
        assert parsed["response"] == "hello"
        assert parsed["token_usage"]["prompt_tokens"] == 10
        assert parsed["token_usage"]["total_tokens"] == 15

    def test_preserves_response_resource_output_as_is(self, build_results: Callable[..., list[DataPointResult]]):
        response_resource = {"object": "response", "id": "resp_123", "model": "gpt-4"}
        payload = serialize(build_results(0.9, output=response_resource))
        output = payload["results"][0]["jobResults"][0]["output"]
        assert output == response_resource

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
                    es["score"].pop("token_usage", None)
                    es["score"].pop("raw_output", None)
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

    def test_evaluator_token_usage_and_raw_output_stripped_from_upload(self):
        # Evaluator-cost metadata is kept in local result dumps but must NOT be
        # uploaded to the platform — the send-boundary fixup strips it.
        from evaluatorq.contracts import TokenUsage

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
                                score=EvaluationResult(
                                    value=True,
                                    explanation="resistant",
                                    token_usage=TokenUsage(
                                        prompt_tokens=4, completion_tokens=2, total_tokens=6, calls=1
                                    ),
                                    raw_output={"raw_content": "{}"},
                                ),
                            ),
                        ],
                    ),
                ],
            ),
        ]
        # Plain serialize keeps them (local dumps like 02_attack_results.json).
        assert "token_usage" in extract_score(serialize(results))
        # The upload fixup strips them.
        score = extract_score(serialize_with_fixup(results))
        assert "token_usage" not in score
        assert "raw_output" not in score

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


class TestSendResultsUploadFailures:
    @pytest.mark.asyncio
    async def test_non_success_response_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        build_results: Callable[..., list[DataPointResult]],
    ):
        async def fake_post(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> httpx.Response:
            request = httpx.Request("POST", "https://my.orq.ai/v2/spreadsheets/evaluations/receive")
            return httpx.Response(
                401,
                request=request,
                json={"error": "unauthorized"},
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        with pytest.raises(SendResultsError, match="401 Unauthorized"):
            await send_results_to_orq(
                api_key="bad-key",
                evaluation_name="eval",
                evaluation_description=None,
                dataset_id=None,
                results=build_results(0.5),
                start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
                raise_on_error=True,
            )

    @pytest.mark.asyncio
    async def test_network_error_is_not_swallowed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        build_results: Callable[..., list[DataPointResult]],
    ):
        async def fake_post(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> httpx.Response:
            request = httpx.Request("POST", "https://my.orq.ai/v2/spreadsheets/evaluations/receive")
            raise httpx.RequestError("connection failed", request=request)

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        with pytest.raises(httpx.RequestError, match="connection failed"):
            await send_results_to_orq(
                api_key="key",
                evaluation_name="eval",
                evaluation_description=None,
                dataset_id=None,
                results=build_results(0.5),
                start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
                raise_on_error=True,
            )

    @pytest.mark.asyncio
    async def test_non_success_response_returns_none_when_raise_on_error_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
        build_results: Callable[..., list[DataPointResult]],
    ):
        async def fake_post(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> httpx.Response:
            request = httpx.Request("POST", "https://my.orq.ai/v2/spreadsheets/evaluations/receive")
            return httpx.Response(503, request=request, json={"error": "service unavailable"})

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        result = await send_results_to_orq(
            api_key="key",
            evaluation_name="eval",
            evaluation_description=None,
            dataset_id=None,
            results=build_results(0.5),
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            raise_on_error=False,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_evaluator_metadata_absent_from_posted_payload(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # End-to-end guard on the real upload boundary: token_usage/raw_output must
        # never reach the platform. Exercises the prod strip in send_results_to_orq
        # (not the test-local fixup), so deleting those .pop()s fails this test.
        from evaluatorq.contracts import TokenUsage

        captured: dict[str, Any] = {}

        async def fake_post(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> httpx.Response:
            captured["json"] = kwargs["json"]
            request = httpx.Request("POST", "https://my.orq.ai/v2/spreadsheets/evaluations/receive")
            return httpx.Response(
                200,
                request=request,
                json={
                    "sheet_id": "s1",
                    "manifest_id": "m1",
                    "experiment_name": "eval",
                    "rows_created": 1,
                },
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

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
                                score=EvaluationResult(
                                    value=True,
                                    explanation="resistant",
                                    token_usage=TokenUsage(
                                        prompt_tokens=4, completion_tokens=2, total_tokens=6, calls=1
                                    ),
                                    raw_output={"raw_content": "{}"},
                                ),
                            ),
                        ],
                    ),
                ],
            ),
        ]

        await send_results_to_orq(
            api_key="key",
            evaluation_name="eval",
            evaluation_description=None,
            dataset_id=None,
            results=results,
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            raise_on_error=True,
        )

        score = extract_score(captured["json"])
        assert "token_usage" not in score
        assert "raw_output" not in score
        # Required fields survive the strip.
        assert score["explanation"] == "resistant"
        assert captured["json"]["results"][0]["jobResults"][0]["output"] == "result"

    @pytest.mark.asyncio
    async def test_network_error_returns_none_when_raise_on_error_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
        build_results: Callable[..., list[DataPointResult]],
    ):
        async def fake_post(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> httpx.Response:
            request = httpx.Request("POST", "https://my.orq.ai/v2/spreadsheets/evaluations/receive")
            raise httpx.RequestError("connection failed", request=request)

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        result = await send_results_to_orq(
            api_key="key",
            evaluation_name="eval",
            evaluation_description=None,
            dataset_id=None,
            results=build_results(0.5),
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            raise_on_error=False,
        )
        assert result is None


class TestSendResultsBaseUrl:
    """RES-912: results upload to the same server used for inference."""

    @staticmethod
    def _ok_post(captured: dict[str, Any]):
        async def fake_post(self: httpx.AsyncClient, url: str, *args: Any, **kwargs: Any) -> httpx.Response:
            captured["url"] = url
            request = httpx.Request("POST", url)
            return httpx.Response(
                200,
                request=request,
                json={"sheet_id": "s1", "manifest_id": "m1", "experiment_name": "eval", "rows_created": 1},
            )

        return fake_post

    @pytest.mark.asyncio
    async def test_base_url_kwarg_overrides_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
        build_results: Callable[..., list[DataPointResult]],
    ):
        monkeypatch.setenv("ORQ_BASE_URL", "https://my.orq.ai")
        captured: dict[str, Any] = {}
        monkeypatch.setattr(httpx.AsyncClient, "post", self._ok_post(captured))

        await send_results_to_orq(
            api_key="key",
            evaluation_name="eval",
            evaluation_description=None,
            dataset_id=None,
            results=build_results(0.5),
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            base_url="https://my.staging.orq.ai",
        )
        assert captured["url"] == "https://my.staging.orq.ai/v2/spreadsheets/evaluations/receive"

    @pytest.mark.asyncio
    async def test_base_url_trailing_slash_normalized(
        self,
        monkeypatch: pytest.MonkeyPatch,
        build_results: Callable[..., list[DataPointResult]],
    ):
        captured: dict[str, Any] = {}
        monkeypatch.setattr(httpx.AsyncClient, "post", self._ok_post(captured))

        await send_results_to_orq(
            api_key="key",
            evaluation_name="eval",
            evaluation_description=None,
            dataset_id=None,
            results=build_results(0.5),
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            base_url="https://my.staging.orq.ai/",
        )
        assert captured["url"] == "https://my.staging.orq.ai/v2/spreadsheets/evaluations/receive"

    @pytest.mark.asyncio
    async def test_falls_back_to_env_when_base_url_not_supplied(
        self,
        monkeypatch: pytest.MonkeyPatch,
        build_results: Callable[..., list[DataPointResult]],
    ):
        monkeypatch.setenv("ORQ_BASE_URL", "https://my.staging.orq.ai")
        captured: dict[str, Any] = {}
        monkeypatch.setattr(httpx.AsyncClient, "post", self._ok_post(captured))

        await send_results_to_orq(
            api_key="key",
            evaluation_name="eval",
            evaluation_description=None,
            dataset_id=None,
            results=build_results(0.5),
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        )
        assert captured["url"] == "https://my.staging.orq.ai/v2/spreadsheets/evaluations/receive"
