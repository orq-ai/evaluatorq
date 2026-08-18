"""Tests for set_evaluation_attributes in tracing/spans."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from opentelemetry.trace import StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from evaluatorq.common.tracing import set_span_error
from evaluatorq.tracing import spans as spans_module
from evaluatorq.tracing.spans import (
    EvaluationSpanOptions,
    JobSpanOptions,
    RunSpanOptions,
    set_evaluation_attributes,
    with_evaluation_span,
    with_job_span,
    with_run_span,
)


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
        set_evaluation_attributes(mock_span, 0.85, explanation="good score", pass_=True)

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
        set_evaluation_attributes(None, 1.0, explanation="test", pass_=True)


class TestSpanBodyImportError:
    """An ImportError from the body must reach the caller, not be swallowed.

    The OTel imports are guarded so tracing degrades when opentelemetry is
    absent; that guard used to wrap the ``yield`` too, so a job lazily importing
    a missing optional extra was caught and the generator resumed, surfacing as
    ``RuntimeError: generator didn't stop after athrow()``.
    """

    @pytest.fixture()
    def _stub_tracer(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(spans_module, "get_tracer", lambda: MagicMock())

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_stub_tracer")
    async def test_job_span_propagates_body_import_error(self):
        with pytest.raises(ImportError, match="marker"):
            async with with_job_span(JobSpanOptions(run_id="abc", row_index=0)):
                raise ImportError("marker")

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_stub_tracer")
    async def test_evaluation_span_propagates_body_import_error(self):
        options = EvaluationSpanOptions(run_id="abc", evaluator_name="judge")
        with pytest.raises(ImportError, match="marker"):
            async with with_evaluation_span(options):
                raise ImportError("marker")


@pytest.mark.asyncio
async def test_job_span_preserves_deliberate_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer('test')
    monkeypatch.setattr(spans_module, 'get_tracer', lambda: tracer)
    monkeypatch.setattr('evaluatorq.common.tracing.get_tracer', lambda: tracer)

    async with with_job_span(JobSpanOptions(run_id='abc', row_index=0)) as span:
        set_span_error(span, 'judge failed but was handled')

    provider.shutdown()
    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    assert finished[0].status.status_code is StatusCode.ERROR


class TestSpanCancellation:
    @pytest.fixture()
    def _stub_tracer(self, monkeypatch: pytest.MonkeyPatch):
        tracer = MagicMock()
        first_span = MagicMock()
        spans = [first_span]
        started = 0

        def _start_span(*args, **kwargs):
            nonlocal started
            span = first_span if started == 0 else MagicMock()
            if started > 0:
                spans.append(span)
            started += 1
            context_manager = MagicMock()
            context_manager.__enter__.return_value = span
            context_manager.__exit__.return_value = False
            return context_manager

        tracer.start_as_current_span.side_effect = _start_span
        monkeypatch.setattr(spans_module, "get_tracer", lambda: tracer)
        monkeypatch.setattr("evaluatorq.common.tracing.get_tracer", lambda: tracer)
        first_span._all_spans = spans
        return first_span

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_stub_tracer")
    async def test_job_span_marks_cancellation_as_error(self, _stub_tracer: MagicMock):
        with pytest.raises(asyncio.CancelledError):
            async with with_job_span(JobSpanOptions(run_id="abc", row_index=0)):
                raise asyncio.CancelledError

        status = _stub_tracer.set_status.call_args.args[0]
        assert status.status_code is StatusCode.ERROR
        _stub_tracer.set_attribute.assert_called_once_with("error.type", "CancelledError")

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_stub_tracer")
    async def test_nested_cancellation_marks_run_job_and_evaluation_once(self, _stub_tracer: MagicMock):
        with pytest.raises(asyncio.CancelledError):
            async with with_run_span(RunSpanOptions(run_id="abc", run_name="run")):
                async with with_job_span(JobSpanOptions(run_id="abc", row_index=0)):
                    async with with_evaluation_span(EvaluationSpanOptions(run_id="abc", evaluator_name="judge")):
                        raise asyncio.CancelledError

        spans = _stub_tracer._all_spans
        assert len(spans) == 3
        for span in spans:
            assert span.set_status.call_count == 1
            status = span.set_status.call_args.args[0]
            assert status.status_code is StatusCode.ERROR
            span.set_attribute.assert_called_once_with("error.type", "CancelledError")

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_stub_tracer")
    async def test_evaluation_span_marks_cancellation_as_error(self, _stub_tracer: MagicMock):
        options = EvaluationSpanOptions(run_id="abc", evaluator_name="judge")
        with pytest.raises(asyncio.CancelledError):
            async with with_evaluation_span(options):
                raise asyncio.CancelledError

        status = _stub_tracer.set_status.call_args.args[0]
        assert status.status_code is StatusCode.ERROR
        _stub_tracer.set_attribute.assert_called_once_with("error.type", "CancelledError")
