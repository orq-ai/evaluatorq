# tests/common/test_tracing.py
"""Tests for evaluatorq.common.tracing unified layer.

Required by RES-899: prove that the unified record_llm_response covers
all attributes emitted by both the former redteam and simulation impls,
for both Chat Completions and Responses API shapes.
"""
from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult


class _Exporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


@pytest.fixture
def span_collector():
    exporter = _Exporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    with patch("evaluatorq.simulation.tracing.get_tracer", return_value=tracer):
        yield exporter, tracer
    provider.shutdown()


def _span(exporter: _Exporter) -> ReadableSpan:
    assert len(exporter.spans) == 1
    return exporter.spans[0]


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


# ---------------------------------------------------------------------------
# truncate_for_span
# ---------------------------------------------------------------------------

def test_truncate_for_span_short_unchanged() -> None:
    from evaluatorq.common.tracing import truncate_for_span
    assert truncate_for_span("hello", max_chars=100) == "hello"


def test_truncate_for_span_long_ends_with_marker() -> None:
    from evaluatorq.common.tracing import _TRUNCATION_MARKER, truncate_for_span
    result = truncate_for_span("x" * 200, max_chars=50)
    assert len(result) == 50
    assert result.endswith(_TRUNCATION_MARKER)


def test_truncate_for_span_zero_disables() -> None:
    from evaluatorq.common.tracing import truncate_for_span
    text = "x" * 1000
    assert truncate_for_span(text, max_chars=0) == text


def test_truncate_for_span_non_string_coerced() -> None:
    from evaluatorq.common.tracing import truncate_for_span
    assert truncate_for_span(42, max_chars=100) == "42"


def test_truncate_default_is_8192(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.common.tracing import (
        _DEFAULT_SPAN_MAX_TEXT_CHARS,
        _default_span_max_text_chars,
        truncate_for_span,
    )
    monkeypatch.delenv("EVALUATORQ_SPAN_MAX_TEXT_CHARS", raising=False)
    _default_span_max_text_chars.cache_clear()
    try:
        result = truncate_for_span("x" * 10_000)
        assert len(result) == _DEFAULT_SPAN_MAX_TEXT_CHARS == 8192
    finally:
        _default_span_max_text_chars.cache_clear()


# ---------------------------------------------------------------------------
# _capture_message_content
# ---------------------------------------------------------------------------

def test_capture_message_content_default_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False)
    from evaluatorq.common.tracing import _capture_message_content
    assert _capture_message_content() is True


def test_capture_message_content_false_when_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false")
    from evaluatorq.common.tracing import _capture_message_content
    assert _capture_message_content() is False


# ---------------------------------------------------------------------------
# record_token_usage — superset: all attrs from both impls
# ---------------------------------------------------------------------------

def test_record_token_usage_superset_attrs() -> None:
    """Unified record_token_usage emits every attribute both impls used to emit."""
    from unittest.mock import MagicMock
    from evaluatorq.common.tracing import record_token_usage

    span = MagicMock()
    record_token_usage(span, prompt_tokens=10, completion_tokens=20, total_tokens=30, calls=2)

    set_attrs: dict[str, Any] = {
        call.args[0]: call.args[1]
        for call in span.set_attribute.call_args_list
    }
    # OTel GenAI semantic convention
    assert set_attrs["gen_ai.usage.input_tokens"] == 10
    assert set_attrs["gen_ai.usage.output_tokens"] == 20
    assert set_attrs["gen_ai.usage.total_tokens"] == 30
    # Aliases (from redteam impl)
    assert set_attrs["gen_ai.usage.prompt_tokens"] == 10
    assert set_attrs["gen_ai.usage.completion_tokens"] == 20
    # Bare keys (platform compat)
    assert set_attrs["prompt_tokens"] == 10
    assert set_attrs["completion_tokens"] == 20
    assert set_attrs["input_tokens"] == 10
    assert set_attrs["output_tokens"] == 20
    assert set_attrs["total_tokens"] == 30
    # Call count (from redteam impl)
    assert set_attrs["gen_ai.usage.calls"] == 2
    assert set_attrs["calls"] == 2


def test_record_token_usage_cache_attrs() -> None:
    """Cache token attrs (from simulation impl) survive in unified version."""
    from unittest.mock import MagicMock
    from evaluatorq.common.tracing import record_token_usage

    span = MagicMock()
    record_token_usage(span, prompt_tokens=5, cache_read_input_tokens=3)

    set_attrs: dict[str, Any] = {
        call.args[0]: call.args[1]
        for call in span.set_attribute.call_args_list
    }
    assert set_attrs["gen_ai.usage.cache_read.input_tokens"] == 3


def test_record_token_usage_zero_prompt_preserved() -> None:
    """Zero prompt_tokens must not fall back to 0 (regression guard)."""
    from unittest.mock import MagicMock
    from evaluatorq.common.tracing import record_token_usage

    span = MagicMock()
    record_token_usage(span, prompt_tokens=0, completion_tokens=5, total_tokens=5)
    set_attrs: dict[str, Any] = {
        call.args[0]: call.args[1]
        for call in span.set_attribute.call_args_list
    }
    assert set_attrs["gen_ai.usage.input_tokens"] == 0
    assert set_attrs["gen_ai.usage.output_tokens"] == 5


# ---------------------------------------------------------------------------
# record_llm_response — superset: chat-completions shape
# ---------------------------------------------------------------------------

def test_record_llm_response_chat_completions_attr_set() -> None:
    """Unified record_llm_response emits all attrs for Chat Completions shape."""
    from unittest.mock import MagicMock
    from evaluatorq.common.tracing import record_llm_response

    class _Usage:
        prompt_tokens = 7
        completion_tokens = 11
        total_tokens = 18
        prompt_tokens_details = None
        completion_tokens_details = None

    class _Msg:
        role = "assistant"
        content = "hello"
        tool_calls = None

    class _Choice:
        finish_reason = "stop"
        message = _Msg()

    class _Resp:
        id = "resp-123"
        model = "azure/gpt-4o-mini"
        usage = _Usage()
        choices = [_Choice()]

    span = MagicMock()
    record_llm_response(span, _Resp())

    set_attrs: dict[str, Any] = {
        call.args[0]: call.args[1]
        for call in span.set_attribute.call_args_list
    }
    # Response metadata
    assert set_attrs["gen_ai.response.id"] == "resp-123"
    assert set_attrs["gen_ai.response.model"] == "azure/gpt-4o-mini"
    # Token attrs (via record_token_usage — both OTel and aliases)
    assert set_attrs["gen_ai.usage.input_tokens"] == 7
    assert set_attrs["gen_ai.usage.output_tokens"] == 11
    assert set_attrs["gen_ai.usage.prompt_tokens"] == 7
    assert set_attrs["gen_ai.usage.completion_tokens"] == 11
    assert set_attrs["gen_ai.usage.total_tokens"] == 18
    assert set_attrs["prompt_tokens"] == 7
    assert set_attrs["completion_tokens"] == 11
    assert set_attrs["input_tokens"] == 7
    assert set_attrs["output_tokens"] == 11
    assert set_attrs["total_tokens"] == 18
    # Finish reason
    assert set_attrs["gen_ai.response.finish_reasons"] == ["stop"]
    # Output content (capture gate default=True)
    assert "gen_ai.output.messages" in set_attrs
    parsed = json.loads(set_attrs["gen_ai.output.messages"])
    assert parsed == [{"role": "assistant", "content": "hello"}]
    assert set_attrs["output"] == set_attrs["gen_ai.output.messages"]


def test_record_llm_response_reasoning_tokens_attr() -> None:
    """reasoning_tokens from completion_tokens_details must be recorded (from redteam impl)."""
    from unittest.mock import MagicMock
    from evaluatorq.common.tracing import record_llm_response

    class _CompDetails:
        reasoning_tokens = 42

    class _Usage:
        prompt_tokens = 5
        completion_tokens = 10
        total_tokens = 15
        prompt_tokens_details = None
        completion_tokens_details = _CompDetails()

    class _Resp:
        id = "r"
        model = "gpt-5"
        usage = _Usage()
        choices = []

    span = MagicMock()
    record_llm_response(span, _Resp())
    set_attrs: dict[str, Any] = {
        call.args[0]: call.args[1]
        for call in span.set_attribute.call_args_list
    }
    assert set_attrs["gen_ai.usage.completion_tokens_details.reasoning_tokens"] == 42


# ---------------------------------------------------------------------------
# record_llm_response — superset: Responses API shape
# ---------------------------------------------------------------------------

def test_record_llm_response_responses_api_attr_set() -> None:
    """Unified record_llm_response emits all attrs for Responses API shape."""
    from unittest.mock import MagicMock
    from evaluatorq.common.tracing import record_llm_response

    class _ContentPart:
        text = "hello world"

    class _OutputItem:
        content = [_ContentPart()]

    class _Usage:
        input_tokens = 4
        output_tokens = 2
        total_tokens = 6
        prompt_tokens_details = None
        completion_tokens_details = None

    class _Resp:
        id = "resp_api_1"
        model = "openai/gpt-4o"
        usage = _Usage()
        output = [_OutputItem()]
        status = "completed"

    span = MagicMock()
    record_llm_response(span, _Resp())

    set_attrs: dict[str, Any] = {
        call.args[0]: call.args[1]
        for call in span.set_attribute.call_args_list
    }
    assert set_attrs["gen_ai.response.id"] == "resp_api_1"
    assert set_attrs["gen_ai.response.model"] == "openai/gpt-4o"
    # Falls back to input_tokens / output_tokens when prompt_tokens absent
    assert set_attrs["gen_ai.usage.input_tokens"] == 4
    assert set_attrs["gen_ai.usage.output_tokens"] == 2
    # Bare aliases
    assert set_attrs["prompt_tokens"] == 4
    assert set_attrs["completion_tokens"] == 2
    # Finish reason from .status
    assert set_attrs["gen_ai.response.finish_reasons"] == ["completed"]
    # Output content
    assert "gen_ai.output.messages" in set_attrs
    parsed = json.loads(set_attrs["gen_ai.output.messages"])
    assert parsed == [{"role": "assistant", "content": "hello world"}]


def test_record_llm_response_dict_shape() -> None:
    """record_llm_response works on plain dicts (duck-typed via _field)."""
    from unittest.mock import MagicMock
    from evaluatorq.common.tracing import record_llm_response

    response = {
        "id": "dict_resp",
        "model": "gpt-4o",
        "status": "completed",
        "usage": {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8},
        "output": [{"content": [{"text": "hi dict"}]}],
    }
    span = MagicMock()
    record_llm_response(span, response)
    set_attrs: dict[str, Any] = {
        call.args[0]: call.args[1]
        for call in span.set_attribute.call_args_list
    }
    assert set_attrs["gen_ai.response.id"] == "dict_resp"
    parsed = json.loads(set_attrs["gen_ai.output.messages"])
    assert parsed[0]["content"] == "hi dict"


def test_record_llm_response_output_content_override() -> None:
    """output_content param overrides response extraction (backward compat with redteam callers)."""
    from unittest.mock import MagicMock
    from evaluatorq.common.tracing import record_llm_response

    class _Resp:
        id = "r"
        model = "m"
        usage = None
        choices = []

    span = MagicMock()
    record_llm_response(span, _Resp(), output_content="my explicit output")
    set_attrs: dict[str, Any] = {
        call.args[0]: call.args[1]
        for call in span.set_attribute.call_args_list
    }
    assert "gen_ai.output.messages" in set_attrs
    parsed = json.loads(set_attrs["gen_ai.output.messages"])
    assert parsed[0]["content"] == "my explicit output"


def test_record_llm_response_suppressed_by_capture_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """When OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false, no output recorded."""
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false")
    from unittest.mock import MagicMock
    from evaluatorq.common.tracing import record_llm_response

    class _Msg:
        role = "assistant"
        content = "secret"
        tool_calls = None

    class _Choice:
        finish_reason = "stop"
        message = _Msg()

    class _Resp:
        id = "r"
        model = "m"
        usage = None
        choices = [_Choice()]

    span = MagicMock()
    record_llm_response(span, _Resp())
    set_attr_keys = {call.args[0] for call in span.set_attribute.call_args_list}
    assert "gen_ai.output.messages" not in set_attr_keys
    assert "output" not in set_attr_keys


# ---------------------------------------------------------------------------
# set_span_attrs
# ---------------------------------------------------------------------------

def test_set_span_attrs_skips_none() -> None:
    from unittest.mock import MagicMock
    from evaluatorq.common.tracing import set_span_attrs

    span = MagicMock()
    set_span_attrs(span, {"key": "val", "skip": None})
    calls = {c.args[0] for c in span.set_attribute.call_args_list}
    assert "key" in calls
    assert "skip" not in calls


def test_set_span_attrs_noop_on_none_span() -> None:
    from evaluatorq.common.tracing import set_span_attrs
    set_span_attrs(None, {"key": "val"})  # must not raise


# ---------------------------------------------------------------------------
# record_llm_input / record_llm_output
# ---------------------------------------------------------------------------

def test_record_llm_input_serializes_and_gates() -> None:
    from unittest.mock import MagicMock
    from evaluatorq.common.tracing import record_llm_input

    span = MagicMock()
    record_llm_input(span, [{"role": "user", "content": "hello"}])
    set_attrs = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
    assert "gen_ai.input.messages" in set_attrs
    assert set_attrs["input"] == set_attrs["gen_ai.input.messages"]


def test_record_llm_input_suppressed_by_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false")
    from unittest.mock import MagicMock
    from evaluatorq.common.tracing import record_llm_input

    span = MagicMock()
    record_llm_input(span, [{"role": "user", "content": "secret"}])
    assert span.set_attribute.call_count == 0


def test_record_llm_output_serializes() -> None:
    from unittest.mock import MagicMock
    from evaluatorq.common.tracing import record_llm_output

    span = MagicMock()
    record_llm_output(span, "response text")
    set_attrs = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
    assert "gen_ai.output.messages" in set_attrs
    parsed = json.loads(set_attrs["gen_ai.output.messages"])
    assert parsed == [{"role": "assistant", "content": "response text"}]


# ---------------------------------------------------------------------------
# get_trace_context_headers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_trace_context_headers_returns_dict() -> None:
    from evaluatorq.common.tracing import get_trace_context_headers
    headers = await get_trace_context_headers()
    assert isinstance(headers, dict)
