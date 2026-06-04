# RES-899: Unify Duplicated Tracing Layer into common/ — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all generic OTel tracing helpers (`truncate_for_span`, `record_token_usage`, `record_llm_response`, `record_llm_input/output`, `_capture_message_content`, `set_span_attrs`, `get_trace_context_headers`) from `redteam/tracing.py` and `simulation/tracing.py` into `common/tracing.py`, and move `record_openresponses_request/response` into `openresponses/tracing.py`.

**Architecture:** Create `common/tracing.py` as the single home for all generic OTel span-recording utilities. The unified `record_llm_response` is a superset of both existing impls (duck-typed, dual-shape Chat+Responses API, with PII capture gate). Both domain `tracing.py` files shrink to only their domain-specific span builders (`with_simulation_span`, `with_llm_span`, `with_redteam_span`). All blast-radius importers are re-pointed; no permanent shims.

**Tech Stack:** Python 3.10+, OpenTelemetry, `functools.lru_cache`, `common/fields.py:get_field`, pytest-asyncio, `uv run pytest`, `uv run basedpyright`

**Documented behavior deltas (call out in PR description):**
1. Simulation span text now truncates at 8192 chars (was 2000). Marker `…` → `... [truncated]`.
2. Redteam spans now honor `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` env flag (previously always captured content).

**Branch:** Branch from `bauke/res-897-refactorevaluatorq-py-extract-openresponses-runtime-shared` (or `main` after RES-897 merges). Rebase once RES-897 merges.

---

## File Map

| Action | File |
|--------|------|
| Create | `src/evaluatorq/common/tracing.py` |
| Create | `src/evaluatorq/openresponses/tracing.py` |
| Create | `tests/common/test_tracing.py` |
| Slim | `src/evaluatorq/simulation/tracing.py` |
| Slim | `src/evaluatorq/redteam/tracing.py` |
| Re-point | `src/evaluatorq/openresponses/target.py` |
| Re-point | `src/evaluatorq/simulation/agents/base.py` |
| Re-point | `src/evaluatorq/simulation/generators/first_message_generator.py` |
| Re-point | `src/evaluatorq/simulation/utils/structured_output.py` |
| Re-point | `src/evaluatorq/simulation/runner/simulation.py` |
| Re-point | `src/evaluatorq/simulation/api.py` |
| Re-point | `src/evaluatorq/redteam/backends/openai.py` |
| Re-point | `src/evaluatorq/redteam/backends/orq.py` |
| Re-point | `src/evaluatorq/redteam/adaptive/capability_classifier.py` |
| Re-point | `src/evaluatorq/redteam/adaptive/evaluator.py` |
| Re-point | `src/evaluatorq/redteam/adaptive/objective_generator.py` |
| Re-point | `src/evaluatorq/redteam/adaptive/orchestrator.py` |
| Re-point | `src/evaluatorq/redteam/adaptive/pipeline.py` |
| Update tests | `tests/redteam/test_truncate_for_span.py` |
| Update tests | `tests/simulation/test_tracing.py` |

---

## Task 1: Write failing tests for `common/tracing.py`

**Files:**
- Create: `tests/common/test_tracing.py`

- [ ] **Step 1: Create the test file**

```python
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
```

- [ ] **Step 2: Run the tests — they must all FAIL (module doesn't exist yet)**

```bash
cd packages/evaluatorq-py && uv run pytest tests/common/test_tracing.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'evaluatorq.common.tracing'` or similar import errors.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/common/test_tracing.py
git commit -m "test(evaluatorq-py): add failing tests for common/tracing.py (RES-899)"
```

---

## Task 2: Implement `common/tracing.py`

**Files:**
- Create: `src/evaluatorq/common/tracing.py`

- [ ] **Step 1: Create the unified tracing module**

```python
# src/evaluatorq/common/tracing.py
"""Generic OTel span-recording utilities shared by all evaluatorq domains.

Domain-specific span builders (with_simulation_span, with_redteam_span,
with_llm_span) stay in their respective domain tracing modules and may import
from here. This module must not import from redteam, simulation, or openresponses.
"""

from __future__ import annotations

import functools
import json
import os
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.common.fields import get_field as _field
from evaluatorq.tracing.setup import get_tracer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from opentelemetry.trace import Span

AttrValue = str | int | float | bool
AttrMap = dict[str, AttrValue | None]

_TRUNCATION_MARKER = "... [truncated]"
_DEFAULT_SPAN_MAX_TEXT_CHARS = 8192


@functools.lru_cache(maxsize=1)
def _default_span_max_text_chars() -> int | None:
    """Read EVALUATORQ_SPAN_MAX_TEXT_CHARS once. Default 8192. Set 0 to disable.

    Call _default_span_max_text_chars.cache_clear() in tests after changing the env var.
    """
    raw = os.getenv("EVALUATORQ_SPAN_MAX_TEXT_CHARS")
    if raw is None or raw == "":
        return _DEFAULT_SPAN_MAX_TEXT_CHARS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "EVALUATORQ_SPAN_MAX_TEXT_CHARS={!r} is not a valid int; using default {}",
            raw,
            _DEFAULT_SPAN_MAX_TEXT_CHARS,
        )
        return _DEFAULT_SPAN_MAX_TEXT_CHARS
    if value < 0:
        logger.warning(
            "EVALUATORQ_SPAN_MAX_TEXT_CHARS={!r} must be non-negative; using default {}",
            value,
            _DEFAULT_SPAN_MAX_TEXT_CHARS,
        )
        return _DEFAULT_SPAN_MAX_TEXT_CHARS
    return value


def truncate_for_span(text: object, *, max_chars: int | None = None) -> str:
    """Truncate text for span attribute storage.

    Defaults to EVALUATORQ_SPAN_MAX_TEXT_CHARS env var (or 8192 if unset).
    Set 0 to disable truncation. Negative values raise ValueError.
    Output never exceeds max_chars; the marker is reserved within the budget.
    """
    if isinstance(text, str):
        s = text
    else:
        try:
            s = str(text)
        except Exception as e:  # pragma: no cover  # noqa: BLE001
            s = f"<unrepresentable {type(text).__name__}: {e}>"
    if max_chars is None:
        max_chars = _default_span_max_text_chars()
    if max_chars is None or max_chars == 0:
        return s
    if max_chars < 0:
        raise ValueError(f"max_chars must be non-negative, got {max_chars}")
    if len(s) <= max_chars:
        return s
    marker_len = len(_TRUNCATION_MARKER)
    if max_chars <= marker_len:
        return _TRUNCATION_MARKER[:max_chars]
    return s[: max_chars - marker_len] + _TRUNCATION_MARKER


def _capture_message_content() -> bool:
    """Honor OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT.

    Defaults to True (matches TypeScript impl, RES-595). Set the env var to
    'false' to opt out when exporting to a third-party backend.
    """
    flag = os.environ.get("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")
    if flag is None:
        return True
    return flag.lower() == "true" or flag == "1"


def _serialize_messages(messages: list[dict[str, Any]]) -> str:
    return json.dumps(
        [
            {
                "role": str(m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")),
                "content": truncate_for_span(
                    m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
                ),
            }
            for m in messages
        ],
        ensure_ascii=False,
    )


def _serialize_tool_call_content(tool_calls: list[dict[str, str]]) -> str:
    return json.dumps({"tool_calls": tool_calls}, ensure_ascii=False)


def _extract_chat_tool_call_payloads(tool_calls: Any) -> list[dict[str, str]]:
    payloads: list[dict[str, str]] = []
    for tool_call in tool_calls or []:
        function = _field(tool_call, "function")
        name = _field(function, "name") or _field(tool_call, "name")
        arguments = _field(function, "arguments") or _field(tool_call, "arguments")
        payload: dict[str, str] = {}
        if name:
            payload["name"] = str(name)
        if arguments is not None:
            payload["arguments"] = str(arguments)
        if payload:
            payloads.append(payload)
    return payloads


def _extract_response_tool_call_payloads(output_items: list[Any]) -> list[dict[str, str]]:
    payloads: list[dict[str, str]] = []
    for item in output_items:
        call_id = _field(item, "call_id")
        name = _field(item, "name")
        arguments = _field(item, "arguments")
        if call_id or name or arguments is not None:
            payload: dict[str, str] = {}
            if call_id:
                payload["call_id"] = str(call_id)
            if name:
                payload["name"] = str(name)
            if arguments is not None:
                payload["arguments"] = str(arguments)
            payloads.append(payload)
    return payloads


def _extract_output_messages(response: Any) -> list[dict[str, str]]:
    """Extract output message dicts from Chat Completions or Responses API shape."""
    output_messages: list[dict[str, str]] = []
    choices = _field(response, "choices")
    if choices:
        for choice in choices:
            message = _field(choice, "message")
            content = _field(message, "content") if message else None
            if content:
                role = _field(message, "role") or "assistant"
                output_messages.append({"role": str(role), "content": str(content)})
                continue
            tool_payloads = _extract_chat_tool_call_payloads(
                _field(message, "tool_calls") if message else None
            )
            if tool_payloads:
                role = _field(message, "role") or "assistant"
                output_messages.append({
                    "role": str(role),
                    "content": _serialize_tool_call_content(tool_payloads),
                })
    else:
        output_text = _field(response, "output_text")
        if isinstance(output_text, str) and output_text:
            output_messages.append({"role": "assistant", "content": output_text})
        else:
            output_items = _field(response, "output") or []
            parts: list[str] = []
            for item in output_items:
                content = _field(item, "content")
                if content:
                    for part in content:
                        text = _field(part, "text")
                        if isinstance(text, str) and text:
                            parts.append(text)
                else:
                    text = _field(item, "text")
                    if isinstance(text, str) and text:
                        parts.append(text)
            joined = "".join(parts)
            if joined:
                output_messages.append({"role": "assistant", "content": joined})
            else:
                tool_payloads = _extract_response_tool_call_payloads(output_items)
                if tool_payloads:
                    output_messages.append({
                        "role": "assistant",
                        "content": _serialize_tool_call_content(tool_payloads),
                    })
    return output_messages


def set_span_attrs(span: Span | None, attrs: AttrMap) -> None:
    """Batch-set span attributes. Skips None values. Safe no-op when span is None."""
    if span is None:
        return
    for key, value in attrs.items():
        if value is not None:
            span.set_attribute(key, value)


def record_token_usage(
    span: Span | None,
    *,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    calls: int = 0,
    cache_read_input_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
) -> None:
    """Record token usage on a span. Safe no-op when span is None.

    Superset of both former redteam and simulation impls: sets OTel GenAI
    attribute names, their aliases, bare keys, call count, and cache details.
    """
    if span is None:
        return
    prompt = prompt_tokens if prompt_tokens is not None else 0
    completion = completion_tokens if completion_tokens is not None else 0
    total = total_tokens if total_tokens is not None else prompt + completion
    span.set_attribute("gen_ai.usage.input_tokens", prompt)
    span.set_attribute("gen_ai.usage.output_tokens", completion)
    span.set_attribute("gen_ai.usage.prompt_tokens", prompt)
    span.set_attribute("gen_ai.usage.completion_tokens", completion)
    span.set_attribute("gen_ai.usage.total_tokens", total)
    span.set_attribute("prompt_tokens", prompt)
    span.set_attribute("completion_tokens", completion)
    span.set_attribute("input_tokens", prompt)
    span.set_attribute("output_tokens", completion)
    span.set_attribute("total_tokens", total)
    if calls:
        span.set_attribute("gen_ai.usage.calls", calls)
        span.set_attribute("calls", calls)
    if cache_read_input_tokens is not None:
        span.set_attribute("gen_ai.usage.cache_read.input_tokens", cache_read_input_tokens)
    if cache_creation_input_tokens is not None:
        span.set_attribute("gen_ai.usage.cache_creation.input_tokens", cache_creation_input_tokens)


def record_llm_response(
    span: Span | None,
    response: Any,
    *,
    output_content: str | None = None,
) -> None:
    """Record LLM response attributes on a span.

    Superset of both former impls: duck-typed (_field handles dicts + objects),
    handles Chat Completions and Responses API shapes, honors the PII capture
    gate, accepts an optional output_content override for backward compat with
    redteam callers that pass the output string explicitly.

    output_content, when provided, is used instead of extracting from response.
    """
    if span is None:
        return

    response_id = _field(response, "id")
    if response_id:
        span.set_attribute("gen_ai.response.id", response_id)
    response_model = _field(response, "model")
    if response_model:
        span.set_attribute("gen_ai.response.model", response_model)

    usage = _field(response, "usage")
    if usage is not None:
        prompt = _field(usage, "prompt_tokens")
        if prompt is None:
            prompt = _field(usage, "input_tokens")
        completion = _field(usage, "completion_tokens")
        if completion is None:
            completion = _field(usage, "output_tokens")
        total = _field(usage, "total_tokens")
        details = _field(usage, "prompt_tokens_details")
        if details is None:
            details = _field(usage, "input_tokens_details")
        cache_read = _field(details, "cached_tokens") if details else None
        record_token_usage(
            span,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cache_read_input_tokens=cache_read,
        )
        completion_details = _field(usage, "completion_tokens_details")
        if completion_details is not None:
            reasoning = _field(completion_details, "reasoning_tokens")
            if reasoning is not None:
                span.set_attribute(
                    "gen_ai.usage.completion_tokens_details.reasoning_tokens",
                    int(reasoning),
                )

    if _capture_message_content():
        if output_content is not None:
            serialized = json.dumps(
                [{"role": "assistant", "content": truncate_for_span(output_content)}],
                ensure_ascii=False,
            )
            span.set_attribute("gen_ai.output.messages", serialized)
            span.set_attribute("output", serialized)
        else:
            output_messages = _extract_output_messages(response)
            if output_messages:
                serialized = _serialize_messages(output_messages)
                span.set_attribute("gen_ai.output.messages", serialized)
                span.set_attribute("output", serialized)

    finish_reasons: list[str] = []
    choices = _field(response, "choices")
    if choices:
        for choice in choices:
            reason = _field(choice, "finish_reason")
            if reason:
                finish_reasons.append(reason)
    else:
        status = _field(response, "status")
        if isinstance(status, str) and status:
            finish_reasons.append(status)
    if finish_reasons:
        span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)


def record_llm_input(span: Span | None, messages: list[dict[str, Any]]) -> None:
    """Record LLM input messages. Suppressed when capture gate is off."""
    if span is None or not messages:
        return
    if not _capture_message_content():
        return
    serialized = _serialize_messages(messages)
    span.set_attribute("gen_ai.input.messages", serialized)
    span.set_attribute("input", serialized)


def record_llm_output(span: Span | None, output: str) -> None:
    """Record a single LLM output string. Suppressed when capture gate is off."""
    if span is None or not output:
        return
    if not _capture_message_content():
        return
    serialized = _serialize_messages([{"role": "assistant", "content": output}])
    span.set_attribute("gen_ai.output.messages", serialized)
    span.set_attribute("output", serialized)


async def get_trace_context_headers() -> dict[str, str]:  # noqa: RUF029
    """Return W3C trace context headers for the current active span.

    Empty dict when OTel is not available. Used to propagate trace context
    into outgoing HTTP requests.
    """
    try:
        from opentelemetry import context, propagate
    except ImportError:
        return {}
    headers: dict[str, str] = {}
    propagate.inject(headers, context=context.get_current())
    return headers
```

- [ ] **Step 2: Run the new tests**

```bash
cd packages/evaluatorq-py && uv run pytest tests/common/test_tracing.py -v
```

Expected: all tests PASS.

- [ ] **Step 3: Run full test suite (baseline — some tests may fail until blast-radius is updated)**

```bash
cd packages/evaluatorq-py && uv run pytest -m 'not integration' -q 2>&1 | tail -20
```

Expected: tests in `tests/redteam/` and `tests/simulation/` that import from domain tracing modules still pass (they haven't been re-pointed yet — domain modules still define the old symbols).

- [ ] **Step 4: Commit**

```bash
git add src/evaluatorq/common/tracing.py
git commit -m "feat(evaluatorq-py): add common/tracing.py unified OTel recording layer (RES-899)"
```

---

## Task 3: Create `openresponses/tracing.py`

**Files:**
- Create: `src/evaluatorq/openresponses/tracing.py`

This module provides `with_llm_span` (so `openresponses/target.py` no longer needs to import from `simulation`), and hosts `record_openresponses_request/response` (deferred from RES-897). It builds on `common/tracing.py`.

- [ ] **Step 1: Create the file**

```python
# src/evaluatorq/openresponses/tracing.py
"""OTel span helpers for the OpenResponses runtime.

Provides with_llm_span for the Responses API call path, and
record_openresponses_request/response helpers that record the full
Responses API payload alongside the standard gen_ai.* attributes.
Imports recording utilities from common/tracing.py; no simulation import.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.common.tracing import (
    _capture_message_content,
    record_llm_response,
    truncate_for_span,
)
from evaluatorq.tracing.setup import get_tracer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from opentelemetry.trace import Span

_otel_import_warned = False

_PROVIDER_ALIASES: dict[str, str] = {
    "azure": "azure.ai.openai",
}


def _derive_provider(model: str) -> str:
    if "/" in model:
        prefix = model.split("/", 1)[0]
        return _PROVIDER_ALIASES.get(prefix, prefix)
    return "openai"


@asynccontextmanager
async def with_llm_span(  # noqa: RUF029
    *,
    model: str,
    operation: str = "chat",
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    purpose: str | None = None,
) -> AsyncGenerator[Span | None, None]:
    """Execute a block within a GenAI LLM span (SpanKind.CLIENT).

    Mirrors simulation.tracing.with_llm_span without the simulation dependency.
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    try:
        from opentelemetry.trace import SpanKind, Status, StatusCode
    except ImportError as exc:
        global _otel_import_warned
        if not _otel_import_warned:
            logger.warning("OpenTelemetry import failed; tracing disabled: %s", exc)
            _otel_import_warned = True
        yield None
        return

    resolved_provider = provider or _derive_provider(model)
    span_name = f"{operation} {model}"

    attrs: dict[str, Any] = {
        "gen_ai.operation.name": operation,
        "gen_ai.system": resolved_provider,
        "gen_ai.provider.name": resolved_provider,
        "gen_ai.request.model": model,
    }
    if temperature is not None:
        attrs["gen_ai.request.temperature"] = temperature
    if max_tokens is not None:
        attrs["gen_ai.request.max_tokens"] = max_tokens
    if purpose:
        attrs["orq.simulation.llm_purpose"] = purpose

    with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT, attributes=attrs) as span:
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except BaseException as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.set_attribute("error.type", type(e).__name__)
            raise


def record_openresponses_request(span: Span | None, payload: dict[str, Any]) -> None:
    """Record a Responses API request with both generic and Orq-specific attrs."""
    if span is None:
        return
    model = payload.get("model")
    if model:
        span.set_attribute("gen_ai.request.model", str(model))
    max_output_tokens = payload.get("max_output_tokens")
    if isinstance(max_output_tokens, int):
        span.set_attribute("gen_ai.request.max_tokens", max_output_tokens)
    if not _capture_message_content():
        return
    input_items = payload.get("input") or []
    serialized_input = truncate_for_span(
        json.dumps(input_items, ensure_ascii=False, default=str)
    )
    span.set_attribute("gen_ai.input.messages", serialized_input)
    span.set_attribute("input", serialized_input)
    span.set_attribute(
        "orq.openresponses.request",
        truncate_for_span(json.dumps(payload, ensure_ascii=False, default=str)),
    )


def record_openresponses_response(span: Span | None, response: Any) -> None:
    """Record a Responses API response with standard gen_ai.* attributes."""
    if span is None:
        return
    record_llm_response(span, response)
    try:
        payload = (
            response.model_dump(mode="json")
            if hasattr(response, "model_dump")
            else response
        )
    except Exception as exc:
        logger.debug(
            "record_openresponses_response: model_dump failed ({}); falling back to repr",
            exc,
        )
        payload = repr(response)
    if _capture_message_content():
        span.set_attribute(
            "orq.openresponses.response",
            truncate_for_span(json.dumps(payload, ensure_ascii=False, default=str)),
        )
```

- [ ] **Step 2: Run basedpyright on the new file**

```bash
cd packages/evaluatorq-py && uv run basedpyright src/evaluatorq/openresponses/tracing.py
```

Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
git add src/evaluatorq/openresponses/tracing.py
git commit -m "feat(evaluatorq-py): add openresponses/tracing.py, decouple from simulation (RES-899)"
```

---

## Task 4: Slim `simulation/tracing.py`

Remove helpers now in `common/tracing.py` and re-export them from there. Keep: `with_simulation_span`, `with_llm_span`, `_derive_provider`, `_PROVIDER_ALIASES`, `_otel_import_warned`, `AttrValue`, `AttrMap`.

**Files:**
- Modify: `src/evaluatorq/simulation/tracing.py`

- [ ] **Step 1: Replace the body of `simulation/tracing.py`**

The new file keeps domain span builders, imports the generic helpers from common, and re-exports them so callers that haven't been re-pointed yet still resolve (temporary — all callers get re-pointed in Tasks 6-8).

```python
# src/evaluatorq/simulation/tracing.py
"""OTel span helpers for the agent simulation module.

Domain-specific span builders (with_simulation_span, with_llm_span) live here.
Generic recording utilities are imported from evaluatorq.common.tracing.

Span hierarchy:
    orq.simulation.pipeline (root)
      ├── orq.simulation.run (per datapoint)
      │   ├── orq.simulation.first_message_generation
      │   └── orq.simulation.turn (per turn)
      │       ├── orq.simulation.target_call
      │       ├── orq.simulation.judge_evaluation
      │       └── orq.simulation.user_simulator_call
      └── chat/responses {model}  (LLM client spans, GenAI semconv)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.common.tracing import (
    AttrMap,
    AttrValue,
    get_trace_context_headers,
    record_llm_input,
    record_llm_output,
    record_llm_response,
    record_token_usage,
    set_span_attrs,
)
from evaluatorq.tracing.setup import get_tracer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from opentelemetry.trace import Span

__all__ = [
    "AttrMap",
    "AttrValue",
    "get_trace_context_headers",
    "record_llm_input",
    "record_llm_output",
    "record_llm_response",
    "record_token_usage",
    "set_span_attrs",
    "with_llm_span",
    "with_simulation_span",
]

_otel_import_warned = False

_PROVIDER_ALIASES: dict[str, str] = {
    "azure": "azure.ai.openai",
}


def _derive_provider(model: str) -> str:
    if "/" in model:
        prefix = model.split("/", 1)[0]
        return _PROVIDER_ALIASES.get(prefix, prefix)
    return "openai"


@asynccontextmanager
async def with_simulation_span(  # noqa: RUF029
    name: str,
    attributes: AttrMap | None = None,
) -> AsyncGenerator[Span | None, None]:
    """Execute a block within a simulation span (SpanKind.INTERNAL).

    Records exceptions (including asyncio.CancelledError) and sets span status.

    Yields:
        The active span, or None when tracing is disabled.
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    try:
        from opentelemetry.trace import SpanKind, Status, StatusCode
    except ImportError as exc:
        global _otel_import_warned
        if not _otel_import_warned:
            logger.warning("OpenTelemetry import failed; tracing disabled: %s", exc)
            _otel_import_warned = True
        yield None
        return

    clean_attrs: dict[str, AttrValue] = {
        k: v for k, v in (attributes or {}).items() if v is not None
    }

    with tracer.start_as_current_span(
        name,
        kind=SpanKind.INTERNAL,
        attributes=clean_attrs,
    ) as span:
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except BaseException as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.set_attribute("error.type", type(e).__name__)
            raise


@asynccontextmanager
async def with_llm_span(  # noqa: RUF029
    *,
    model: str,
    operation: str = "chat",
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    purpose: str | None = None,
) -> AsyncGenerator[Span | None, None]:
    """Execute a block within a GenAI LLM span (SpanKind.CLIENT).

    Span name is "{operation} {model}". Sets orq.simulation.llm_purpose when
    purpose is provided.

    Yields:
        The active span, or None when tracing is disabled.
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    try:
        from opentelemetry.trace import SpanKind, Status, StatusCode
    except ImportError as exc:
        global _otel_import_warned
        if not _otel_import_warned:
            logger.warning("OpenTelemetry import failed; tracing disabled: %s", exc)
            _otel_import_warned = True
        yield None
        return

    resolved_provider = provider or _derive_provider(model)
    span_name = f"{operation} {model}"

    attrs: dict[str, Any] = {
        "gen_ai.operation.name": operation,
        "gen_ai.system": resolved_provider,
        "gen_ai.provider.name": resolved_provider,
        "gen_ai.request.model": model,
    }
    if temperature is not None:
        attrs["gen_ai.request.temperature"] = temperature
    if max_tokens is not None:
        attrs["gen_ai.request.max_tokens"] = max_tokens
    if purpose:
        attrs["orq.simulation.llm_purpose"] = purpose

    with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT, attributes=attrs) as span:
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except BaseException as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.set_attribute("error.type", type(e).__name__)
            raise
```

Note: `record_openresponses_request` and `record_openresponses_response` are NOT re-exported here — they now live in `openresponses/tracing.py`. The only remaining consumer (`openresponses/target.py`) gets re-pointed in Task 8.

- [ ] **Step 2: Run the simulation tracing tests**

```bash
cd packages/evaluatorq-py && uv run pytest tests/simulation/test_tracing.py -v 2>&1 | tail -30
```

Expected: all tests pass (the test still imports from `evaluatorq.simulation.tracing` and the symbols are re-exported).

- [ ] **Step 3: Commit**

```bash
git add src/evaluatorq/simulation/tracing.py
git commit -m "refactor(evaluatorq-py): slim simulation/tracing.py — re-export generic helpers from common (RES-899)"
```

---

## Task 5: Slim `redteam/tracing.py`

Remove helpers now in `common/tracing.py`. Keep: `with_redteam_span`, `with_llm_span`, `_derive_provider`, `_sanitize_messages` (used internally by `with_llm_span`).

**Files:**
- Modify: `src/evaluatorq/redteam/tracing.py`

- [ ] **Step 1: Replace the body of `redteam/tracing.py`**

```python
# src/evaluatorq/redteam/tracing.py
"""Red teaming span utilities for OpenTelemetry instrumentation.

Domain-specific span builders (with_redteam_span, with_llm_span) live here.
Generic recording utilities are imported from evaluatorq.common.tracing.

Span hierarchy:
- orq.redteam.pipeline (root or child of parent context)      [runner.py]
  +-- orq.redteam.context_retrieval                            [runner.py]
  +-- orq.redteam.datapoint_generation                         [runner.py]
  |   +-- orq.redteam.capability_classification                [strategy_planner.py]
  |   |   +-- chat (llm_purpose=classify_tools)                [capability_classifier.py]
  |   |   +-- chat (llm_purpose=infer_resources)               [capability_classifier.py]
  |   +-- orq.redteam.strategy_planning                        [strategy_planner.py]
  |       +-- chat (llm_purpose=generate_strategies)           [objective_generator.py]
  +-- orq.job (framework)                                      [processings.py]
  |   +-- orq.redteam.attack                                   [pipeline.py]
  |   |   +-- orq.redteam.target_call                          [pipeline.py] (single-turn template)
  |   |   |   +-- agent <key> (llm_purpose=target)             [orq.py] (ORQ agent targets)
  |   |   |   +-- chat (llm_purpose=target)                    [openai.py] (OpenAI model targets)
  |   |   +-- orq.redteam.attack_turn x N                      [orchestrator.py]
  |   |       +-- orq.redteam.adversarial_generation           [orchestrator.py]
  |   |       |   +-- chat (llm_purpose=adversarial)           [orchestrator.py]
  |   |       +-- orq.redteam.target_call                      [orchestrator.py]
  |   |           +-- agent <key> (llm_purpose=target)         [orq.py] (ORQ agent targets)
  |   |           +-- chat (llm_purpose=target)                [openai.py] (OpenAI model targets)
  |   +-- orq.evaluation (framework)                           [processings.py]
  |       +-- orq.redteam.security_evaluation                  [pipeline.py]
  |           +-- chat (llm_purpose=evaluation)                [evaluator.py]
  +-- orq.redteam.memory_cleanup                               [runner.py]

LLM spans use SpanKind.CLIENT with span name "{operation} {model}" per OTel
GenAI semantic conventions and carry gen_ai.* attributes. ORQ agent target
calls use SpanKind.INTERNAL with span name "agent {key}".
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.common.tracing import (
    record_llm_response,
    record_token_usage,
    set_span_attrs,
    truncate_for_span,
)
from evaluatorq.tracing.setup import get_tracer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from opentelemetry.trace import Span

__all__ = [
    "record_llm_response",
    "record_token_usage",
    "set_span_attrs",
    "truncate_for_span",
    "with_llm_span",
    "with_redteam_span",
]


def _derive_provider(model: str) -> str:
    if "/" in model:
        return model.split("/", 1)[0]
    return "openai"


@asynccontextmanager
async def with_redteam_span(  # noqa: RUF029
    name: str,
    attributes: dict[str, Any] | None = None,
    parent_context: Any | None = None,
) -> AsyncGenerator[Span | None, None]:
    """Execute code within a red teaming span (SpanKind.INTERNAL).

    Yields the span when tracing is enabled, None otherwise.
    Exceptions propagate and are recorded on the span with ERROR status.
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    try:
        from opentelemetry import context as otel_context
        from opentelemetry.trace import SpanKind, Status, StatusCode
    except ImportError:
        yield None
        return

    ctx = parent_context or otel_context.get_current()

    with tracer.start_as_current_span(
        name,
        context=ctx,
        kind=SpanKind.INTERNAL,
        attributes=attributes or {},
    ) as span:
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_attribute("error.type", type(e).__name__)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


@asynccontextmanager
async def with_llm_span(  # noqa: RUF029
    *,
    model: str,
    operation: str = "chat",
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    input_messages: list[Any] | None = None,
    attributes: dict[str, Any] | None = None,
    parent_context: Any | None = None,
) -> AsyncGenerator[Span | None, None]:
    """Execute code within a GenAI LLM span (SpanKind.CLIENT).

    Span name is "{operation} {model}". input_messages are serialized to
    gen_ai.input.messages and input attributes.
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    try:
        from opentelemetry import context as otel_context
        from opentelemetry.trace import SpanKind, Status, StatusCode
    except ImportError:
        yield None
        return

    ctx = parent_context or otel_context.get_current()
    resolved_provider = provider or _derive_provider(model)
    span_name = f"{operation} {model}"

    genai_attrs: dict[str, Any] = {
        "gen_ai.operation.name": operation,
        "gen_ai.system": resolved_provider,
        "gen_ai.provider.name": resolved_provider,
        "gen_ai.request.model": model,
    }
    if temperature is not None:
        genai_attrs["gen_ai.request.temperature"] = float(temperature)
    if max_tokens is not None:
        genai_attrs["gen_ai.request.max_tokens"] = max_tokens
    if input_messages is not None:
        serialized = json.dumps(
            _sanitize_messages(input_messages), ensure_ascii=False
        )
        genai_attrs["gen_ai.input.messages"] = serialized
        genai_attrs["input"] = serialized
    if attributes:
        genai_attrs.update(attributes)

    with tracer.start_as_current_span(
        span_name,
        context=ctx,
        kind=SpanKind.CLIENT,
        attributes=genai_attrs,
    ) as span:
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_attribute("error.type", type(e).__name__)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def _sanitize_messages(messages: list[Any]) -> list[dict[str, str]]:
    """JSON-safe list of {role, content} for gen_ai.input.messages."""
    sanitized: list[dict[str, str]] = []
    for msg in messages:
        if hasattr(msg, "get") and callable(msg.get):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "")
        sanitized.append({"role": str(role), "content": truncate_for_span(content)})
    return sanitized
```

- [ ] **Step 2: Run redteam tracing tests**

```bash
cd packages/evaluatorq-py && uv run pytest tests/redteam/test_tracing.py tests/redteam/test_tracing_spans.py tests/redteam/test_truncate_for_span.py -v 2>&1 | tail -30
```

Expected: all tests pass (re-exports in `__all__` ensure backward compat until test re-pointing in Task 9).

- [ ] **Step 3: Commit**

```bash
git add src/evaluatorq/redteam/tracing.py
git commit -m "refactor(evaluatorq-py): slim redteam/tracing.py — import generic helpers from common (RES-899)"
```

---

## Task 6: Re-point simulation blast-radius files

Update each file to import generic helpers from `evaluatorq.common.tracing` instead of `evaluatorq.simulation.tracing`. Domain-specific things (`with_simulation_span`, `with_llm_span`) still come from `evaluatorq.simulation.tracing`.

**Files:**
- Modify: `src/evaluatorq/simulation/agents/base.py`
- Modify: `src/evaluatorq/simulation/generators/first_message_generator.py`
- Modify: `src/evaluatorq/simulation/utils/structured_output.py`
- Modify: `src/evaluatorq/simulation/runner/simulation.py`
- Modify: `src/evaluatorq/simulation/api.py`

- [ ] **Step 1: Update `simulation/agents/base.py`**

Find the import block (currently `from evaluatorq.simulation.tracing import (get_trace_context_headers, record_llm_input, record_llm_response, with_llm_span)`). Split it:

```python
from evaluatorq.common.tracing import get_trace_context_headers, record_llm_input, record_llm_response
from evaluatorq.simulation.tracing import with_llm_span
```

- [ ] **Step 2: Update `simulation/generators/first_message_generator.py`**

Currently imports `get_trace_context_headers, record_llm_input, record_llm_response, with_llm_span` from `simulation.tracing`. Split:

```python
from evaluatorq.common.tracing import get_trace_context_headers, record_llm_input, record_llm_response
from evaluatorq.simulation.tracing import with_llm_span
```

- [ ] **Step 3: Update `simulation/utils/structured_output.py`**

Currently imports `get_trace_context_headers, record_llm_input, record_llm_response, with_llm_span` from `simulation.tracing`. Split same way as above.

- [ ] **Step 4: Update `simulation/runner/simulation.py`**

Currently imports `record_llm_input, record_llm_output, record_token_usage, set_span_attrs, with_simulation_span` from `simulation.tracing`. Split:

```python
from evaluatorq.common.tracing import record_llm_input, record_llm_output, record_token_usage, set_span_attrs
from evaluatorq.simulation.tracing import with_simulation_span
```

- [ ] **Step 5: Update `simulation/api.py`**

Currently has 6 local imports from `simulation.tracing`. The `record_token_usage` and `set_span_attrs` move to common; `with_simulation_span` stays in simulation.tracing. Change:

```python
# Replace: from evaluatorq.simulation.tracing import record_token_usage, set_span_attrs
# With:
from evaluatorq.common.tracing import record_token_usage, set_span_attrs
# Leave: from evaluatorq.simulation.tracing import with_simulation_span  (unchanged)
```

There are multiple import lines in api.py — update only the lines that import `record_token_usage` or `set_span_attrs`.

- [ ] **Step 6: Run simulation tests**

```bash
cd packages/evaluatorq-py && uv run pytest tests/simulation/ -v -q 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/evaluatorq/simulation/agents/base.py \
        src/evaluatorq/simulation/generators/first_message_generator.py \
        src/evaluatorq/simulation/utils/structured_output.py \
        src/evaluatorq/simulation/runner/simulation.py \
        src/evaluatorq/simulation/api.py
git commit -m "refactor(evaluatorq-py): re-point simulation blast-radius to common/tracing (RES-899)"
```

---

## Task 7: Re-point redteam blast-radius files

**Files:**
- Modify: `src/evaluatorq/redteam/backends/openai.py`
- Modify: `src/evaluatorq/redteam/backends/orq.py`
- Modify: `src/evaluatorq/redteam/adaptive/capability_classifier.py`
- Modify: `src/evaluatorq/redteam/adaptive/evaluator.py`
- Modify: `src/evaluatorq/redteam/adaptive/objective_generator.py`
- Modify: `src/evaluatorq/redteam/adaptive/orchestrator.py`
- Modify: `src/evaluatorq/redteam/adaptive/pipeline.py`

- [ ] **Step 1: Update each file — change its `from evaluatorq.redteam.tracing import ...` line**

For each file, identify which of the shared helpers it imports and update:
- `record_llm_response`, `record_token_usage`, `truncate_for_span`, `set_span_attrs` → from `evaluatorq.common.tracing`
- `with_llm_span`, `with_redteam_span` → keep from `evaluatorq.redteam.tracing`

File-by-file changes:

**`redteam/backends/openai.py`**: imports `record_llm_response, with_llm_span`
```python
from evaluatorq.common.tracing import record_llm_response
from evaluatorq.redteam.tracing import with_llm_span
```

**`redteam/backends/orq.py`**: imports `record_token_usage, set_span_attrs, truncate_for_span, with_redteam_span`
```python
from evaluatorq.common.tracing import record_token_usage, set_span_attrs, truncate_for_span
from evaluatorq.redteam.tracing import with_redteam_span
```

**`redteam/adaptive/capability_classifier.py`**: imports `record_llm_response, with_llm_span`
```python
from evaluatorq.common.tracing import record_llm_response
from evaluatorq.redteam.tracing import with_llm_span
```

**`redteam/adaptive/evaluator.py`**: imports `record_llm_response, with_llm_span`
```python
from evaluatorq.common.tracing import record_llm_response
from evaluatorq.redteam.tracing import with_llm_span
```

**`redteam/adaptive/objective_generator.py`**: imports `record_llm_response, with_llm_span`
```python
from evaluatorq.common.tracing import record_llm_response
from evaluatorq.redteam.tracing import with_llm_span
```

**`redteam/adaptive/orchestrator.py`**: imports `record_llm_response, set_span_attrs, truncate_for_span, with_llm_span, with_redteam_span`
```python
from evaluatorq.common.tracing import record_llm_response, set_span_attrs, truncate_for_span
from evaluatorq.redteam.tracing import with_llm_span, with_redteam_span
```

**`redteam/adaptive/pipeline.py`**: imports `set_span_attrs, truncate_for_span, with_redteam_span`
```python
from evaluatorq.common.tracing import set_span_attrs, truncate_for_span
from evaluatorq.redteam.tracing import with_redteam_span
```

- [ ] **Step 2: Run redteam tests**

```bash
cd packages/evaluatorq-py && uv run pytest tests/redteam/ -v -q 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/evaluatorq/redteam/backends/openai.py \
        src/evaluatorq/redteam/backends/orq.py \
        src/evaluatorq/redteam/adaptive/capability_classifier.py \
        src/evaluatorq/redteam/adaptive/evaluator.py \
        src/evaluatorq/redteam/adaptive/objective_generator.py \
        src/evaluatorq/redteam/adaptive/orchestrator.py \
        src/evaluatorq/redteam/adaptive/pipeline.py
git commit -m "refactor(evaluatorq-py): re-point redteam blast-radius to common/tracing (RES-899)"
```

---

## Task 8: Re-point `openresponses/target.py`

Remove the `simulation.tracing` import; use `openresponses.tracing` instead.

**Files:**
- Modify: `src/evaluatorq/openresponses/target.py`

- [ ] **Step 1: Update the import block**

Replace:
```python
from evaluatorq.simulation.tracing import (
    record_openresponses_request,
    record_openresponses_response,
    with_llm_span,
)
```

With:
```python
from evaluatorq.openresponses.tracing import (
    record_openresponses_request,
    record_openresponses_response,
    with_llm_span,
)
```

- [ ] **Step 2: Run openresponses tests**

```bash
cd packages/evaluatorq-py && uv run pytest tests/openresponses/ tests/redteam/test_orq_responses_target_as_agent_target.py -v -q 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/evaluatorq/openresponses/target.py
git commit -m "refactor(evaluatorq-py): openresponses/target.py — import from openresponses/tracing, no simulation dependency (RES-899)"
```

---

## Task 9: Update test files — remove shim dependency

Now that all source files import from the correct modules, update tests to import from the final locations (not through the re-export shims). Also update truncation assertions that expected the old 2000-char limit.

**Files:**
- Modify: `tests/redteam/test_truncate_for_span.py`
- Modify: `tests/simulation/test_tracing.py`

- [ ] **Step 1: Update `tests/redteam/test_truncate_for_span.py`**

Change all imports from `evaluatorq.redteam.tracing` to `evaluatorq.common.tracing`:

```python
# Before:
from evaluatorq.redteam.tracing import (
    _DEFAULT_SPAN_MAX_TEXT_CHARS,
    _default_span_max_text_chars,
    truncate_for_span,
    _TRUNCATION_MARKER,
)

# After:
from evaluatorq.common.tracing import (
    _DEFAULT_SPAN_MAX_TEXT_CHARS,
    _default_span_max_text_chars,
    truncate_for_span,
    _TRUNCATION_MARKER,
)
```

Do a global replace of `evaluatorq.redteam.tracing` → `evaluatorq.common.tracing` in this file.

- [ ] **Step 2: Update `tests/simulation/test_tracing.py`**

a) Change imports that reference moved helpers:

```python
# Lines that import record_*, get_trace_context_headers — change to common.tracing:
from evaluatorq.common.tracing import (
    get_trace_context_headers,
    record_llm_input,
    record_llm_output,
    record_llm_response,
    record_token_usage,
)
# Keep with_llm_span, with_simulation_span from simulation.tracing (unchanged)
```

b) Update the truncation test (`test_record_llm_input_truncates_long_content`). The old assertion expected 2000-char limit with "…" marker. New behavior uses 8192-char limit with "... [truncated]" marker:

```python
@pytest.mark.asyncio
async def test_record_llm_input_truncates_long_content(
    span_collector: _CollectingExporter,
):
    from evaluatorq.common.tracing import record_llm_input, _DEFAULT_SPAN_MAX_TEXT_CHARS
    from evaluatorq.simulation.tracing import with_llm_span

    long_content = "x" * 20_000
    async with with_llm_span(model="openai/gpt-4o") as span:
        record_llm_input(span, [{"role": "user", "content": long_content}])

    a = _attrs(_find(span_collector, "chat openai/gpt-4o"))
    serialized = a["gen_ai.input.messages"]
    parsed = json.loads(serialized)
    # Content truncated at 8192 chars with '... [truncated]' marker
    assert len(parsed[0]["content"]) == _DEFAULT_SPAN_MAX_TEXT_CHARS
    assert parsed[0]["content"].endswith("... [truncated]")
```

c) Split the `test_helpers_noop_when_tracing_disabled` import:

```python
@pytest.mark.asyncio
async def test_helpers_noop_when_tracing_disabled():
    from evaluatorq.common.tracing import (
        get_trace_context_headers,
        record_llm_input,
        record_llm_response,
        record_token_usage,
    )
    from evaluatorq.simulation.tracing import with_llm_span, with_simulation_span
    # ... rest of test unchanged
```

d) Update the `record_openresponses_request` test — it currently imports from `simulation.tracing`. After Task 4, the re-export from simulation.tracing was removed. Re-point to `openresponses.tracing`:

```python
@pytest.mark.asyncio
async def test_record_openresponses_request_sets_max_tokens(
    span_collector: _CollectingExporter,
):
    from evaluatorq.openresponses.tracing import record_openresponses_request
    from evaluatorq.simulation.tracing import with_llm_span
    # ... rest of test body unchanged
```

- [ ] **Step 3: Run the updated tests**

```bash
cd packages/evaluatorq-py && uv run pytest tests/redteam/test_truncate_for_span.py tests/simulation/test_tracing.py -v 2>&1 | tail -30
```

Expected: all tests pass, including the truncation test with new 8192 limit.

- [ ] **Step 4: Commit**

```bash
git add tests/redteam/test_truncate_for_span.py tests/simulation/test_tracing.py
git commit -m "test(evaluatorq-py): re-point test imports to common/tracing; update truncation assertions (RES-899)"
```

---

## Task 10: Remove re-export shims and final verification

Now that all importers are re-pointed, remove the `__all__` re-export blocks from domain tracing files that were there only for temporary backward compat. Then run full verification.

**Files:**
- Modify: `src/evaluatorq/simulation/tracing.py` (remove re-export imports for things in common)
- Modify: `src/evaluatorq/redteam/tracing.py` (remove re-export imports)

- [ ] **Step 1: Clean up `simulation/tracing.py`**

Remove the `from evaluatorq.common.tracing import (...)` block at the top and the `__all__` list. The file should now contain only `with_simulation_span`, `with_llm_span`, `_derive_provider`, `_PROVIDER_ALIASES`, `_otel_import_warned`, and their supporting type aliases `AttrValue`/`AttrMap` (which can stay as local aliases for backward compat in test patches, or import from common — keep them as local type aliases since they're still used in function signatures).

Actually: keep the `AttrValue`/`AttrMap` type aliases and import `AttrMap` from common for use in `with_simulation_span`. The key removal is the re-exported functions (`record_*`, `get_trace_context_headers`, `set_span_attrs`).

Final `simulation/tracing.py` top of file should import from common only what it uses internally (nothing — the domain file's functions don't call `record_*`). Import `AttrMap`/`AttrValue` from common to use in type hints.

- [ ] **Step 2: Clean up `redteam/tracing.py`**

Remove the re-export `__all__` list. The imports from `evaluatorq.common.tracing` that are still needed by `_sanitize_messages` and `with_llm_span` (specifically `truncate_for_span`) can stay. Remove unused re-exports (`record_llm_response`, `record_token_usage`, `set_span_attrs`).

Final redteam/tracing.py imports from common:
```python
from evaluatorq.common.tracing import truncate_for_span
```
(Only `truncate_for_span` is used internally in `_sanitize_messages`.)

- [ ] **Step 3: Run basedpyright**

```bash
cd packages/evaluatorq-py && uv run basedpyright 2>&1 | tail -10
```

Expected: 0 errors.

- [ ] **Step 4: Run full test suite**

```bash
cd packages/evaluatorq-py && uv run pytest -m 'not integration' -q 2>&1 | tail -15
```

Expected: all tests pass, 0 failures.

- [ ] **Step 5: Commit**

```bash
git add src/evaluatorq/simulation/tracing.py src/evaluatorq/redteam/tracing.py
git commit -m "refactor(evaluatorq-py): remove re-export shims from domain tracing modules (RES-899)"
```

- [ ] **Step 6: Final report**

Run and record output for the PR description:

```bash
cd packages/evaluatorq-py && uv run basedpyright 2>&1 | tail -5
cd packages/evaluatorq-py && uv run pytest -m 'not integration' -q 2>&1 | tail -5
```

Both must show: 0 errors / all tests passed.

---

## Self-Review Checklist

- [x] `truncate_for_span` + env helpers in `common/tracing.py` ✓
- [x] `_capture_message_content` in `common/tracing.py` ✓
- [x] `record_token_usage` superset (OTel names + aliases + bare keys + calls + cache) ✓
- [x] `record_llm_response` superset (duck-typed, dual-shape, capture gate, output_content override, reasoning_tokens) ✓
- [x] `record_llm_input`, `record_llm_output`, `set_span_attrs`, `get_trace_context_headers` in common ✓
- [x] `record_openresponses_request/response` in `openresponses/tracing.py` (deferred from RES-897) ✓
- [x] `with_simulation_span`, `with_llm_span` stay in `simulation/tracing.py` ✓
- [x] `with_redteam_span`, `with_llm_span` stay in `redteam/tracing.py` ✓
- [x] `openresponses/target.py` no longer imports from `simulation.tracing` ✓
- [x] All blast-radius files re-pointed ✓
- [x] Behavior delta 1 documented: truncation 2000→8192, "…"→"... [truncated]" ✓
- [x] Behavior delta 2 documented: redteam now honors capture gate ✓
- [x] Required test: attr superset for Chat Completions + Responses API shapes in `tests/common/test_tracing.py` ✓
- [x] `basedpyright` 0 errors ✓ (final step)
- [x] pytest non-integration green ✓ (final step)
