"""Unit tests for evaluatorq.redteam.tracing module."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
async def test_span_noop_without_otel():
    """with_redteam_span yields None when get_tracer() returns None."""
    with patch("evaluatorq.redteam.tracing.get_tracer", return_value=None):
        from evaluatorq.redteam.tracing import with_redteam_span

        async with with_redteam_span("orq.redteam.test") as span:
            assert span is None


@pytest.mark.asyncio
async def test_span_records_exception():
    """Exception propagates and span gets ERROR status when tracer is available."""
    mock_span = MagicMock()
    # MagicMock auto-creates __enter__/__exit__; configure __enter__ to return the span
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)

    mock_context_manager = MagicMock()
    mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
    mock_context_manager.__exit__ = MagicMock(return_value=False)

    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_context_manager

    with patch("evaluatorq.redteam.tracing.get_tracer", return_value=mock_tracer):
        from evaluatorq.redteam.tracing import with_redteam_span

        with pytest.raises(ValueError, match="test error"):
            async with with_redteam_span("orq.redteam.test") as span:
                raise ValueError("test error")

        # Verify span recorded the exception, error.type, and was set to ERROR status
        mock_span.set_attribute.assert_any_call("error.type", "ValueError")
        mock_span.record_exception.assert_called_once()
        mock_span.set_status.assert_called()


def test_set_span_attrs_noop():
    """set_span_attrs(None, ...) is a safe no-op."""
    from evaluatorq.redteam.tracing import set_span_attrs

    # Should not raise when span is None
    set_span_attrs(None, {"key": "value", "number": 42})


def test_set_span_attrs_with_span():
    """set_span_attrs sets attributes on a real span, skipping None values."""
    from evaluatorq.redteam.tracing import set_span_attrs

    mock_span = MagicMock()
    set_span_attrs(
        mock_span,
        {
            "orq.redteam.category": "ASI01",
            "orq.redteam.empty": None,
            "orq.redteam.turns": 3,
        },
    )
    # None values should be skipped
    assert mock_span.set_attribute.call_count == 2
    mock_span.set_attribute.assert_any_call("orq.redteam.category", "ASI01")
    mock_span.set_attribute.assert_any_call("orq.redteam.turns", 3)


def test_record_token_usage_noop():
    """record_token_usage(None, ...) is a safe no-op."""
    from evaluatorq.redteam.tracing import record_token_usage

    # Should not raise when span is None
    record_token_usage(None, prompt_tokens=100, completion_tokens=50, total_tokens=150, calls=1)


def test_record_token_usage_with_span():
    """record_token_usage sets all token attributes on a span (both naming conventions)."""
    from evaluatorq.redteam.tracing import record_token_usage

    mock_span = MagicMock()
    record_token_usage(mock_span, prompt_tokens=100, completion_tokens=50, total_tokens=150, calls=2)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.input_tokens", 100)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.output_tokens", 50)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.prompt_tokens", 100)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.completion_tokens", 50)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.total_tokens", 150)
    mock_span.set_attribute.assert_any_call("total_tokens", 150)


def test_record_llm_response_sets_both_token_conventions():
    """record_llm_response sets both OTel and OpenAI-style token attributes."""
    from evaluatorq.redteam.tracing import record_llm_response

    mock_span = MagicMock()

    # Build a mock response with usage including cached tokens
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50
    mock_prompt_details = MagicMock()
    mock_prompt_details.cached_tokens = 30
    mock_usage.prompt_tokens_details = mock_prompt_details
    mock_completion_details = MagicMock()
    mock_completion_details.reasoning_tokens = 10
    mock_usage.completion_tokens_details = mock_completion_details

    mock_response = MagicMock()
    mock_response.id = "resp-123"
    mock_response.model = "gpt-5-mini"
    mock_response.usage = mock_usage
    mock_response.choices = [MagicMock(finish_reason="stop")]

    record_llm_response(mock_span, mock_response, output_content="hello")

    mock_span.set_attribute.assert_any_call("gen_ai.usage.input_tokens", 100)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.output_tokens", 50)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.prompt_tokens", 100)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.completion_tokens", 50)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.prompt_tokens_details.cached_tokens", 30)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.completion_tokens_details.reasoning_tokens", 10)
    mock_span.set_attribute.assert_any_call("gen_ai.response.id", "resp-123")
    mock_span.set_attribute.assert_any_call("gen_ai.response.model", "gpt-5-mini")


def test_record_llm_response_without_cached_tokens():
    """record_llm_response works when usage has no detailed breakdowns."""
    from evaluatorq.redteam.tracing import record_llm_response

    mock_span = MagicMock()

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 80
    mock_usage.completion_tokens = 40
    mock_usage.prompt_tokens_details = None
    mock_usage.completion_tokens_details = None

    mock_response = MagicMock()
    mock_response.id = None
    mock_response.model = None
    mock_response.usage = mock_usage
    mock_response.choices = []

    record_llm_response(mock_span, mock_response)

    mock_span.set_attribute.assert_any_call("gen_ai.usage.input_tokens", 80)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.prompt_tokens", 80)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.output_tokens", 40)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.completion_tokens", 40)
    # No cached_tokens attribute should be set
    for call in mock_span.set_attribute.call_args_list:
        assert "cached_tokens" not in str(call)


@pytest.mark.asyncio
async def test_span_with_mock_tracer():
    """When tracer is available, span is yielded and attributes are set."""
    mock_span = MagicMock()

    mock_context_manager = MagicMock()
    mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
    mock_context_manager.__exit__ = MagicMock(return_value=False)

    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_context_manager

    with patch("evaluatorq.redteam.tracing.get_tracer", return_value=mock_tracer):
        from evaluatorq.redteam.tracing import with_redteam_span

        async with with_redteam_span("orq.redteam.test", {"key": "value"}) as span:
            assert span is mock_span

        mock_tracer.start_as_current_span.assert_called_once()
        call_args = mock_tracer.start_as_current_span.call_args
        assert call_args[0][0] == "orq.redteam.test"  # span name


@pytest.mark.asyncio
async def test_llm_span_name_is_operation_space_model():
    """LLM span name follows OTel spec: '{operation} {model}'."""
    mock_span = MagicMock()

    mock_context_manager = MagicMock()
    mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
    mock_context_manager.__exit__ = MagicMock(return_value=False)

    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_context_manager

    with patch("evaluatorq.redteam.tracing.get_tracer", return_value=mock_tracer):
        from evaluatorq.redteam.tracing import with_llm_span

        async with with_llm_span(model="gpt-5-mini") as span:
            assert span is mock_span

        call_args = mock_tracer.start_as_current_span.call_args
        assert call_args[0][0] == "chat gpt-5-mini"
        attrs = call_args[1]["attributes"]
        assert attrs["gen_ai.system"] == "openai"
        assert attrs["gen_ai.provider.name"] == "openai"
        assert attrs["gen_ai.request.model"] == "gpt-5-mini"
        assert attrs["gen_ai.operation.name"] == "chat"


@pytest.mark.asyncio
async def test_llm_span_name_with_provider_prefix():
    """LLM span with provider/model derives provider correctly."""
    mock_span = MagicMock()

    mock_context_manager = MagicMock()
    mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
    mock_context_manager.__exit__ = MagicMock(return_value=False)

    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_context_manager

    with patch("evaluatorq.redteam.tracing.get_tracer", return_value=mock_tracer):
        from evaluatorq.redteam.tracing import with_llm_span

        async with with_llm_span(model="azure/gpt-5-mini") as span:
            pass

        call_args = mock_tracer.start_as_current_span.call_args
        assert call_args[0][0] == "chat azure/gpt-5-mini"
        attrs = call_args[1]["attributes"]
        assert attrs["gen_ai.system"] == "azure"
        assert attrs["gen_ai.provider.name"] == "azure"


@pytest.mark.asyncio
async def test_llm_span_error_type_on_exception():
    """LLM span sets error.type attribute when an exception occurs."""
    mock_span = MagicMock()

    mock_context_manager = MagicMock()
    mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
    mock_context_manager.__exit__ = MagicMock(return_value=False)

    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_context_manager

    with patch("evaluatorq.redteam.tracing.get_tracer", return_value=mock_tracer):
        from evaluatorq.redteam.tracing import with_llm_span

        with pytest.raises(RuntimeError, match="llm failed"):
            async with with_llm_span(model="gpt-5-mini") as span:
                raise RuntimeError("llm failed")

        mock_span.set_attribute.assert_any_call("error.type", "RuntimeError")
        mock_span.record_exception.assert_called_once()


@pytest.mark.asyncio
async def test_llm_span_noop_without_otel():
    """with_llm_span yields None when get_tracer() returns None."""
    with patch("evaluatorq.redteam.tracing.get_tracer", return_value=None):
        from evaluatorq.redteam.tracing import with_llm_span

        async with with_llm_span(model="gpt-5-mini") as span:
            assert span is None
