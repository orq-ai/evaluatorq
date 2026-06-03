"""Coverage for review-surfaced gaps in PR #114.

Groups:
- SendResult equality and metadata field propagation
- TokenUsage __radd__ NotImplemented fallthrough, ge=0 validation,
  from_completion zero-total fallback
- truncate_for_span negative-arg warn-and-return path
- _default_span_max_text_chars LRU memoization (no re-read without cache_clear)
- OpenAI backend response_id / finish_reason / model propagation
- Vercel _parse_data_stream all-zeros usage guard
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from evaluatorq.contracts import Message
from evaluatorq.redteam.contracts import SendResult, TokenUsage


# ---------------------------------------------------------------------------
# SendResult: equality + frozen-slots semantics
# ---------------------------------------------------------------------------
class TestSendResultEquality:
    def test_equal_when_fields_match(self) -> None:
        a = SendResult(text="hi", usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, calls=1))
        b = SendResult(text="hi", usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, calls=1))
        assert a == b

    def test_not_equal_when_text_differs(self) -> None:
        assert SendResult(text="a") != SendResult(text="b")

    def test_not_equal_when_usage_differs(self) -> None:
        a = SendResult(text="x", usage=TokenUsage(prompt_tokens=1))
        b = SendResult(text="x", usage=TokenUsage(prompt_tokens=2))
        assert a != b

    def test_metadata_fields_round_trip(self) -> None:
        result = SendResult(
            text="ok",
            model="gpt-4o",
            response_id="resp-123",
            finish_reason="stop",
        )
        assert result.model == "gpt-4o"
        assert result.response_id == "resp-123"
        assert result.finish_reason == "stop"


# ---------------------------------------------------------------------------
# TokenUsage: arithmetic edge cases + validation constraints
# ---------------------------------------------------------------------------
class TestTokenUsageArithmeticEdges:
    def test_radd_returns_notimplemented_for_string(self) -> None:
        usage = TokenUsage(prompt_tokens=1)
        # __radd__ with a non-int/non-TokenUsage left operand must return
        # NotImplemented so Python can fall back to the left operand's
        # __add__ (and ultimately TypeError) instead of silently producing
        # a misleading result.
        assert usage.__radd__("not a number") is NotImplemented  # pyright: ignore[reportArgumentType]

    def test_radd_returns_notimplemented_for_nonzero_int(self) -> None:
        usage = TokenUsage(prompt_tokens=1)
        # Only the int=0 case from sum()'s seed should be handled; any other
        # int must fall through.
        assert usage.__radd__(5) is NotImplemented


class TestTokenUsageValidation:
    @pytest.mark.parametrize(
        "field",
        ["prompt_tokens", "completion_tokens", "total_tokens", "calls"],
    )
    def test_negative_values_rejected(self, field: str) -> None:
        # ge=0 invariant: a buggy provider returning negative counts must be
        # caught at construction, not silently aggregated into report totals.
        with pytest.raises(ValidationError):
            TokenUsage(**{field: -1})


class TestTokenUsageFromCompletion:
    def test_zero_total_falls_back_to_sum(self) -> None:
        # Provider reports total_tokens=0 with non-zero prompt/completion
        # (e.g. some providers return 0 instead of None when omitted).
        # Mirror the > 0 guard used by other integrations and fall back
        # to prompt+completion rather than propagating the zero.
        usage_obj = MagicMock(spec=["prompt_tokens", "completion_tokens", "total_tokens"])
        usage_obj.prompt_tokens = 10
        usage_obj.completion_tokens = 5
        usage_obj.total_tokens = 0

        completion = MagicMock(spec=["usage"])
        completion.usage = usage_obj

        result = TokenUsage.from_completion(completion)
        assert result is not None
        assert result.total_tokens == 15  # fallback applied

    def test_provider_total_preserved_when_positive(self) -> None:
        # Provider-reported total > 0 wins, even when it doesn't equal
        # prompt+completion (e.g. cached/reasoning tokens not broken out).
        usage_obj = MagicMock(spec=["prompt_tokens", "completion_tokens", "total_tokens"])
        usage_obj.prompt_tokens = 10
        usage_obj.completion_tokens = 5
        usage_obj.total_tokens = 42

        completion = MagicMock(spec=["usage"])
        completion.usage = usage_obj

        result = TokenUsage.from_completion(completion)
        assert result is not None
        assert result.total_tokens == 42


# ---------------------------------------------------------------------------
# truncate_for_span + _default_span_max_text_chars
# ---------------------------------------------------------------------------
class TestTruncateForSpanNegativeArg:
    def test_negative_max_chars_raises(self) -> None:
        # truncate_for_span raises ValueError for negative max_chars.
        # _default_span_max_text_chars (env var path) warns-and-returns-None instead;
        # the two codepaths differ by design.
        from evaluatorq.redteam.tracing import truncate_for_span

        with pytest.raises(ValueError, match="non-negative"):
            truncate_for_span("x" * 50, max_chars=-1)


class TestSpanMaxTextCharsCacheMemoization:
    def test_repeated_calls_do_not_re_read_env(self, monkeypatch: Any) -> None:
        # @lru_cache(maxsize=1) is load-bearing: without it every span
        # attribute write would re-os.getenv and re-parse. Verify a runtime
        # env change does NOT take effect without explicit cache_clear().
        from evaluatorq.common.tracing import _default_span_max_text_chars

        _default_span_max_text_chars.cache_clear()

        monkeypatch.setenv("EVALUATORQ_SPAN_MAX_TEXT_CHARS", "100")
        first = _default_span_max_text_chars()
        assert first == 100

        # Change the env without clearing — cached value must stick.
        monkeypatch.setenv("EVALUATORQ_SPAN_MAX_TEXT_CHARS", "999")
        second = _default_span_max_text_chars()
        assert second == 100  # cached, not re-read

        _default_span_max_text_chars.cache_clear()


# ---------------------------------------------------------------------------
# OpenAI backend: response_id / finish_reason / model propagation through SendResult
# ---------------------------------------------------------------------------
class TestOpenAIBackendMetadataPropagation:
    @pytest.mark.asyncio
    async def test_send_result_carries_response_id_finish_reason_model(self) -> None:
        from evaluatorq.redteam.backends.openai import OpenAIModelTarget

        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="hi"), finish_reason="length")],
                usage=MagicMock(prompt_tokens=2, completion_tokens=3, total_tokens=5),
                id="resp-abc",
                model="gpt-from-server",
            )
        )
        target = OpenAIModelTarget(model="gpt-x", client=client)
        result = await target.respond([Message(role="user", content="hi")])
        assert result.response_id == "resp-abc"
        assert result.finish_reason == "length"
        # backend prefers server-reported model over the configured one
        assert result.model == "gpt-from-server"


# ---------------------------------------------------------------------------
# Vercel _parse_data_stream: all-zeros usage guard
# ---------------------------------------------------------------------------
class TestVercelDataStreamZeroUsageGuard:
    def test_all_zero_usage_yields_no_token_usage(self) -> None:
        # Stream parser used to emit calls=1 with zero tokens for every
        # all-zeros usage block, polluting aggregate counts. Mirror the
        # JSON parser's `p > 0 or c > 0` guard.
        from evaluatorq.integrations.vercel_ai_sdk_integration.target import (
            _parse_data_stream,
        )

        body = (
            '0:"hello"\n'
            'd:{"usage":{"promptTokens":0,"completionTokens":0,"totalTokens":0}}\n'
        )
        text, usage = _parse_data_stream(body)
        assert text == "hello"
        assert usage is None  # no spurious zero entry
