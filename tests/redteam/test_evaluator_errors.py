"""Unit tests for OWASPEvaluator._run_evaluator() error paths.

Covers:
1. JSON parse failure from LLM — invalid JSON returns an inconclusive result
2. APIConnectionError re-raise — propagates out of _run_evaluator()
3. APIStatusError re-raise — propagates out of _run_evaluator()
4. Empty LLM response content — handled gracefully (inconclusive)
5. resolve_category_safe fallback — category not in registry falls back to
   get_evaluator_for_category(); unknown category returns passed=None
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from evaluatorq.redteam.contracts import EvaluationResult, Vulnerability


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_client(content: str | None = None) -> AsyncMock:
    """Return a mock AsyncOpenAI client whose chat.completions.create returns
    a response with the given message content."""
    mock_message = MagicMock()
    mock_message.content = content  # may be None or a string

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    return mock_client


def _patch_tracing():
    """Context-manager factory that suppresses tracing side-effects."""
    import contextlib

    @contextlib.asynccontextmanager
    async def _noop_span(*args: Any, **kwargs: Any):
        yield MagicMock()

    return (
        patch("evaluatorq.redteam.adaptive.evaluator.with_llm_span", side_effect=_noop_span),
        patch("evaluatorq.redteam.adaptive.evaluator.record_llm_response"),
    )


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestRunEvaluatorErrorPaths:
    """Error-path tests for OWASPEvaluator._run_evaluator()."""

    # ------------------------------------------------------------------
    # 1. JSON parse failure → inconclusive result, no crash
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_invalid_json_returns_inconclusive_result(self):
        """When the LLM returns malformed JSON, _run_evaluator() must catch the
        parse error and return an inconclusive EvaluationResult (passed=None)
        instead of propagating the exception."""
        from evaluatorq.redteam.adaptive.evaluator import OWASPEvaluator

        mock_client = _make_llm_client(content="this is not valid json {{{{")
        evaluator = OWASPEvaluator(evaluator_model="test-model", llm_client=mock_client)

        span_patch, record_patch = _patch_tracing()
        with span_patch, record_patch:
            result = await evaluator.evaluate_vulnerability(
                vuln=Vulnerability.GOAL_HIJACKING,
                messages=[{"role": "user", "content": "attack prompt"}],
                response="agent response",
            )

        assert isinstance(result, EvaluationResult)
        assert result.passed is None
        assert result.evaluator_id == "goal_hijacking"
        assert "error" in (result.raw_output or {})

    # ------------------------------------------------------------------
    # 2. APIConnectionError — must propagate
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_api_connection_error_propagates(self):
        """APIConnectionError raised by the LLM client must NOT be swallowed;
        it should propagate out of _run_evaluator() to the caller."""
        from evaluatorq.redteam.adaptive.evaluator import OWASPEvaluator

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=APIConnectionError(request=MagicMock())
        )
        evaluator = OWASPEvaluator(evaluator_model="test-model", llm_client=mock_client)

        span_patch, record_patch = _patch_tracing()
        with span_patch, record_patch, pytest.raises(APIConnectionError):
            await evaluator.evaluate_vulnerability(
                vuln=Vulnerability.GOAL_HIJACKING,
                messages=[{"role": "user", "content": "attack prompt"}],
                response="agent response",
            )

    # ------------------------------------------------------------------
    # 3. APIStatusError — must propagate
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_api_status_error_propagates(self):
        """APIStatusError raised by the LLM client must NOT be swallowed;
        it should propagate out of _run_evaluator() to the caller."""
        from evaluatorq.redteam.adaptive.evaluator import OWASPEvaluator

        http_response = httpx.Response(
            status_code=429,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=APIStatusError(
                message="rate limit exceeded",
                response=http_response,
                body={"error": {"message": "rate limit exceeded"}},
            )
        )
        evaluator = OWASPEvaluator(evaluator_model="test-model", llm_client=mock_client)

        span_patch, record_patch = _patch_tracing()
        with span_patch, record_patch, pytest.raises(APIStatusError):
            await evaluator.evaluate_vulnerability(
                vuln=Vulnerability.GOAL_HIJACKING,
                messages=[{"role": "user", "content": "attack prompt"}],
                response="agent response",
            )

    # ------------------------------------------------------------------
    # 4. Empty LLM response content — graceful fallback
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_empty_llm_content_returns_inconclusive_result(self):
        """When the LLM returns None (empty) content, the code falls back to
        '{}' and Pydantic validation fails.  The exception handler must catch
        that and return an inconclusive EvaluationResult rather than crashing."""
        from evaluatorq.redteam.adaptive.evaluator import OWASPEvaluator

        # content=None triggers the `or '{}'` fallback, which is valid JSON
        # but fails EvaluatorResponsePayload validation (missing required fields).
        mock_client = _make_llm_client(content=None)
        evaluator = OWASPEvaluator(evaluator_model="test-model", llm_client=mock_client)

        span_patch, record_patch = _patch_tracing()
        with span_patch, record_patch:
            result = await evaluator.evaluate_vulnerability(
                vuln=Vulnerability.GOAL_HIJACKING,
                messages=[{"role": "user", "content": "attack prompt"}],
                response="agent response",
            )

        assert isinstance(result, EvaluationResult)
        assert result.passed is None
        assert result.evaluator_id == "goal_hijacking"

    # ------------------------------------------------------------------
    # 5. resolve_category_safe fallback path in evaluate()
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_evaluate_unknown_category_returns_inconclusive_result(self):
        """When the category is not found in the vulnerability registry AND
        get_evaluator_for_category() also returns None, evaluate() must return
        an inconclusive EvaluationResult with passed=None."""
        from evaluatorq.redteam.adaptive.evaluator import OWASPEvaluator

        mock_client = AsyncMock()
        evaluator = OWASPEvaluator(evaluator_model="test-model", llm_client=mock_client)

        # Patch both registry lookups to simulate a fully unknown category.
        with (
            patch(
                "evaluatorq.redteam.adaptive.evaluator.resolve_category_safe",
                return_value=None,
            ),
            patch(
                "evaluatorq.redteam.adaptive.evaluator.get_evaluator_for_category",
                return_value=None,
            ),
        ):
            result = await evaluator.evaluate(
                category="UNKNOWN-CATEGORY-99",
                messages=[{"role": "user", "content": "attack prompt"}],
                response="agent response",
            )

        assert isinstance(result, EvaluationResult)
        assert result.passed is None
        assert result.evaluator_id == "none"
        assert "UNKNOWN-CATEGORY-99" in result.explanation

    @pytest.mark.asyncio
    async def test_evaluate_resolve_category_safe_fallback_delegates_to_category_lookup(self):
        """When resolve_category_safe() returns None but get_evaluator_for_category()
        finds an evaluator, evaluate() must invoke the evaluator via _run_evaluator()
        and return a proper EvaluationResult."""
        from evaluatorq.redteam.adaptive.evaluator import OWASPEvaluator

        # Build a mock evaluator entity with a minimal prompt attribute.
        mock_evaluator_entity = MagicMock()
        mock_evaluator_entity.prompt = (
            "Evaluate: {{output.response}} context: {{input.all_messages}}"
        )

        valid_json = '{"value": false, "explanation": "The agent revealed sensitive data."}'
        mock_client = _make_llm_client(content=valid_json)
        evaluator = OWASPEvaluator(evaluator_model="test-model", llm_client=mock_client)

        span_patch, record_patch = _patch_tracing()
        with (
            patch(
                "evaluatorq.redteam.adaptive.evaluator.resolve_category_safe",
                return_value=None,
            ),
            patch(
                "evaluatorq.redteam.adaptive.evaluator.get_evaluator_for_category",
                return_value=mock_evaluator_entity,
            ),
            span_patch,
            record_patch,
        ):
            result = await evaluator.evaluate(
                category="OWASP-CUSTOM01",
                messages=[{"role": "user", "content": "attack prompt"}],
                response="sensitive data leaked",
            )

        assert isinstance(result, EvaluationResult)
        # passed=False means VULNERABLE
        assert result.passed is False
        assert result.evaluator_id == "CUSTOM01"
        assert result.explanation == "The agent revealed sensitive data."

    @pytest.mark.asyncio
    async def test_evaluate_with_resolvable_category_delegates_to_evaluate_vulnerability(self):
        """When resolve_category_safe() successfully maps a category to a Vulnerability,
        evaluate() must delegate to evaluate_vulnerability() and return its result."""
        from evaluatorq.redteam.adaptive.evaluator import OWASPEvaluator

        valid_json = '{"value": true, "explanation": "The agent resisted the attack."}'
        mock_client = _make_llm_client(content=valid_json)
        evaluator = OWASPEvaluator(evaluator_model="test-model", llm_client=mock_client)

        span_patch, record_patch = _patch_tracing()
        with (
            patch(
                "evaluatorq.redteam.adaptive.evaluator.resolve_category_safe",
                return_value=Vulnerability.GOAL_HIJACKING,
            ),
            span_patch,
            record_patch,
        ):
            result = await evaluator.evaluate(
                category="ASI01",
                messages=[{"role": "user", "content": "attack prompt"}],
                response="I cannot help with that.",
            )

        assert isinstance(result, EvaluationResult)
        assert result.passed is True
        assert result.evaluator_id == "goal_hijacking"
