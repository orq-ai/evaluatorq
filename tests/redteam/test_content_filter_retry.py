"""Tests for the shared content-filter detection + regeneration helper, and its use in
the live objective-generation call (objective_generator)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APIStatusError

from evaluatorq.common.content_filter import (
    classify_finish_reason,
    is_content_filter_error,
    regenerate_on_content_filter,
)


def _cf_error(code: str = 'content_filter') -> APIStatusError:
    """A real ``APIStatusError`` carrying a structured ``code`` (how Azure surfaces a filter).

    Built from a genuine httpx.Response so ``isinstance(..., APIStatusError)`` holds and the
    SDK derives ``.code``/``.body`` — the helper's ``except (APIConnectionError, APIStatusError)``
    catches it exactly like a live block.
    """
    request = httpx.Request('POST', 'https://router.example/v3')
    response = httpx.Response(400, request=request)
    return APIStatusError(f'blocked: {code}', response=response, body={'code': code})


class _BodyOnlyError(Exception):
    """Exception exposing only a structured ``body`` (no ``.code``) to exercise the body fallback."""

    def __init__(self, body: dict[str, object]) -> None:
        super().__init__('content filtered')
        self.body = body


def _resp(finish_reason: str | None, *, parsed: object = None, content: str | None = None) -> SimpleNamespace:
    message = SimpleNamespace(parsed=parsed, content=content)
    choice = SimpleNamespace(finish_reason=finish_reason, message=message)
    return SimpleNamespace(choices=[choice])


class TestClassifiers:
    def test_classify_only_matches_content_filter(self):
        assert classify_finish_reason('content_filter') == 'content_filter'

    @pytest.mark.parametrize('fr', ['stop', 'length', 'tool_calls', None])
    def test_classify_passes_through_non_filter(self, fr):
        assert classify_finish_reason(fr) is None

    def test_is_content_filter_error_on_code_attr(self):
        assert is_content_filter_error(_cf_error()) is True

    def test_is_content_filter_error_on_body(self):
        assert is_content_filter_error(_BodyOnlyError({'code': 'content_filter'})) is True

    def test_is_content_filter_error_false_for_other(self):
        assert is_content_filter_error(_cf_error('rate_limit')) is False
        assert is_content_filter_error(ValueError('nope')) is False


class TestRegenerateOnContentFilter:
    @pytest.mark.asyncio
    async def test_returns_first_usable_without_retry(self):
        usable = _resp('stop')
        call = AsyncMock(return_value=usable)
        result = await regenerate_on_content_filter(call, max_attempts=3)
        assert result is usable
        assert call.call_count == 1

    @pytest.mark.asyncio
    async def test_self_refusal_is_not_retried(self):
        # finish_reason='stop' is a prose refusal — must NOT be treated as a block.
        refusal = _resp('stop', content="I can't help with that.")
        call = AsyncMock(return_value=refusal)
        result = await regenerate_on_content_filter(call, max_attempts=3)
        assert result is refusal
        assert call.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_finish_reason_then_succeeds(self):
        usable = _resp('stop')
        call = AsyncMock(side_effect=[_resp('content_filter'), _resp('content_filter'), usable])
        result = await regenerate_on_content_filter(call, max_attempts=3)
        assert result is usable
        assert call.call_count == 3

    @pytest.mark.asyncio
    async def test_exhausted_finish_reason_returns_last_filtered(self):
        last = _resp('content_filter')
        call = AsyncMock(side_effect=[_resp('content_filter'), _resp('content_filter'), last])
        result = await regenerate_on_content_filter(call, max_attempts=3)
        assert result is last
        assert call.call_count == 3

    @pytest.mark.asyncio
    async def test_retries_on_error_form_then_succeeds(self):
        usable = _resp('stop')
        call = AsyncMock(side_effect=[_cf_error(), usable])
        result = await regenerate_on_content_filter(call, max_attempts=3)
        assert result is usable
        assert call.call_count == 2

    @pytest.mark.asyncio
    async def test_error_form_on_last_attempt_propagates(self):
        call = AsyncMock(side_effect=_cf_error())
        with pytest.raises(APIStatusError):
            await regenerate_on_content_filter(call, max_attempts=2)
        assert call.call_count == 2

    @pytest.mark.asyncio
    async def test_non_filter_error_propagates_without_retry(self):
        call = AsyncMock(side_effect=_cf_error('rate_limit'))
        with pytest.raises(APIStatusError):
            await regenerate_on_content_filter(call, max_attempts=3)
        assert call.call_count == 1

    @pytest.mark.asyncio
    async def test_max_attempts_clamped_to_one(self):
        usable = _resp('stop')
        call = AsyncMock(return_value=usable)
        result = await regenerate_on_content_filter(call, max_attempts=0)
        assert result is usable
        assert call.call_count == 1


class TestObjectiveGenerationRetry:
    """The live attacker-objective generation call regenerates on a content-filter block."""

    @staticmethod
    def _client(side_effect) -> AsyncMock:
        client = AsyncMock()
        client.chat.completions.parse = AsyncMock(side_effect=side_effect)
        return client

    @staticmethod
    async def _call(client, *, retries: int = 2):
        from evaluatorq.redteam.adaptive.objective_generator import _call_llm_for_objectives_single
        from evaluatorq.redteam.contracts import PIPELINE_CONFIG

        cfg = PIPELINE_CONFIG.model_copy(update={'max_content_filter_retries': retries})
        return await _call_llm_for_objectives_single(
            prompt_template='Generate {count} objectives',
            llm_client=client,
            model='openai/gpt-4o-mini',
            count=2,
            span_attributes={},
            log_label='ASI01',
            cfg=cfg,
        )

    @pytest.mark.asyncio
    async def test_retries_then_returns_objectives(self):
        parsed = SimpleNamespace(objectives=[SimpleNamespace(), SimpleNamespace()])
        client = self._client([_resp('content_filter'), _resp('stop', parsed=parsed)])
        result = await self._call(client)
        assert len(result) == 2
        assert client.chat.completions.parse.call_count == 2

    @pytest.mark.asyncio
    async def test_exhausted_200_filter_degrades_to_empty(self):
        client = self._client([_resp('content_filter'), _resp('content_filter'), _resp('content_filter')])
        result = await self._call(client, retries=2)
        assert result == []
        assert client.chat.completions.parse.call_count == 3

    @pytest.mark.asyncio
    async def test_error_form_filter_degrades_to_empty(self):
        client = self._client(_cf_error())
        result = await self._call(client, retries=1)
        assert result == []
        assert client.chat.completions.parse.call_count == 2

    @pytest.mark.asyncio
    async def test_non_filter_status_error_propagates(self):
        client = self._client(_cf_error('rate_limit'))
        with pytest.raises(APIStatusError):
            await self._call(client)
        assert client.chat.completions.parse.call_count == 1
