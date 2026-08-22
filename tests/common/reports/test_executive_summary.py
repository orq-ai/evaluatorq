from __future__ import annotations

from typing import Any, cast

import pytest
from openai import AsyncOpenAI

from evaluatorq.common.reports.executive_summary import (
    EXECUTIVE_SUMMARY_SYSTEM_PROMPT,
    generate_executive_summary,
    truncate_text,
)


class _StubMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _StubChoice:
    def __init__(self, content: str | None) -> None:
        self.message = _StubMessage(content)


class _StubResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [_StubChoice(content)]


class _StubCompletions:
    def __init__(self, content: str | None, *, raise_exc: Exception | None = None) -> None:
        self._content = content
        self._raise = raise_exc
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _StubResponse:
        self.calls.append(kwargs)
        if self._raise is not None:
            raise self._raise
        return _StubResponse(self._content)


class _StubChat:
    def __init__(self, completions: _StubCompletions) -> None:
        self.completions = completions


class _StubClient:
    def __init__(self, content: str | None, *, raise_exc: Exception | None = None) -> None:
        self.chat = _StubChat(_StubCompletions(content, raise_exc=raise_exc))


def test_truncate_text():
    assert truncate_text('hello   world') == 'hello world'
    assert truncate_text('x' * 300, limit=10) == 'xxxxxxxxxx…'


@pytest.mark.asyncio
async def test_generate_returns_prose_and_passes_prompt():
    client = _StubClient('  Across 10 attacks, the agent resisted 80%.  ')
    out = await generate_executive_summary(
        'Total attacks: 10',
        llm_client=cast(AsyncOpenAI, cast(object, client)),
        model='openai/gpt-4o-mini',
        temperature=0.3,
        extra_body={'foo': 'bar'},
        extra_kwargs={'seed': 7},
    )
    assert out.text == 'Across 10 attacks, the agent resisted 80%.'
    call = client.chat.completions.calls[0]
    assert call['model'] == 'openai/gpt-4o-mini'
    assert call['temperature'] == 0.3
    assert call['messages'][0]['content'] == EXECUTIVE_SUMMARY_SYSTEM_PROMPT
    assert call['messages'][1]['content'] == 'Total attacks: 10'
    assert call['extra_body'] == {'foo': 'bar'}
    assert call['seed'] == 7


@pytest.mark.asyncio
async def test_generate_returns_none_on_blank_facts():
    client = _StubClient('should not be called')
    out = await generate_executive_summary('   ', llm_client=cast(AsyncOpenAI, cast(object, client)), model='m')
    assert out.text is None
    assert client.chat.completions.calls == []


@pytest.mark.asyncio
async def test_generate_returns_none_on_empty_completion():
    client = _StubClient(None)
    out = await generate_executive_summary('facts', llm_client=cast(AsyncOpenAI, cast(object, client)), model='m')
    assert out.text is None


@pytest.mark.asyncio
async def test_generate_returns_none_on_exception():
    client = _StubClient(None, raise_exc=RuntimeError('boom'))
    out = await generate_executive_summary('facts', llm_client=cast(AsyncOpenAI, cast(object, client)), model='m')
    assert out.text is None


@pytest.mark.asyncio
async def test_extra_kwargs_temperature_reaches_the_call_instead_of_raising_type_error():
    """``extra_kwargs={'temperature': ...}`` is the documented escape hatch for
    reasoning-class models that reject a lowered ``temperature=``. The old
    hand-built-dict shape splatted ``extra_kwargs`` next to an explicit
    ``temperature=`` keyword, which raised ``TypeError: got multiple values for
    keyword argument`` — swallowed by the blanket ``except Exception`` into a
    silently ``None`` summary. `LLMCallConfig.completion_params` merges
    ``extra_kwargs`` last, so the caller-supplied value must win instead.
    """
    client = _StubClient('Some summary.')
    out = await generate_executive_summary(
        'Total attacks: 10',
        llm_client=cast(AsyncOpenAI, cast(object, client)),
        model='openai/gpt-5.6-luna',
        temperature=0.0,
        extra_kwargs={'temperature': 1},
    )
    assert out.text == 'Some summary.'
    call = client.chat.completions.calls[0]
    assert call['temperature'] == 1


@pytest.mark.asyncio
async def test_extra_kwargs_extra_body_is_rejected_not_clobbered():
    """``extra_body`` is a structural, call-site-owned field (it carries the Orq
    router retry body); a caller routing it through ``extra_kwargs`` instead of
    the dedicated ``extra_body=`` parameter must be rejected, not silently
    allowed to clobber the router-supplied retry config.
    """
    client = _StubClient('unused')
    out = await generate_executive_summary(
        'Total attacks: 10',
        llm_client=cast(AsyncOpenAI, cast(object, client)),
        model='m',
        extra_body={'retry': {'count': 3}},
        extra_kwargs={'extra_body': {'malicious': True}},
    )
    # The ValueError from check_reserved_keys is caught by the module's own
    # best-effort `except Exception`, so the contract observable from the
    # public function is "no summary produced", not a raised exception — but
    # the call must never reach the client with the clobbered body.
    assert out.text is None
    assert client.chat.completions.calls == []
