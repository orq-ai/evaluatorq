"""`common.redact`: batching, fail-closed behaviour, the cut-after-redaction bound and the Orq client derivation."""

from __future__ import annotations

from typing import Any

import pytest

from evaluatorq.common import redact as redact_module
from evaluatorq.common.redact import (
    BATCH_CHARS,
    EDGE_WINDOW,
    SEPARATOR,
    normalize_placeholders,
    redact_cut,
    redact_texts,
    redaction_client,
)

from .fake_orq import FakeOrq

SECRET = 'sk-live-0123456789abcdef'


def _join(head: str, tail: str, omitted: int) -> str:
    return f'{head}|{omitted}|{tail}'


@pytest.mark.asyncio
async def test_redacts_each_text_and_keeps_input_order() -> None:
    orq = FakeOrq()

    redacted = await redact_texts(['curl -H "x: sk-abcdefgh12"', 'git status', 'echo hi'], orq=orq.client)

    assert redacted == ['curl -H "x: <API_KEY_1>"', 'git status', 'echo hi']


@pytest.mark.asyncio
async def test_texts_share_one_request_and_duplicates_are_sent_once() -> None:
    orq = FakeOrq()

    redacted = await redact_texts(['a', 'b', 'a', 'c'], orq=orq.client)

    assert redacted == ['a', 'b', 'a', 'c']
    assert orq.requests == [SEPARATOR.join(['a', 'b', 'c'])]


@pytest.mark.asyncio
async def test_blank_texts_are_not_sent() -> None:
    orq = FakeOrq()

    assert await redact_texts(['', '  \n'], orq=orq.client) == ['', '  \n']
    assert orq.requests == []


@pytest.mark.asyncio
async def test_batches_stay_within_the_character_cap_and_a_giant_text_travels_alone() -> None:
    orq = FakeOrq()
    medium = [f'{index} ' + 'x' * 20_000 for index in range(5)]
    giant = 'g' * (BATCH_CHARS + 1)

    redacted = await redact_texts([*medium, giant], orq=orq.client)

    assert redacted == [*medium, giant]
    assert all(len(request) <= BATCH_CHARS for request in orq.requests if request != giant)
    assert giant in orq.requests
    assert len(orq.requests) == 4  # two medium texts per request, the fifth alone, the giant alone


@pytest.mark.asyncio
async def test_a_text_holding_the_separator_cannot_break_its_neighbours() -> None:
    orq = FakeOrq()
    hostile = f'before{SEPARATOR}after {SECRET}'

    redacted = await redact_texts(['safe', hostile], orq=orq.client)

    assert redacted == ['safe', f'before{SEPARATOR}after <API_KEY_1>']
    assert SEPARATOR.join(['safe', hostile]) not in orq.requests


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('orq', 'cause'),
    [
        (FakeOrq(error=RuntimeError('boom')), 'RuntimeError'),
        (FakeOrq(reply='one part only'), 'split into 1 parts, expected 2'),
        (FakeOrq(reply=None.__class__), 'instead of redacted text'),
    ],
    ids=['exception', 'mismatched-split', 'not-a-string'],
)
async def test_a_failed_batch_returns_none_never_the_raw_text(
    orq: FakeOrq, cause: str, caplog: pytest.LogCaptureFixture
) -> None:
    redacted = await redact_texts([f'one {SECRET}', 'two'], orq=orq.client)

    assert redacted == [None, None]
    assert cause in caplog.text
    assert SECRET not in caplog.text


@pytest.mark.asyncio
async def test_an_api_error_body_is_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    class ApiError(Exception):
        status_code = 413

    orq = FakeOrq(error=ApiError(f'rejected: {SECRET}'))

    assert await redact_texts([SECRET], orq=orq.client) == [None]
    assert 'ApiError (HTTP 413)' in caplog.text
    assert SECRET not in caplog.text


@pytest.mark.asyncio
async def test_without_an_orq_client_every_text_is_withheld_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    assert await redact_texts(['a', 'b', ' '], orq=None) == [None, None, ' ']
    assert 'PII redaction is unavailable' in caplog.text


@pytest.mark.asyncio
async def test_one_failed_batch_does_not_lose_the_others() -> None:
    first = 'a' * 30_000
    second = 'b' * 30_000
    orq = FakeOrq(error=RuntimeError('boom'), error_when=lambda text: text == second)

    assert await redact_texts([first, second], orq=orq.client) == [first, None]


@pytest.mark.asyncio
async def test_cut_redacts_before_cutting_so_a_secret_across_the_cut_leaves_no_fragment() -> None:
    orq = FakeOrq()
    text = 'a' * 140 + SECRET + ' ' + 'b' * 400 + SECRET + ' ' + 'c' * 140

    [cut] = await redact_cut([text], keep=150, join=_join, orq=orq.client)

    assert cut is not None
    assert SECRET[-10:] not in cut
    assert SECRET[:10] not in cut
    assert cut.startswith('a' * 140 + '<API_KEY_')


@pytest.mark.asyncio
async def test_cut_of_a_very_long_text_redacts_only_its_two_ends() -> None:
    orq = FakeOrq()
    text = 'a' * 140 + SECRET + ' ' + 'm' * 40_000 + SECRET + ' ' + 'c' * 140

    [cut] = await redact_cut([text], keep=150, join=_join, orq=orq.client)

    assert cut is not None
    assert len(orq.requests) == 1  # both ends travel in one batch
    assert all(len(part) <= EDGE_WINDOW for part in orq.requests[0].split(SEPARATOR))
    assert len(orq.requests[0].split(SEPARATOR)) == 2
    assert SECRET[-10:] not in cut
    assert f'|{len(text) - 300}|' in cut
    assert cut.endswith(' ' + 'c' * 140)
    assert cut.startswith('a' * 140 + '<API_KEY_')


@pytest.mark.asyncio
async def test_cut_keeps_a_short_text_whole_and_returns_none_for_a_failed_one() -> None:
    assert await redact_cut(['short'], keep=150, join=_join, orq=FakeOrq().client) == ['short']
    assert await redact_cut(['x' * 500], keep=150, join=_join, orq=None) == [None]


def test_normalize_placeholders_drops_the_numbering() -> None:
    assert normalize_placeholders('<UUID_3> and <EMAIL_ADDRESS_12> via <IP_ADDRESS_1>') == (
        '<UUID> and <EMAIL_ADDRESS> via <IP_ADDRESS>'
    )
    assert normalize_placeholders('a < b_2 > c <not_a_placeholder_1>') == 'a < b_2 > c <not_a_placeholder_1>'


class _LlmClient:
    def __init__(self, base_url: str, api_key: str | None) -> None:
        self.base_url = base_url
        self.api_key = api_key


@pytest.mark.asyncio
async def test_redaction_client_is_built_from_a_router_client_and_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[tuple[str, str]] = []
    closed: list[object] = []
    orq = object()

    def resolve(api_key: str, base_url: str) -> object:
        built.append((api_key, base_url))
        return orq

    async def close(client: object) -> None:
        closed.append(client)

    monkeypatch.setattr(redact_module, 'resolve_orq_client', resolve)
    monkeypatch.setattr(redact_module, 'close_orq_client', close)

    llm: Any = _LlmClient('https://staging.orq.ai/v3/router', 'key-1')
    async with redaction_client(llm) as client:
        assert client is orq
        assert closed == []

    assert built == [('key-1', 'https://staging.orq.ai')]
    assert closed == [orq]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'llm_client',
    [None, _LlmClient('https://api.openai.com/v1', 'key-1'), _LlmClient('https://my.orq.ai/v3/router', None)],
    ids=['none', 'not-orq', 'no-key'],
)
async def test_redaction_client_is_none_when_the_llm_client_has_no_orq_credentials(llm_client: Any) -> None:
    async with redaction_client(llm_client) as client:
        assert client is None
