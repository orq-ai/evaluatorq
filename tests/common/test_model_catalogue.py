"""Orq model catalogue: client-side pricing and model qualification (RES-1295)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from evaluatorq.common import model_catalogue as pricing
from evaluatorq.common.model_catalogue import ModelInfo
from evaluatorq.contracts import Usage


@pytest.fixture(autouse=True)
def _clear_catalogue():
    pricing.reset_catalogue_cache()
    yield
    pricing.reset_catalogue_cache()


@pytest.fixture
def _catalogue(monkeypatch: pytest.MonkeyPatch):
    async def fake_load(client=None):  # noqa: ANN001, ARG001
        return {'gpt-5-mini': ModelInfo(0.00025, 0.002, 'openai', supports_responses=True)}

    monkeypatch.setattr(pricing, '_load_catalogue', fake_load)


def _usage(**kw: object) -> Usage:
    return Usage(input_tokens=1000, output_tokens=500, total_tokens=1500, calls=1, priced_calls=0, **kw)  # pyright: ignore[reportArgumentType]


@pytest.mark.asyncio
@pytest.mark.usefixtures('_catalogue')
async def test_prices_unpriced_usage():
    priced = await pricing.price_usage(_usage(), 'gpt-5-mini')
    assert priced is not None
    assert priced.input_cost == pytest.approx(0.00025)
    assert priced.output_cost == pytest.approx(0.001)
    assert priced.total_cost == pytest.approx(0.00125)
    assert priced.priced_calls == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures('_catalogue')
async def test_strips_provider_prefix():
    priced = await pricing.price_usage(_usage(), 'openai/gpt-5-mini')
    assert priced is not None
    assert priced.total_cost == pytest.approx(0.00125)


@pytest.mark.asyncio
@pytest.mark.usefixtures('_catalogue')
async def test_leaves_provider_reported_cost_alone():
    priced = await pricing.price_usage(_usage(input_cost=1.0, output_cost=2.0, total_cost=3.0), 'gpt-5-mini')
    assert priced is not None
    assert priced.total_cost == pytest.approx(3.0)


@pytest.mark.asyncio
@pytest.mark.usefixtures('_catalogue')
async def test_unknown_model_stays_unpriced():
    priced = await pricing.price_usage(_usage(), 'some/unlisted-model')
    assert priced is not None
    assert priced.total_cost is None
    assert priced.priced_calls == 0


@pytest.mark.asyncio
async def test_no_credentials_yields_empty_table(monkeypatch: pytest.MonkeyPatch):
    # Swap the suite-wide offline cache for a real one so the loader actually runs.
    monkeypatch.setattr(pricing, '_catalogues', {})
    monkeypatch.delenv('ORQ_API_KEY', raising=False)
    priced = await pricing.price_usage(_usage(), 'gpt-5-mini')
    assert priced is not None
    assert priced.total_cost is None


@pytest.mark.asyncio
@pytest.mark.usefixtures('_catalogue')
async def test_qualifies_bare_model_with_provider():
    assert await pricing.qualified_model('gpt-5-mini') == 'openai/gpt-5-mini'


@pytest.mark.asyncio
@pytest.mark.usefixtures('_catalogue')
async def test_qualified_model_passes_through_and_reports_unknown():
    assert await pricing.qualified_model('anthropic/claude-4-opus') == 'anthropic/claude-4-opus'
    assert await pricing.qualified_model('unlisted-model') is None


def test_parse_catalogue_skips_malformed_entries():
    prices = pricing._parse_catalogue(  # pyright: ignore[reportPrivateUsage]
        [
            {'model_id': 'a', 'provider': 'openai', 'input_cost': 0.1, 'output_cost': 0.2},
            {'model_id': 'b', 'provider': 'openai', 'input_cost': None, 'output_cost': 0.2},
            {'model_id': 'c', 'input_cost': 0.1, 'output_cost': 0.2},
            {'input_cost': 0.1, 'output_cost': 0.2},
            'nonsense',
        ]
    )
    assert prices == {'a': ModelInfo(0.1, 0.2, 'openai', supports_responses=False)}


def test_parse_catalogue_prefers_the_developers_own_provider():
    prices = pricing._parse_catalogue(  # pyright: ignore[reportPrivateUsage]
        [
            {'model_id': 'm', 'provider': 'azure', 'model_developer': 'openai', 'input_cost': 1.0, 'output_cost': 2.0},
            {'model_id': 'm', 'provider': 'openai', 'model_developer': 'openai', 'input_cost': 1.0, 'output_cost': 2.0},
        ]
    )
    assert prices['m'].provider == 'openai'


# --- _parse_catalogue edge cases ---------------------------------------------


def test_parse_catalogue_rejects_non_list_payload():
    assert pricing._parse_catalogue({'data': [{'model_id': 'a', 'provider': 'openai', 'input_cost': 0.1, 'output_cost': 0.2}]}) == {}  # pyright: ignore[reportPrivateUsage]


def test_parse_catalogue_skips_bool_cost():
    # bool is an int subclass: input_cost=True must not price at $1.00/1k.
    prices = pricing._parse_catalogue(  # pyright: ignore[reportPrivateUsage]
        [{'model_id': 'a', 'provider': 'openai', 'input_cost': True, 'output_cost': 0.2}]
    )
    assert prices == {}


def test_parse_catalogue_skips_negative_cost():
    prices = pricing._parse_catalogue(  # pyright: ignore[reportPrivateUsage]
        [{'model_id': 'a', 'provider': 'openai', 'input_cost': -0.1, 'output_cost': 0.2}]
    )
    assert prices == {}


def test_parse_catalogue_skips_non_usd_currency_but_prices_empty_and_usd():
    prices = pricing._parse_catalogue(  # pyright: ignore[reportPrivateUsage]
        [
            {
                'model_id': 'eur-model',
                'provider': 'openai',
                'input_cost': 0.1,
                'output_cost': 0.2,
                'input_currency': 'eur',
            },
            {
                'model_id': 'empty-currency',
                'provider': 'openai',
                'input_cost': 0.1,
                'output_cost': 0.2,
                'input_currency': '',
            },
            {
                'model_id': 'usd-model',
                'provider': 'openai',
                'input_cost': 0.1,
                'output_cost': 0.2,
                'input_currency': 'usd',
                'output_currency': 'usd',
            },
        ]
    )
    assert set(prices) == {'empty-currency', 'usd-model'}


def test_parse_catalogue_reads_supports_responses_from_metadata():
    prices = pricing._parse_catalogue(  # pyright: ignore[reportPrivateUsage]
        [
            {
                'model_id': 'a',
                'provider': 'openai',
                'input_cost': 0.1,
                'output_cost': 0.2,
                'metadata': {'supports_responses_api': True},
            }
        ]
    )
    assert prices['a'].supports_responses is True


def test_parse_catalogue_warns_when_nonempty_payload_yields_zero_entries(monkeypatch: pytest.MonkeyPatch):
    warnings: list[str] = []
    monkeypatch.setattr(pricing.logger, 'warning', lambda msg, *a, **kw: warnings.append(msg))  # pyright: ignore[reportUnknownLambdaType]
    prices = pricing._parse_catalogue([{'model_id': 'a', 'provider': 'openai'}])  # missing costs  # pyright: ignore[reportPrivateUsage]
    assert prices == {}
    assert any('none parsed' in w for w in warnings)


# --- price_usage aggregate guard / qualified_model responses gate -----------


@pytest.mark.asyncio
@pytest.mark.usefixtures('_catalogue')
async def test_price_usage_leaves_aggregate_usage_unchanged():
    aggregate = Usage(input_tokens=1000, output_tokens=500, total_tokens=1500, calls=2, priced_calls=0)
    priced = await pricing.price_usage(aggregate, 'gpt-5-mini')
    assert priced is aggregate
    assert aggregate.total_cost is None
    assert aggregate.priced_calls == 0


@pytest.mark.asyncio
async def test_qualified_model_none_when_model_does_not_support_responses(monkeypatch: pytest.MonkeyPatch):
    async def fake_load(client=None):  # noqa: ANN001, ARG001
        return {'chat-only-model': ModelInfo(0.0001, 0.0002, 'openai', supports_responses=False)}

    monkeypatch.setattr(pricing, '_load_catalogue', fake_load)
    assert await pricing.qualified_model('chat-only-model') is None


# --- _load_catalogue: failure caching, host keying, concurrency -------------


class _FakeResponse:
    def __init__(self, status_code: int, payload: object = None):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request('GET', 'https://example.invalid/v2/models')
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError('error', request=request, response=response)

    def json(self) -> object:
        return self._payload


class _FakeAsyncClient:
    """Stand-in for httpx.AsyncClient that records GET calls and returns a canned response."""

    def __init__(self, calls: list[str], response_factory):  # noqa: ANN001
        self._calls = calls
        self._response_factory = response_factory

    def __call__(self, *args: object, **kwargs: object) -> _FakeAsyncClient:
        return self

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def get(self, url: str, headers: dict[str, str] | None = None) -> _FakeResponse:  # noqa: ARG002
        self._calls.append(url)
        return self._response_factory()


@pytest.mark.asyncio
async def test_failed_fetch_caches_empty_and_does_not_refetch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pricing, '_catalogues', {})
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    calls: list[str] = []
    fake_client = _FakeAsyncClient(calls, lambda: _FakeResponse(500))
    monkeypatch.setattr(httpx, 'AsyncClient', fake_client)

    first = await pricing.price_usage(_usage(), 'gpt-5-mini')
    second = await pricing.price_usage(_usage(), 'gpt-5-mini')

    assert first is not None and first.total_cost is None
    assert second is not None and second.total_cost is None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_successful_fetch_parsed_and_cached_once_across_concurrent_callers(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pricing, '_catalogues', {})
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    calls: list[str] = []

    def _respond() -> _FakeResponse:
        return _FakeResponse(
            200,
            [{'model_id': 'gpt-5-mini', 'provider': 'openai', 'input_cost': 0.00025, 'output_cost': 0.002}],
        )

    fake_client = _FakeAsyncClient(calls, _respond)
    monkeypatch.setattr(httpx, 'AsyncClient', fake_client)

    results = await asyncio.gather(*(pricing.price_usage(_usage(), 'gpt-5-mini') for _ in range(5)))

    assert len(calls) == 1
    for priced in results:
        assert priced is not None
        assert priced.total_cost == pytest.approx(0.00125)


@pytest.mark.asyncio
async def test_cache_is_keyed_by_host(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pricing, '_catalogues', {})
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    calls: list[str] = []

    def _respond() -> _FakeResponse:
        return _FakeResponse(
            200,
            [{'model_id': 'gpt-5-mini', 'provider': 'openai', 'input_cost': 0.00025, 'output_cost': 0.002}],
        )

    fake_client = _FakeAsyncClient(calls, _respond)
    monkeypatch.setattr(httpx, 'AsyncClient', fake_client)

    monkeypatch.setenv('ORQ_BASE_URL', 'https://host-a.example')
    await pricing._load_catalogue()  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setenv('ORQ_BASE_URL', 'https://host-b.example')
    await pricing._load_catalogue()  # pyright: ignore[reportPrivateUsage]

    assert len(calls) == 2
    assert set(pricing._catalogues) == {'https://host-a.example', 'https://host-b.example'}  # pyright: ignore[reportPrivateUsage]
