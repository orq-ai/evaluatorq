"""Orq model catalogue: client-side pricing and model qualification (RES-1295)."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from openai import AsyncOpenAI

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


# --- reasoning effort pre-flight ----------------------------------------------


def _entry_with_efforts(*values: str) -> dict[str, object]:
    return {
        'model_id': 'thinky',
        'provider': 'openai',
        'input_cost': 0.1,
        'output_cost': 0.2,
        'parameters': [
            {'parameter': 'temperature', 'config': {'min': 0, 'max': 2}},
            {
                'parameter': 'reasoningEffort',
                'config': {
                    'default': 'medium',
                    'options': [{'display_name': v.title(), 'value': v} for v in values],
                },
            },
        ],
    }


def test_parse_catalogue_reads_reasoning_effort_options():
    prices = pricing._parse_catalogue([_entry_with_efforts('low', 'medium', 'high', 'xhigh')])  # pyright: ignore[reportPrivateUsage]
    assert prices['thinky'].reasoning_efforts == frozenset({'low', 'medium', 'high', 'xhigh'})


def test_parse_catalogue_reasoning_efforts_is_none_without_the_parameter():
    prices = pricing._parse_catalogue(  # pyright: ignore[reportPrivateUsage]
        [{'model_id': 'plain', 'provider': 'openai', 'input_cost': 0.1, 'output_cost': 0.2}]
    )
    assert prices['plain'].reasoning_efforts is None


def test_parse_catalogue_ignores_the_supports_reasoning_effort_flags():
    """The flags disagree with the options on a third of the live catalogue;
    the options list is the one that is right."""
    entry = _entry_with_efforts('low', 'high', 'max')
    entry['metadata'] = {'supports_reasoning_effort_low': True, 'supports_reasoning_effort_high': True}
    prices = pricing._parse_catalogue([entry])  # pyright: ignore[reportPrivateUsage]
    assert prices['thinky'].reasoning_efforts == frozenset({'low', 'high', 'max'})


@pytest.fixture
def _reasoning_catalogue(monkeypatch: pytest.MonkeyPatch):
    async def fake_load(client=None):  # noqa: ANN001, ARG001
        return {
            'thinky': ModelInfo(
                0.1, 0.2, 'openai', supports_responses=True, reasoning_efforts=frozenset({'low', 'high'})
            ),
            'plain': ModelInfo(0.1, 0.2, 'openai', supports_responses=True),
        }

    monkeypatch.setattr(pricing, '_load_catalogue', fake_load)


@pytest.mark.asyncio
@pytest.mark.usefixtures('_reasoning_catalogue')
async def test_validate_reasoning_effort_accepts_a_listed_value():
    await pricing.validate_reasoning_effort('high', 'openai/thinky')


@pytest.mark.asyncio
@pytest.mark.usefixtures('_reasoning_catalogue')
async def test_validate_reasoning_effort_rejects_an_unlisted_value():
    with pytest.raises(ValueError, match='not accepted by'):
        await pricing.validate_reasoning_effort('xhigh', 'thinky')


@pytest.mark.asyncio
@pytest.mark.usefixtures('_reasoning_catalogue')
async def test_validate_reasoning_effort_passes_when_the_catalogue_cannot_say():
    """An agent/<key> id is never in the catalogue, and 86 of 148 live entries
    carry no reasoningEffort parameter — neither may block a run."""
    await pricing.validate_reasoning_effort('xhigh', 'agent/support')
    await pricing.validate_reasoning_effort('xhigh', 'plain')


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

    def __init__(self, calls: list[str], response_factory, headers: list[dict[str, str]] | None = None):  # noqa: ANN001
        self._calls = calls
        self._response_factory = response_factory
        self._headers = headers

    def __call__(self, *args: object, **kwargs: object) -> _FakeAsyncClient:
        return self

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def get(self, url: str, headers: dict[str, str] | None = None) -> _FakeResponse:
        self._calls.append(url)
        if self._headers is not None:
            self._headers.append(headers or {})
        return self._response_factory()


@pytest.mark.asyncio
async def test_failed_fetch_retries_then_gives_up_for_the_process(monkeypatch: pytest.MonkeyPatch):
    """A transient fetch failure must not degrade the whole run to unpriced.

    The catalogue is retried up to ``_MAX_FETCH_FAILURES`` times; only then is the
    empty result cached for the process. Caching {} on the first hiccup used to
    silently pin every later call in the run to unpriced and chat-completions-only.
    """
    monkeypatch.setattr(pricing, '_catalogues', {})
    monkeypatch.setattr(pricing, '_fetch_failures', {})
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    calls: list[str] = []
    fake_client = _FakeAsyncClient(calls, lambda: _FakeResponse(500))
    monkeypatch.setattr(httpx, 'AsyncClient', fake_client)

    for _ in range(pricing._MAX_FETCH_FAILURES):
        result = await pricing.price_usage(_usage(), 'gpt-5-mini')
        assert result is not None and result.total_cost is None
    assert len(calls) == pricing._MAX_FETCH_FAILURES

    # Given up: the empty catalogue is now cached, so no further HTTP happens.
    after = await pricing.price_usage(_usage(), 'gpt-5-mini')
    assert after is not None and after.total_cost is None
    assert len(calls) == pricing._MAX_FETCH_FAILURES


@pytest.mark.asyncio
async def test_transient_failure_then_success_prices_normally(monkeypatch: pytest.MonkeyPatch):
    """One hiccup, then a good response: the run gets its prices back."""
    monkeypatch.setattr(pricing, '_catalogues', {})
    monkeypatch.setattr(pricing, '_fetch_failures', {})
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    calls: list[str] = []
    responses = [
        _FakeResponse(500),
        _FakeResponse(
            200,
            [{'model_id': 'gpt-5-mini', 'provider': 'openai', 'input_cost': 0.00025, 'output_cost': 0.002}],
        ),
    ]
    monkeypatch.setattr(httpx, 'AsyncClient', _FakeAsyncClient(calls, lambda: responses.pop(0)))

    failed = await pricing.price_usage(_usage(), 'gpt-5-mini')
    assert failed is not None and failed.total_cost is None

    priced = await pricing.price_usage(_usage(), 'gpt-5-mini')
    assert priced is not None and priced.total_cost is not None
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_register_model_overrides_the_fetched_catalogue(monkeypatch: pytest.MonkeyPatch):
    """A registered entry prices a model the catalogue does not list, with no HTTP."""
    monkeypatch.setattr(pricing, '_catalogues', {})
    monkeypatch.setattr(pricing, '_overrides', {})
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    calls: list[str] = []
    monkeypatch.setattr(httpx, 'AsyncClient', _FakeAsyncClient(calls, lambda: _FakeResponse(500)))

    pricing.register_model(
        'my-self-hosted',
        pricing.ModelInfo(
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
            provider='self',
            supports_responses=True,
            reasoning_efforts=frozenset({'low', 'high'}),
        ),
    )

    priced = await pricing.price_usage(_usage(), 'my-self-hosted')
    assert priced is not None and priced.total_cost is not None
    assert await pricing.qualified_model('my-self-hosted') == 'self/my-self-hosted'
    assert (await pricing.get_model_info('my-self-hosted')) is not None
    # An override answers without ever reaching the network.
    assert calls == []

    with pytest.raises(ValueError, match='not accepted'):
        await pricing.validate_reasoning_effort('medium', 'my-self-hosted')


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


@pytest.mark.asyncio
async def test_client_key_preferred_when_host_derived_from_client(monkeypatch: pytest.MonkeyPatch):
    """A staging client's own key must be used, not an ambient prod ORQ_API_KEY.

    Otherwise the catalogue fetch either 401s against the wrong workspace
    (caching {} and silently disabling pricing + Responses routing) or reads
    prices from the wrong workspace entirely.
    """
    monkeypatch.setattr(pricing, '_catalogues', {})
    monkeypatch.setenv('ORQ_API_KEY', 'prod-env-key')

    calls: list[str] = []
    headers: list[dict[str, str]] = []

    def _respond() -> _FakeResponse:
        return _FakeResponse(
            200,
            [{'model_id': 'gpt-5-mini', 'provider': 'openai', 'input_cost': 0.00025, 'output_cost': 0.002}],
        )

    fake_client = _FakeAsyncClient(calls, _respond, headers)
    monkeypatch.setattr(httpx, 'AsyncClient', fake_client)

    staging_client = AsyncOpenAI(api_key='staging-client-key', base_url='https://staging.example/v3/router')
    await pricing._load_catalogue(staging_client)  # pyright: ignore[reportPrivateUsage]

    assert calls == ['https://staging.example/v2/models']
    assert headers[0]['Authorization'] == 'Bearer staging-client-key'


@pytest.mark.asyncio
async def test_env_key_used_when_no_client_given(monkeypatch: pytest.MonkeyPatch):
    """The client-less path (no injected client to derive host or key from) still uses the env var."""
    monkeypatch.setattr(pricing, '_catalogues', {})
    monkeypatch.setenv('ORQ_API_KEY', 'env-key')
    monkeypatch.setenv('ORQ_BASE_URL', 'https://prod.example')

    calls: list[str] = []
    headers: list[dict[str, str]] = []

    def _respond() -> _FakeResponse:
        return _FakeResponse(
            200,
            [{'model_id': 'gpt-5-mini', 'provider': 'openai', 'input_cost': 0.00025, 'output_cost': 0.002}],
        )

    fake_client = _FakeAsyncClient(calls, _respond, headers)
    monkeypatch.setattr(httpx, 'AsyncClient', fake_client)

    await pricing._load_catalogue()  # pyright: ignore[reportPrivateUsage]

    assert calls == ['https://prod.example/v2/models']
    assert headers[0]['Authorization'] == 'Bearer env-key'


def test_parse_catalogue_survives_a_non_mapping_metadata():
    """A provider returning `metadata` as a list must cost that one entry, not the catalogue.

    `(entry.get('metadata') or {}).get(...)` survives an EMPTY list and raises
    `AttributeError` on a populated one, so the empty case hid this until a live
    payload happened to carry a non-empty one.
    """
    prices = pricing._parse_catalogue(  # pyright: ignore[reportPrivateUsage]
        [
            {
                'model_id': 'listy',
                'provider': 'openai',
                'input_cost': 0.1,
                'output_cost': 0.2,
                'metadata': [{'supports_responses_api': True}],
            },
            {
                'model_id': 'stringy',
                'provider': 'openai',
                'input_cost': 0.1,
                'output_cost': 0.2,
                'metadata': 'supports_responses_api',
            },
            {
                'model_id': 'good',
                'provider': 'openai',
                'input_cost': 0.1,
                'output_cost': 0.2,
                'metadata': {'supports_responses_api': True},
            },
        ]
    )
    assert set(prices) == {'listy', 'stringy', 'good'}
    assert prices['listy'].supports_responses is False
    assert prices['stringy'].supports_responses is False
    assert prices['good'].supports_responses is True
