"""Client-side pricing of chat-completion usage (RES-1295)."""

from __future__ import annotations

import pytest

from evaluatorq.common import pricing
from evaluatorq.contracts import Usage


@pytest.fixture(autouse=True)
def _clear_cache():
    pricing.reset_price_cache()
    yield
    pricing.reset_price_cache()


@pytest.fixture
def _catalogue(monkeypatch: pytest.MonkeyPatch):
    async def fake_load():
        return {'gpt-5-mini': (0.00025, 0.002)}

    monkeypatch.setattr(pricing, '_load_prices', fake_load)


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
    monkeypatch.delenv('ORQ_API_KEY', raising=False)
    priced = await pricing.price_usage(_usage(), 'gpt-5-mini')
    assert priced is not None
    assert priced.total_cost is None


def test_parse_catalogue_skips_malformed_entries():
    prices = pricing._parse_catalogue(  # pyright: ignore[reportPrivateUsage]
        [
            {'model_id': 'a', 'input_cost': 0.1, 'output_cost': 0.2},
            {'model_id': 'b', 'input_cost': None, 'output_cost': 0.2},
            {'input_cost': 0.1, 'output_cost': 0.2},
            'nonsense',
        ]
    )
    assert prices == {'a': (0.1, 0.2)}
