"""Tiered, cache and audio pricing from `/v2/models` card metadata (RES-1346). Hand-built cards, no network."""

from __future__ import annotations

import typing

import pytest

from evaluatorq.common import model_catalogue as catalogue
from evaluatorq.common.model_catalogue import ModelInfo, cost_of
from evaluatorq.contracts import Usage

GEMINI_METADATA = {
    'million_tokens_input_cost': 1.25,
    'million_tokens_output_cost': 10,
    'million_tokens_cache_read_cost': 0.125,
    'million_tokens_cache_write_cost': 4.5,
    'million_tokens_above_200k_input_cost': 2.5,
    'million_tokens_above_200k_output_cost': 15,
    'million_tokens_above_200k_cache_read_cost': 0.25,
    'million_tokens_audio_input_cost': 3,
    'extended_context_threshold': 200000,
    'extended_context_pricing_mode': 'all_tokens',
}


def _card(metadata: object, model_id: str = 'gemini-2.5-pro') -> dict[str, object]:
    return {
        'model_id': model_id,
        'provider': 'google-ai',
        'input_cost': 0.00125,
        'output_cost': 0.01,
        'metadata': metadata,
    }


def _parse(metadata: object) -> ModelInfo:
    info = catalogue._parse_catalogue([_card(metadata)])['gemini-2.5-pro']
    return info


def _info(**kw: float | int | str | None) -> ModelInfo:
    """A flat 0.00125 / 0.01 card plus any extra field, built through the parser's own types."""
    fields = typing.cast('dict[str, typing.Any]', kw)
    return ModelInfo(0.00125, 0.01, 'google-ai', True, **fields)  # noqa: FBT003


def _tiered() -> ModelInfo:
    return _parse(GEMINI_METADATA)


# --- parsing ---------------------------------------------------------------------------------


def test_parses_every_field_per_1k():
    info = _tiered()
    assert info.extended_context_threshold == 200000
    assert info.extended_context_mode == 'all_tokens'
    assert info.above_threshold_input_cost_per_1k == pytest.approx(0.0025)
    assert info.above_threshold_output_cost_per_1k == pytest.approx(0.015)
    assert info.above_threshold_cache_read_cost_per_1k == pytest.approx(0.00025)
    assert info.audio_input_cost_per_1k == pytest.approx(0.003)
    assert info.cache_read_cost_per_1k == pytest.approx(0.000125)
    assert info.cache_write_cost_per_1k == pytest.approx(0.0045)


def test_absent_fields_are_none():
    info = _parse({})
    assert info.extended_context_threshold is None
    assert info.extended_context_mode is None
    assert info.cache_read_cost_per_1k is None
    assert info.audio_input_cost_per_1k is None
    assert info.input_cost_per_1k == pytest.approx(0.00125)


@pytest.mark.parametrize(
    'bad', [True, False, -1, -0.5, float('inf'), float('nan'), 'abc', '1.0', None, [1], {'a': 1}, 10**400]
)
@pytest.mark.parametrize(
    ('key', 'field'),
    [
        ('million_tokens_above_200k_input_cost', 'above_threshold_input_cost_per_1k'),
        ('million_tokens_above_200k_output_cost', 'above_threshold_output_cost_per_1k'),
        ('million_tokens_above_200k_cache_read_cost', 'above_threshold_cache_read_cost_per_1k'),
        ('million_tokens_audio_input_cost', 'audio_input_cost_per_1k'),
        ('million_tokens_cache_read_cost', 'cache_read_cost_per_1k'),
        ('million_tokens_cache_write_cost', 'cache_write_cost_per_1k'),
    ],
)
def test_malformed_rate_drops_only_that_field(key: str, field: str, bad: object):
    info = _parse({**GEMINI_METADATA, key: bad})
    assert getattr(info, field) is None
    # The card and every other field survive.
    assert info.input_cost_per_1k == pytest.approx(0.00125)
    reference = _tiered()
    for other in (
        'extended_context_threshold',
        'extended_context_mode',
        'above_threshold_input_cost_per_1k',
        'above_threshold_output_cost_per_1k',
        'above_threshold_cache_read_cost_per_1k',
        'audio_input_cost_per_1k',
        'cache_read_cost_per_1k',
        'cache_write_cost_per_1k',
    ):
        if other != field:
            assert getattr(info, other) == getattr(reference, other)


def test_zero_rate_is_kept():
    assert _parse({**GEMINI_METADATA, 'million_tokens_cache_read_cost': 0}).cache_read_cost_per_1k == 0.0


@pytest.mark.parametrize('bad', [True, 0, -5, 1.5, float('nan'), float('inf'), 'big', None, [200000]])
def test_malformed_threshold_is_dropped(bad: object):
    info = _parse({**GEMINI_METADATA, 'extended_context_threshold': bad})
    assert info.extended_context_threshold is None
    assert info.extended_context_mode == 'all_tokens'
    assert info.cache_write_cost_per_1k == pytest.approx(0.0045)


@pytest.mark.parametrize('bad', [True, 5, '', '   ', None, ['all_tokens']])
def test_malformed_mode_is_dropped(bad: object):
    info = _parse({**GEMINI_METADATA, 'extended_context_pricing_mode': bad})
    assert info.extended_context_mode is None
    assert info.extended_context_threshold == 200000


@pytest.mark.parametrize('metadata', [[], 'text', None, 7])
def test_non_mapping_metadata_keeps_the_card(metadata: object):
    info = _parse(metadata)
    assert info.input_cost_per_1k == pytest.approx(0.00125)
    assert info.cache_read_cost_per_1k is None


# --- cost_of ---------------------------------------------------------------------------------


def test_flat_when_no_tier_or_cache_rates():
    assert cost_of(_info(), 1000, 500) == pytest.approx((0.00125, 0.005))


def test_none_without_flat_rates():
    assert cost_of(ModelInfo(None, None, 'x', supports_responses=False), 1000, 500) is None
    assert cost_of(ModelInfo(0.001, None, 'x', supports_responses=False), 1000, 500) is None


def test_cached_tokens_use_cache_read_rate_as_subset_of_input():
    info = _info(cache_read_cost_per_1k=0.000125)
    cost = cost_of(info, 10_000, 0, cached_tokens=8_000)
    assert cost is not None
    assert cost[0] == pytest.approx(2 * 0.00125 + 8 * 0.000125)


def test_cached_tokens_fall_back_to_input_rate_without_cache_read_rate():
    assert cost_of(_info(), 10_000, 0, cached_tokens=8_000) == pytest.approx((10 * 0.00125, 0.0))


def test_cache_write_uses_cache_write_rate():
    info = _info(cache_write_cost_per_1k=0.0045)
    cost = cost_of(info, 10_000, 0, cache_creation_tokens=4_000)
    assert cost is not None
    assert cost[0] == pytest.approx(6 * 0.00125 + 4 * 0.0045)


def test_cache_write_falls_back_to_input_rate():
    assert cost_of(_info(), 10_000, 0, cache_creation_tokens=4_000) == pytest.approx((10 * 0.00125, 0.0))


def test_audio_uses_audio_rate():
    info = _info(audio_input_cost_per_1k=0.003)
    cost = cost_of(info, 10_000, 0, audio_input_tokens=3_000)
    assert cost is not None
    assert cost[0] == pytest.approx(7 * 0.00125 + 3 * 0.003)


def test_subsets_are_clamped_to_the_input():
    info = _info(cache_read_cost_per_1k=0.000125)
    cost = cost_of(info, 1_000, 0, cached_tokens=5_000)
    assert cost is not None
    assert cost[0] == pytest.approx(1 * 0.000125)


def test_below_threshold_bills_flat():
    cost = cost_of(_tiered(), 100_000, 1_000)
    assert cost == pytest.approx((100 * 0.00125, 1 * 0.01))


def test_at_threshold_bills_flat():
    cost = cost_of(_tiered(), 200_000, 1_000)
    assert cost == pytest.approx((200 * 0.00125, 1 * 0.01))


def test_above_threshold_reprices_the_whole_call():
    # input 200_001 > threshold: every input token at 2.5/1M, output at 15/1M.
    cost = cost_of(_tiered(), 200_001, 1_000)
    assert cost == pytest.approx((200.001 * 0.0025, 1 * 0.015))


def test_above_threshold_cache_read_and_unrepriced_cache_write():
    cost = cost_of(_tiered(), 300_000, 0, cached_tokens=100_000, cache_creation_tokens=50_000)
    assert cost is not None
    assert cost[0] == pytest.approx(150 * 0.0025 + 100 * 0.00025 + 50 * 0.0045)


def test_above_threshold_cache_write_and_audio_without_own_rate_bill_at_standard_input_rate():
    info = _parse(
        {
            k: v
            for k, v in GEMINI_METADATA.items()
            if k not in ('million_tokens_cache_write_cost', 'million_tokens_audio_input_cost')
        }
    )
    assert info.cache_write_cost_per_1k is None
    assert info.audio_input_cost_per_1k is None
    cost = cost_of(info, 300_000, 0, cache_creation_tokens=50_000, audio_input_tokens=50_000)
    assert cost is not None
    assert cost[0] == pytest.approx(200 * 0.0025 + 100 * 0.00125)


def test_negative_output_tokens_never_reduce_the_cost():
    cost = cost_of(_info(), 1_000, -500)
    assert cost == pytest.approx((0.00125, 0.0))


def test_above_threshold_without_above_rates_stays_standard():
    info = _info(extended_context_threshold=1000, extended_context_mode='all_tokens', cache_read_cost_per_1k=0.0001)
    cost = cost_of(info, 2_000, 1_000, cached_tokens=1_000)
    assert cost is not None
    assert cost[0] == pytest.approx(1 * 0.00125 + 1 * 0.0001)
    assert cost[1] == pytest.approx(0.01)


def test_unknown_mode_bills_flat_and_logs():
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level='DEBUG')
    try:
        info = _parse({**GEMINI_METADATA, 'extended_context_pricing_mode': 'marginal_tokens'})
        cost = cost_of(info, 300_000, 1_000)
    finally:
        logger.remove(sink)
    assert cost == pytest.approx((300 * 0.00125, 0.01))
    assert any('marginal_tokens' in m for m in messages)


def test_mode_none_bills_flat():
    info = _parse({**GEMINI_METADATA, 'extended_context_pricing_mode': 'none'})
    assert cost_of(info, 300_000, 1_000) == pytest.approx((300 * 0.00125, 0.01))


def test_missing_mode_bills_flat():
    info = _parse({k: v for k, v in GEMINI_METADATA.items() if k != 'extended_context_pricing_mode'})
    assert cost_of(info, 300_000, 1_000) == pytest.approx((300 * 0.00125, 0.01))


# --- price_usage -----------------------------------------------------------------------------


@pytest.fixture
def _tiered_catalogue(monkeypatch: pytest.MonkeyPatch):
    table = catalogue._parse_catalogue([_card(GEMINI_METADATA)])

    async def fake_load(client=None):  # noqa: ANN001, ARG001
        return table

    monkeypatch.setattr(catalogue, '_load_catalogue', fake_load)
    catalogue.reset_catalogue_cache()
    yield
    catalogue.reset_catalogue_cache()


@pytest.mark.asyncio
@pytest.mark.usefixtures('_tiered_catalogue')
async def test_price_usage_uses_tiered_and_cache_rates():
    usage = Usage(
        input_tokens=300_000,
        output_tokens=1_000,
        total_tokens=301_000,
        cached_tokens=100_000,
        cache_creation_tokens=50_000,
        calls=1,
        priced_calls=0,
    )
    priced = await catalogue.price_usage(usage, 'gemini-2.5-pro')
    assert priced is not None
    assert priced.input_cost == pytest.approx(150 * 0.0025 + 100 * 0.00025 + 50 * 0.0045)
    assert priced.output_cost == pytest.approx(1 * 0.015)
    assert priced.total_cost == pytest.approx((priced.input_cost or 0) + (priced.output_cost or 0))
    assert priced.priced_calls == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures('_tiered_catalogue')
async def test_price_usage_below_threshold_is_flat_for_uncached_calls():
    usage = Usage(input_tokens=1_000, output_tokens=500, total_tokens=1_500, calls=1, priced_calls=0)
    priced = await catalogue.price_usage(usage, 'gemini-2.5-pro')
    assert priced is not None
    assert priced.input_cost == pytest.approx(0.00125)
    assert priced.output_cost == pytest.approx(0.005)
