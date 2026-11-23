"""Time-of-day pricing: peak and off-peak variants priced at their mean (RES-1929). No network.

`fixtures/v2_models_time_of_day_cards.json` holds the two DeepSeek cards recorded from
``GET https://api.orq.ai/v2/models`` on 2026-11-20, trimmed to identity, flat prices and
``metadata.pricing``. Their variants carry only ``when``, so they price through
`_TIME_OF_DAY_PRICE_OVERRIDES`; the hand-built cards below carry variant prices in the
platform's ``{cost, currency, unit, per}`` shape and exercise the generic reader.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from evaluatorq.common import model_catalogue as catalogue
from evaluatorq.common.model_catalogue import cost_of
from evaluatorq.common.model_frontier import scored_model_from_card

FIXTURE = Path(__file__).parent / 'fixtures' / 'v2_models_time_of_day_cards.json'

# 2026-11-19, after both dated sets on the recorded cards.
NOW = 1_795_000_000
V4_PRO_START = 1786896000  # 2026-08-16 16:00 UTC
FLASH_START = 1789012800  # 2026-09-10 04:00 UTC
PEAK_HOURS = (
    '((request_unix / 86400 + 3) % 7 < 5) && ((request_unix % 86400 >= 3600 && request_unix % 86400 < 14400) '
    '|| (request_unix % 86400 >= 21600 && request_unix % 86400 < 36000))'
)


def _cards() -> list[dict[str, Any]]:
    return json.loads(FIXTURE.read_text())


def _card(model_id: str) -> dict[str, Any]:
    return next(card for card in _cards() if card['model_id'] == model_id)


def _price(per_million: float) -> dict[str, object]:
    return {'cost': per_million, 'currency': 'USD', 'unit': 'tokens', 'per': 1_000_000}


def _variant(when: str, **rates: float) -> dict[str, object]:
    return {'when': when, **{dim: _price(rate) for dim, rate in rates.items()}}


def _priced_card(variants: list[dict[str, object]], model_id: str = 'deepseek-v4-pro') -> dict[str, Any]:
    """A DeepSeek-shaped card with flat $0.435 / $0.87 / $0.003625 per 1M and the given variants."""
    return {
        'model_id': model_id,
        'provider': 'deepseek',
        'model_developer': 'deepseek',
        'input_cost': 0.000435,
        'output_cost': 0.00087,
        'metadata': {'million_tokens_cache_read_cost': 0.003625, 'pricing': {'variants': variants}},
    }


def _parse(card: dict[str, object], now: float = NOW) -> catalogue.ModelInfo:
    return catalogue._parse_catalogue([card], now=now)[f'{card["provider"]}/{card["model_id"]}']


@pytest.fixture
def logs() -> Iterator[list[tuple[str, str]]]:
    from loguru import logger

    records: list[tuple[str, str]] = []
    sink = logger.add(lambda m: records.append((m.record['level'].name, m.record['message'])), level='DEBUG')
    yield records
    logger.remove(sink)


# --- override table (cards without variant prices) ------------------------------------------


def test_recorded_v4_pro_card_prices_at_the_mean_of_peak_and_off_peak():
    info = _parse(_card('deepseek-v4-pro'))
    # Peak $1.32 / $3.96 / $0.044, off-peak $0.66 / $1.98 / $0.022 per 1M.
    assert info.input_cost_per_1k == pytest.approx(0.99 / 1000)
    assert info.output_cost_per_1k == pytest.approx(2.97 / 1000)
    assert info.pricing.cache_read == pytest.approx(0.033 / 1000)
    assert info.pricing.time_of_day_average is True


def test_recorded_flash_card_prices_at_the_mean_of_its_current_set():
    info = _parse(_card('deepseek-flash'))
    # Peak $0.30 / $1.20 / $0.006, off-peak $0.15 / $0.60 / $0.003 per 1M.
    assert info.input_cost_per_1k == pytest.approx(0.225 / 1000)
    assert info.output_cost_per_1k == pytest.approx(0.9 / 1000)
    assert info.pricing.cache_read == pytest.approx(0.0045 / 1000)
    assert info.pricing.time_of_day_average is True


def test_bare_id_resolves_to_the_averaged_developer_card():
    prices = catalogue._parse_catalogue(_cards(), now=NOW)
    assert prices['deepseek-v4-pro'].input_cost_per_1k == pytest.approx(0.99 / 1000)


def test_cost_of_v4_pro_uses_the_averaged_rates():
    info = _parse(_card('deepseek-v4-pro'))
    cost = cost_of(info, 1_000_000, 1_000_000, cached_tokens=200_000)
    assert cost == pytest.approx((0.8 * 0.99 + 0.2 * 0.033, 2.97))


def test_stale_override_is_not_applied_and_warns(logs: list[tuple[str, str]]):
    card = _card('deepseek-v4-pro')
    newer = NOW - 86400
    card['metadata']['pricing']['variants'] = [
        {'when': f'request_unix >= {newer}'},
        {'when': f'request_unix >= {newer} && {PEAK_HOURS}'},
    ]
    info = _parse(card)
    assert info.input_cost_per_1k == pytest.approx(0.000435)
    assert info.output_cost_per_1k == pytest.approx(0.00087)
    assert info.pricing.time_of_day_average is False
    warnings = [message for level, message in logs if level == 'WARNING']
    assert len(warnings) == 1
    assert 'deepseek/deepseek-v4-pro' in warnings[0]
    assert 'stale' in warnings[0]


def test_stale_override_warns_even_when_the_new_set_has_no_peak_variant(logs: list[tuple[str, str]]):
    card = _card('deepseek-v4-pro')
    card['metadata']['pricing']['variants'] = [{'when': f'request_unix >= {NOW - 86400}'}]
    info = _parse(card)
    assert info.input_cost_per_1k == pytest.approx(0.000435)
    assert info.pricing.time_of_day_average is False
    warnings = [message for level, message in logs if level == 'WARNING']
    assert len(warnings) == 1
    assert 'stale' in warnings[0]


def test_override_waits_for_its_dated_set_to_start():
    info = _parse(_card('deepseek-v4-pro'), now=V4_PRO_START - 1)
    assert info.input_cost_per_1k == pytest.approx(0.000435)
    assert info.pricing.time_of_day_average is False


# --- generic reader (cards with variant prices) ---------------------------------------------


def test_variant_prices_beat_the_override(logs: list[tuple[str, str]]):
    card = _priced_card([
        _variant(f'request_unix >= {V4_PRO_START}', input=1.0, output=2.0, cache_read=0.01),
        _variant(f'request_unix >= {V4_PRO_START} && {PEAK_HOURS}', input=3.0, output=6.0, cache_read=0.03),
    ])
    info = _parse(card)
    assert info.input_cost_per_1k == pytest.approx(2.0 / 1000)
    assert info.output_cost_per_1k == pytest.approx(4.0 / 1000)
    assert info.pricing.cache_read == pytest.approx(0.02 / 1000)
    assert info.pricing.time_of_day_average is True
    assert any(level == 'DEBUG' and 'ignoring its time-of-day override' in m for level, m in logs)


def test_an_expired_dated_set_is_ignored():
    old, new = FLASH_START - 30 * 86400, FLASH_START
    variants = [
        _variant(f'request_unix >= {old}', input=0.22, output=0.66, cache_read=0.007, cache_write_5m=0.5),
        _variant(f'request_unix >= {old} && {PEAK_HOURS}', input=0.44, output=1.32, cache_read=0.014),
        _variant(f'request_unix >= {new}', input=0.15, output=0.6, cache_read=0.003, cache_write_5m=0.1),
        _variant(f'request_unix >= {new} && {PEAK_HOURS}', input=0.3, output=1.2, cache_read=0.006, cache_write_5m=0.3),
    ]
    info = _parse(_priced_card(variants, model_id='other-model'))
    assert info.input_cost_per_1k == pytest.approx(0.225 / 1000)
    assert info.output_cost_per_1k == pytest.approx(0.9 / 1000)
    assert info.pricing.cache_read == pytest.approx(0.0045 / 1000)
    assert info.pricing.cache_write == pytest.approx(0.2 / 1000)
    # Before the newer set starts, the older one is the current set.
    earlier = _parse(_priced_card(variants, model_id='other-model'), now=new - 1)
    assert earlier.input_cost_per_1k == pytest.approx(0.33 / 1000)


def test_bare_number_variant_prices_read_as_per_million():
    card = _priced_card([
        {'when': f'request_unix >= {V4_PRO_START}', 'input': 0.66, 'output': 1.98},
        {'when': f'request_unix >= {V4_PRO_START} && {PEAK_HOURS}', 'input': 1.32, 'output': 3.96},
    ])
    info = _parse(card)
    assert info.input_cost_per_1k == pytest.approx(0.99 / 1000)
    assert info.output_cost_per_1k == pytest.approx(2.97 / 1000)
    # Neither variant prices the cache read, so both halves inherit the card's flat rate.
    assert info.pricing.cache_read == pytest.approx(0.003625 / 1000)


def test_dated_set_without_peak_prices_at_its_own_rates():
    info = _parse(_priced_card([_variant(f'request_unix >= {V4_PRO_START}', input=0.5, output=1.0)]))
    assert info.input_cost_per_1k == pytest.approx(0.5 / 1000)
    assert info.output_cost_per_1k == pytest.approx(1.0 / 1000)
    assert info.pricing.time_of_day_average is False


def test_batch_and_long_context_variants_are_left_alone():
    card = _priced_card(
        [
            _variant('tier == "batch"', input=0.1, output=0.2),
            _variant('input_tokens > 200000', input=9.0, output=9.0),
            _variant(f'request_unix >= {V4_PRO_START} && tier == "batch"', input=0.2, output=0.4),
        ],
        model_id='other-model',
    )
    info = _parse(card)
    assert info.input_cost_per_1k == pytest.approx(0.000435)
    assert info.output_cost_per_1k == pytest.approx(0.00087)
    assert info.pricing.time_of_day_average is False


def test_batch_variant_does_not_leak_into_the_time_of_day_mean():
    card = _priced_card([
        _variant(f'request_unix >= {V4_PRO_START}', input=0.66, output=1.98),
        _variant(f'request_unix >= {V4_PRO_START} && {PEAK_HOURS}', input=1.32, output=3.96),
        _variant('tier == "batch"', input=0.01, output=0.01),
    ])
    info = _parse(card)
    assert info.input_cost_per_1k == pytest.approx(0.99 / 1000)


def test_card_without_variants_is_unchanged():
    card = _priced_card([])
    del card['metadata']['pricing']
    info = _parse(card)
    assert info.input_cost_per_1k == pytest.approx(0.000435)
    assert info.output_cost_per_1k == pytest.approx(0.00087)
    assert info.pricing.cache_read == pytest.approx(0.003625 / 1000)
    assert info.pricing.time_of_day_average is False


@pytest.mark.parametrize('when', ['', 'request_unix >= (', 'request_unix >= 1 &&', 'request_unix > 1786896000'])
def test_malformed_conditions_are_ignored(when: str):
    info = _parse(_priced_card([_variant(when, input=5.0, output=5.0)], model_id='other-model'))
    assert info.input_cost_per_1k == pytest.approx(0.000435)


def test_parenthesised_dated_peak_variant_keeps_its_date():
    # Wrapped in one outer pair, the peak variant must still read as dated, not as an undated peak.
    card = _priced_card(
        [
            _variant(f'request_unix >= {V4_PRO_START}', input=0.66, output=1.98),
            _variant(f'(request_unix >= {V4_PRO_START} && {PEAK_HOURS})', input=1.32, output=3.96),
        ],
        model_id='other-model',
    )
    assert _parse(card).input_cost_per_1k == pytest.approx(0.99 / 1000)
    # Before the set starts neither variant applies, so the card prices flat.
    earlier = _parse(card, now=V4_PRO_START - 1)
    assert earlier.input_cost_per_1k == pytest.approx(0.000435)
    assert earlier.pricing.time_of_day_average is False


def test_modulo_condition_without_an_hour_window_is_not_a_peak():
    card = _priced_card(
        [
            _variant(f'request_unix >= {V4_PRO_START}', input=0.66, output=1.98),
            _variant(f'request_unix >= {V4_PRO_START} && request_unix % 2 == 0', input=9.0, output=9.0),
        ],
        model_id='other-model',
    )
    info = _parse(card)
    assert info.input_cost_per_1k == pytest.approx(0.66 / 1000)
    assert info.pricing.time_of_day_average is False


# --- the Pareto view reads the same rates -----------------------------------------------------


def test_frontier_prices_the_card_at_the_averaged_rates():
    card = _card('deepseek-v4-pro')
    card['metadata']['autorouter'] = {'v2': {'intelligence_index': 36.0}}
    scored = scored_model_from_card(card)
    assert scored is not None
    # Uses the real clock, which is after the recorded dated set.
    assert scored.input_price == pytest.approx(0.99)
    assert scored.output_price == pytest.approx(2.97)
