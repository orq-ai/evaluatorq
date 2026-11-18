"""Contract test: real `/v2/models` cards through `_parse_catalogue` and `cost_of` (RES-1346).

`fixtures/v2_models_pricing_cards.json` holds four entries recorded verbatim from
``GET https://api.orq.ai/v2/models`` on 2026-11-12, trimmed to identity, price and
``metadata`` fields: a tiered Gemini card (mode ``all_tokens``, threshold 200000),
a mode ``none`` card, an Anthropic card with cache read and write rates, and the one
card that publishes ``million_tokens_audio_input_cost``. The hand-built cards in
`test_model_catalogue_tiered_pricing.py` prove the logic; this proves the parser
still agrees with what the API actually sends. Re-record the fixture if the card
schema changes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from evaluatorq.common import model_catalogue as catalogue
from evaluatorq.common.model_catalogue import ModelInfo, Pricing, cost_of

FIXTURE = Path(__file__).parent / 'fixtures' / 'v2_models_pricing_cards.json'

GEMINI = 'google-ai/gemini-2.5-pro'
NO_TIER = 'orq-research@orq/autorouter-anthropic-cost'
ANTHROPIC = 'anthropic/claude-sonnet-5-5'
AUDIO = 'google-ai/gemini-3.5-transcribe-live'

# metadata key -> getter on `Pricing`, all USD per 1M published and per 1k parsed.
RATE_FIELDS: dict[str, Callable[[Pricing], float | None]] = {
    'million_tokens_cache_read_cost': lambda p: p.cache_read,
    'million_tokens_cache_write_cost': lambda p: p.cache_write,
    'million_tokens_audio_input_cost': lambda p: p.audio_input,
    'million_tokens_above_200k_input_cost': lambda p: p.tier.input if p.tier else None,
    'million_tokens_above_200k_output_cost': lambda p: p.tier.output if p.tier else None,
    'million_tokens_above_200k_cache_read_cost': lambda p: p.tier.cache_read if p.tier else None,
}


@pytest.fixture(scope='module')
def cards() -> dict[str, dict[str, object]]:
    return {card['refId']: card for card in json.loads(FIXTURE.read_text())}


@pytest.fixture(scope='module')
def parsed(cards: dict[str, dict[str, object]]) -> dict[str, ModelInfo]:
    table = catalogue._parse_catalogue(list(cards.values()))
    return {ref: table[f'{card["provider"]}/{card["model_id"]}'] for ref, card in cards.items()}


def test_fixture_holds_the_four_recorded_cards(cards: dict[str, dict[str, object]]):
    assert set(cards) == {GEMINI, NO_TIER, ANTHROPIC, AUDIO}


@pytest.mark.parametrize('ref', [GEMINI, NO_TIER, ANTHROPIC, AUDIO])
def test_each_rate_is_the_published_per_million_over_1000(
    ref: str, cards: dict[str, dict[str, object]], parsed: dict[str, ModelInfo]
):
    metadata = cards[ref]['metadata']
    assert isinstance(metadata, dict)
    for key, get in RATE_FIELDS.items():
        published = metadata.get(key)
        if published is None:
            assert get(parsed[ref].pricing) is None, (ref, key)
        else:
            assert get(parsed[ref].pricing) == pytest.approx(published / 1000), (ref, key)


def test_the_audio_card_is_the_only_one_with_an_audio_rate(parsed: dict[str, ModelInfo]):
    assert [ref for ref, info in parsed.items() if info.pricing.audio_input is not None] == [AUDIO]
    assert parsed[AUDIO].pricing.audio_input == pytest.approx(0.0035)


def test_anthropic_card_has_cache_rates_and_no_tier(parsed: dict[str, ModelInfo]):
    info = parsed[ANTHROPIC]
    assert info.pricing.cache_read == pytest.approx(0.0001)
    assert info.pricing.cache_write == pytest.approx(0.0025)
    assert info.pricing.tier is None


def test_tiered_gemini_card_parses_its_tier(parsed: dict[str, ModelInfo]):
    info = parsed[GEMINI]
    tier = info.pricing.tier
    assert tier is not None
    assert tier.threshold == 200000
    assert tier.input == pytest.approx(0.0025)
    assert tier.output == pytest.approx(0.015)
    assert tier.cache_read == pytest.approx(0.00025)


def test_mode_none_card_has_no_tier_rates(parsed: dict[str, ModelInfo]):
    info = parsed[NO_TIER]
    assert info.pricing.tier is None


def test_cost_of_below_the_gemini_threshold_bills_standard_rates(parsed: dict[str, ModelInfo]):
    costs = cost_of(parsed[GEMINI], 200_000, 1_000)
    assert costs is not None
    assert costs.input == pytest.approx(200 * 0.00125)
    assert costs.output == pytest.approx(1 * 0.01)


def test_cost_of_above_the_gemini_threshold_reprices_the_whole_call(parsed: dict[str, ModelInfo]):
    costs = cost_of(parsed[GEMINI], 200_001, 1_000, cached_tokens=100_000)
    assert costs is not None
    assert costs.input == pytest.approx(100.001 * 0.0025 + 100 * 0.00025)
    assert costs.output == pytest.approx(1 * 0.015)
