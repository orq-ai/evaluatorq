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
from pathlib import Path

import pytest

from evaluatorq.common import model_catalogue as catalogue
from evaluatorq.common.model_catalogue import ModelInfo, cost_of

FIXTURE = Path(__file__).parent / 'fixtures' / 'v2_models_pricing_cards.json'

GEMINI = 'google-ai/gemini-2.5-pro'
NO_TIER = 'orq-research@orq/autorouter-anthropic-cost'
ANTHROPIC = 'anthropic/claude-sonnet-5-5'
AUDIO = 'google-ai/gemini-3.5-transcribe-live'

# metadata key -> ModelInfo field, all USD per 1M published and per 1k parsed.
RATE_FIELDS = {
    'million_tokens_cache_read_cost': 'cache_read_cost_per_1k',
    'million_tokens_cache_write_cost': 'cache_write_cost_per_1k',
    'million_tokens_audio_input_cost': 'audio_input_cost_per_1k',
    'million_tokens_above_200k_input_cost': 'above_threshold_input_cost_per_1k',
    'million_tokens_above_200k_output_cost': 'above_threshold_output_cost_per_1k',
    'million_tokens_above_200k_cache_read_cost': 'above_threshold_cache_read_cost_per_1k',
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
    for key, field in RATE_FIELDS.items():
        published = metadata.get(key)
        if published is None:
            assert getattr(parsed[ref], field) is None, (ref, field)
        else:
            assert getattr(parsed[ref], field) == pytest.approx(published / 1000), (ref, field)


def test_the_audio_card_is_the_only_one_with_an_audio_rate(parsed: dict[str, ModelInfo]):
    assert [ref for ref, info in parsed.items() if info.audio_input_cost_per_1k is not None] == [AUDIO]
    assert parsed[AUDIO].audio_input_cost_per_1k == pytest.approx(0.0035)


def test_anthropic_card_has_cache_rates_and_no_tier(parsed: dict[str, ModelInfo]):
    info = parsed[ANTHROPIC]
    assert info.cache_read_cost_per_1k == pytest.approx(0.0001)
    assert info.cache_write_cost_per_1k == pytest.approx(0.0025)
    assert info.extended_context_threshold is None
    assert info.above_threshold_input_cost_per_1k is None


def test_tiered_gemini_card_parses_its_tier(parsed: dict[str, ModelInfo]):
    info = parsed[GEMINI]
    assert info.extended_context_threshold == 200000
    assert info.extended_context_mode == 'all_tokens'
    assert info.above_threshold_input_cost_per_1k == pytest.approx(0.0025)
    assert info.above_threshold_output_cost_per_1k == pytest.approx(0.015)
    assert info.above_threshold_cache_read_cost_per_1k == pytest.approx(0.00025)


def test_mode_none_card_has_no_tier_rates(parsed: dict[str, ModelInfo]):
    info = parsed[NO_TIER]
    assert info.extended_context_mode == 'none'
    assert info.above_threshold_input_cost_per_1k is None
    assert info.above_threshold_output_cost_per_1k is None
    assert info.above_threshold_cache_read_cost_per_1k is None


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
