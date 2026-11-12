"""`Usage.extract` keeps the subset contract (cache tokens inside input_tokens) for every provider shape (RES-1346)."""

from __future__ import annotations

import pytest

from evaluatorq.common import model_catalogue as catalogue
from evaluatorq.common.model_catalogue import ModelInfo, cost_of
from evaluatorq.contracts import Usage

ANTHROPIC = {
    'input_tokens': 10,
    'output_tokens': 5,
    'cache_read_input_tokens': 100,
    'cache_creation_input_tokens': 50,
    'cache_creation': {'ephemeral_1h_input_tokens': 20, 'ephemeral_5m_input_tokens': 30},
}


def _extract(payload: object) -> Usage:
    usage = Usage.extract(payload)
    assert usage is not None
    return usage


def test_anthropic_dict_folds_cache_into_input():
    usage = _extract(ANTHROPIC)
    assert usage.input_tokens == 160
    assert usage.cached_tokens == 100
    assert usage.cache_creation_tokens == 50
    assert (usage.cache_creation_1h_tokens, usage.cache_creation_5m_tokens) == (20, 30)
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens == 165


def test_anthropic_sdk_object():
    anthropic_types = pytest.importorskip('anthropic.types')
    raw = anthropic_types.Usage(
        input_tokens=10,
        output_tokens=5,
        cache_read_input_tokens=100,
        cache_creation_input_tokens=50,
        cache_creation=anthropic_types.CacheCreation(ephemeral_1h_input_tokens=20, ephemeral_5m_input_tokens=30),
    )
    usage = _extract(raw)
    assert (usage.input_tokens, usage.cached_tokens, usage.cache_creation_tokens) == (160, 100, 50)
    assert usage.total_tokens == 165


def test_anthropic_tier_object_alone_counts_as_cache_writes():
    usage = _extract({
        'input_tokens': 10,
        'output_tokens': 5,
        'cache_creation': {'ephemeral_1h_input_tokens': 20, 'ephemeral_5m_input_tokens': 30},
    })
    assert usage.cache_creation_tokens == 50
    assert usage.input_tokens == 60


def test_anthropic_without_cache_is_unchanged():
    usage = _extract({'input_tokens': 10, 'output_tokens': 5, 'cache_read_input_tokens': 0})
    assert (usage.input_tokens, usage.total_tokens) == (10, 15)


def test_anthropic_reported_total_is_kept_as_provided():
    usage = _extract({**ANTHROPIC, 'total_tokens': 999})
    assert usage.total_tokens == 999
    assert usage.input_tokens == 160


def test_openai_chat_shape_is_unchanged():
    usage = _extract({
        'prompt_tokens': 1000,
        'completion_tokens': 50,
        'total_tokens': 1050,
        'prompt_tokens_details': {'cached_tokens': 800},
    })
    assert (usage.input_tokens, usage.cached_tokens, usage.total_tokens) == (1000, 800, 1050)


def test_orq_responses_shape_is_unchanged():
    usage = _extract({
        'input_tokens': 1000,
        'output_tokens': 50,
        'total_tokens': 1050,
        'input_tokens_details': {
            'cached_tokens': 600,
            'cache_creation_tokens': 200,
            'cache_creation_1h_tokens': 150,
            'cache_creation_5m_tokens': 50,
        },
    })
    assert usage.input_tokens == 1000
    assert (usage.cached_tokens, usage.cache_creation_tokens) == (600, 200)
    assert (usage.cache_creation_1h_tokens, usage.cache_creation_5m_tokens) == (150, 50)


def test_langchain_shape_is_unchanged():
    usage = _extract({
        'input_tokens': 1000,
        'output_tokens': 50,
        'total_tokens': 1050,
        'input_token_details': {'cache_read': 700},
    })
    assert usage.input_tokens == 1000
    assert usage.cached_tokens == 700


def test_codex_flat_cached_input_tokens_stays_a_subset():
    usage = _extract({'input_tokens': 1000, 'output_tokens': 50, 'cached_input_tokens': 800})
    assert (usage.input_tokens, usage.cached_tokens) == (1000, 800)


def test_both_shapes_present_are_not_double_counted():
    usage = _extract({
        'input_tokens': 1000,
        'output_tokens': 50,
        'input_tokens_details': {'cached_tokens': 800, 'cache_creation_tokens': 100},
        'cache_read_input_tokens': 800,
        'cache_creation_input_tokens': 100,
    })
    assert usage.input_tokens == 1000
    assert (usage.cached_tokens, usage.cache_creation_tokens) == (800, 100)


def test_cost_of_prices_an_anthropic_cache_heavy_call_on_all_tokens():
    info = ModelInfo(
        0.003,
        0.015,
        'anthropic',
        True,  # noqa: FBT003
        cache_read_cost_per_1k=0.0003,
        cache_write_cost_per_1k=0.00375,
    )
    usage = _extract({
        'input_tokens': 10,
        'output_tokens': 5,
        'cache_read_input_tokens': 100_000,
        'cache_creation_input_tokens': 50_000,
    })
    costs = cost_of(
        info,
        usage.input_tokens,
        usage.output_tokens,
        cached_tokens=usage.cached_tokens,
        cache_creation_tokens=usage.cache_creation_tokens,
    )
    assert costs is not None
    assert costs[0] == pytest.approx(0.01 * 0.003 + 100 * 0.0003 + 50 * 0.00375)


@pytest.mark.asyncio
async def test_price_usage_prices_anthropic_cache_tokens(monkeypatch: pytest.MonkeyPatch):
    table = {
        'claude-x': ModelInfo(
            0.003,
            0.015,
            'anthropic',
            True,  # noqa: FBT003
            cache_read_cost_per_1k=0.0003,
            cache_write_cost_per_1k=0.00375,
        )
    }

    async def fake_load(client=None):  # noqa: ANN001, ARG001
        return table

    monkeypatch.setattr(catalogue, '_load_catalogue', fake_load)
    usage = _extract({
        'input_tokens': 10,
        'output_tokens': 1000,
        'cache_read_input_tokens': 100_000,
        'cache_creation_input_tokens': 50_000,
    })
    priced = await catalogue.price_usage(usage, 'claude-x')
    assert priced is not None
    assert priced.input_cost == pytest.approx(0.01 * 0.003 + 100 * 0.0003 + 50 * 0.00375)
    assert priced.output_cost == pytest.approx(0.015)
