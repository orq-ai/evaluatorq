"""`Usage.extract` keeps the subset contract (cache tokens inside input_tokens) for every provider shape (RES-1346)."""

from __future__ import annotations

import pytest

from evaluatorq.common import model_catalogue as catalogue
from evaluatorq.common.model_catalogue import ModelInfo, cost_of
from evaluatorq.contracts import Usage, _cache_counts

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


@pytest.mark.parametrize('face', ['chat', 'responses'])
def test_anthropic_adapter_usage_is_not_double_counted(face):
    """The adapter folds cache counters into its prompt count and emits only OpenAI-shaped keys
    (the cache write rides in the details object); ``Usage.extract`` must not add the cache again."""
    anthropic_types = pytest.importorskip('anthropic.types')
    from evaluatorq.anthropic.convert import convert_usage
    from evaluatorq.anthropic.responses import convert_response_usage

    message = anthropic_types.Message(
        id='msg_1',
        type='message',
        role='assistant',
        model='claude-sonnet-5-5',
        content=[anthropic_types.TextBlock(type='text', text='hi')],
        stop_reason='end_turn',
        stop_sequence=None,
        usage=anthropic_types.Usage(
            input_tokens=10, output_tokens=5, cache_read_input_tokens=100, cache_creation_input_tokens=50
        ),
    )
    converted = convert_usage(message) if face == 'chat' else convert_response_usage(message)
    usage = _extract(converted)
    assert (usage.input_tokens, usage.cached_tokens, usage.cache_creation_tokens) == (160, 100, 50)
    assert usage.total_tokens == 165


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


def test_openai_chat_audio_prompt_tokens_are_extracted():
    usage = _extract({
        'prompt_tokens': 1000,
        'completion_tokens': 50,
        'total_tokens': 1050,
        'prompt_tokens_details': {'audio_tokens': 300, 'cached_tokens': 100},
    })
    assert usage.input_tokens == 1000
    assert (usage.audio_input_tokens, usage.cached_tokens) == (300, 100)


def test_langchain_input_token_details_audio_is_extracted():
    usage = _extract({
        'input_tokens': 1000,
        'output_tokens': 50,
        'total_tokens': 1050,
        'input_token_details': {'audio': 250},
    })
    assert usage.audio_input_tokens == 250
    assert usage.input_tokens == 1000


def test_output_side_audio_is_not_counted_as_audio_input():
    usage = _extract({
        'input_tokens': 100,
        'output_tokens': 50,
        'output_tokens_details': {'audio_tokens': 40},
    })
    assert usage.audio_input_tokens == 0


def test_audio_input_tokens_add_and_subtract_clamped():
    a = Usage(input_tokens=10, audio_input_tokens=4, calls=1)
    b = Usage(input_tokens=10, audio_input_tokens=7, calls=1)
    assert (a + b).audio_input_tokens == 11
    assert (b - a).audio_input_tokens == 3
    assert (a - b).audio_input_tokens == 0


# --- _cache_counts: one row per shape convention (RES-1346) -------------------------------------


def test_cache_counts_details_object_is_inclusive():
    counts = _cache_counts({
        'input_tokens_details': {'cached_tokens': 800, 'cache_creation_tokens': 100},
    })
    assert (counts.cached, counts.written, counts.exclusive) == (800, 100, 0)


def test_cache_counts_langchain_details_are_inclusive():
    counts = _cache_counts({'input_token_details': {'cache_read': 7, 'cache_creation': 3}})
    assert (counts.cached, counts.exclusive) == (7, 0)


def test_cache_counts_native_anthropic_flat_is_exclusive():
    counts = _cache_counts(ANTHROPIC)
    assert (counts.cached, counts.written, counts.written_1h, counts.written_5m) == (100, 50, 20, 30)
    assert counts.exclusive == 150


def test_cache_counts_anthropic_tier_object_alone_is_exclusive():
    counts = _cache_counts({'cache_creation': {'ephemeral_1h_input_tokens': 20, 'ephemeral_5m_input_tokens': 30}})
    assert (counts.written, counts.exclusive) == (50, 50)


def test_cache_counts_codex_cli_flat_is_inclusive():
    counts = _cache_counts({'cached_input_tokens': 40, 'cache_write_input_tokens': 9})
    assert (counts.cached, counts.written, counts.exclusive) == (40, 9, 0)


def test_cache_counts_vercel_camel_case_is_inclusive():
    counts = _cache_counts({'inputTokens': 100, 'outputTokens': 5, 'cachedInputTokens': 30})
    assert (counts.cached, counts.exclusive) == (30, 0)


def test_cache_counts_bedrock_camel_case_is_exclusive():
    counts = _cache_counts({'inputTokens': 10, 'cacheReadInputTokens': 100, 'cacheWriteInputTokens': 50})
    assert (counts.cached, counts.written, counts.exclusive) == (100, 50, 150)


def test_cache_counts_mixed_shapes_do_not_double_count():
    """An inclusive source for a counter wins: that counter is never also added."""
    counts = _cache_counts({
        'input_tokens_details': {'cached_tokens': 800, 'cache_creation_tokens': 100},
        'cache_read_input_tokens': 800,
        'cache_creation_input_tokens': 100,
    })
    assert (counts.cached, counts.written, counts.exclusive) == (800, 100, 0)
    # no details object, but a Codex-style inclusive read beside an Anthropic-style flat read
    counts = _cache_counts({'cached_input_tokens': 40, 'cache_read_input_tokens': 40, 'cache_creation_input_tokens': 5})
    assert (counts.cached, counts.exclusive) == (40, 5)


def test_extract_bedrock_converse_folds_cache_into_input():
    usage = _extract({
        'inputTokens': 10,
        'outputTokens': 5,
        'totalTokens': 15,
        'cacheReadInputTokens': 100,
        'cacheWriteInputTokens': 50,
    })
    assert (usage.input_tokens, usage.output_tokens) == (160, 5)
    assert (usage.cached_tokens, usage.cache_creation_tokens) == (100, 50)
    assert usage.total_tokens == 165


def test_extract_bedrock_total_that_already_includes_cache_is_kept():
    usage = _extract({
        'inputTokens': 10,
        'outputTokens': 5,
        'totalTokens': 165,
        'cacheReadInputTokens': 100,
        'cacheWriteInputTokens': 50,
    })
    assert (usage.input_tokens, usage.total_tokens) == (160, 165)


def test_extract_vercel_camel_case_input_is_not_folded():
    usage = _extract({'inputTokens': 100, 'outputTokens': 5, 'totalTokens': 105, 'cachedInputTokens': 30})
    assert (usage.input_tokens, usage.cached_tokens, usage.total_tokens) == (100, 30, 105)
