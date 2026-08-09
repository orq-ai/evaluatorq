"""Usage cost breakdown: extraction from Orq Responses v3 usage, span recording."""

from unittest.mock import MagicMock

import pytest

from evaluatorq.common.tracing import record_token_usage
from evaluatorq.contracts import TokenUsage, Usage

V3_USAGE = {
    'input_tokens': 5227,
    'output_tokens': 1233,
    'total_tokens': 6460,
    'input_cost': 0.0004434,
    'output_cost': 0.002785,
    'total_cost': 0.0032284,
    'input_tokens_details': {'cached_tokens': 4096, 'cache_creation_tokens': 512},
    'output_tokens_details': {'reasoning_tokens': 640},
}


def test_extract_v3_usage_full_richness():
    u = Usage.extract(V3_USAGE)
    assert u is not None
    assert u.input_tokens == 5227
    assert u.output_tokens == 1233
    assert u.cached_tokens == 4096
    assert u.cache_creation_tokens == 512
    assert u.reasoning_tokens == 640
    assert u.input_cost == 0.0004434
    assert u.output_cost == 0.002785
    assert u.total_cost == 0.0032284
    assert u.cost_usd == 0.0032284  # deprecated alias


def test_extract_total_cost_falls_back_to_component_sum():
    u = Usage.extract({'input_tokens': 1, 'output_tokens': 1, 'input_cost': 0.1, 'output_cost': 0.2})
    assert u is not None
    assert u.total_cost == pytest.approx(0.3)


def test_token_usage_alias_and_cost_usd_construction():
    assert TokenUsage is Usage
    u = Usage(cost_usd=0.5, total_tokens=1)
    assert u.total_cost == 0.5
    dumped = u.model_dump()
    assert dumped['cost_usd'] == 0.5
    assert dumped['total_cost'] == 0.5
    assert Usage.model_validate(dumped).total_cost == 0.5


def test_arithmetic_carries_cost_breakdown():
    a = Usage.extract(V3_USAGE)
    assert a is not None
    total = a + a
    assert total.input_cost == a.input_cost * 2  # pyright: ignore[reportOptionalOperand]
    assert total.total_cost == a.total_cost * 2  # pyright: ignore[reportOptionalOperand]
    assert total.cache_creation_tokens == 1024
    delta = total - a
    assert delta.total_cost == a.total_cost
    # unknown + unknown stays unknown
    assert (Usage() + Usage()).total_cost is None


def test_record_token_usage_sets_cost_attributes():
    span = MagicMock()
    record_token_usage(span, usage=Usage.extract(V3_USAGE))
    span.set_attribute.assert_any_call('gen_ai.usage.input_cost', 0.0004434)
    span.set_attribute.assert_any_call('gen_ai.usage.output_cost', 0.002785)
    span.set_attribute.assert_any_call('gen_ai.usage.total_cost', 0.0032284)
    span.set_attribute.assert_any_call('gen_ai.usage.cost', 0.0032284)
    span.set_attribute.assert_any_call('gen_ai.usage.cache_creation.input_tokens', 512)


def test_record_token_usage_omits_cost_attributes_when_unknown():
    span = MagicMock()
    record_token_usage(span, usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, calls=1))
    for call in span.set_attribute.call_args_list:
        assert 'cost' not in call.args[0]


# ---------------------------------------------------------------------------
# priced_calls — cost coverage tracking
# ---------------------------------------------------------------------------


def test_extract_prices_all_calls_when_cost_reported():
    u = Usage.extract({'input_tokens': 1, 'output_tokens': 1, 'total_cost': 0.5}, calls=1)
    assert u is not None
    assert u.priced_calls == 1
    assert u.cost_is_partial is False


def test_extract_prices_nothing_when_cost_absent():
    u = Usage.extract({'input_tokens': 1, 'output_tokens': 1}, calls=1)
    assert u is not None
    assert u.priced_calls == 0
    assert u.cost_is_partial is False  # no cost at all is not "partial"


def test_summing_costed_and_uncosted_calls_marks_cost_partial():
    costed = Usage.extract({'input_tokens': 1, 'output_tokens': 1, 'total_cost': 0.5}, calls=1)
    free = Usage.extract({'input_tokens': 1, 'output_tokens': 1}, calls=1)
    assert costed is not None and free is not None

    total = costed + free + free

    assert total.calls == 3
    assert total.priced_calls == 1
    assert total.total_cost == pytest.approx(0.5)
    # The $0.50 is a lower bound, not a total — the two free calls contributed
    # nothing because their cost was unknown, not because it was zero.
    assert total.cost_is_partial is True


def test_fully_priced_aggregate_is_not_partial():
    a = Usage.extract({'input_tokens': 1, 'output_tokens': 1, 'total_cost': 0.5}, calls=1)
    b = Usage.extract({'input_tokens': 1, 'output_tokens': 1, 'total_cost': 0.25}, calls=1)
    assert a is not None and b is not None

    total = a + b

    assert total.priced_calls == total.calls == 2
    assert total.cost_is_partial is False


def test_legacy_report_without_priced_calls_is_not_flagged_partial():
    """Reports saved before priced_calls existed must not be labelled partial.

    They deserialize with priced_calls=0 alongside a real cost, which is a shape
    a freshly-built aggregate can never produce.
    """
    u = Usage.model_validate({'input_tokens': 10, 'output_tokens': 5, 'cost_usd': 1.25, 'calls': 4})

    assert u.priced_calls == 0
    assert u.total_cost == 1.25
    assert u.cost_is_partial is False


def test_with_calls_keeps_priced_calls_consistent():
    parsed = Usage.extract({'input_tokens': 1, 'output_tokens': 1, 'total_cost': 0.5}, calls=0)
    assert parsed is not None
    assert parsed.priced_calls == 0  # nothing billed yet

    stamped = parsed.with_calls(1)

    assert stamped.calls == 1
    assert stamped.priced_calls == 1
    assert stamped.cost_is_partial is False


def test_with_calls_leaves_priced_calls_zero_when_cost_unknown():
    parsed = Usage.extract({'input_tokens': 1, 'output_tokens': 1}, calls=0)
    assert parsed is not None

    stamped = parsed.with_calls(1)

    assert stamped.calls == 1
    assert stamped.priced_calls == 0


def test_priced_calls_survives_round_trip():
    original = Usage(input_tokens=1, output_tokens=1, total_cost=0.5, calls=3, priced_calls=1)
    assert Usage.model_validate(original.model_dump()) == original


def test_subtraction_clamps_priced_calls_at_zero():
    a = Usage(input_tokens=2, output_tokens=2, total_cost=0.5, calls=1, priced_calls=1)
    b = Usage(input_tokens=1, output_tokens=1, total_cost=0.2, calls=3, priced_calls=3)
    assert (a - b).priced_calls == 0


def test_v3_responses_cost_survives_openai_sdk_parsing():
    """Orq's cost fields ride the v3/router path as pydantic *extras*.

    `OrqResponsesTarget` talks to the Orq router through `openai.AsyncOpenAI`,
    whose models are `extra='allow'` — so cost keys Orq adds to `usage` survive
    parsing even though the openai SDK does not declare them. If openai ever
    flips to `extra='ignore'`, cost silently vanishes from every v3 trace and
    this test is the tripwire.
    """
    response_usage = pytest.importorskip('openai.types.responses.response_usage')

    # model_validate, not construct: construct skips validation entirely, so it
    # would keep the extra keys even under extra='ignore' — i.e. it would pass
    # for the exact regression this test exists to catch.
    usage = response_usage.ResponseUsage.model_validate({
        'input_tokens': 10,
        'output_tokens': 5,
        'total_tokens': 15,
        'input_tokens_details': {'cached_tokens': 0},
        'output_tokens_details': {'reasoning_tokens': 0},
        'input_cost': 0.001,
        'output_cost': 0.002,
        'total_cost': 0.003,
    })
    extracted = Usage.extract(usage, calls=1)

    assert extracted is not None
    assert extracted.input_cost == pytest.approx(0.001)
    assert extracted.output_cost == pytest.approx(0.002)
    assert extracted.total_cost == pytest.approx(0.003)
    assert extracted.priced_calls == 1


# ---------------------------------------------------------------------------
# fmt_cost — one formatter, one precision, every surface
# ---------------------------------------------------------------------------


def test_fmt_cost_is_shared_by_reports_and_dashboard():
    """The dashboard must not carry its own cost formatter.

    Two formatters meant the same value rendered `$0.0032` in a markdown report
    and `$0.00` on the dashboard. They are now the same function object.
    """
    from evaluatorq.common.reports import fmt_cost
    from evaluatorq.dashboard import view

    assert view._fmt_cost is fmt_cost


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (None, '—'),  # unknown — never fabricated as a number
        (0.0, '$0.0000'),  # a real zero, distinct from unknown
        (0.0032, '$0.0032'),  # sub-cent per-call costs keep their digits
        (1234.5, '$1,234.5000'),  # large rollups stay at 4dp, still thousands-separated
    ],
)
def test_fmt_cost_formats_at_four_decimals(value: float | None, expected: str):
    from evaluatorq.common.reports import fmt_cost

    assert fmt_cost(value) == expected


def test_cost_aliases_work_without_going_through_extract():
    """`prompt_cost`/`completion_cost` resolve at the pydantic layer.

    They used to be honored only inside `Usage.extract`, so a direct
    `model_validate` of a provider dict silently dropped both to None.
    """
    u = Usage.model_validate({'prompt_cost': 0.01, 'completion_cost': 0.02, 'cost': 0.03})

    assert u.input_cost == 0.01
    assert u.output_cost == 0.02
    assert u.total_cost == 0.03


def test_orq_sdk_public_usage_declares_the_cost_names_we_read():
    """Pin `Usage.extract`'s cost keys against the Orq SDK's own v3 schema.

    `PublicUsage` is the usage model on the v3 router `/responses` endpoint. If
    Orq renames or nests these, this fails here instead of silently producing
    cost-free traces.
    """
    public_usage = pytest.importorskip('orq_ai_sdk.models.publicusage')

    declared = set(public_usage.PublicUsage.model_fields)
    assert {'input_cost', 'output_cost', 'total_cost'} <= declared


def test_extract_captures_cache_ttl_tier_tokens():
    """Anthropic prices 1h cache writes above 5m ones, so keep the tiers apart.

    Orq folds both into the flat `input_cost`, so this is attribution only — the
    money is already right either way.
    """
    u = Usage.extract({
        'input_tokens': 1000,
        'output_tokens': 10,
        'input_tokens_details': {
            'cache_creation_tokens': 800,
            'cache_creation_1h_tokens': 500,
            'cache_creation_5m_tokens': 300,
        },
    })

    assert u is not None
    assert u.cache_creation_tokens == 800
    assert u.cache_creation_1h_tokens == 500
    assert u.cache_creation_5m_tokens == 300


def test_cache_ttl_tier_tokens_aggregate():
    u = Usage(cache_creation_1h_tokens=5, cache_creation_5m_tokens=3)
    total = u + u
    assert total.cache_creation_1h_tokens == 10
    assert total.cache_creation_5m_tokens == 6
    assert (total - u).cache_creation_1h_tokens == 5


def test_orq_sdk_input_tokens_details_declares_the_tier_names_we_read():
    """Pin the tier key names against the Orq SDK's v3 schema."""
    details = pytest.importorskip('orq_ai_sdk.models.inputtokensdetails')

    declared = set(details.InputTokensDetails.model_fields)
    assert {'cache_creation_tokens', 'cache_creation_1h_tokens', 'cache_creation_5m_tokens'} <= declared


def test_real_anthropic_usage_model_extracts_every_cache_field():
    """Built from the installed SDK's own model, not a hand-guessed dict.

    `anthropic.types.Usage` keeps cache reads and cache writes as flat top-level
    fields and nests only the TTL split under `cache_creation`. Constructing the
    real model means this test fails if that schema ever moves.
    """
    from anthropic.types import Usage as AnthropicUsage
    from anthropic.types.cache_creation import CacheCreation

    u = Usage.extract(
        AnthropicUsage(
            input_tokens=1000,
            output_tokens=10,
            cache_creation_input_tokens=800,
            cache_read_input_tokens=2000,
            cache_creation=CacheCreation(ephemeral_1h_input_tokens=500, ephemeral_5m_input_tokens=300),
        )
    )

    assert u is not None
    assert u.cached_tokens == 2000
    assert u.cache_creation_tokens == 800
    assert u.cache_creation_1h_tokens == 500
    assert u.cache_creation_5m_tokens == 300


def test_cache_tiers_roll_up_when_no_total_is_reported():
    """Fallback only: a payload carrying the TTL split with no pre-summed total.

    Both Anthropic and Orq v3 do send a total, so this path is defensive — but
    without it those cache-write tokens would vanish from `cache_creation_tokens`.
    """
    u = Usage.extract({
        'input_tokens': 1000,
        'output_tokens': 10,
        'cache_creation': {'ephemeral_1h_input_tokens': 500, 'ephemeral_5m_input_tokens': 300},
    })

    assert u is not None
    assert u.cache_creation_1h_tokens == 500
    assert u.cache_creation_5m_tokens == 300
    assert u.cache_creation_tokens == 800


def test_reported_cache_creation_total_wins_over_the_tier_sum():
    """Orq v3 pre-sums the total; trust it rather than recomputing from tiers."""
    u = Usage.extract({
        'input_tokens': 1000,
        'output_tokens': 10,
        'input_tokens_details': {
            'cache_creation_tokens': 900,
            'cache_creation_1h_tokens': 500,
            'cache_creation_5m_tokens': 300,
        },
    })

    assert u is not None
    assert u.cache_creation_tokens == 900
