"""Usage cost breakdown: extraction from Orq Responses v3 usage, span recording."""

from unittest.mock import MagicMock

from evaluatorq.contracts import TokenUsage, Usage
from evaluatorq.common.tracing import record_token_usage

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
    assert u.total_cost == 0.30000000000000004


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
