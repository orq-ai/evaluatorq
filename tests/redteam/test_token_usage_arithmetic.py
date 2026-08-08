"""Unit tests for TokenUsage arithmetic from evaluatorq.redteam.contracts."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from evaluatorq.redteam.contracts import TokenUsage


class TestTokenUsageAddition:
    def test_add_two_usages_sums_all_fields(self) -> None:
        a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, calls=1)
        b = TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28, calls=2)
        result = a + b
        assert result.prompt_tokens == 30
        assert result.completion_tokens == 13
        assert result.total_tokens == 43
        assert result.calls == 3

    def test_add_none_returns_copy_of_lhs(self) -> None:
        a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, calls=1)
        result = a + None
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.total_tokens == 15
        assert result.calls == 1
        assert result is not a

    def test_add_zeros(self) -> None:
        a = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0, calls=0)
        b = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0, calls=0)
        result = a + b
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0
        assert result.calls == 0


class TestTokenUsageSum:
    def test_sum_list_without_start_raises_type_error(self) -> None:
        """sum([TokenUsage, ...]) without explicit start calls __radd__(0) first."""
        a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, calls=1)
        b = TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28, calls=2)
        # sum() starts with int(0); __radd__ must handle it
        result: TokenUsage = sum([a, b])  # pyright: ignore[reportAssignmentType]
        assert result.prompt_tokens == 30
        assert result.completion_tokens == 13
        assert result.calls == 3

    def test_sum_three_usages(self) -> None:
        a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, calls=1)
        b = TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28, calls=2)
        c = TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7, calls=1)
        result: TokenUsage = sum([a, b, c])  # pyright: ignore[reportAssignmentType]
        assert result.prompt_tokens == 35
        assert result.completion_tokens == 15
        assert result.calls == 4

    def test_sum_empty_list_with_explicit_start(self) -> None:
        """sum([], TokenUsage()) returns zero TokenUsage without error."""
        result = sum([], TokenUsage())
        assert isinstance(result, TokenUsage)
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0
        assert result.calls == 0

    def test_radd_with_zero_int(self) -> None:
        """__radd__(0) returns a copy (not self), enabling sum() compatibility while avoiding shared refs."""
        a = TokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10, calls=1)
        result = a.__radd__(0)
        assert result is not a
        assert result.prompt_tokens == a.prompt_tokens
        assert result.completion_tokens == a.completion_tokens
        assert result.total_tokens == a.total_tokens
        assert result.calls == a.calls

    def test_radd_with_another_usage(self) -> None:
        """__radd__ with another TokenUsage calls __add__ in reverse."""
        a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, calls=1)
        b = TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5, calls=1)
        result = a.__radd__(b)
        assert result.prompt_tokens == 13
        assert result.completion_tokens == 7
        assert result.calls == 2


class TestTokenUsageTotalTokensInvariant:
    def test_matching_values_unchanged(self) -> None:
        """When total == prompt + completion, value is left as-is."""
        u = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, calls=1)
        assert u.total_tokens == 15

    def test_provider_total_trusted_even_when_it_differs_from_sum(self) -> None:
        """Provider-supplied total_tokens is trusted as-is (validator removed).

        Providers may legitimately report totals that differ from the naive
        prompt+completion sum when cached tokens, reasoning tokens, or audio
        tokens are included. We must not silently corrupt that data.
        """
        u = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=999, calls=1)
        assert u.total_tokens == 999

    def test_zero_total_preserved(self) -> None:
        """total_tokens=0 is preserved."""
        u = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0, calls=0)
        assert u.total_tokens == 0

    def test_total_only_caller_preserved_when_prompt_completion_zero(self) -> None:
        """When prompt=completion=0 but total>0, the explicit total is kept."""
        u = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=100, calls=1)
        assert u.total_tokens == 100

    def test_default_construction_all_zero(self) -> None:
        u = TokenUsage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0
        assert u.calls == 0


class TestTokenUsageFromCompletion:
    def _make_usage(
        self,
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        total_tokens: int | None = 15,
    ) -> MagicMock:
        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        usage.total_tokens = total_tokens
        return usage

    def test_returns_none_when_usage_attribute_absent(self) -> None:
        response = MagicMock(spec=[])  # No attributes
        result = TokenUsage.from_completion(response)
        assert result is None

    def test_returns_none_when_usage_is_none(self) -> None:
        response = MagicMock()
        response.usage = None
        result = TokenUsage.from_completion(response)
        assert result is None

    def test_returns_proper_token_usage(self) -> None:
        response = MagicMock()
        response.usage = self._make_usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        result = TokenUsage.from_completion(response)
        assert result is not None
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150
        assert result.calls == 1

    def test_returns_none_when_all_usage_fields_none(self) -> None:
        """A usage object whose token fields are all None carries no signal and is
        treated as a present-but-empty payload — extract returns None rather than
        fabricating a billed-zero record (see TokenUsage.extract gate)."""
        response = MagicMock()
        usage = MagicMock()
        usage.prompt_tokens = None
        usage.completion_tokens = None
        usage.total_tokens = None
        response.usage = usage
        assert TokenUsage.from_completion(response) is None

    def test_calls_is_always_one(self) -> None:
        response = MagicMock()
        response.usage = self._make_usage()
        result = TokenUsage.from_completion(response)
        assert result is not None
        assert result.calls == 1


class TestExtractAcrossShapes:
    """RES-906: one shared extractor maps every provider usage shape with a single
    fallback rule, carries cached/reasoning, and the result reconciles."""

    def test_chat_shape_with_details(self) -> None:
        usage = {
            'prompt_tokens': 100,
            'completion_tokens': 40,
            'total_tokens': 140,
            'prompt_tokens_details': {'cached_tokens': 30},
            'completion_tokens_details': {'reasoning_tokens': 12},
        }
        u = TokenUsage.extract(usage)
        assert u is not None
        assert (u.input_tokens, u.output_tokens, u.total_tokens) == (100, 40, 140)
        assert u.cached_tokens == 30
        assert u.reasoning_tokens == 12
        # reconciliation invariants
        assert u.total_tokens == u.input_tokens + u.output_tokens
        assert u.cached_tokens <= u.input_tokens
        assert u.reasoning_tokens <= u.output_tokens

    def test_responses_shape_with_details(self) -> None:
        usage = {
            'input_tokens': 80,
            'output_tokens': 20,
            'total_tokens': 100,
            'input_tokens_details': {'cached_tokens': 16},
            'output_tokens_details': {'reasoning_tokens': 5},
        }
        u = TokenUsage.extract(usage)
        assert u is not None
        assert (u.input_tokens, u.output_tokens, u.cached_tokens, u.reasoning_tokens) == (80, 20, 16, 5)
        assert u.total_tokens == u.input_tokens + u.output_tokens

    def test_langchain_usage_metadata_detail_keys(self) -> None:
        usage = {
            'input_tokens': 50,
            'output_tokens': 10,
            'total_tokens': 60,
            'input_token_details': {'cache_read': 8},
            'output_token_details': {'reasoning': 3},
        }
        u = TokenUsage.extract(usage)
        assert u is not None
        assert u.cached_tokens == 8
        assert u.reasoning_tokens == 3

    def test_vercel_camelcase_shape(self) -> None:
        u = TokenUsage.extract({'promptTokens': 12, 'completionTokens': 4, 'totalTokens': 16})
        assert u is not None
        assert (u.input_tokens, u.output_tokens, u.total_tokens) == (12, 4, 16)

    def test_unified_fallback_zero_total_synthesized(self) -> None:
        """A contradictory total=0 with nonzero components → input+output (one rule)."""
        u = TokenUsage.extract({'prompt_tokens': 7, 'completion_tokens': 5, 'total_tokens': 0})
        assert u is not None
        assert u.total_tokens == 12

    def test_provider_total_above_components_is_trusted(self) -> None:
        """A provider total larger than input+output (e.g. audio tokens) is kept."""
        u = TokenUsage.extract({'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 20})
        assert u is not None
        assert u.total_tokens == 20

    def test_extract_reads_calls_from_payload(self) -> None:
        with_calls = TokenUsage.extract({'input_tokens': 1, 'output_tokens': 1, 'calls': 3})
        assert with_calls is not None
        assert with_calls.calls == 3
        # default when absent
        default_calls = TokenUsage.extract({'input_tokens': 1, 'output_tokens': 1})
        assert default_calls is not None
        assert default_calls.calls == 1

    def test_extract_none_input_returns_none(self) -> None:
        assert TokenUsage.extract(None) is None

    def test_extract_empty_payload_returns_none(self) -> None:
        """A present-but-empty/unparseable usage block yields no signal and must NOT
        be fabricated into a confident 'billed-zero, 1 call' record — that would
        undercount tokens on a genuinely billed call while still counting the call.
        Mirrors from_openresponses' usage=None contract."""
        assert TokenUsage.extract({}) is None
        assert TokenUsage.extract({'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}) is None

    def test_extract_zero_tokens_kept_when_cost_or_cached_present(self) -> None:
        """A genuine zero-token call carrying a cost or cached tokens is real and kept."""
        with_cost = TokenUsage.extract({'input_tokens': 0, 'output_tokens': 0, 'cost': 0.002})
        assert with_cost is not None
        assert with_cost.cost_usd == pytest.approx(0.002)
        with_cached = TokenUsage.extract({
            'input_tokens': 0,
            'output_tokens': 0,
            'prompt_tokens_details': {'cached_tokens': 8},
        })
        assert with_cached is not None
        assert with_cached.cached_tokens == 8


class TestDetailsAndCostArithmetic:
    def test_add_sums_cached_reasoning_and_cost(self) -> None:
        a = TokenUsage(
            input_tokens=10, output_tokens=5, total_tokens=15, cached_tokens=4, reasoning_tokens=2, cost_usd=0.01
        )
        b = TokenUsage(
            input_tokens=20, output_tokens=8, total_tokens=28, cached_tokens=6, reasoning_tokens=1, cost_usd=0.02
        )
        c = a + b
        assert (c.input_tokens, c.output_tokens, c.total_tokens) == (30, 13, 43)
        assert (c.cached_tokens, c.reasoning_tokens) == (10, 3)
        assert c.cost_usd == pytest.approx(0.03)

    def test_cost_none_aware(self) -> None:
        a = TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2)  # cost None
        b = TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=0.05)
        assert (a + a).cost_usd is None
        assert (a + b).cost_usd == pytest.approx(0.05)
        assert (b + a).cost_usd == pytest.approx(0.05)

    def test_negative_provider_cost_clamped_not_raised(self) -> None:
        """A provider credit/adjustment (negative cost) clamps to 0 rather than
        tripping the cost_usd ge=0 constraint and crashing extraction."""
        u = TokenUsage.extract({'input_tokens': 5, 'output_tokens': 3, 'cost': -0.01})
        assert u is not None
        assert u.cost_usd == 0.0

    def test_sub_component_wise_clamped(self) -> None:
        after = TokenUsage(
            input_tokens=30, output_tokens=12, total_tokens=42, cached_tokens=8, reasoning_tokens=3, calls=2
        )
        before = TokenUsage(
            input_tokens=10, output_tokens=4, total_tokens=14, cached_tokens=3, reasoning_tokens=1, calls=1
        )
        d = after - before
        assert (d.input_tokens, d.output_tokens, d.total_tokens) == (20, 8, 28)
        assert (d.cached_tokens, d.reasoning_tokens, d.calls) == (5, 2, 1)
        # clamp: subtracting a larger value never goes negative
        assert (before - after).input_tokens == 0

    def test_sub_cost_propagates_unknown(self) -> None:
        """Cost stays None (unknown) only when BOTH sides are unknown. If either side
        carries a known cost the missing side is treated as 0.0, so a real known cost
        is never discarded — mirroring __add__'s asymmetric None-awareness."""
        known = TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=5.0)
        unknown = TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2)  # cost None
        assert (unknown - unknown).cost_usd is None
        # known - unknown: baseline unknown, after known → keep the known cost (0 baseline).
        assert (known - unknown).cost_usd == 5.0
        # unknown - known: would be negative, clamped to 0.0 (not poisoned to None).
        assert (unknown - known).cost_usd == 0.0
        assert (known - known).cost_usd == 0.0


class TestBackCompatSerialization:
    def test_construct_with_legacy_names(self) -> None:
        u = TokenUsage(prompt_tokens=9, completion_tokens=3, total_tokens=12)
        assert u.input_tokens == 9 and u.output_tokens == 3
        assert u.prompt_tokens == 9 and u.completion_tokens == 3

    def test_dump_emits_both_old_and_new_keys(self) -> None:
        d = TokenUsage(input_tokens=9, output_tokens=3, total_tokens=12).model_dump(mode='json')
        assert d['input_tokens'] == 9 and d['output_tokens'] == 3
        assert d['prompt_tokens'] == 9 and d['completion_tokens'] == 3


class TestCostBreakdownExtract:
    """RES-1194-adjacent: Orq Responses v3 per-component cost + cache-write tokens."""

    def test_extract_reads_component_costs(self) -> None:
        u = TokenUsage.extract({'input_tokens': 100, 'output_tokens': 40, 'input_cost': 0.01, 'output_cost': 0.02})
        assert u is not None
        assert u.input_cost == pytest.approx(0.01)
        assert u.output_cost == pytest.approx(0.02)
        # total_cost absent from payload -> synthesized as the sum of the components.
        assert u.total_cost == pytest.approx(0.03)

    def test_extract_reads_legacy_component_cost_aliases(self) -> None:
        u = TokenUsage.extract({'input_tokens': 1, 'output_tokens': 1, 'prompt_cost': 0.1, 'completion_cost': 0.2})
        assert u is not None
        assert u.input_cost == pytest.approx(0.1)
        assert u.output_cost == pytest.approx(0.2)
        assert u.total_cost == pytest.approx(0.3)

    def test_extract_prefers_explicit_total_cost_over_component_sum(self) -> None:
        """A provider-reported total_cost is trusted as-is, even if it does not
        equal input_cost + output_cost (e.g. it includes a surcharge)."""
        u = TokenUsage.extract(
            {'input_tokens': 1, 'output_tokens': 1, 'input_cost': 0.01, 'output_cost': 0.02, 'total_cost': 0.5}
        )
        assert u is not None
        assert u.total_cost == pytest.approx(0.5)

    def test_extract_total_cost_alias_chain(self) -> None:
        by_total_cost = TokenUsage.extract({'input_tokens': 1, 'total_cost': 0.4})
        by_cost_usd = TokenUsage.extract({'input_tokens': 1, 'cost_usd': 0.5})
        by_cost = TokenUsage.extract({'input_tokens': 1, 'cost': 0.6})
        assert by_total_cost is not None and by_total_cost.total_cost == pytest.approx(0.4)
        assert by_cost_usd is not None and by_cost_usd.total_cost == pytest.approx(0.5)
        assert by_cost is not None and by_cost.total_cost == pytest.approx(0.6)

    def test_extract_clamps_each_negative_cost_component_independently(self) -> None:
        u = TokenUsage.extract(
            {
                'input_tokens': 5,
                'output_tokens': 3,
                'input_cost': -0.01,
                'output_cost': -0.02,
                'total_cost': -0.03,
            }
        )
        assert u is not None
        assert u.input_cost == 0.0
        assert u.output_cost == 0.0
        assert u.total_cost == 0.0

    def test_extract_cache_creation_tokens_from_v3_details_shape(self) -> None:
        u = TokenUsage.extract(
            {
                'input_tokens': 100,
                'output_tokens': 40,
                'input_tokens_details': {'cache_creation_tokens': 25},
            }
        )
        assert u is not None
        assert u.cache_creation_tokens == 25

    def test_extract_cache_creation_tokens_falls_back_to_anthropic_top_level_shape(self) -> None:
        u = TokenUsage.extract(
            {
                'input_tokens': 100,
                'output_tokens': 40,
                'cache_creation_input_tokens': 17,
            }
        )
        assert u is not None
        assert u.cache_creation_tokens == 17

    def test_extract_cache_creation_tokens_defaults_to_zero(self) -> None:
        u = TokenUsage.extract({'input_tokens': 10, 'output_tokens': 5})
        assert u is not None
        assert u.cache_creation_tokens == 0

    def test_multi_component_extract_round_trip(self) -> None:
        """Full Orq Responses v3 usage payload survives extract -> dump -> reload."""
        payload = {
            'input_tokens': 200,
            'output_tokens': 80,
            'total_tokens': 280,
            'input_tokens_details': {'cached_tokens': 30, 'cache_creation_tokens': 12},
            'output_tokens_details': {'reasoning_tokens': 5},
            'input_cost': 0.002,
            'output_cost': 0.004,
            'total_cost': 0.006,
        }
        u = TokenUsage.extract(payload)
        assert u is not None
        dumped = u.model_dump(mode='json')
        restored = TokenUsage.model_validate(dumped)
        assert restored == u
        assert restored.cache_creation_tokens == 12
        assert restored.input_cost == pytest.approx(0.002)
        assert restored.output_cost == pytest.approx(0.004)
        assert restored.total_cost == pytest.approx(0.006)
        assert restored.cost_usd == pytest.approx(0.006)


class TestCostBreakdownArithmetic:
    def test_add_sums_cache_creation_tokens_and_component_costs(self) -> None:
        a = TokenUsage(
            input_tokens=10,
            output_tokens=5,
            cache_creation_tokens=3,
            input_cost=0.01,
            output_cost=0.02,
            total_cost=0.03,
        )
        b = TokenUsage(
            input_tokens=20,
            output_tokens=8,
            cache_creation_tokens=4,
            input_cost=0.05,
            output_cost=0.06,
            total_cost=0.11,
        )
        c = a + b
        assert c.cache_creation_tokens == 7
        assert c.input_cost == pytest.approx(0.06)
        assert c.output_cost == pytest.approx(0.08)
        assert c.total_cost == pytest.approx(0.14)

    def test_add_component_cost_none_aware(self) -> None:
        a = TokenUsage(input_tokens=1, output_tokens=1)  # all costs None
        b = TokenUsage(input_tokens=1, output_tokens=1, input_cost=0.5, output_cost=0.25)
        assert (a + a).input_cost is None
        assert (a + a).output_cost is None
        assert (a + b).input_cost == pytest.approx(0.5)
        assert (a + b).output_cost == pytest.approx(0.25)

    def test_sub_clamps_cache_creation_tokens_and_costs(self) -> None:
        after = TokenUsage(
            input_tokens=30, cache_creation_tokens=10, input_cost=0.5, output_cost=0.3, total_cost=0.8
        )
        before = TokenUsage(
            input_tokens=10, cache_creation_tokens=4, input_cost=0.1, output_cost=0.1, total_cost=0.2
        )
        d = after - before
        assert d.cache_creation_tokens == 6
        assert d.input_cost == pytest.approx(0.4)
        assert d.output_cost == pytest.approx(0.2)
        assert d.total_cost == pytest.approx(0.6)
        # subtracting a larger cache_creation_tokens value never goes negative
        assert (before - after).cache_creation_tokens == 0
