# ruff: noqa: S101

from __future__ import annotations

from evaluatorq.contracts import Usage


class TestUsageCostUsdSerialization:
    """Pin the exact contract for cost_usd key serialization.

    The dashboard reads saved report JSON via `_cost_usd()`
    (src/evaluatorq/dashboard/metrics.py), which does `.get('cost_usd')` rather
    than reading a model field. If the `cost_usd` key disappears, cost data
    silently vanishes from every report.

    `cost_usd` is always emitted, even when `total_cost` is None, so a bare
    `'cost_usd' in usage` membership check does not mean cost is known — the
    dashboard must (and does) distinguish None from 0.0.
    """

    def test_cost_usd_always_in_dump(self) -> None:
        """The 'cost_usd' key must always be present in model_dump(mode='json')."""
        u_with_cost = Usage(
            input_tokens=10, output_tokens=5, total_tokens=15, total_cost=0.5
        )
        u_without_cost = Usage(
            input_tokens=10, output_tokens=5, total_tokens=15, total_cost=None
        )
        u_zero_cost = Usage(
            input_tokens=10, output_tokens=5, total_tokens=15, total_cost=0.0
        )

        assert 'cost_usd' in u_with_cost.model_dump(mode='json')
        assert 'cost_usd' in u_without_cost.model_dump(mode='json')
        assert 'cost_usd' in u_zero_cost.model_dump(mode='json')

    def test_cost_usd_none_when_unknown_not_missing(self) -> None:
        """When cost is unknown (None), dumped cost_usd must be None, never omitted.

        Dashboard's membership check ('cost_usd' in usage) must match regardless
        of whether cost is known. If key is missing, membership fails.
        """
        u = Usage(input_tokens=100, output_tokens=40, total_cost=None)
        dumped = u.model_dump(mode='json')

        assert 'cost_usd' in dumped
        assert dumped['cost_usd'] is None

    def test_cost_usd_none_vs_zero_distinction(self) -> None:
        """Distinguish None (unknown) from 0.0 (reported as free).

        Dashboard receives both values and must handle them distinctly.
        """
        u_unknown = Usage(input_tokens=10, output_tokens=5, total_cost=None)
        u_free = Usage(input_tokens=10, output_tokens=5, total_cost=0.0)

        dumped_unknown = u_unknown.model_dump(mode='json')
        dumped_free = u_free.model_dump(mode='json')

        assert dumped_unknown['cost_usd'] is None
        assert dumped_free['cost_usd'] == 0.0
        assert dumped_free['cost_usd'] is not None

    def test_dump_contains_exact_expected_key_set(self) -> None:
        """Verify the exact set of keys in model_dump(mode='json').

        Guarantee 1: dumped Usage must emit both canonical field names AND
        the three legacy keys (prompt_tokens, completion_tokens, cost_usd)
        that the @model_serializer(mode='wrap') injects.

        This test fails if any key is missing (new fields not injected) or
        if unexpected keys appear (serializer behavior changed).
        """
        u = Usage(
            input_tokens=100,
            output_tokens=40,
            total_tokens=140,
            cached_tokens=30,
            cache_creation_tokens=12,
            reasoning_tokens=5,
            input_cost=0.002,
            output_cost=0.004,
            total_cost=0.006,
            calls=2,
        )
        dumped = u.model_dump(mode='json')

        # Expected keys: all canonical fields + legacy injected keys
        expected_keys = {
            # Canonical field names
            'input_tokens',
            'output_tokens',
            'total_tokens',
            'cached_tokens',
            'cache_creation_tokens',
            'cache_creation_1h_tokens',
            'cache_creation_5m_tokens',
            'reasoning_tokens',
            'input_cost',
            'output_cost',
            'total_cost',
            'calls',
            'priced_calls',
            # Legacy keys injected by @model_serializer(mode='wrap')
            'prompt_tokens',
            'completion_tokens',
            'cost_usd',
        }

        assert set(dumped.keys()) == expected_keys


class TestUsageOldFormatDeserialization:
    """Pin that old-format saved payloads (pre-breakdown) still load.

    Before this task, saved reports contained only prompt_tokens, completion_tokens,
    and cost_usd. The new breakdown fields (input_cost, output_cost) did not exist.
    Old reports must still deserialize without error.
    """

    def test_old_format_via_model_validate(self) -> None:
        """Old-format dict (prompt_tokens/completion_tokens/cost_usd) loads via model_validate."""
        old_payload = {
            'prompt_tokens': 100,
            'completion_tokens': 40,
            'total_tokens': 140,
            'cost_usd': 0.5,
            # No input_cost, output_cost, cache_creation_tokens, reasoning_tokens
        }
        u = Usage.model_validate(old_payload)

        # Legacy keys map to canonical names via validation aliases
        assert u.input_tokens == 100
        assert u.output_tokens == 40
        assert u.total_cost == 0.5

        # New breakdown fields default: cost components are None, token details are 0
        assert u.input_cost is None
        assert u.output_cost is None
        assert u.cache_creation_tokens == 0
        assert u.cached_tokens == 0
        assert u.reasoning_tokens == 0

    def test_old_format_without_cost_via_model_validate(self) -> None:
        """Old-format dict without cost_usd key loads with total_cost=None."""
        old_payload = {
            'prompt_tokens': 100,
            'completion_tokens': 40,
            'total_tokens': 140,
        }
        u = Usage.model_validate(old_payload)

        assert u.input_tokens == 100
        assert u.output_tokens == 40
        assert u.total_cost is None
        assert u.input_cost is None
        assert u.output_cost is None


class TestUsageSerializationRoundTrip:
    """Verify that dumped Usage round-trips unchanged through model_validate.

    Complements the existing test_multi_component_extract_round_trip in
    test_token_usage_arithmetic.py (which uses extract() path). This pins
    round-trip behavior when not going through extract().
    """

    def test_round_trip_cost_none_unchanged(self) -> None:
        """Round-trip preserves None cost (unknown), distinguishing from 0.0 (free)."""
        original = Usage(
            input_tokens=100, output_tokens=40, total_tokens=140, total_cost=None
        )
        dumped = original.model_dump(mode='json')
        restored = Usage.model_validate(dumped)

        assert restored == original
        assert restored.total_cost is None

    def test_round_trip_cost_zero_unchanged(self) -> None:
        """Round-trip preserves 0.0 cost distinct from None."""
        original = Usage(
            input_tokens=100, output_tokens=40, total_tokens=140, total_cost=0.0
        )
        dumped = original.model_dump(mode='json')
        restored = Usage.model_validate(dumped)

        assert restored == original
        assert restored.total_cost == 0.0
        assert restored.total_cost is not None
