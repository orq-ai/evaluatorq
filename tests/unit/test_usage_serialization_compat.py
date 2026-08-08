# ruff: noqa: S101

from __future__ import annotations

import pytest

from evaluatorq.contracts import Usage


class TestUsageSerializationLegacyKeys:
    """Verify that dumped Usage contains legacy keys for dashboard compatibility.

    The dashboard (src/evaluatorq/dashboard/metrics.py:92-101,519,664) reads
    saved report JSON by dict key membership and .get(), not by model field.
    These tests pin the exact serialization contract that downstream readers depend on.
    """

    def test_model_dump_includes_prompt_tokens_completion_tokens_cost_usd(self) -> None:
        """Dumped Usage must include legacy keys alongside canonical names."""
        u = Usage(input_tokens=100, output_tokens=40, total_tokens=140, total_cost=0.006)
        dumped = u.model_dump(mode='json')

        # Legacy keys must be present
        assert 'prompt_tokens' in dumped
        assert 'completion_tokens' in dumped
        assert 'cost_usd' in dumped

        # Legacy keys must match canonical names
        assert dumped['prompt_tokens'] == dumped['input_tokens'] == 100
        assert dumped['completion_tokens'] == dumped['output_tokens'] == 40
        assert dumped['cost_usd'] == dumped['total_cost'] == 0.006

    def test_model_dump_includes_new_cost_breakdown_keys(self) -> None:
        """Dumped Usage must include new cost breakdown fields."""
        u = Usage(
            input_tokens=100,
            output_tokens=40,
            total_tokens=140,
            input_cost=0.002,
            output_cost=0.004,
            total_cost=0.006,
            cache_creation_tokens=12,
        )
        dumped = u.model_dump(mode='json')

        # New breakdown keys must be present
        assert 'input_cost' in dumped
        assert 'output_cost' in dumped
        assert 'total_cost' in dumped
        assert 'cache_creation_tokens' in dumped

        # Verify values
        assert dumped['input_cost'] == pytest.approx(0.002)
        assert dumped['output_cost'] == pytest.approx(0.004)
        assert dumped['total_cost'] == pytest.approx(0.006)
        assert dumped['cache_creation_tokens'] == 12

    def test_cost_usd_present_when_cost_known(self) -> None:
        """When cost is known, 'cost_usd' key must be present and non-None.

        This is the exact shape dashboard/metrics.py:92-101,519,664 checks via
        'cost_usd' in usage and usage.get('cost_usd').
        """
        u = Usage(input_tokens=10, output_tokens=5, total_tokens=15, total_cost=0.5)
        dumped = u.model_dump(mode='json')

        assert 'cost_usd' in dumped
        assert dumped['cost_usd'] is not None
        assert dumped['cost_usd'] == 0.5

    def test_cost_usd_present_with_none_when_cost_unknown(self) -> None:
        """When cost is unknown, 'cost_usd' key must be present with None value.

        None means "not reported" (not fabricated as 0). The dashboard checks
        membership ('cost_usd' in usage) to decide if a run counts as "costed".
        """
        u = Usage(input_tokens=10, output_tokens=5, total_tokens=15)
        dumped = u.model_dump(mode='json')

        assert 'cost_usd' in dumped
        assert dumped['cost_usd'] is None

    def test_zero_cost_reported_as_zero_not_none(self) -> None:
        """When cost is explicitly reported as free (0.0), distinguish from unknown (None).

        0.0 means "reported as free", None means "not reported".
        """
        u = Usage(input_tokens=10, output_tokens=5, total_tokens=15, total_cost=0.0)
        dumped = u.model_dump(mode='json')

        assert 'cost_usd' in dumped
        assert dumped['cost_usd'] == 0.0
        assert dumped['cost_usd'] is not None


class TestUsageSerializationRoundTrip:
    """Verify that dumped Usage round-trips unchanged through model_validate."""

    def test_round_trip_preserves_all_fields(self) -> None:
        """Dumped Usage must round-trip unchanged through model_validate."""
        original = Usage(
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
        dumped = original.model_dump(mode='json')
        restored = Usage.model_validate(dumped)

        assert restored == original
        assert restored.input_tokens == 100
        assert restored.output_tokens == 40
        assert restored.total_tokens == 140
        assert restored.cached_tokens == 30
        assert restored.cache_creation_tokens == 12
        assert restored.reasoning_tokens == 5
        assert restored.input_cost == pytest.approx(0.002)
        assert restored.output_cost == pytest.approx(0.004)
        assert restored.total_cost == pytest.approx(0.006)
        assert restored.calls == 2

    def test_round_trip_with_unknown_cost(self) -> None:
        """Round-trip preserves None cost values."""
        original = Usage(
            input_tokens=100, output_tokens=40, total_tokens=140, total_cost=None
        )
        dumped = original.model_dump(mode='json')
        restored = Usage.model_validate(dumped)

        assert restored == original
        assert restored.total_cost is None

    def test_round_trip_with_zero_cost(self) -> None:
        """Round-trip preserves 0.0 cost (free) distinct from None (unknown)."""
        original = Usage(
            input_tokens=100, output_tokens=40, total_tokens=140, total_cost=0.0
        )
        dumped = original.model_dump(mode='json')
        restored = Usage.model_validate(dumped)

        assert restored == original
        assert restored.total_cost == 0.0
        assert restored.total_cost is not None


class TestUsageBackCompatDeserialization:
    """Verify that old-format saved payloads (pre-breakdown) still load.

    An old-format payload contains only prompt_tokens/completion_tokens/cost_usd
    (no cost breakdown). The new breakdown fields (input_cost, output_cost)
    should come back as None.
    """

    def test_old_format_payload_loads_with_legacy_keys(self) -> None:
        """Old-format payload with legacy keys must load and populate new fields as None."""
        old_payload = {
            'prompt_tokens': 100,
            'completion_tokens': 40,
            'total_tokens': 140,
            'cost_usd': 0.5,
            # No input_cost, output_cost, cache_creation_tokens
        }
        u = Usage.model_validate(old_payload)

        # Legacy keys load into new canonical names
        assert u.input_tokens == 100
        assert u.output_tokens == 40
        assert u.total_tokens == 140
        assert u.total_cost == 0.5

        # New breakdown fields default to None (unknown)
        assert u.input_cost is None
        assert u.output_cost is None
        assert u.cache_creation_tokens == 0  # Defaults to 0, not None
        assert u.cached_tokens == 0  # Defaults to 0, not None

    def test_old_format_payload_with_unknown_cost(self) -> None:
        """Old-format payload without cost loads with total_cost=None."""
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

    def test_old_format_payload_constructs_via_legacy_aliases(self) -> None:
        """Construction with legacy prompt_tokens/completion_tokens/cost_usd works."""
        u = Usage(prompt_tokens=100, completion_tokens=40, total_tokens=140, cost_usd=0.5)

        assert u.input_tokens == 100
        assert u.output_tokens == 40
        assert u.total_cost == 0.5
        # Check the deprecated property aliases work
        assert u.prompt_tokens == 100
        assert u.completion_tokens == 40
        assert u.cost_usd == 0.5


class TestUsageSerializationDashboardContract:
    """Verify the exact shapes the dashboard depends on.

    The dashboard checks membership ('cost_usd' in usage_dict) and calls
    usage.get('cost_usd') to decide whether a run counts as "costed" and to
    extract its cost. These tests ensure the serialization never silently
    breaks that contract.
    """

    def test_dashboard_cost_membership_check_when_known(self) -> None:
        """Dashboard check: 'cost_usd' in usage dict when cost is known."""
        u = Usage(input_tokens=100, output_tokens=40, total_cost=0.5)
        dumped = u.model_dump(mode='json')

        # This is exactly what dashboard/metrics.py:519,664 do
        if isinstance(dumped, dict) and 'cost_usd' in dumped:
            cost = dumped.get('cost_usd')
            assert cost == 0.5
        else:
            pytest.fail("Dashboard cost membership check failed")

    def test_dashboard_cost_membership_check_when_unknown(self) -> None:
        """Dashboard check: 'cost_usd' in usage dict even when cost is unknown (None)."""
        u = Usage(input_tokens=100, output_tokens=40)
        dumped = u.model_dump(mode='json')

        # Dashboard should still see 'cost_usd' key, just with None value
        assert 'cost_usd' in dumped
        # When checking membership, dashboard skips the run
        if isinstance(dumped, dict) and 'cost_usd' in dumped:
            cost = dumped.get('cost_usd')
            # But when cost is None, it shouldn't count as "costed"
            assert cost is None

    def test_dashboard_can_extract_cost_with_get(self) -> None:
        """Dashboard calls usage.get('cost_usd') to extract cost safely."""
        u = Usage(input_tokens=100, output_tokens=40, total_cost=0.123)
        dumped = u.model_dump(mode='json')

        # This is what dashboard/metrics.py:92-101 does via _cost_usd()
        if isinstance(dumped, dict):
            cost_value = dumped.get('cost_usd')
            assert cost_value == pytest.approx(0.123)

    def test_dashboard_handles_missing_cost_gracefully(self) -> None:
        """If 'cost_usd' key is somehow missing, .get() returns None gracefully."""
        u = Usage(input_tokens=100, output_tokens=40)
        dumped = u.model_dump(mode='json')

        # Ensure the key is actually present (it should be per the contract)
        assert 'cost_usd' in dumped

        # But if it were missing, .get() would return None
        if isinstance(dumped, dict):
            cost_value = dumped.get('cost_usd')
            assert cost_value is None


class TestUsageNewBreakdownFieldsBothPresent:
    """Verify both new breakdown AND legacy keys coexist in the dump."""

    def test_dump_contains_both_legacy_and_new_keys(self) -> None:
        """Full dump must contain all legacy and new keys together."""
        u = Usage(
            input_tokens=100,
            output_tokens=40,
            total_tokens=140,
            cached_tokens=30,
            cache_creation_tokens=12,
            input_cost=0.002,
            output_cost=0.004,
            total_cost=0.006,
        )
        dumped = u.model_dump(mode='json')

        # Legacy keys
        assert dumped['prompt_tokens'] == 100
        assert dumped['completion_tokens'] == 40
        assert dumped['cost_usd'] == pytest.approx(0.006)

        # New keys
        assert dumped['input_tokens'] == 100
        assert dumped['output_tokens'] == 40
        assert dumped['total_cost'] == pytest.approx(0.006)
        assert dumped['input_cost'] == pytest.approx(0.002)
        assert dumped['output_cost'] == pytest.approx(0.004)
        assert dumped['cache_creation_tokens'] == 12

    def test_component_costs_present_even_without_total_cost(self) -> None:
        """Component costs can be present independently of total_cost."""
        u = Usage(
            input_tokens=100,
            output_tokens=40,
            total_tokens=140,
            input_cost=0.002,
            output_cost=0.004,
        )
        dumped = u.model_dump(mode='json')

        # All new breakdown keys present
        assert 'cost_usd' in dumped
        assert 'total_cost' in dumped
        assert 'input_cost' in dumped
        assert 'output_cost' in dumped

        # Component costs are reported
        assert dumped['input_cost'] == pytest.approx(0.002)
        assert dumped['output_cost'] == pytest.approx(0.004)

        # But total_cost is None when not explicitly provided
        assert dumped['total_cost'] is None
        assert dumped['cost_usd'] is None
