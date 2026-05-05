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

    def test_returns_zero_fields_when_usage_fields_none(self) -> None:
        """Fields that are None on usage object fall back to 0."""
        response = MagicMock()
        usage = MagicMock()
        usage.prompt_tokens = None
        usage.completion_tokens = None
        usage.total_tokens = None
        response.usage = usage
        result = TokenUsage.from_completion(response)
        assert result is not None
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0

    def test_calls_is_always_one(self) -> None:
        response = MagicMock()
        response.usage = self._make_usage()
        result = TokenUsage.from_completion(response)
        assert result is not None
        assert result.calls == 1
