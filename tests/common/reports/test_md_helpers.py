"""Direct coverage for `fmt_cached_tokens`.

The report-level tests only ever exercise one clean ratio. The branches that
decide whether an operator reads a real measurement or a fabricated one — a
missing denominator, a sub-1% share — are only reachable from here.
"""

from __future__ import annotations

import pytest

from evaluatorq.common.reports import fmt_cached_tokens


def test_share_is_rendered_against_input():
    assert fmt_cached_tokens(75, 100) == '75 (75% of input tokens)'


@pytest.mark.parametrize('input_tokens', [0, -1])
def test_no_denominator_means_no_share(input_tokens: int):
    """A missing input count is unknown, not zero — and dividing by it would raise."""
    assert fmt_cached_tokens(500, input_tokens) == '500'


def test_a_tiny_share_is_not_rounded_away_into_looking_uncached():
    """`0%` beside a non-zero count reads as a bug; the count is what carries the signal."""
    assert fmt_cached_tokens(4, 10_000) == '4 (0% of input tokens)'


def test_counts_are_rounded_not_truncated():
    """Saved reports carry these as JSON numbers, so a float reaches the formatter."""
    assert fmt_cached_tokens(74.6, 100) == '75 (75% of input tokens)'


def test_thousands_separator_on_both_surfaces():
    assert fmt_cached_tokens(1_234_567, 2_000_000) == '1,234,567 (62% of input tokens)'
