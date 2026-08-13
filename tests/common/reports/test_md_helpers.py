# tests/common/reports/test_md_helpers.py
from __future__ import annotations

from evaluatorq.common.reports.md_helpers import cost_coverage


def test_cost_coverage_no_priced_calls():
    """priced_calls == 0 means no coverage data at all — no noise."""
    assert cost_coverage(0, 10) == ''
    assert cost_coverage(0, 10, estimated_calls=5) == ''


def test_cost_coverage_fully_priced_by_provider():
    """Every call priced, all by the provider — nothing to qualify."""
    assert cost_coverage(10, 10) == ''
    assert cost_coverage(10, 10, estimated_calls=0) == ''


def test_cost_coverage_partial_provider_only():
    """Existing behavior: partial coverage, no estimation involved."""
    assert cost_coverage(3, 10) == ' (3 of 10 calls)'


def test_cost_coverage_fully_priced_fully_estimated():
    assert cost_coverage(10, 10, estimated_calls=10) == ' (estimated)'


def test_cost_coverage_fully_priced_mixed_estimated():
    assert cost_coverage(10, 10, estimated_calls=4) == ' (partly estimated)'


def test_cost_coverage_partial_fully_estimated():
    assert cost_coverage(3, 10, estimated_calls=3) == ' (3 of 10 calls, estimated)'


def test_cost_coverage_partial_mixed_estimated():
    assert cost_coverage(3, 10, estimated_calls=1) == ' (3 of 10 calls, partly estimated)'


def test_cost_coverage_estimated_calls_exceeding_priced_still_reads_as_fully_estimated():
    # Defensive: an inconsistent caller passing estimated_calls > priced_calls
    # should still resolve to "fully estimated" rather than a bogus label.
    assert cost_coverage(3, 10, estimated_calls=5) == ' (3 of 10 calls, estimated)'
