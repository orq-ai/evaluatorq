# tests/common/reports/test_md_helpers.py
from __future__ import annotations

import pytest

from evaluatorq.common.reports.md_helpers import cost_coverage


@pytest.mark.parametrize(
    ('priced', 'calls', 'estimated', 'expected'),
    [
        # priced_calls == 0 means no coverage data at all — no noise.
        pytest.param(0, 10, 0, '', id='no_priced_calls'),
        pytest.param(0, 10, 5, '', id='no_priced_calls_ignores_estimated'),
        # Every call priced, all by the provider — nothing to qualify.
        pytest.param(10, 10, 0, '', id='fully_priced_by_provider'),
        # Existing behavior: partial coverage, no estimation involved.
        pytest.param(3, 10, 0, ' (3 of 10 calls)', id='partial_provider_only'),
        pytest.param(10, 10, 10, ' (estimated)', id='fully_priced_fully_estimated'),
        pytest.param(10, 10, 4, ' (partly estimated)', id='fully_priced_mixed_estimated'),
        pytest.param(3, 10, 3, ' (3 of 10 calls, estimated)', id='partial_fully_estimated'),
        pytest.param(3, 10, 1, ' (3 of 10 calls, partly estimated)', id='partial_mixed_estimated'),
        # Defensive: an inconsistent caller passing estimated_calls > priced_calls
        # should still resolve to "fully estimated" rather than a bogus label.
        pytest.param(3, 10, 5, ' (3 of 10 calls, estimated)', id='estimated_exceeding_priced_reads_as_fully_estimated'),
    ],
)
def test_cost_coverage(priced: int, calls: int, estimated: int, expected: str) -> None:
    assert cost_coverage(priced, calls, estimated_calls=estimated) == expected
