from __future__ import annotations

from evaluatorq.contracts import Usage
from evaluatorq.insights.usage import UsageLedger


def test_ledger_sums_per_stage_and_keeps_unpriced_visible() -> None:
    ledger = UsageLedger()
    ledger.add(
        'summary', Usage(input_tokens=10, output_tokens=5, total_tokens=15, total_cost=0.01, calls=1, priced_calls=1)
    )
    ledger.add(
        'summary', Usage(input_tokens=10, output_tokens=5, total_tokens=15, total_cost=None, calls=1, priced_calls=0)
    )
    ledger.add('merge', None)

    totals = ledger.totals()

    assert totals['summary'] is not None and totals['summary'].calls == 2 and totals['summary'].cost_is_partial
    assert totals['merge'] is None
    assert 'label' not in totals
