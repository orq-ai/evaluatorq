from __future__ import annotations

from evaluatorq.contracts import Usage
from evaluatorq.insights.usage import UsageLedger


def test_ledger_sums_per_stage_and_keeps_unpriced_visible() -> None:
    ledger = UsageLedger()
    ledger.add(
        'summary', Usage(input_tokens=10, output_tokens=5, total_tokens=15, total_cost=0.01, calls=1, priced_calls=1)
    )
    ledger.add('summary', None)
    ledger.add('merge', None)

    totals = ledger.totals()

    assert totals['summary'] is not None and totals['summary'].calls == 2 and totals['summary'].cost_is_partial
    assert totals['summary'] is not None and totals['summary'].priced_calls == 1
    assert totals['merge'] is not None and totals['merge'].calls == 1 and totals['merge'].total_cost is None
    assert 'label' not in totals


def test_ledger_claims_each_warning_key_once_per_run() -> None:
    ledger = UsageLedger()

    assert ledger.claim_warning('unpriced_embedding_cost') is True
    assert ledger.claim_warning('unpriced_embedding_cost') is False
    assert ledger.claim_warning('other_warning') is True
