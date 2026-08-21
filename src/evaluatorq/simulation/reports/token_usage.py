"""Shared token-usage table rows for simulation report renderers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

from evaluatorq.common.reports import cost_coverage, fmt_cached_tokens, fmt_cost
from evaluatorq.simulation.token_usage import token_value


def build_token_usage_rows(data: Mapping[str, Any]) -> list[list[str]]:
    """Build canonical token-usage rows, reading legacy keys from saved reports."""
    rows = [
        ['Input Tokens (total)', f'{token_value(data, "input_tokens", "prompt_tokens"):,}'],
        ['Output Tokens (total)', f'{token_value(data, "output_tokens", "completion_tokens"):,}'],
        ['Total Tokens', f'{token_value(data, "total_tokens"):,}'],
        ['Avg Total / Conversation', f'{token_value(data, "avg_total_per_conversation"):.0f}'],
        [
            'Avg Input / Conversation',
            f'{token_value(data, "avg_input_per_conversation", "avg_prompt_per_conversation"):.0f}',
        ],
        [
            'Avg Output / Conversation',
            f'{token_value(data, "avg_output_per_conversation", "avg_completion_per_conversation"):.0f}',
        ],
    ]
    unknown = data.get('unknown_usage_conversations', 0)
    if isinstance(unknown, int) and unknown > 0:
        label = 'conversation' if unknown == 1 else 'conversations'
        rows.append(['Usage Coverage', f'unknown for {unknown} {label}'])
    cached = token_value(data, 'cached_tokens')
    if cached:
        input_tokens = token_value(data, 'input_tokens', 'prompt_tokens')
        rows.append(['Cached Tokens (retrieved)', fmt_cached_tokens(int(cached), int(input_tokens))])
    cache_creation = token_value(data, 'cache_creation_tokens')
    if cache_creation:
        rows.append(['Cache-Write Tokens', f'{cache_creation:,}'])
    reasoning = token_value(data, 'reasoning_tokens')
    if reasoning:
        rows.append(['Reasoning Tokens', f'{reasoning:,}'])

    # Cost is only ever shown when the provider actually reported it — `None`
    # ("unknown") must never be rendered as `$0.00`. Old saved reports predate
    # the cost breakdown entirely, so `.get` returning None here is expected
    # and simply omits the row rather than fabricating a value.
    input_cost = data.get('input_cost')
    if input_cost is not None and isinstance(input_cost, int | float) and not isinstance(input_cost, bool):
        rows.append(['Input Cost', fmt_cost(input_cost)])
    output_cost = data.get('output_cost')
    if output_cost is not None and isinstance(output_cost, int | float) and not isinstance(output_cost, bool):
        rows.append(['Output Cost', fmt_cost(output_cost)])
    total_cost = data.get('total_cost')
    if total_cost is not None and isinstance(total_cost, int | float) and not isinstance(total_cost, bool):
        # A cost summed across calls where only some reported one is a lower
        # bound, not a total — say so rather than let it read as authoritative.
        coverage = cost_coverage(int(token_value(data, 'priced_calls')), int(token_value(data, 'calls')))
        rows.append(['Total Cost', f'{fmt_cost(total_cost)}{coverage}'])
    return rows
