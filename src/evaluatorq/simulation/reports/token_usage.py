"""Shared token-usage table rows for simulation report renderers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

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
    cached = token_value(data, 'cached_tokens')
    if cached:
        rows.append(['Cached Tokens (retrieved)', f'{cached:,}'])
    reasoning = token_value(data, 'reasoning_tokens')
    if reasoning:
        rows.append(['Reasoning Tokens', f'{reasoning:,}'])
    return rows
