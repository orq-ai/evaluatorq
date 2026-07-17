"""Shared token-usage table rows for simulation report renderers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def _number(data: Mapping[str, Any], canonical_key: str, legacy_key: str | None = None) -> int | float:
    value = data.get(canonical_key)
    if value is None and legacy_key is not None:
        value = data.get(legacy_key)
    return value if isinstance(value, int | float) and not isinstance(value, bool) else 0


def build_token_usage_rows(data: Mapping[str, Any]) -> list[list[str]]:
    """Build canonical token-usage rows, reading legacy keys from saved reports."""
    rows = [
        ['Input Tokens (total)', f'{_number(data, "input_tokens", "prompt_tokens"):,}'],
        ['Output Tokens (total)', f'{_number(data, "output_tokens", "completion_tokens"):,}'],
        ['Total Tokens', f'{_number(data, "total_tokens"):,}'],
        ['Avg Total / Conversation', f'{_number(data, "avg_total_per_conversation"):.0f}'],
        [
            'Avg Input / Conversation',
            f'{_number(data, "avg_input_per_conversation", "avg_prompt_per_conversation"):.0f}',
        ],
        [
            'Avg Output / Conversation',
            f'{_number(data, "avg_output_per_conversation", "avg_completion_per_conversation"):.0f}',
        ],
    ]
    cached = _number(data, 'cached_tokens')
    if cached:
        rows.append(['Cached Tokens (retrieved)', f'{cached:,}'])
    reasoning = _number(data, 'reasoning_tokens')
    if reasoning:
        rows.append(['Reasoning Tokens', f'{reasoning:,}'])
    return rows
