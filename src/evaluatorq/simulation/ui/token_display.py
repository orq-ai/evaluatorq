"""Dependency-free token display helpers for the simulation dashboard."""

from __future__ import annotations

from typing import Any


def _token_value(data: dict[str, Any], canonical_key: str, legacy_key: str | None = None) -> int:
    value = data.get(canonical_key)
    if value is None and legacy_key is not None:
        value = data.get(legacy_key)
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0


def token_metric_specs(data: dict[str, Any]) -> list[tuple[str, str]]:
    """Return dashboard metric labels and values using canonical token names."""
    input_tokens = _token_value(data, 'input_tokens', 'prompt_tokens')
    output_tokens = _token_value(data, 'output_tokens', 'completion_tokens')
    metrics = [
        ('Input', f'{input_tokens:,}'),
        ('Output', f'{output_tokens:,}'),
        ('Total', f'{_token_value(data, "total_tokens"):,}'),
    ]
    cached = _token_value(data, 'cached_tokens')
    if cached:
        metrics.append(('Cached (retrieved)', f'{cached:,}'))
    reasoning = _token_value(data, 'reasoning_tokens')
    if reasoning:
        metrics.append(('Reasoning', f'{reasoning:,}'))
    return metrics


def token_overview_caption(data: dict[str, Any]) -> str:
    """Build the compact token summary shown on the dashboard overview."""
    input_tokens = _token_value(data, 'input_tokens', 'prompt_tokens')
    output_tokens = _token_value(data, 'output_tokens', 'completion_tokens')
    parts = [f'Input {input_tokens:,}', f'Output {output_tokens:,}']
    cached = _token_value(data, 'cached_tokens')
    if cached:
        parts.append(f'Cached (retrieved) {cached:,}')
    reasoning = _token_value(data, 'reasoning_tokens')
    if reasoning:
        parts.append(f'Reasoning {reasoning:,}')
    avg = data.get('avg_total_per_conversation', 0.0)
    avg_value = float(avg) if isinstance(avg, int | float) and not isinstance(avg, bool) else 0.0
    parts.append(f'Avg {avg_value:.0f}/conv')
    return ' · '.join(parts)
