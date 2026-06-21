"""Shared Rich color styling for evaluatorq report summaries.

Single source of truth for the threshold→color mapping used by both the
redteam and simulation console renderers. Brand colors live in ``palette.py``;
this maps a 0..1 rate to a Rich style name for terminal tables.
"""

from __future__ import annotations


def rate_style(value: float, *, higher_is_better: bool = True) -> str:
    """Rich color for a 0..1 rate. Green good, yellow middling, red bad.

    ``higher_is_better=True`` suits success/coverage rates; ``False`` suits
    error/attack rates where low is good (replaces the old ``_asr_style``).
    """
    score = value if higher_is_better else 1.0 - value
    if score >= 0.8:
        return "green"
    if score >= 0.5:
        return "yellow"
    return "red"
