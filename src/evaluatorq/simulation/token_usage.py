"""Shared normalization helpers for simulation token-usage data."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any


def token_value(data: Mapping[str, Any], canonical_key: str, legacy_key: str | None = None) -> int | float:
    """Read a numeric token value, falling back to a legacy saved-report key."""
    value = data.get(canonical_key)
    if value is None and legacy_key is not None:
        value = data.get(legacy_key)
    return value if isinstance(value, int | float) and not isinstance(value, bool) else 0
