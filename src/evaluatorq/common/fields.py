"""Shared field-accessor utility for dict/object dual-natured payloads."""

from __future__ import annotations

from typing import Any


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Get a named field from a dict or object attribute."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
