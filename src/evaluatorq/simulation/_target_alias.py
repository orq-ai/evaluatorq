"""Compatibility handling for the former ``target_callback`` keyword."""

from __future__ import annotations

import warnings
from typing import Any

_MISSING = object()


def resolve_target_alias(*, target: Any, deprecated_kwargs: dict[str, Any], caller: str) -> Any:
    """Return ``target`` while accepting the deprecated callback keyword."""
    legacy_target = deprecated_kwargs.pop('target_callback', _MISSING)
    if deprecated_kwargs:
        names = ', '.join(repr(name) for name in sorted(deprecated_kwargs))
        raise TypeError(f'{caller}() got unexpected keyword argument(s): {names}')
    if legacy_target is not _MISSING:
        warnings.warn(
            f'{caller}(target_callback=...) is deprecated; use {caller}(target=...) instead.',
            DeprecationWarning,
            stacklevel=3,
        )
        if target is None:
            return legacy_target
    return target


def resolve_renamed_kwarg(
    *, new_value: Any, deprecated_kwargs: dict[str, Any], old_name: str, new_name: str, caller: str
) -> Any:
    """Return new_value, or the popped legacy value if the caller passed the old
    keyword. Emits a one-time DeprecationWarning when the old name is used.
    Does NOT raise on other leftover kwargs — call resolve_target_alias last for that.
    """
    legacy = deprecated_kwargs.pop(old_name, _MISSING)
    if legacy is _MISSING:
        return new_value
    warnings.warn(
        f'{caller}({old_name}=...) is deprecated; use {caller}({new_name}=...) instead.',
        DeprecationWarning,
        stacklevel=3,
    )
    return legacy if new_value is None else new_value
