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
