"""Async helpers shared across the evaluatorq subpackages.

Hooks (and other injected callbacks) may be implemented as either ``def`` or
``async def``. ``await_maybe`` lets a single call site drive both shapes, and
``warn_if_sync_hooks`` nudges implementers toward the async form.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Awaitable
from typing import TypeAlias, TypeVar

_T = TypeVar('_T')

MaybeAsync: TypeAlias = _T | Awaitable[_T]
"""A value that may be returned directly or as an awaitable.

Use as a hook/callback return annotation so both sync and async implementations
type-check, e.g. ``def on_confirm(self, ...) -> MaybeAsync[bool]: ...``. Drive
it with :func:`await_maybe`.
"""


async def await_maybe(value: MaybeAsync[_T]) -> _T:
    """Await ``value`` if it is awaitable, else return it unchanged.

    Lets one call site drive both sync and async implementations: a sync hook
    returns its result directly, an async hook returns a coroutine that is
    awaited here. The return type is the resolved ``_T`` either way, so the
    caller keeps the real type (e.g. ``bool`` for an ``on_confirm`` gate).
    Exceptions propagate identically for both shapes.
    """
    if inspect.isawaitable(value):
        return await value
    # isawaitable is False here, so value is the bare _T; the checker cannot
    # narrow the union via isawaitable, hence the cast-free ignore.
    return value  # type: ignore[return-value]


def warn_if_sync_hooks(hooks: object, method_names: tuple[str, ...]) -> None:
    """Emit a one-shot ``DeprecationWarning`` if any hook method is synchronous.

    Sync hooks remain supported (driven via :func:`await_maybe`); this is purely
    a nudge toward ``async def``. Inspects the bound methods directly with
    ``iscoroutinefunction`` — we check the method, not a return value.
    """
    sync = [
        name
        for name in method_names
        if callable(getattr(hooks, name, None)) and not inspect.iscoroutinefunction(getattr(hooks, name))
    ]
    if sync:
        warnings.warn(
            f'{type(hooks).__name__} implements sync hook(s) {sync}; '
            "sync hooks are supported but deprecated — define them as 'async def'.",
            DeprecationWarning,
            stacklevel=3,
        )
