"""Async helpers shared across the evaluatorq subpackages.

Hooks (and other injected callbacks) may be implemented as either ``def`` or
``async def``. ``await_maybe`` lets a single call site drive both shapes, and
``warn_if_sync_hooks`` nudges implementers toward the async form.
"""

from __future__ import annotations

import asyncio
import inspect
import warnings
from collections.abc import Awaitable, Iterable, Sequence
from typing import Any, TypeAlias, TypeVar

from loguru import logger

_T = TypeVar('_T')

MaybeAsync: TypeAlias = _T | Awaitable[_T]
"""A value that may be returned directly or as an awaitable.

Use as a hook/callback return annotation so both sync and async implementations
type-check, e.g. ``def on_confirm(self, ...) -> MaybeAsync[bool]: ...``. Drive
it with `await_maybe`.
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


async def fan_out(children: Iterable[Any], method_name: str, *args: Any, **kwargs: Any) -> None:
    """Call ``method_name`` on every child (via `await_maybe`).

    Runs ALL children, captures the FIRST exception, then re-raises it after the
    loop — a uniform run-all-then-reraise policy for every void hook method. No
    ``getattr(..., None)`` skip branch: children are full protocol
    implementations, so a missing method is a real bug and should raise.
    """
    first_exc: BaseException | None = None
    for child in children:
        try:
            await await_maybe(getattr(child, method_name)(*args, **kwargs))
        except asyncio.CancelledError:  # noqa: PERF203 — per-child capture is the point of fan-out
            # Cancellation must propagate immediately — never swallow it into the
            # run-all-then-reraise loop (which would keep driving later children).
            raise
        except BaseException as exc:
            if first_exc is None:
                first_exc = exc
            else:
                # Only the FIRST exception is re-raised; log the rest so a later
                # child's failure isn't lost silently (FIX 6).
                logger.opt(exception=exc).warning(
                    f'fan_out({method_name!r}): child {type(child).__name__} raised after an earlier '
                    'exception; suppressed (the first exception is re-raised).'
                )
    if first_exc is not None:
        raise first_exc


async def combine_confirm(children: Iterable[Any], *args: Any, **kwargs: Any) -> bool:
    """Fan an ``on_confirm`` gate out to every child and AND the verdicts.

    Same run-all-then-reraise policy as `fan_out`: call ``on_confirm`` on
    every child, capture the FIRST exception and re-raise it after the loop
    (logging any later exceptions so they aren't lost, FIX 6). The run proceeds
    only if **every** child approves (``all(...)``); a single child behaves
    identically to calling it directly (``all([x]) == bool(x)``).
    """
    results: list[bool] = []
    first_exc: BaseException | None = None
    for child in children:
        try:
            results.append(bool(await await_maybe(child.on_confirm(*args, **kwargs))))
        except asyncio.CancelledError:  # noqa: PERF203 — per-child capture is the point
            # Cancellation must propagate immediately — never swallow it into the
            # run-all-then-reraise loop (which would keep polling later children).
            raise
        except BaseException as exc:
            if first_exc is None:
                first_exc = exc
            else:
                logger.opt(exception=exc).warning(
                    f'combine_confirm: child {type(child).__name__} raised after an earlier '
                    'exception; suppressed (the first exception is re-raised).'
                )
    if first_exc is not None:
        raise first_exc
    return all(results)


def normalize_to_list(value: Any) -> list[Any]:
    """Coerce a single item, a sequence, or ``None`` into a list.

    ``None`` → ``[]``; a list/tuple → ``list(value)``; any other single object
    (including ``str``/``bytes``, treated as scalar) → ``[value]``. Modeled on
    the list-or-single idiom used for ``target`` in the redteam runner. Empty
    inputs are returned empty — callers apply their own defaults.
    """
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def warn_if_sync_hooks(hooks: object, method_names: tuple[str, ...]) -> None:
    """Emit a one-shot ``DeprecationWarning`` if any hook method is synchronous.

    Sync hooks remain supported (driven via `await_maybe`); this is purely
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
