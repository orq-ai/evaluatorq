"""Shared hook composition for the run surfaces (redteam + simulation).

Both ``red_team()`` and ``simulate()`` / ``generate_and_simulate()`` accept one
or many user hooks and, when persisting, register a manifest stage-recorder as
the *first* hook. This module centralises that normalise → validate → warn →
mint-manifest → compose pipeline so the two surfaces can't drift, and validates
structurally that each user hook is actually hook-like (spec §5).

Surface-specific pieces (which ``Composite*Hooks`` class, which ``DefaultHooks``,
which ``ManifestStageHooks``, and how a ``ManifestWriter`` is minted) are passed
in as parameters, so this module stays free of any redteam/simulation imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evaluatorq.common.async_utils import normalize_to_list, warn_if_sync_hooks

if TYPE_CHECKING:
    from collections.abc import Callable

    from evaluatorq.common.run_manifest import ManifestWriter


def require_hooks_like(value: Any) -> None:
    """Raise ``TypeError`` unless *value* looks like a hooks object.

    A hooks object exposes a callable ``on_confirm`` — the run gate present on
    every surface Protocol. ``normalize_to_list`` stays permissive/generic; this
    is the structural gate at the compose layer (spec §5), so a wrong type (e.g.
    ``hooks=123`` or a stray string in a list) fails fast with a clear message
    naming the bad value.
    """
    if not callable(getattr(value, 'on_confirm', None)):
        raise TypeError(
            'hooks entries must be hooks objects (exposing a callable on_confirm); '
            f'got {value!r} of type {type(value).__name__}.'
        )


def compose_run_hooks(
    hooks: Any,
    *,
    method_names: tuple[str, ...],
    composite_cls: Callable[[list[Any]], Any],
    default_hooks_factory: Callable[[], Any],
    manifest_factory: Callable[[], ManifestWriter] | None,
    manifest_hook_factory: Callable[[ManifestWriter], Any],
) -> tuple[Any, ManifestWriter | None]:
    """Normalise, validate, warn, mint the manifest, and compose user hooks.

    ``hooks`` may be a single hook, a sequence, or ``None``. Each child is
    structurally validated (:func:`require_hooks_like`) and checked for the
    sync-hook deprecation nudge BEFORE anything else — once composed the async
    composite would mask a sync child, and validating first means a bad hook
    type never mints a (then-stuck-'running') manifest.

    With no user hooks a ``default_hooks_factory()`` is composed in so the
    default logging/confirm behaviour is preserved. ``manifest_factory`` is
    called (once) only when persisting; the resulting writer's stage-recorder
    hook is registered FIRST so stage status is durable before any user hook
    runs. Returns ``(composed_hooks, manifest_writer_or_None)`` — the raw writer
    is handed back for the runner's terminal complete/cancel/fail calls.
    """
    user_hooks = normalize_to_list(hooks)
    for child in user_hooks:
        require_hooks_like(child)
        warn_if_sync_hooks(child, method_names)
    base: list[Any] = user_hooks or [default_hooks_factory()]
    writer = manifest_factory() if manifest_factory is not None else None
    children = [manifest_hook_factory(writer), *base] if writer is not None else base
    return composite_cls(children), writer
