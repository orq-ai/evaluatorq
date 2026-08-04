"""Delivery-method registry — the single source of truth for delivery methods.

Mirrors :mod:`evaluatorq.redteam.vulnerability_registry`: a canonical set (the
:class:`DeliveryMethod` enum, grouped by technique family) plus a mutable
registry for custom entries, with ``register``/``resolve`` helpers so the CLI
and API validate against **registry plus enum** rather than a frozen enum.

Strictness policy (RES-966 decision)
------------------------------------
Delivery methods are **coerce-known + passthrough-unknown**: an enum member
resolves to its canonical :class:`DeliveryMethod` object; anything else passes
through as a raw string. A *registered custom* is known (it validates without a
warning) but stays a string — there is no enum member for it to resolve to, so
registration's observable effect is :func:`is_known_delivery_method` returning
``True``. This preserves the open set RES-295 deliberately chose (a dataset may
carry a delivery method the enum does not list), now extensible via
:func:`register_delivery_method`.

This differs on purpose from :func:`vulnerability_registry.resolve_vulnerabilities`,
which **rejects** unknown values. The *pattern* is the same (enum plus registered,
resolve/register); the *strictness* differs because the semantics do. An unknown
delivery method is a harmless filter/label — it either matches a dataset row
spelled the same or it does not. An unknown vulnerability has no strategies and
no evaluator, so passing it through would silently produce an unscoreable run;
rejecting it is the safe boundary there. Same registry shape, different failure
mode, each chosen for its own contract.

No fuzzy normalization: a value equals a known method exactly, or it is unknown
(RES-295 non-goal, unchanged).

Registry lifecycle: registrations are **process-global** — a custom method
registered by one call is known to every later call in the same process, by
design (that is what a registry is for). Re-registering the same value is
last-write-wins on its category. Registrations are never persisted to disk, so a
fresh process starts with only the enum: a value registered in one script is
*not* known to a separate ``eq redteam`` invocation. Tests that register a custom
method should reset ``_CUSTOM_DELIVERY_METHODS`` between cases to stay isolated.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING

from evaluatorq.redteam.contracts import DeliveryMethod

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    'DELIVERY_METHOD_CATEGORY',
    'delivery_method_str',
    'is_known_delivery_method',
    'list_available_delivery_methods',
    'register_delivery_method',
    'resolve_delivery_method',
    'resolve_delivery_methods',
]


def delivery_method_str(value: DeliveryMethod | str) -> str:
    """Canonical string for a delivery method — the value, version-independently.

    ``str(DeliveryMethod.DAN)`` returns ``'DAN'`` on native 3.11+ ``StrEnum`` but
    the ``'DeliveryMethod.DAN'`` repr on the 3.10 ``StrEnum`` polyfill (which has
    no ``__str__`` override). So never call ``str()`` on a member for display or
    string-keying — use this. Filtering is unaffected (a member compares/hashes
    equal to its value on every version); only string rendering diverges.
    """
    return value.value if isinstance(value, DeliveryMethod) else value


# Technique family for each canonical method (the groupings the enum documents).
# The one piece of real structure delivery methods carry; kept so ``register``
# has somewhere to place a custom method and the registry mirrors the
# vulnerability registry's structured defs rather than being a bare set.
# Category is the only metadata a consumer needs today; add richer per-method
# defs here if one ever does.
DELIVERY_METHOD_CATEGORY: Mapping[DeliveryMethod, str] = {
    DeliveryMethod.DAN: 'persona',
    DeliveryMethod.ROLE_PLAY: 'persona',
    DeliveryMethod.SKELETON_KEY: 'persona',
    DeliveryMethod.BASE64: 'obfuscation',
    DeliveryMethod.LEETSPEAK: 'obfuscation',
    DeliveryMethod.MULTILINGUAL: 'obfuscation',
    DeliveryMethod.CHARACTER_SPACING: 'obfuscation',
    DeliveryMethod.CRESCENDO: 'multi-turn',
    DeliveryMethod.MANY_SHOT: 'multi-turn',
    DeliveryMethod.AUTHORITY_IMPERSONATION: 'social-engineering',
    DeliveryMethod.REFUSAL_SUPPRESSION: 'social-engineering',
    DeliveryMethod.DIRECT_REQUEST: 'direct',
    DeliveryMethod.CODE_ELICITATION: 'direct',
    DeliveryMethod.CODE_ASSISTANCE: 'direct',
    DeliveryMethod.TOOL_RESPONSE: 'tool-agent',
    DeliveryMethod.WORD_SUBSTITUTION: 'tool-agent',
}

# Fail at import, not in CI: a new enum member without a technique family would
# otherwise ship silently. Frozen afterwards so no caller can mutate the
# canonical set at runtime (mirrors vulnerability_registry.py).
_missing = sorted(delivery_method_str(m) for m in set(DeliveryMethod) - set(DELIVERY_METHOD_CATEGORY))
if _missing:
    raise RuntimeError(f'Missing DELIVERY_METHOD_CATEGORY entries for: {_missing}')

DELIVERY_METHOD_CATEGORY = types.MappingProxyType(dict(DELIVERY_METHOD_CATEGORY))  # pyright: ignore[reportConstantRedefinition]

# Registered custom methods: value -> category. Seeded empty; the enum members
# are always known without being listed here. ``register_delivery_method``
# appends to this so ``is_known``/``resolve`` see the addition immediately.
_CUSTOM_DELIVERY_METHODS: dict[str, str] = {}


def register_delivery_method(value: str, *, category: str = 'custom') -> str:
    """Register a custom delivery method so it validates as known.

    A registered value stays a plain string — there is no enum member for
    :func:`resolve_delivery_method` to coerce it to. What registration buys is
    :func:`is_known_delivery_method` returning ``True``, which suppresses the
    CLI's unknown-method warning; filtering already worked without it.

    A value equal to an existing :class:`DeliveryMethod` member is a no-op (the
    enum is already canonical). Registration is process-local and not persisted,
    so it must happen in the same process that later runs the CLI. Returns the
    registered value.
    """
    if not value:
        msg = 'Cannot register an empty delivery method.'
        raise ValueError(msg)
    try:
        DeliveryMethod(value)
    except ValueError:
        _CUSTOM_DELIVERY_METHODS[value] = category
    return value


def is_known_delivery_method(value: str) -> bool:
    """True when *value* is a canonical enum member or a registered custom method."""
    if value in _CUSTOM_DELIVERY_METHODS:
        return True
    try:
        DeliveryMethod(value)
    except ValueError:
        return False
    return True


def resolve_delivery_method(value: DeliveryMethod | str) -> DeliveryMethod | str:
    """Resolve one value: an enum member (by value) becomes its
    :class:`DeliveryMethod` object, anything else passes through unchanged (open
    set) — including a registered custom, which is *known* but has no member.

    Already a :class:`DeliveryMethod`? Returned as-is — this is the identity that
    lets registry-backed objects flow through the pipeline without conversion.
    """
    if isinstance(value, DeliveryMethod):
        return value
    try:
        return DeliveryMethod(value)
    except ValueError:
        return value


def resolve_delivery_methods(values: list[DeliveryMethod | str]) -> list[DeliveryMethod | str]:
    """Resolve a list, coercing known values to enum objects and keeping unknown
    ones as strings. Order preserved, duplicates removed."""
    seen: set[str] = set()
    result: list[DeliveryMethod | str] = []
    for value in values:
        resolved = resolve_delivery_method(value)
        key = delivery_method_str(resolved)
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return result


def list_available_delivery_methods() -> list[DeliveryMethod | str]:
    """Every known method: the canonical enum members plus registered customs."""
    return [*DeliveryMethod, *_CUSTOM_DELIVERY_METHODS]
