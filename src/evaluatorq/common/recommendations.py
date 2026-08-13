"""Shared resolution for the ``recommendations=`` flag.

Both surfaces spell the toggle the same way — ``red_team(recommendations=...)`` and the
simulation run entry points — so a caller learns one rule: ``True`` means "on, with
defaults", ``False`` means "skip the LLM call", and a config instance means "on, tuned".
The config *types* differ per surface (red teaming tunes how many risk areas and traces
are analyzed; simulation tunes trigger thresholds), which is why this resolves through a
factory rather than owning a single model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar('T')


def resolve_recommendations(value: bool | T, default: Callable[[], T]) -> T | None:  # noqa: FBT001 — the flag IS the API
    """The config to generate recommendations with, or ``None`` to skip generation."""
    if value is True:
        return default()
    if value is False:
        return None
    return value
