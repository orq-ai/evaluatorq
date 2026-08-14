"""Shared shape for the ``recommendations=`` flag and the config behind it.

Both surfaces spell the toggle the same way — ``red_team(recommendations=...)`` and the
simulation run entry points — so a caller learns one rule: ``True`` means "on, with
defaults", ``False`` means "skip the LLM call", and a config instance means "on, tuned".

The two configs share only what genuinely means the same thing on both surfaces (the
suggestion cap and the analysis call's token budget), which is what
``RecommendationConfigBase`` holds. Everything else is surface-specific — red
teaming samples failed traces per risk area, simulation gates on judge metric thresholds
— so each surface subclasses with its own fields rather than sharing one wide model where
half the knobs are inert.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar('T')


class RecommendationConfigBase(BaseModel):
    """Recommendation-generation knobs that mean the same thing on every surface.

    Bounds are enforced because a parseable value is not necessarily a meaningful one:
    a suggestion cap of 0 pays for the LLM call and then drops every result.
    """

    model_config = ConfigDict(extra='forbid')

    max_suggestions: int = Field(default=3, ge=1)
    """How many suggestions the analysis LLM is asked for, and kept from, per call."""

    max_tokens: int = Field(default=800, ge=1)
    """Completion budget for one analysis call. Raise it if suggestions get truncated."""


def resolve_recommendations(value: bool | T, default: Callable[[], T]) -> T | None:  # noqa: FBT001 — the flag IS the API
    """The config to generate recommendations with, or ``None`` to skip generation."""
    if value is True:
        return default()
    if value is False:
        return None
    return value
