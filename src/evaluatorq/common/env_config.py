"""Typed environment-variable readers for user-configurable limits.

Small helpers so hardcoded module constants (max lengths, thresholds, and similar limits) can be
overridden via ``EVALUATORQ_*`` environment variables without threading a config object everywhere.
An unset, malformed, or out-of-range value falls back to the default (with a warning), so behaviour
is unchanged unless a user opts in with a value that can actually mean something. Mirrors the
existing ``EVALUATORQ_SPAN_MAX_TEXT_CHARS`` pattern in tracing.
"""

from __future__ import annotations

import math
import os

from loguru import logger


def env_int(name: str, default: int, *, min_value: int | None = None) -> int:
    """The ``int`` value of env var ``name``, or ``default`` if unset/invalid/below ``min_value``.

    ``min_value`` exists because a parseable value is not necessarily a meaningful one: a
    suggestion cap of 0 pays for the LLM call and then drops every result, and a negative
    transcript budget silently truncates from the wrong end (review on PR #141).
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning('{}={!r} is not a valid int; using default {}', name, raw, default)
        return default
    if min_value is not None and value < min_value:
        logger.warning('{}={} is below the minimum {}; using default {}', name, value, min_value, default)
        return default
    return value


def env_float(name: str, default: float, *, min_value: float | None = None, max_value: float | None = None) -> float:
    """The ``float`` value of env var ``name``, or ``default`` if unset/invalid/out of range.

    Non-finite values (``nan``/``inf``) parse but cannot mean anything for a threshold - ``nan``
    makes every comparison False and quietly disables the trigger - so they fall back too.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning('{}={!r} is not a valid float; using default {}', name, raw, default)
        return default
    if not math.isfinite(value):
        logger.warning('{}={!r} is not finite; using default {}', name, raw, default)
        return default
    if (min_value is not None and value < min_value) or (max_value is not None and value > max_value):
        logger.warning(
            '{}={} is outside the valid range [{}, {}]; using default {}', name, value, min_value, max_value, default
        )
        return default
    return value
