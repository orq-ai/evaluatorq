"""Typed environment-variable readers for user-configurable limits.

Small helpers so hardcoded module constants (max lengths, thresholds, and similar limits) can be
overridden via ``EVALUATORQ_*`` environment variables without threading a config object everywhere.
An unset or malformed value falls back to the default (with a warning), so behaviour is unchanged
unless a user opts in. Mirrors the existing ``EVALUATORQ_SPAN_MAX_TEXT_CHARS`` pattern in tracing.
"""

from __future__ import annotations

import os

from loguru import logger


def env_int(name: str, default: int) -> int:
    """Return the ``int`` value of env var ``name``, or ``default`` if unset/invalid."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning('{}={!r} is not a valid int; using default {}', name, raw, default)
        return default


def env_float(name: str, default: float) -> float:
    """Return the ``float`` value of env var ``name``, or ``default`` if unset/invalid."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning('{}={!r} is not a valid float; using default {}', name, raw, default)
        return default
