"""Single contract for reading validated env-var overrides.

One place the package parses and validates environment overrides, so a tuning knob behaves the
same wherever it is read. The contract:

- unset or empty string -> the default.
- a set-but-invalid value (unparseable, or outside an optional ``[min_value, max_value]`` range)
  logs a WARNING and falls back to the default. It never raises, so a misconfigured knob is
  actionable but non-fatal.

Prefer ``env_int`` / ``env_float`` / ``env_bool`` over ad hoc ``os.getenv`` + ``int()`` / ``float()``
in the package.
"""

from __future__ import annotations

import os

from loguru import logger

_TRUE = {'1', 'true', 'yes', 'on'}
_FALSE = {'0', 'false', 'no', 'off'}


def _bounded(name: str, value: float, default: float, min_value: float | None, max_value: float | None) -> float:
    if min_value is not None and value < min_value:
        logger.warning('{} must be >= {} (got {}); using default {}.', name, min_value, value, default)
        return default
    if max_value is not None and value > max_value:
        logger.warning('{} must be <= {} (got {}); using default {}.', name, max_value, value, default)
        return default
    return value


def env_int(name: str, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    """Read an int override. Unset/empty -> default; invalid or out-of-range -> WARNING + default."""
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning('{} is not an integer ({!r}); using default {}.', name, raw, default)
        return default
    return int(_bounded(name, value, default, min_value, max_value))


def env_float(name: str, default: float, *, min_value: float | None = None, max_value: float | None = None) -> float:
    """Read a float override. Unset/empty -> default; invalid or out-of-range -> WARNING + default."""
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning('{} is not a number ({!r}); using default {}.', name, raw, default)
        return default
    return float(_bounded(name, value, default, min_value, max_value))


def env_bool(name: str, *, default: bool) -> bool:
    """Read a bool override. Unset/empty -> default; unrecognised -> WARNING + default.

    Truthy: 1/true/yes/on. Falsy: 0/false/no/off (case-insensitive).
    """
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    logger.warning('{} is not a boolean ({!r}); using default {}.', name, raw, default)
    return default
