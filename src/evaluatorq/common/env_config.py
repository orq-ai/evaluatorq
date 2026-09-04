"""Single contract for reading validated env-var overrides.

One place the package parses and validates environment overrides, so a tuning knob behaves the
same wherever it is read. The contract:

- unset (variable absent) -> the default, silently.
- a set-but-invalid value logs a WARNING and falls back to the default. It never raises, so a
  misconfigured knob is actionable but non-fatal. "Invalid" for a number means: empty/whitespace
  (in a CI ``env:`` block an unresolved ``${{ vars.X }}`` expands to empty, which should be a
  signal, not a silent default), unparseable, or outside an optional ``[min_value, max_value]``
  range. For a bool, empty is treated as unset (-> default).

Prefer ``env_int`` / ``env_float`` / ``env_bool`` over ad hoc ``os.getenv`` + ``int()`` / ``float()``
in the package.
"""

from __future__ import annotations

import math
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


def _raw_number(name: str, default: float) -> str | None:
    """Shared prelude for env_int/env_float: None if unset (silent), else the stripped value;
    an empty/whitespace value warns and returns None so the caller falls back to the default."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        logger.warning('{} is set but empty; using default {}.', name, default)
        return None
    return raw


def env_int(name: str, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    """Read an int override. Unset -> default; empty/invalid/out-of-range -> WARNING + default."""
    raw = _raw_number(name, default)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning('{} is not an integer ({!r}); using default {}.', name, raw, default)
        return default
    return int(_bounded(name, value, default, min_value, max_value))


def env_float(name: str, default: float, *, min_value: float | None = None, max_value: float | None = None) -> float:
    """Read a float override. Unset -> default; empty/invalid/out-of-range -> WARNING + default."""
    raw = _raw_number(name, default)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning('{} is not a number ({!r}); using default {}.', name, raw, default)
        return default
    if not math.isfinite(value):  # float() accepts nan/inf; a knob is never one of those
        logger.warning('{} is not a finite number ({!r}); using default {}.', name, raw, default)
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
