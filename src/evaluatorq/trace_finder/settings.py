"""Persisted defaults shared by the trace-finder dashboard and CLI."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field, ValidationError, field_validator

from evaluatorq.contracts import DEFAULT_PIPELINE_MODEL

SETTINGS_PATH_ENV = 'EVALUATORQ_DASHBOARD_SETTINGS'
MIN_WINDOW_DAYS = 1
MAX_WINDOW_DAYS = 90
MIN_LIMIT = 1
MAX_LIMIT = 500
MIN_PARALLELISM = 1
MAX_PARALLELISM = 200


class DashboardSettings(BaseModel):
    """Configurable models and finder limits used by the dashboard and CLI."""

    compiler_model: str = DEFAULT_PIPELINE_MODEL
    jev_model: str = 'typesafe/jev-latest'
    apply_model: str = DEFAULT_PIPELINE_MODEL
    window_days: int = Field(7, ge=MIN_WINDOW_DAYS, le=MAX_WINDOW_DAYS)
    limit: int = Field(MAX_LIMIT, ge=MIN_LIMIT, le=MAX_LIMIT)
    parallelism: int = Field(100, ge=MIN_PARALLELISM, le=MAX_PARALLELISM)

    @field_validator('compiler_model', 'jev_model', 'apply_model', mode='before')
    @classmethod
    def strip_model_identifier(cls, value: object) -> object:
        """Reject blank model identifiers after removing surrounding space."""

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError('model identifier must not be blank')
            return stripped
        return value


def _default_settings() -> DashboardSettings:
    return DashboardSettings.model_validate({})


def settings_path() -> Path:
    """Return the settings file path from the environment or its default."""
    configured = os.environ.get(SETTINGS_PATH_ENV, '').strip()
    return Path(configured) if configured else Path('.evaluatorq/dashboard-settings.json')


def load_settings(path: Path | None = None) -> DashboardSettings:
    """Load settings from *path*, falling back to defaults when unavailable.

    A missing file is normal on first use. An unreadable or invalid file emits a
    warning naming the problem and also falls back to defaults so the dashboard
    can still start.
    """
    target = path or settings_path()
    try:
        return DashboardSettings.model_validate_json(target.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return _default_settings()
    except (OSError, ValidationError, ValueError) as exc:
        logger.warning('Could not load dashboard settings from {}: {}; using defaults', target, exc)
        return _default_settings()


def save_settings(s: DashboardSettings, path: Path | None = None) -> None:
    """Atomically write *s* to *path*, creating parent directories as needed."""
    target = path or settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=target.parent,
            prefix=f'.{target.name}.',
            suffix='.tmp',
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(s.model_dump_json(indent=2))
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def effective_settings(overrides: dict[str, Any] | None = None) -> DashboardSettings:
    """Return settings with overrides applied in precedence order.

    Explicit *overrides* (typically CLI flags) beat environment variables,
    which beat the saved file, which beats model defaults. ``None`` values in
    *overrides* are ignored so callers can pass optional flags directly.
    Environment model variables are ``EVALUATORQ_APPLY_MODEL``,
    ``EVALUATORQ_COMPILER_MODEL``, and ``EVALUATORQ_JEV_MODEL``. Finder limit
    variables are ``EVALUATORQ_FINDER_WINDOW_DAYS``, ``EVALUATORQ_FINDER_LIMIT``,
    and ``EVALUATORQ_FINDER_PARALLELISM``; invalid integer or out-of-range values
    are ignored with a warning.
    """
    values = load_settings().model_dump()
    for field, env_name in (
        ('apply_model', 'EVALUATORQ_APPLY_MODEL'),
        ('compiler_model', 'EVALUATORQ_COMPILER_MODEL'),
        ('jev_model', 'EVALUATORQ_JEV_MODEL'),
    ):
        env_value = os.environ.get(env_name, '').strip()
        if env_value:
            values[field] = env_value
    for field, env_name, minimum, maximum in (
        ('window_days', 'EVALUATORQ_FINDER_WINDOW_DAYS', MIN_WINDOW_DAYS, MAX_WINDOW_DAYS),
        ('limit', 'EVALUATORQ_FINDER_LIMIT', MIN_LIMIT, MAX_LIMIT),
        ('parallelism', 'EVALUATORQ_FINDER_PARALLELISM', MIN_PARALLELISM, MAX_PARALLELISM),
    ):
        env_value = os.environ.get(env_name, '').strip()
        if not env_value:
            continue
        try:
            parsed = int(env_value)
        except ValueError:
            logger.warning('Ignoring invalid integer {}={} in finder settings', env_name, env_value)
            continue
        if not minimum <= parsed <= maximum:
            logger.warning(
                'Ignoring out-of-range {}={} in finder settings; expected {}..{}',
                env_name,
                env_value,
                minimum,
                maximum,
            )
            continue
        values[field] = parsed
    if overrides:
        values.update({key: value for key, value in overrides.items() if value is not None})
    return DashboardSettings.model_validate(values)
