"""Persisted defaults shared by the trace-finder dashboard and CLI."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from evaluatorq.contracts import DEFAULT_PIPELINE_MODEL

SETTINGS_PATH_ENV = 'EVALUATORQ_DASHBOARD_SETTINGS'


class DashboardSettings(BaseModel):
    """Configurable models and finder limits used by the dashboard and CLI."""

    compiler_model: str = DEFAULT_PIPELINE_MODEL
    jev_model: str = 'typesafe/jev-latest'
    apply_model: str = DEFAULT_PIPELINE_MODEL
    window_days: int = Field(7, ge=1, le=90)
    limit: int = Field(500, ge=1, le=500)
    parallelism: int = Field(100, ge=1, le=200)


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
    ``EVALUATORQ_COMPILER_MODEL``, and ``EVALUATORQ_JEV_MODEL``.
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
    if overrides:
        values.update({key: value for key, value in overrides.items() if value is not None})
    return DashboardSettings.model_validate(values)
