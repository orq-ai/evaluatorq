"""Persisted defaults shared by the trace-finder dashboard and CLI."""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

SETTINGS_PATH_ENV = 'EVALUATORQ_DASHBOARD_SETTINGS'
MIN_WINDOW_DAYS = 1
MAX_WINDOW_DAYS = 90
MIN_LIMIT = 1
MAX_LIMIT = 5000
DEFAULT_LIMIT = 500
MIN_PARALLELISM = 1
MAX_PARALLELISM = 200
# Pre-roles settings files saved these as plain fields, defaults included.
_LEGACY_MODEL_KEYS = frozenset({'compiler_model', 'apply_model', 'classifier_model'})
_LEGACY_DEFAULT_MODEL = 'openai/gpt-5.6-luna'
_LEGACY_CLASSIFIER_MODEL = 'typesafe/jev-latest'


class DashboardSettings(BaseModel):
    """Configurable models and finder limits used by the dashboard and CLI."""

    model_config = ConfigDict(extra='forbid')

    fast_model: str | None = None
    smart_model: str | None = None
    classifier_model: str | None = None
    embedding_model: str | None = None
    model_overrides: dict[str, str] = Field(default_factory=dict)
    window_days: int = Field(7, ge=MIN_WINDOW_DAYS, le=MAX_WINDOW_DAYS)
    limit: int = Field(DEFAULT_LIMIT, ge=MIN_LIMIT, le=MAX_LIMIT)
    parallelism: int = Field(100, ge=MIN_PARALLELISM, le=MAX_PARALLELISM)
    ask_ai_mode: Literal['immediate', 'review'] = 'immediate'
    orq_auth_method: Literal['environment', 'cli_profile', 'cli_oauth', 'stored_api_key'] = 'environment'
    orq_api_key_ciphertext: str | None = None
    orq_oauth_server: str | None = None
    orq_profile: str | None = None
    orq_profile_host: str | None = None
    orq_credential_fingerprint: str | None = None
    orq_workspace: str | None = None
    orq_project_id: str | None = None
    orq_project_name: str | None = None
    explorer_columns: tuple[str, ...] | None = None

    @field_validator('explorer_columns', mode='after')
    @classmethod
    def drop_unknown_columns(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        """Drop unknown saved columns with a warning instead of rejecting all settings."""
        if value is None:
            return None
        from .columns import COLUMNS

        unknown = [key for key in value if key not in COLUMNS]
        if unknown:
            logger.warning('Dropping unknown explorer column(s) {} from dashboard settings', ', '.join(unknown))
        return tuple(key for key in value if key in COLUMNS)

    @model_validator(mode='before')
    @classmethod
    def migrate_auth_method(cls, value: object) -> object:
        """Infer the legacy profile auth choice when older config lacks the field."""

        if isinstance(value, dict) and 'orq_auth_method' not in value and value.get('orq_profile'):
            return {**value, 'orq_auth_method': 'cli_profile'}
        return value

    @model_validator(mode='before')
    @classmethod
    def migrate_legacy_models(cls, value: object) -> object:
        """Fold the pre-roles model fields into role fields and task overrides.

        A legacy value equal to its old default is dropped so the old default is
        not frozen into the file; any other value becomes a task override.
        """

        if not isinstance(value, dict) or not _LEGACY_MODEL_KEYS & value.keys():
            return value
        migrated = dict(value)
        raw_overrides = migrated.get('model_overrides')
        overrides = dict(raw_overrides) if isinstance(raw_overrides, dict) else {}
        for legacy_key, task in (('compiler_model', 'finder.compiler'), ('apply_model', 'apply')):
            legacy = migrated.pop(legacy_key, None)
            if isinstance(legacy, str) and legacy.strip() and legacy.strip() != _LEGACY_DEFAULT_MODEL:
                existing = overrides.get(task)
                if not (isinstance(existing, str) and existing.strip()):
                    overrides[task] = legacy.strip()
        if overrides:
            migrated['model_overrides'] = overrides
        classifier = migrated.get('classifier_model')
        if isinstance(classifier, str) and classifier.strip() == _LEGACY_CLASSIFIER_MODEL:
            migrated['classifier_model'] = None
        return migrated

    @field_validator(
        'orq_profile',
        'orq_profile_host',
        'orq_credential_fingerprint',
        'orq_workspace',
        'orq_project_id',
        'orq_project_name',
        'orq_oauth_server',
        mode='before',
    )
    @classmethod
    def blank_selection_is_none(cls, value: object) -> object:
        """An empty credential or scope selector means no saved choice."""

        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator('fast_model', 'smart_model', 'classifier_model', 'embedding_model', mode='before')
    @classmethod
    def strip_model_identifier(cls, value: object) -> object:
        """Strip surrounding space; a blank role model means use the default."""

        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator('model_overrides', mode='after')
    @classmethod
    def drop_unknown_overrides(cls, value: dict[str, str]) -> dict[str, str]:
        """Drop overrides for unknown tasks and blank models with a warning instead of rejecting all settings."""
        from evaluatorq.common.model_roles import TASKS

        kept: dict[str, str] = {}
        for task, model in value.items():
            if task not in TASKS:
                logger.warning('Dropping model override for unknown task {!r} from dashboard settings', task)
            elif model.strip():
                kept[task] = model.strip()
            else:
                logger.warning('Dropping blank model override for task {!r} from dashboard settings', task)
        return kept


def _default_settings() -> DashboardSettings:
    return DashboardSettings.model_validate({})


def settings_path() -> Path:
    """Return the settings file path from the environment or its default."""
    configured = os.environ.get(SETTINGS_PATH_ENV, '').strip()
    return Path(configured) if configured else Path('.evaluatorq/dashboard-settings.json')


def credential_fingerprint(api_key: str | None, base_url: str | None) -> str | None:
    """Bind a saved project to the key and API host that exposed it, without persisting the key."""
    if not api_key:
        return None
    from evaluatorq.common.orq_client import DEFAULT_ORQ_BASE_URL

    host = (base_url or DEFAULT_ORQ_BASE_URL).rstrip('/')
    return sha256(f'{api_key}\0{host}'.encode()).hexdigest()


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
    except Exception:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
        raise


def effective_settings(overrides: dict[str, Any] | None = None) -> DashboardSettings:
    """Return settings with overrides applied in precedence order.

    Explicit *overrides* (typically CLI flags) beat environment variables,
    which beat the saved file, which beats model defaults. ``None`` values in
    *overrides* are ignored so callers can pass optional flags directly.
    Models are not resolved here: ``evaluatorq.common.model_roles.role_model``
    owns their precedence. Finder limit variables are
    ``EVALUATORQ_FINDER_WINDOW_DAYS``, ``EVALUATORQ_FINDER_LIMIT``, and
    ``EVALUATORQ_FINDER_PARALLELISM``; invalid integer or out-of-range values
    are ignored with a warning.
    """
    values = load_settings().model_dump()
    # Older Settings files may contain a saved workspace or project. Authentication
    # now uses the selected credential's full scope; callers can still override
    # these fields explicitly for a single invocation.
    values.update(orq_workspace=None, orq_project_id=None, orq_project_name=None)
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


def store_api_key(settings: DashboardSettings, key: str) -> DashboardSettings:
    """Encrypt a user-entered API key and return settings containing ciphertext only.

    The encryption key is stored in macOS Keychain on macOS. Other platforms
    require ``EVALUATORQ_DASHBOARD_KEY_ENCRYPTION_KEY`` to be set to a Fernet key.
    """
    from evaluatorq.trace_finder.secure_credentials import encrypt_api_key

    normalized = key.strip()
    if not normalized:
        raise ValueError('API key must not be blank')
    return settings.model_copy(
        update={
            'orq_auth_method': 'stored_api_key',
            'orq_api_key_ciphertext': encrypt_api_key(normalized),
        }
    )


def read_stored_api_key(settings: DashboardSettings) -> str | None:
    """Decrypt the saved user-entered API key, if one is configured."""
    if not settings.orq_api_key_ciphertext:
        return None
    from evaluatorq.trace_finder.secure_credentials import decrypt_api_key

    return decrypt_api_key(settings.orq_api_key_ciphertext)
