"""Persisted defaults shared by the trace-finder dashboard and CLI."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, field_validator, model_validator

from evaluatorq.common.file_lock import strict_file_lock

SETTINGS_PATH_ENV = 'EVALUATORQ_DASHBOARD_SETTINGS'
MIN_WINDOW_DAYS = 1
MAX_WINDOW_DAYS = 90
MIN_LIMIT = 1
MAX_LIMIT = 5000
DEFAULT_LIMIT = 500
MIN_PARALLELISM = 1
MAX_PARALLELISM = 200
MIN_TRACE_INPUT_CHARS = 4096
MAX_TRACE_INPUT_CHARS = 5_000_000
DEFAULT_TRACE_INPUT_CHARS = 500_000
# Pre-roles settings files saved these as plain fields, defaults included.
_LEGACY_MODEL_KEYS = frozenset({'compiler_model', 'apply_model', 'classifier_model'})
_LEGACY_DEFAULT_MODEL = 'openai/gpt-5.6-luna'
_LEGACY_CLASSIFIER_MODEL = 'typesafe/jev-latest'
CURRENT_SETTINGS_SCHEMA_VERSION = 1
AUTH_SETTINGS_FIELDS = frozenset({
    'orq_auth_method',
    'orq_api_key_ciphertext',
    'orq_oauth_server',
    'orq_profile',
    'orq_profile_host',
    'orq_credential_fingerprint',
    'orq_workspace',
    'orq_project_id',
    'orq_project_name',
})


class DashboardSettings(BaseModel):
    """Configurable models, filter selection and finder limits used by the dashboard and CLI."""

    model_config = ConfigDict(extra='forbid')
    _future_fields: dict[str, Any] = PrivateAttr(default_factory=dict)
    _source_document: dict[str, Any] | None = PrivateAttr(default=None)
    _baseline_values: dict[str, Any] | None = PrivateAttr(default=None)
    _explicit_updates: set[str] = PrivateAttr(default_factory=set)

    schema_version: int = Field(CURRENT_SETTINGS_SCHEMA_VERSION, strict=True, gt=0)
    fast_model: str | None = None
    smart_model: str | None = None
    classifier_model: str | None = None
    embedding_model: str | None = None
    model_overrides: dict[str, str] = Field(default_factory=dict)
    window_days: int = Field(7, ge=MIN_WINDOW_DAYS, le=MAX_WINDOW_DAYS)
    limit: int = Field(DEFAULT_LIMIT, ge=MIN_LIMIT, le=MAX_LIMIT)
    parallelism: int = Field(100, ge=MIN_PARALLELISM, le=MAX_PARALLELISM)
    trace_input_chars: int = Field(DEFAULT_TRACE_INPUT_CHARS, ge=MIN_TRACE_INPUT_CHARS, le=MAX_TRACE_INPUT_CHARS)
    filter_selection: Literal['llm', 'llm+jev'] = 'llm+jev'
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
    compression_inspector_enabled: bool = False

    def with_updates(self, updates: dict[str, Any], *, persistent: bool = True) -> DashboardSettings:
        """Validate updates while retaining saved source; caller updates win over saved values."""
        updated = DashboardSettings.model_validate({**self.model_dump(), **updates})
        updated._future_fields = dict(self._future_fields)  # noqa: SLF001 - opaque persisted metadata
        updated._source_document = self._source_document  # noqa: SLF001 - preserve loader metadata
        updated._baseline_values = dict(self._baseline_values or self.model_dump(mode='json'))  # noqa: SLF001
        updated._explicit_updates = set(self._explicit_updates)  # noqa: SLF001 - preserve caller intent
        if persistent:
            updated._explicit_updates.update(updates)  # noqa: SLF001 - track caller intent
            if self._source_document is None and self._baseline_values is None:
                updated._explicit_updates.update(self.model_fields_set)  # noqa: SLF001 - preserve authored inputs
        else:
            updated._baseline_values = updated.model_dump(mode='json')  # noqa: SLF001 - reset runtime baseline
            updated._explicit_updates.clear()  # noqa: SLF001 - runtime overrides are not edits
        return updated

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
        return _migrate_legacy_auth_fields(value)

    @model_validator(mode='before')
    @classmethod
    def migrate_legacy_models(cls, value: object) -> object:
        """Fold the pre-roles model fields into role fields and task overrides.

        A legacy value equal to its old default is dropped so the old default is
        not frozen into the file; any other value becomes a task override.
        """

        return _migrate_legacy_model_fields(value, preserve_invalid=False)

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


if AUTH_SETTINGS_FIELDS - DashboardSettings.model_fields.keys():
    raise RuntimeError('AUTH_SETTINGS_FIELDS contains fields missing from DashboardSettings')


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


def load_settings(path: Path | None = None) -> DashboardSettings:  # noqa: C901
    """Load settings from *path*, falling back to defaults when unavailable.

    A missing file is normal on first use. An unreadable or invalid file emits a
    warning naming the path and error type and also falls back to defaults so
    the dashboard can still start. Newer schemas are read best-effort; their
    unknown top-level fields are preserved when the settings are saved again.
    """
    target = path or settings_path()
    future_schema_version: int | None = None
    future_fields: dict[str, Any] = {}
    source_document: dict[str, Any] | None = None
    settings: DashboardSettings | None = None
    try:
        raw = json.loads(target.read_text(encoding='utf-8'))
        if not isinstance(raw, dict):
            return DashboardSettings.model_validate(raw)
        source_document = dict(raw)
        version = raw.get('schema_version', 0)
        # Validate version before filtering keys so booleans and strings cannot
        # masquerade as supported integer schema versions.
        if isinstance(version, bool) or not isinstance(version, int) or (version <= 0 and 'schema_version' in raw):
            return DashboardSettings.model_validate({**raw, 'schema_version': version})
        if version == 0:
            raw = migrate_settings_v0_to_v1(raw)
            version = 1
        elif version > CURRENT_SETTINGS_SCHEMA_VERSION:
            future_schema_version = version
            logger.warning(
                'Dashboard settings schema version {} is newer than supported version {}; reading known fields and preserving unknown top-level fields for save',
                version,
                CURRENT_SETTINGS_SCHEMA_VERSION,
            )
        known = DashboardSettings.model_fields.keys()
        unknown = sorted(raw.keys() - known)
        future_fields = {key: raw[key] for key in unknown}
        if unknown:
            logger.warning('Unknown dashboard settings field(s) preserved for save: {}', ', '.join(unknown))
            raw = {key: value for key, value in raw.items() if key in known}
        try:
            settings = DashboardSettings.model_validate(raw)
        except ValidationError as exc:
            errors = exc.errors(include_input=False, include_context=False, include_url=False)
            invalid_fields = {str(error['loc'][0]) for error in errors if error.get('loc')}
            if not invalid_fields or invalid_fields & AUTH_SETTINGS_FIELDS:
                raise
            defaults = DashboardSettings().model_dump(mode='json')
            recovered = dict(raw)
            failure_types: dict[str, set[str]] = {}
            while invalid_fields:
                for field in invalid_fields:
                    failure_types.setdefault(field, set()).update(
                        str(error['type']) for error in errors if error.get('loc') and error['loc'][0] == field
                    )
                    if field in defaults:
                        recovered[field] = defaults[field]
                    else:
                        recovered.pop(field, None)
                try:
                    settings = DashboardSettings.model_validate(recovered)
                    break
                except ValidationError as retry_exc:
                    errors = retry_exc.errors(include_input=False, include_context=False, include_url=False)
                    invalid_fields = {str(error['loc'][0]) for error in errors if error.get('loc')}
                    if not invalid_fields or invalid_fields & AUTH_SETTINGS_FIELDS:
                        raise
            logger.warning(
                'Invalid dashboard settings field(s); using defaults for {}',
                ', '.join(f'{field}:{"/".join(sorted(types))}' for field, types in sorted(failure_types.items())),
            )
        if settings is None:
            raise ValueError('Could not recover dashboard settings')
        settings._future_fields = future_fields  # noqa: SLF001 - attached only by the file loader
        settings._source_document = source_document  # noqa: SLF001 - attach loader metadata
        settings._baseline_values = settings.model_dump(mode='json')  # noqa: SLF001 - establish diff baseline
        return settings
    except FileNotFoundError:
        return _default_settings()
    except ValidationError as exc:
        failures = ', '.join(
            f'{".".join(str(part) for part in error["loc"])}:{error["type"]}'
            for error in exc.errors(include_input=False, include_context=False, include_url=False)
        )
        logger.warning('Could not load dashboard settings from {} ({}); using defaults', target, failures)
        settings = _default_settings()
        if future_schema_version is not None:
            settings.schema_version = future_schema_version
            settings._future_fields = future_fields  # noqa: SLF001 - preserved for newer-schema saves
        if source_document is not None:
            settings._source_document = source_document  # noqa: SLF001 - retain source after fallback
            settings._baseline_values = settings.model_dump(mode='json')  # noqa: SLF001 - establish diff baseline
        return settings
    except (OSError, ValueError) as exc:
        logger.warning('Could not load dashboard settings from {} ({}); using defaults', target, type(exc).__name__)
        return _default_settings()


def save_settings(s: DashboardSettings, path: Path | None = None, *, changed_fields: set[str] | None = None) -> None:
    """Serialize a complete settings read, patch, validation, and replace transaction."""
    target = path or settings_path()
    with strict_file_lock(target.with_name(f'.{target.name}.lock')):
        _save_settings_locked(s, target, changed_fields=changed_fields)


def _save_settings_locked(  # noqa: C901
    s: DashboardSettings, path: Path | None = None, *, changed_fields: set[str] | None = None
) -> None:
    """Atomically write *s* to *path*, including opaque fields from newer schemas."""
    target = path or settings_path()
    values = DashboardSettings.model_validate(s.model_dump()).model_dump(mode='json')
    fields = set(changed_fields if changed_fields is not None else s._explicit_updates)  # noqa: SLF001 - save intent
    if changed_fields is None:
        baseline = s._baseline_values or DashboardSettings().model_dump(mode='json')  # noqa: SLF001 - source baseline
        fields.update(key for key, value in values.items() if baseline.get(key, value) != value)
        if s._source_document is None and s._baseline_values is None:  # noqa: SLF001 - seed a fresh model's explicit fields
            fields.update(s.model_fields_set)
    unknown_fields = fields - DashboardSettings.model_fields.keys()
    if unknown_fields:
        raise ValueError(f'Unknown dashboard settings field(s): {", ".join(sorted(unknown_fields))}')

    if not target.exists():
        persisted = dict(s._source_document or s._future_fields)  # noqa: SLF001 - seed a new destination
        if persisted and persisted.get('schema_version', 0) == 0:
            persisted = migrate_settings_v0_to_v1(persisted)
        if s._source_document is None:  # noqa: SLF001 - seed defaults, not runtime overrides
            persisted.update(DashboardSettings().model_dump(mode='json'))
        if 'model_overrides' in fields and s._source_document is not None:  # noqa: SLF001 - loaded-source patch
            baseline = s._baseline_values or DashboardSettings().model_dump(mode='json')  # noqa: SLF001
            persisted['model_overrides'] = _patch_model_overrides(
                persisted.get('model_overrides'),
                baseline.get('model_overrides'),
                values.get('model_overrides'),
            )
            fields.discard('model_overrides')
        if s._source_document is None:  # noqa: SLF001 - apply only authored choices over the defaults
            persisted.update({field: values[field] for field in fields})
        else:
            persisted.update({field: values[field] for field in fields if field != 'schema_version'})
        if s._source_document is not None and 'schema_version' not in persisted:  # noqa: SLF001
            persisted['schema_version'] = values['schema_version']
    else:
        latest = json.loads(target.read_text(encoding='utf-8'))
        if not isinstance(latest, dict):
            raise ValueError(f'Cannot update dashboard settings at {target}: expected a JSON object')
        version = latest.get('schema_version', 0)
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version < 0
            or (version == 0 and 'schema_version' in latest)
        ):
            raise ValueError(f'Cannot update dashboard settings at {target}: invalid schema_version')
        if 'schema_version' in fields:
            if values['schema_version'] != version:
                raise ValueError('schema_version can only change through a supported migration')
            fields.discard('schema_version')
        if version == 0:
            latest = migrate_settings_v0_to_v1(latest)
        if 'model_overrides' in fields:
            baseline = s._baseline_values or DashboardSettings().model_dump(mode='json')  # noqa: SLF001
            latest['model_overrides'] = _patch_model_overrides(
                latest.get('model_overrides'),
                baseline.get('model_overrides'),
                values.get('model_overrides'),
            )
            fields.discard('model_overrides')
        for field in fields:
            latest[field] = values[field]
        persisted = latest
    if fields & AUTH_SETTINGS_FIELDS:
        auth_candidate = {field: persisted[field] for field in AUTH_SETTINGS_FIELDS if field in persisted}
        DashboardSettings.model_validate(auth_candidate)
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
            handle.write(json.dumps(persisted, indent=2))
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    except Exception:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
        raise


def migrate_settings_v0_to_v1(data: dict[str, Any]) -> dict[str, Any]:
    """Upgrade an unversioned settings mapping to schema version 1.

    Unknown fields and invalid values pass through unchanged. Only the legacy
    model and authentication fields recognized here are rewritten.
    """
    migrated: object = _migrate_legacy_model_fields({**data, 'schema_version': 1}, preserve_invalid=True)
    migrated = _migrate_legacy_auth_fields(migrated)
    return migrated if isinstance(migrated, dict) else {**data, 'schema_version': 1}


def _migrate_legacy_auth_fields(value: object) -> object:
    """Infer CLI profile auth for pre-versioned settings that name a profile."""
    if isinstance(value, dict) and 'orq_auth_method' not in value and value.get('orq_profile'):
        return {**value, 'orq_auth_method': 'cli_profile'}
    return value


def _migrate_legacy_model_fields(value: object, *, preserve_invalid: bool) -> object:
    """Convert recognized legacy model settings while optionally retaining invalid wire values."""
    if not isinstance(value, dict) or not _LEGACY_MODEL_KEYS & value.keys():
        return value
    migrated = dict(value)
    raw_overrides = migrated.get('model_overrides')
    if preserve_invalid and 'model_overrides' in migrated and not isinstance(raw_overrides, dict):
        return migrated
    overrides = dict(raw_overrides) if isinstance(raw_overrides, dict) else {}
    for legacy_key, task in (('compiler_model', 'finder.compiler'), ('apply_model', 'apply')):
        if legacy_key not in migrated:
            continue
        legacy = migrated[legacy_key]
        if preserve_invalid and not isinstance(legacy, str):
            continue
        migrated.pop(legacy_key)
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


def _patch_model_overrides(raw_value: object, previous_value: object, updated_value: object) -> dict[str, Any]:
    """Patch deliberate task changes while preserving untouched raw entries."""
    raw = raw_value if isinstance(raw_value, dict) else {}
    previous = previous_value if isinstance(previous_value, dict) else {}
    updated = updated_value if isinstance(updated_value, dict) else {}
    nested = dict(raw)
    for key in previous.keys() | updated.keys():
        if previous.get(key) != updated.get(key):
            if key in updated:
                nested[key] = updated[key]
            else:
                nested.pop(key, None)
    return nested


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
    saved = load_settings()
    values = saved.model_dump()
    # Legacy saved scope is ignored for API-key methods; OAuth explicitly selects its scope.
    if values['orq_auth_method'] != 'cli_oauth':
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
    return saved.with_updates(values, persistent=False)


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
