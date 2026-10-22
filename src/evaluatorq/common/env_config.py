"""The one table of environment variables evaluatorq owns, and the readers that go through it.

Every read of an owned variable in the package goes through a reader here
(``tests/test_reuse_guardrails.py`` enforces this), and every reader refuses a name that is not
declared in ``ENV_VARS``. To add
a knob: add an ``EnvVar`` row, read it with the matching reader, and add a row to the reference
tables in ``docs/configuration.md`` (``tests/common/test_env_config.py`` checks they agree).

The contract for the typed readers:

- unset (variable absent) -> the default, silently.
- a set-but-invalid value logs a WARNING and falls back to the default. It never raises, so a
  misconfigured knob is actionable but non-fatal. "Invalid" for a number means: empty/whitespace
  (in a CI ``env:`` block an unresolved ``${{ vars.X }}`` expands to empty, which should be a
  signal, not a silent default), unparseable, or outside the row's ``[min_value, max_value]``
  range. For a bool, empty is treated as unset (-> default).

``raw`` rows are passed on exactly as read: the call site keeps its own stripping, case handling and
validation. Rows with an ``owner`` are layered under CLI flags or the settings file; the owner, not
this module, decides their final value, and their ``default`` is the one that owner applies.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Literal, cast, overload

from loguru import logger

DEFAULT_ORQ_BASE_URL = 'https://my.orq.ai'

Kind = Literal['raw', 'int', 'float', 'bool']
Status = Literal['set', 'rejected', 'unset']

_TRUE = {'1', 'true', 'yes', 'on'}
_FALSE = {'0', 'false', 'no', 'off'}


@dataclass(frozen=True)
class EnvVar:
    name: str
    kind: Kind
    default: str | int | float | bool | None
    description: str
    min_value: float | None = None
    max_value: float | None = None
    secret: bool = False
    owner: str | None = None


_ROLE_OWNER = 'evaluatorq.common.model_roles.role_model'
_FINDER_OWNER = 'evaluatorq.trace_finder.settings.effective_settings'

_ROWS = (
    # Backend, credentials and model catalogue
    EnvVar('ORQ_API_KEY', 'raw', None, 'Orq platform API key; also enables tracing.', secret=True),
    EnvVar('ORQ_BASE_URL', 'raw', DEFAULT_ORQ_BASE_URL, 'Orq API base URL.'),
    EnvVar('OPENAI_API_KEY', 'raw', None, 'OpenAI-compatible backend key when ORQ_API_KEY is absent.', secret=True),
    EnvVar('OPENAI_BASE_URL', 'raw', None, 'OpenAI-compatible host for the red teaming LLM client.'),
    EnvVar('EVALUATORQ_CATALOGUE_TIMEOUT_S', 'float', 30.0, 'Model catalogue fetch timeout, read at import.', 0.1),
    # Model roles
    EnvVar('EVALUATORQ_FAST_MODEL', 'raw', 'openai/gpt-6-luna', 'Model for the fast role.', owner=_ROLE_OWNER),
    EnvVar('EVALUATORQ_SMART_MODEL', 'raw', 'openai/gpt-6-sol', 'Model for the smart role.', owner=_ROLE_OWNER),
    EnvVar(
        'EVALUATORQ_CLASSIFIER_MODEL',
        'raw',
        'typesafe/jev-latest',
        'Model for the classifier role.',
        owner=_ROLE_OWNER,
    ),
    EnvVar(
        'EVALUATORQ_EMBEDDING_MODEL',
        'raw',
        'openai/text-embedding-3-small',
        'Model for the embedding role.',
        owner=_ROLE_OWNER,
    ),
    EnvVar('EVALUATORQ_COMPILER_MODEL', 'raw', None, 'Deprecated finder.compiler task pin.', owner=_ROLE_OWNER),
    EnvVar('EVALUATORQ_APPLY_MODEL', 'raw', None, 'Deprecated apply task pin.', owner=_ROLE_OWNER),
    # Runs, storage and output
    EnvVar('EVALUATORQ_DIR', 'raw', None, 'Run store directory; empty is unset.'),
    EnvVar('EQ_DEBUG', 'raw', None, 'Any non-empty value shows full CLI tracebacks.'),
    EnvVar('EVALUATORQ_LOG_LEVEL', 'raw', 'INFO', 'Dashboard server and finder debug log level.'),
    # Dashboard
    EnvVar('ORQ_WORKSPACE', 'raw', None, 'Workspace slug for dashboard deep-links.'),
    EnvVar('ORQ_WORKSPACE_SLUG', 'raw', None, 'Fallback for ORQ_WORKSPACE.'),
    EnvVar('ORQ_UI_BASE_URL', 'raw', None, 'Orq UI host for dashboard deep-links.'),
    EnvVar('EVALUATORQ_DASHBOARD_SETTINGS', 'raw', '.evaluatorq/dashboard-settings.json', 'Settings file path.'),
    EnvVar(
        'EVALUATORQ_DASHBOARD_KEY_ENCRYPTION_KEY',
        'raw',
        None,
        'Fernet key that encrypts stored API keys off macOS.',
        secret=True,
    ),
    EnvVar(
        'EVALUATORQ_FINDER_WINDOW_DAYS', 'int', 7, 'Trace finder search window in days.', 1, 90, owner=_FINDER_OWNER
    ),
    EnvVar('EVALUATORQ_FINDER_LIMIT', 'int', 500, 'Trace finder trace limit.', 1, 5000, owner=_FINDER_OWNER),
    EnvVar(
        'EVALUATORQ_FINDER_PARALLELISM',
        'int',
        100,
        'Trace finder classify concurrency.',
        1,
        200,
        owner=_FINDER_OWNER,
    ),
    # Tracing
    EnvVar('ORQ_DISABLE_TRACING', 'bool', default=False, description='Suppress all OpenTelemetry spans.'),
    EnvVar('ORQ_DEBUG', 'raw', None, 'Any non-empty value prints tracing diagnostics.'),
    EnvVar('OTEL_EXPORTER_OTLP_ENDPOINT', 'raw', None, 'Explicit OTLP HTTP endpoint.'),
    EnvVar('OTEL_EXPORTER_OTLP_HEADERS', 'raw', None, 'Extra OTLP export headers.', secret=True),
    EnvVar('OTEL_SERVICE_NAME', 'raw', 'evaluatorq', 'Span service.name.'),
    EnvVar('OTEL_SERVICE_VERSION', 'raw', '1.0.0', 'Span service.version.'),
    EnvVar('ORQ_OTEL_MAX_QUEUE_SIZE', 'int', 4096, 'BatchSpanProcessor queue size.', 1),
    EnvVar('ORQ_OTEL_MAX_BATCH_SIZE', 'int', 512, 'Spans per OTLP export request.', 1),
    EnvVar('ORQ_OTEL_SCHEDULE_DELAY_MS', 'int', 5000, 'Partial batch export delay.', 1),
    EnvVar('ORQ_OTEL_FLUSH_TIMEOUT_MS', 'int', 5000, 'End-of-run force-flush timeout.', 1),
    EnvVar(
        'EVALUATORQ_CAPTURE_MESSAGE_CONTENT',
        'bool',
        default=True,
        description='Record LLM message content on spans.',
    ),
    EnvVar(
        'EVALUATORQ_PROPAGATE_TRACE_CONTEXT',
        'bool',
        default=True,
        description='Send W3C trace context on outgoing calls.',
    ),
    EnvVar('EVALUATORQ_SPAN_MAX_TEXT_CHARS', 'raw', None, 'Per-attribute span text limit.'),
    # Simulation LLM defaults
    EnvVar('EVALUATORQ_LLM_TIMEOUT_S', 'float', 60.0, 'Simulation per-call LLM timeout.', 1.0),
    EnvVar('EVALUATORQ_LLM_MAX_TOKENS', 'int', 10000, 'Simulation per-call completion token cap.', 1),
    EnvVar('EVALUATORQ_REASONING_EFFORT', 'raw', None, "Reasoning effort for the simulator's own calls."),
)
ENV_VARS: dict[str, EnvVar] = {row.name: row for row in _ROWS}


@dataclass(frozen=True)
class EnvReport:
    name: str
    status: Status
    value: str | None
    default: str | int | float | bool | None
    owner: str | None


def _row(name: str, kind: Kind | None = None) -> EnvVar:
    row = ENV_VARS.get(name)
    if row is None:
        raise KeyError(f'{name} is not declared in evaluatorq.common.env_config.ENV_VARS')
    if kind is not None and row.kind != kind:
        raise KeyError(f'{name} is declared as {row.kind!r}, not {kind!r}')
    return row


@overload
def env_raw(name: str) -> str | None: ...
@overload
def env_raw(name: str, default: str) -> str: ...
@overload
def env_raw(name: str, default: str | None) -> str | None: ...


def env_raw(name: str, default: str | None = None) -> str | None:
    """Return ``os.environ.get(name, default)`` for a declared variable, untouched."""
    _row(name)
    return os.environ.get(name, default)


def env_text(name: str) -> str | None:
    """The stripped value, or ``None`` when unset or blank."""
    _row(name)
    return os.environ.get(name, '').strip() or None


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


def env_int(name: str) -> int:
    """Read an int override. Unset -> default; empty/invalid/out-of-range -> WARNING + default."""
    row = _row(name, 'int')
    default = cast('int', row.default)
    raw = _raw_number(name, default)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning('{} is not an integer ({!r}); using default {}.', name, raw, default)
        return default
    return int(_bounded(name, value, default, row.min_value, row.max_value))


def env_float(name: str) -> float:
    """Read a float override. Unset -> default; empty/invalid/out-of-range -> WARNING + default."""
    row = _row(name, 'float')
    default = float(cast('float', row.default))
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
    return float(_bounded(name, value, default, row.min_value, row.max_value))


def env_bool(name: str) -> bool:
    """Read a bool override. Unset/empty -> default; unrecognised -> WARNING + default.

    Truthy: 1/true/yes/on. Falsy: 0/false/no/off (case-insensitive).
    """
    default = bool(_row(name, 'bool').default)
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if not value:  # empty/whitespace reads as "not set" for a bool, unlike the numeric readers
        return default
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    logger.warning('{} is not a boolean ({!r}); using default {}.', name, raw, default)
    return default


def mask_secret(value: str) -> str:
    """Show only a key's last 4 characters so a user can validate *which* key is
    loaded without exposing the secret. Hidden chars render as ``*`` (capped so a
    long key doesn't blow out the row). Short keys reveal nothing but length."""
    if len(value) <= 4:
        return '*' * len(value)
    stars = '*' * min(len(value) - 4, 16)
    return f'{stars}{value[-4:]}'


def _accepted(row: EnvVar, raw: str) -> bool:
    """Whether the typed readers would take *raw* rather than fall back, without logging."""
    value = raw.strip()
    if row.kind == 'raw':
        return True
    if row.kind == 'bool':
        return not value or value.lower() in _TRUE | _FALSE
    try:
        number = int(value) if row.kind == 'int' else float(value)
    except ValueError:
        return False
    if not math.isfinite(number):
        return False
    return (row.min_value is None or number >= row.min_value) and (row.max_value is None or number <= row.max_value)


def report() -> list[EnvReport]:
    """One row per declared variable: whether the environment set it, and its value, masked if secret."""
    rows: list[EnvReport] = []
    for row in ENV_VARS.values():
        raw = os.environ.get(row.name)
        if raw is None:
            status: Status = 'unset'
        else:
            status = 'set' if _accepted(row, raw) else 'rejected'
        value = mask_secret(raw) if raw is not None and row.secret else raw
        rows.append(EnvReport(row.name, status, value, row.default, row.owner))
    return rows
