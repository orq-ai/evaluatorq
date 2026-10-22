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
    """One environment variable evaluatorq owns.

    ``env_int``, ``env_float`` and ``env_bool`` fall back to ``default``. ``env_raw`` and ``env_text``
    never apply it: for a ``raw`` row, ``default`` records what the call sites fall back to, and
    ``test_literal_env_raw_defaults_match_the_declared_default`` pins literal ones to it.
    """

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
    """Return ``os.environ.get(name, default)`` for a declared variable, untouched.

    The row's own default is not applied; pass it explicitly where the call site needs it.
    """
    _row(name)
    return os.environ.get(name, default)


def env_text(name: str) -> str | None:
    """The stripped value, or ``None`` when unset or blank."""
    _row(name)
    return os.environ.get(name, '').strip() or None


def _parse(row: EnvVar, raw: str) -> tuple[int | float | bool | None, str | None]:
    """Parse a set value of a typed row without logging.

    Returns ``(value, None)`` when the reader takes it, ``(None, problem)`` when the reader warns and
    falls back, and ``(None, None)`` for a blank bool, which reads as unset.
    """
    value = raw.strip()
    if row.kind == 'bool':
        if not value:
            return None, None
        if value.lower() in _TRUE:
            return True, None
        if value.lower() in _FALSE:
            return False, None
        return None, f'is not a boolean ({raw!r})'
    if not value:
        # in a CI ``env:`` block an unresolved ``${{ vars.X }}`` expands to empty: signal it
        return None, 'is set but empty'
    try:
        number = int(value) if row.kind == 'int' else float(value)
    except ValueError:
        return None, f'is not {"an integer" if row.kind == "int" else "a number"} ({value!r})'
    if not math.isfinite(number):  # float() accepts nan/inf; a knob is never one of those
        return None, f'is not a finite number ({value!r})'
    if row.min_value is not None and number < row.min_value:
        return None, f'must be >= {row.min_value} (got {number})'
    if row.max_value is not None and number > row.max_value:
        return None, f'must be <= {row.max_value} (got {number})'
    return number, None


def _read(row: EnvVar, default: float) -> int | float | bool:
    raw = os.environ.get(row.name)
    if raw is None:
        return default
    value, problem = _parse(row, raw)
    if problem is not None:
        # depth=1: the log line names the public reader (env_float, ...), as docs/configuration.md shows
        logger.opt(depth=1).warning('{} {}; using default {}.', row.name, problem, default)
    return default if value is None else value


def env_int(name: str) -> int:
    """Read an int override. Unset -> default; empty/invalid/out-of-range -> WARNING + default."""
    row = _row(name, 'int')
    return int(_read(row, cast('int', row.default)))


def env_float(name: str) -> float:
    """Read a float override. Unset -> default; empty/invalid/out-of-range -> WARNING + default."""
    row = _row(name, 'float')
    return float(_read(row, float(cast('float', row.default))))


def env_bool(name: str) -> bool:
    """Read a bool override. Unset/empty -> default; unrecognised -> WARNING + default.

    Truthy: 1/true/yes/on. Falsy: 0/false/no/off (case-insensitive).
    """
    row = _row(name, 'bool')
    return bool(_read(row, bool(row.default)))


def mask_secret(value: str) -> str:
    """Show only a key's last 4 characters so a user can validate *which* key is
    loaded without exposing the secret. Hidden chars render as ``*`` (capped so a
    long key doesn't blow out the row). Short keys reveal nothing but length."""
    if len(value) <= 4:
        return '*' * len(value)
    stars = '*' * min(len(value) - 4, 16)
    return f'{stars}{value[-4:]}'


def _status(row: EnvVar, raw: str | None) -> Status:
    if raw is None:
        return 'unset'
    if not raw.strip() and (row.kind == 'bool' or row.owner is not None):
        return 'unset'  # env_bool and the owners both skip a blank value
    if row.kind == 'raw':
        return 'set'
    return 'rejected' if _parse(row, raw)[1] is not None else 'set'


def report() -> list[EnvReport]:
    """One row per declared variable: whether the environment set it, and its value, masked if secret."""
    rows: list[EnvReport] = []
    for row in ENV_VARS.values():
        raw = os.environ.get(row.name)
        value = mask_secret(raw) if raw is not None and row.secret else raw
        rows.append(EnvReport(row.name, _status(row, raw), value, row.default, row.owner))
    return rows
