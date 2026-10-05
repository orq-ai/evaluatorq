"""Named model roles and the one place their configured model is resolved.

Precedence, highest first: a per-command flag such as ``--attack-model`` (passed explicitly as
``model=``), a CLI task override (``--model-override``), a CLI role flag, a legacy task environment
variable (``EVALUATORQ_COMPILER_MODEL``, ``EVALUATORQ_APPLY_MODEL``), the settings file's task
override, the role's environment variable, the settings file's role field, the built-in default.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Literal

from loguru import logger

from evaluatorq.contracts import (
    DEFAULT_CLASSIFIER_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FAST_MODEL,
    DEFAULT_SMART_MODEL,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

Role = Literal['fast', 'smart', 'classifier', 'embedding']
ROLES: tuple[Role, ...] = ('fast', 'smart', 'classifier', 'embedding')
BUILTIN: dict[Role, str] = {
    'fast': DEFAULT_FAST_MODEL,
    'smart': DEFAULT_SMART_MODEL,
    'classifier': DEFAULT_CLASSIFIER_MODEL,
    'embedding': DEFAULT_EMBEDDING_MODEL,
}
ROLE_ENV: dict[Role, str] = {
    'fast': 'EVALUATORQ_FAST_MODEL',
    'smart': 'EVALUATORQ_SMART_MODEL',
    'classifier': 'EVALUATORQ_CLASSIFIER_MODEL',
}
TASKS: dict[str, Role] = {
    'redteam.attacker': 'smart',
    'redteam.evaluator': 'smart',
    'sim.user': 'fast',
    'sim.judge': 'smart',
    'sim.generator': 'fast',
    'finder.compiler': 'fast',
    'finder.classifier': 'classifier',
    'apply': 'smart',
    'insights.summary': 'smart',
    'insights.labels': 'classifier',
    'insights.embedding': 'embedding',
    'signals': 'classifier',
}
# Env vars from before roles existed, still honoured as task overrides.
LEGACY_TASK_ENV: dict[str, str] = {
    'finder.compiler': 'EVALUATORQ_COMPILER_MODEL',
    'apply': 'EVALUATORQ_APPLY_MODEL',
}

# Internal: carries the CLI role flags and --model-override pairs (JSON) to subprocesses such as the
# dashboard's Insights worker, which inherit os.environ but not process state.
OVERRIDES_ENV = 'EVALUATORQ_MODEL_OVERRIDES'

_cli_roles: dict[Role, str] = {}
_cli_overrides: dict[str, str] = {}
_warned_legacy: set[str] = set()


def set_cli_models(roles: Mapping[Role, str | None] | None = None, overrides: Mapping[str, str] | None = None) -> None:
    """Record the global CLI flags for this process, replacing any earlier call."""
    _cli_roles.clear()
    _cli_overrides.clear()
    add_cli_models(roles, overrides)


def add_cli_models(roles: Mapping[Role, str | None] | None = None, overrides: Mapping[str, str] | None = None) -> None:
    """Add CLI flags on top of those already recorded, e.g. a subcommand's own model flags."""
    _cli_roles.update({role: model.strip() for role, model in (roles or {}).items() if model and model.strip()})
    _cli_overrides.update(overrides or {})


def parse_overrides(pairs: Iterable[str]) -> dict[str, str]:
    """Parse ``task=model`` pairs, rejecting unknown tasks and malformed pairs."""
    parsed: dict[str, str] = {}
    for pair in pairs:
        task, sep, model = pair.partition('=')
        task, model = task.strip(), model.strip()
        if not sep or not task or not model:
            raise ValueError(f'expected task=model, got {pair!r}')
        if task not in TASKS:
            raise ValueError(f'unknown task {task!r}; valid tasks: {", ".join(TASKS)}')
        parsed[task] = model
    return parsed


def _env(name: str) -> str | None:
    return os.environ.get(name, '').strip() or None


def export_cli_models() -> None:
    """Publish this process's CLI flags in the environment for child processes to inherit."""
    if _cli_roles or _cli_overrides:
        os.environ[OVERRIDES_ENV] = json.dumps({'roles': _cli_roles, 'overrides': _cli_overrides})
    else:
        os.environ.pop(OVERRIDES_ENV, None)


def _inherited_cli_models() -> tuple[dict[str, str], dict[str, str]]:
    """The CLI flags a parent process exported, tolerating anything malformed."""
    raw = _env(OVERRIDES_ENV)
    if not raw:
        return {}, {}
    try:
        loaded = json.loads(raw)
    except ValueError:
        loaded = None
    if not isinstance(loaded, dict):
        logger.warning('Ignoring malformed {}', OVERRIDES_ENV)
        return {}, {}

    def pick(key: str, valid: Iterable[str]) -> dict[str, str]:
        found = loaded.get(key)
        found = found if isinstance(found, dict) else {}
        good = {k: v.strip() for k, v in found.items() if k in valid and isinstance(v, str) and v.strip()}
        if len(good) != len(found):
            logger.warning('Ignoring unknown or blank {} in {}', key, OVERRIDES_ENV)
        return good

    return pick('roles', ROLES), pick('overrides', TASKS)


def _legacy_env(task: str) -> str | None:
    name = LEGACY_TASK_ENV.get(task)
    if name is None:
        return None
    value = _env(name)
    if value and name not in _warned_legacy:
        _warned_legacy.add(name)
        logger.warning('{} is deprecated; use --model-override {}={} or the settings file', name, task, value)
    return value


def _resolve(role: Role, task: str | None) -> tuple[str, str, bool]:
    """The model, its source, and whether a task-level source (not a role-level one) supplied it."""
    if task is not None and TASKS[task] != role:
        raise ValueError(f'task {task!r} belongs to role {TASKS[task]!r}, not {role!r}')
    from evaluatorq.trace_finder.settings import load_settings

    if _cli_roles or _cli_overrides:
        cli_roles, cli_overrides = _cli_roles, _cli_overrides
    else:
        cli_roles, cli_overrides = _inherited_cli_models()
    if task and task in cli_overrides:
        return cli_overrides[task], 'flag', True
    if role in cli_roles:
        return cli_roles[role], 'flag', False
    legacy = _legacy_env(task) if task else None
    if legacy:
        return legacy, 'env', True
    # ponytail: reads the settings file per call; cache by mtime if a hot path shows up.
    settings = load_settings()
    if task and task in settings.model_overrides:
        return settings.model_overrides[task], 'settings', True
    env_name = ROLE_ENV.get(role)
    env_value = _env(env_name) if env_name else None
    if env_value:
        return env_value, 'env', False
    saved = getattr(settings, f'{role}_model')
    if saved:
        return saved, 'settings', False
    return BUILTIN[role], 'default', False


def role_model(role: Role, *, task: str | None = None) -> str:
    """The configured model for *role*, or for *task* when it carries an override.

    Raises ``KeyError`` for a task not in `TASKS`.
    """
    return _resolve(role, task)[0]


def role_source(role: Role, task: str | None = None) -> str:
    """Where `role_model` found its answer: ``flag``, ``env``, ``settings`` or ``default``."""
    return _resolve(role, task)[1]


def task_override(task: str) -> tuple[str, str] | None:
    """The model and a human label of the task-level source that pins *task*, or ``None``."""
    model, source, task_level = _resolve(TASKS[task], task)
    if not task_level:
        return None
    label = {'flag': '--model-override', 'env': LEGACY_TASK_ENV.get(task, 'environment'), 'settings': 'settings file'}
    return model, label[source]
