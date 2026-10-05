"""Named model roles and the one place their configured model is resolved.

Precedence, highest first: a CLI task override, a CLI role flag, the role's
environment variable, the settings file's task override, the settings file's
role field, the built-in default. Per-command flags such as ``--attack-model``
sit above all of these because they are passed explicitly as ``model=``.
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

# Internal: carries --model-override pairs (JSON) to subprocesses such as the
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


def _inherited_overrides() -> dict[str, str]:
    raw = _env(OVERRIDES_ENV)
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except ValueError:
        logger.warning('Ignoring malformed {}', OVERRIDES_ENV)
        return {}
    return (
        {k: v for k, v in loaded.items() if k in TASKS and isinstance(v, str) and v} if isinstance(loaded, dict) else {}
    )


def _legacy_env(task: str) -> str | None:
    name = LEGACY_TASK_ENV.get(task)
    if name is None:
        return None
    value = _env(name)
    if value and name not in _warned_legacy:
        _warned_legacy.add(name)
        logger.warning('{} is deprecated; use --model-override {}={} or the settings file', name, task, value)
    return value


def _resolve(role: Role, task: str | None) -> tuple[str, str]:
    if task is not None and TASKS[task] != role:
        raise ValueError(f'task {task!r} belongs to role {TASKS[task]!r}, not {role!r}')
    from evaluatorq.trace_finder.settings import load_settings

    overrides = _cli_overrides or _inherited_overrides()
    if task and task in overrides:
        return overrides[task], 'flag'
    if role in _cli_roles:
        return _cli_roles[role], 'flag'
    legacy = _legacy_env(task) if task else None
    if legacy:
        return legacy, 'env'
    # ponytail: reads the settings file per call; cache by mtime if a hot path shows up.
    settings = load_settings()
    if task and task in settings.model_overrides:
        return settings.model_overrides[task], 'settings'
    env_name = ROLE_ENV.get(role)
    env_value = _env(env_name) if env_name else None
    if env_value:
        return env_value, 'env'
    saved = getattr(settings, f'{role}_model')
    if saved:
        return saved, 'settings'
    return BUILTIN[role], 'default'


def role_model(role: Role, *, task: str | None = None) -> str:
    """The configured model for *role*, or for *task* when it carries an override.

    Raises ``KeyError`` for a task not in `TASKS`.
    """
    return _resolve(role, task)[0]


def role_source(role: Role, task: str | None = None) -> str:
    """Where `role_model` found its answer: ``flag``, ``env``, ``settings`` or ``default``."""
    return _resolve(role, task)[1]
