"""Structured input and output for the run commands: ``--config``, JSON-valued flags, ``--json`` and ``schema``.

``eq redteam run``, ``eq sim simulate`` and ``eq sim run`` accept their SDK keyword arguments as one JSON
document (``--config``), and print their result model as JSON (``--json``). Each surface owns the pydantic
model naming its keyword arguments and a table of `Flag` rows saying which field each flag sets; this module
turns the two into one validated model with `resolve_config`.

Precedence is the same everywhere: a flag passed on the command line beats the config file, which beats the
model's default. "Passed on the command line" is read from click's parameter source, never by comparing the
value with a default, so ``--max-turns 5`` beats a config's ``"max_turns": 8`` even when 5 is the default.
Config-backed flags therefore declare no default of their own: the model's default is the only one.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import click
import typer
from click.core import ParameterSource
from pydantic import BaseModel, ValidationError

from evaluatorq.common.cli_json import echo_json

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

ModelT = TypeVar('ModelT', bound=BaseModel)

STDIN_SOURCE = '-'

_IMPLICIT_SOURCES = frozenset({ParameterSource.DEFAULT, ParameterSource.DEFAULT_MAP})


def config_option_help(function: str, schema_command: str) -> str:
    """Help text for a ``--config`` option that carries ``function``'s keyword arguments."""
    return (
        f'JSON file of {function} keyword arguments, or "-" to read it from stdin. '
        f'Flags passed on the command line override it, field by field. Unknown keys are rejected; '
        f'print the accepted shape with `{schema_command}`.'
    )


JSON_OUTPUT_HELP = (
    'Print the final result model as JSON on stdout. Progress and messages go to stderr; exit codes are unchanged.'
)


def explicitly_set(ctx: click.Context, param: str) -> bool:
    """Whether ``param`` came from the command line or its env var rather than from its default."""
    source = ctx.get_parameter_source(param)
    return source is not None and source not in _IMPLICIT_SOURCES


# Marks ``Flag.path`` as omitted, so it defaults to ``param``. ``None`` is a real value ("no config field").
_PATH_IS_PARAM: Any = object()


@dataclass(frozen=True)
class Flag:
    """A command-line flag that sets one field of a run config.

    ``path`` is the dotted field path, e.g. ``'llm_config.attacker.model'``. It defaults to ``param``, the
    common case of a flag named after its field. ``None`` marks a flag with no config field of its own, such
    as ``--vercel-url``, which builds a target object outside the config.
    ``to_value`` turns the parsed flag value into the field value. ``replaces`` names top-level fields the
    flag's choice supersedes: passing the flag drops them from the config file, so ``--dataset-id`` replaces a
    file's ``"datapoints"`` instead of colliding with it in the "exactly one source" check.
    """

    param: str
    path: str | None = _PATH_IS_PARAM
    to_value: Callable[[Any], Any] | None = None
    replaces: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.path is _PATH_IS_PARAM:
            object.__setattr__(self, 'path', self.param)


def json_flag(param: str, path: str, model: type[BaseModel], *, flag: str) -> Flag:
    """A flag whose value is a JSON object validated against ``model``, merged into the field at ``path``."""

    def to_value(raw: str) -> dict[str, Any]:
        return parse_json_config(raw, model, flag=flag)

    return Flag(param, path, to_value=to_value)


def resolve_config(
    ctx: click.Context,
    model: type[ModelT],
    flags: Iterable[Flag],
    cli_args: dict[str, Any],
    *,
    config_source: str | None,
) -> ModelT:
    """Build ``model`` from the ``--config`` file with every flag passed on the command line merged on top.

    Flags apply in ``flags`` order and deep-merge into their own path, so ``--attack-model`` keeps the rest of
    a configured ``llm_config``, and a narrow flag listed after ``--llm-config`` wins over it for its field.
    ``cli_args`` holds the command's converted arguments: click's ``ctx.params`` still has raw strings. A
    field nothing sets takes ``model``'s default. ``null`` is accepted only where the field allows ``None``,
    which every SDK keyword whose default is ``None`` does, so there it reads as "unset".
    """
    data: dict[str, Any] = {}
    if config_source is not None:
        data = load_config(config_source, model)
    overrides: dict[str, Any] = {}
    for flag in flags:
        if not explicitly_set(ctx, flag.param):
            continue
        for field in flag.replaces:
            data.pop(field, None)
        if flag.path is None:
            continue
        value = cli_args[flag.param]
        _set_path(overrides, flag.path, value if flag.to_value is None else flag.to_value(value))
    try:
        return model.model_validate(_deep_merge(data, overrides), extra='forbid')
    except ValidationError as exc:
        raise typer.BadParameter(_validation_message(exc)) from exc


def model_kwargs(config: BaseModel) -> dict[str, Any]:
    """Every field of ``config`` as a keyword argument, nested models kept as model instances."""
    return {name: getattr(config, name) for name in type(config).model_fields}


def _read_source(source: str, *, flag: str) -> tuple[str, str]:
    if source == STDIN_SOURCE:
        return click.get_text_stream('stdin').read(), 'stdin'
    try:
        return Path(source).read_text(encoding='utf-8'), source
    except OSError as exc:
        raise typer.BadParameter(f'cannot read {source}: {exc.strerror or exc}', param_hint=flag) from exc
    except UnicodeDecodeError as exc:
        raise typer.BadParameter(f'cannot read {source}: not UTF-8 text', param_hint=flag) from exc


def parse_json_config(raw: str, model: type[BaseModel], *, flag: str, origin: str | None = None) -> dict[str, Any]:
    """Parse a JSON document, validate it against ``model`` and return the parsed dict.

    Unknown keys are rejected at every depth by validating with ``extra='forbid'`` per call, which also covers
    nested models that ignore extras by default (a misspelt ``"temprature"`` would otherwise vanish). The
    document's own JSON is returned, not the validated model re-dumped: a field whose serialization alias
    differs from its validation name would otherwise be silently dropped.
    """
    where = f'{origin}: ' if origin else ''
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise typer.BadParameter(f'{where}{exc}', param_hint=flag) from exc
    try:
        model.model_validate(data, extra='forbid')
    except ValidationError as exc:
        raise typer.BadParameter(_validation_message(exc, where), param_hint=flag) from exc
    return cast('dict[str, Any]', data)


def load_config(source: str, model: type[BaseModel], *, flag: str = '--config') -> dict[str, Any]:
    """Read ``source`` (a path, or ``-`` for stdin), validate it against ``model`` and return its parsed JSON."""
    raw, origin = _read_source(source, flag=flag)
    return parse_json_config(raw, model, flag=flag, origin=origin)


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    *parents, leaf = path.split('.')
    for key in parents:
        target = target.setdefault(key, {})
    target[leaf] = value


def reserve_stdout_for_json(ctx: click.Context) -> Callable[[BaseModel], None]:
    """Send everything printed to stdout during this command to stderr; return the one writer for stdout.

    The swap is undone when the command's context closes. Rich consoles and ``typer.echo`` look up
    ``sys.stdout`` when they print, so progress, banners and "saved to" messages all land on stderr.
    """
    real_stdout = sys.stdout

    def restore() -> None:
        sys.stdout = real_stdout

    sys.stdout = sys.stderr
    ctx.call_on_close(restore)

    def emit(result: BaseModel) -> None:
        click.echo(result.model_dump_json(indent=2), file=real_stdout)

    return emit


def echo_schema(model: type[BaseModel], *, output: bool) -> None:
    """Print ``model``'s JSON schema: the serialised shape for an output model, the accepted shape for input."""
    echo_json(model.model_json_schema(mode='serialization' if output else 'validation'))


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(cast('dict[str, Any]', current), cast('dict[str, Any]', value))
        else:
            merged[key] = value
    return merged


def _dotted(loc: tuple[int | str, ...]) -> str:
    """``('a', 'b', 0, 'c')`` as ``a.b[0].c``."""
    path = ''
    for part in loc:
        if isinstance(part, int):
            path += f'[{part}]'
        else:
            path += f'.{part}' if path else part
    return path


def _validation_message(exc: ValidationError, where: str = '') -> str:
    """``exc`` as text: a pure unknown-key failure names each key's full path, anything else is pydantic's own."""
    errors = exc.errors()
    if errors and all(error['type'] == 'extra_forbidden' for error in errors):
        paths = ', '.join(_dotted(error['loc']) for error in errors)
        return f'{where}unknown key(s): {paths}. Extra inputs are not permitted.'
    return f'{where}{exc}'
