"""Structured input and output for the run commands: ``--config``, JSON-valued flags, ``--json`` and ``schema``.

``eq redteam run``, ``eq sim simulate`` and ``eq sim run`` accept their SDK keyword arguments as one JSON
document (``--config``), and print their result model as JSON (``--json``). This module holds the parts the
two surfaces share; each surface owns the pydantic model that names its own keyword arguments.

Precedence is the same everywhere: a flag passed on the command line beats the config file, which beats the
flag's default. "Passed on the command line" is read from click's parameter source, never by comparing the
value with the default, so ``--max-turns 5`` beats a config's ``"max_turns": 8`` even when 5 is the default.
"""

from __future__ import annotations

import json
import sys
import types
import typing
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
ValueT = TypeVar('ValueT')

STDIN_SOURCE = '-'

_IMPLICIT_SOURCES = frozenset({ParameterSource.DEFAULT, ParameterSource.DEFAULT_MAP})


def config_option_help(function: str, schema_command: str) -> str:
    """Help text for a ``--config`` option that carries ``function``'s keyword arguments."""
    return (
        f'JSON file of {function} keyword arguments, or "-" to read it from stdin. '
        f'Flags passed on the command line override it. Unknown keys are rejected; '
        f'print the accepted shape with `{schema_command}`.'
    )


JSON_OUTPUT_HELP = (
    'Print the final result model as JSON on stdout. Progress and messages go to stderr; exit codes are unchanged.'
)


def explicitly_set(ctx: click.Context, param: str) -> bool:
    """Whether ``param`` came from the command line or its env var rather than from its default."""
    source = ctx.get_parameter_source(param)
    return source is not None and source not in _IMPLICIT_SOURCES


def pick(
    ctx: click.Context, config: BaseModel | None, *, param: str, value: ValueT, field: str | None = None
) -> ValueT:
    """Resolve one CLI parameter against a config file: explicit flag, then config, then the flag default.

    ``field`` names the config field when it differs from the parameter name. A config value of ``null``
    reads as "not set" and falls back to the flag default, which is what ``None`` means to the SDK too.
    """
    field = field or param
    if config is None or explicitly_set(ctx, param) or field not in config.model_fields_set:
        return value
    configured = getattr(config, field)
    return value if configured is None else cast('ValueT', configured)


def unmapped_fields(config: BaseModel | None, *, mapped: Iterable[str]) -> dict[str, Any]:
    """The fields ``config`` set that no CLI flag owns, as typed values ready to forward to the SDK."""
    if config is None:
        return {}
    skip = set(mapped)
    return {name: getattr(config, name) for name in config.model_fields_set if name not in skip}


def _read_source(source: str, *, flag: str) -> tuple[str, str]:
    if source == STDIN_SOURCE:
        return click.get_text_stream('stdin').read(), 'stdin'
    try:
        return Path(source).read_text(encoding='utf-8'), source
    except OSError as exc:
        raise typer.BadParameter(f'cannot read {source}: {exc.strerror or exc}', param_hint=flag) from exc


def parse_json_model(raw: str, model: type[ModelT], *, flag: str, origin: str | None = None) -> ModelT:
    """Validate a JSON document against ``model``, rejecting unknown keys at every depth.

    The top level is rejected by the model's own ``extra='forbid'``. Nested models such as ``LLMCallConfig``
    ignore unknown keys, so a misspelt ``"temprature"`` would otherwise vanish; those are checked here.
    """
    where = f'{origin}: ' if origin else ''
    try:
        parsed = model.model_validate_json(raw)
    except ValidationError as exc:
        raise typer.BadParameter(f'{where}{exc}', param_hint=flag) from exc
    unknown = _unknown_keys(model, json.loads(raw), path='')
    if unknown:
        raise typer.BadParameter(f'{where}unknown key(s): {", ".join(unknown)}', param_hint=flag)
    return parsed


def load_config(source: str, model: type[ModelT], *, flag: str = '--config') -> ModelT:
    """Read ``source`` (a path, or ``-`` for stdin) and validate it against ``model``."""
    raw, origin = _read_source(source, flag=flag)
    return parse_json_model(raw, model, flag=flag, origin=origin)


def overlay_model(model: type[ModelT], base: BaseModel | None, updates: dict[str, Any]) -> ModelT:
    """Build ``model`` from ``base``'s explicitly set fields with ``updates`` deep-merged on top; updates win.

    Only fields the caller set survive the round trip, so a field ``base`` left unset stays unset, which
    matters for models such as ``LLMCallConfig`` that omit an unset parameter from the request.
    """
    data = base.model_dump(exclude_unset=True) if base is not None else {}
    return model.model_validate(_deep_merge(data, updates))


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


def _nested_models(annotation: Any) -> list[type[BaseModel]]:
    """Pydantic models reachable through ``Optional``, union and ``list`` wrappers of ``annotation``."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    origin = typing.get_origin(annotation)
    if origin is None:
        return []
    args = typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType, list, tuple, set, frozenset):
        return [model for arg in args for model in _nested_models(arg)]
    return []


def _unknown_keys(model: type[BaseModel], data: Any, *, path: str) -> list[str]:
    if isinstance(data, list):
        items = cast('list[Any]', data)
        return [key for index, item in enumerate(items) for key in _unknown_keys(model, item, path=f'{path}[{index}]')]
    if not isinstance(data, dict) or model.model_config.get('extra') == 'allow':
        return []
    fields = dict(model.model_fields)
    fields.update({info.alias: info for info in model.model_fields.values() if info.alias})
    unknown: list[str] = []
    for key, value in cast('dict[str, Any]', data).items():
        location = f'{path}.{key}' if path else str(key)
        info = fields.get(key)
        if info is None:
            unknown.append(location)
            continue
        # Only descend when the annotation names exactly one model: in a union of two models a key the
        # first lacks may belong to the second, and pydantic has already chosen between them.
        candidates = _nested_models(info.annotation)
        if len(candidates) == 1:
            unknown.extend(_unknown_keys(candidates[0], value, path=location))
    return unknown
