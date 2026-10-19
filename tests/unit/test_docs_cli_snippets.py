# ruff: noqa: S101
"""Every `eq` command in a docs `bash` fence must parse against the real CLI.

`mkdocs build --strict` never runs a command, so a renamed flag or a wrong `--llm-config` shape in a page
ships silently. This parses each documented command with click (no network, no execution) and validates the
inline JSON handed to `--config` / `--llm-config` against the model that command validates it with.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

import click
import pytest
import typer
from pydantic import BaseModel

from evaluatorq.cli import _register_subapps, app
from evaluatorq.common.cli_config import parse_json_config
from evaluatorq.contracts import LLMCallConfig
from evaluatorq.redteam.cli import RedTeamCliConfig
from evaluatorq.redteam.contracts import LLMConfig
from evaluatorq.simulation.run_config import GenerateAndSimulateCliConfig, SimulateCliConfig

ROOT = Path(__file__).resolve().parents[2]
FENCE = re.compile(r'^(?P<indent>\s*)(?P<ticks>`{3,})(?P<lang>bash|sh|shell|console)\s*$')
# Placeholders and deliberately failing examples a reader is told not to run verbatim.
SKIP = ('eq find', 'eq insights', 'eq agent-sessions')

# (command path) -> model its --config validates against; --llm-config differs per surface.
CONFIG_MODEL: dict[tuple[str, ...], type[BaseModel]] = {
    ('redteam', 'run'): RedTeamCliConfig,
    ('sim', 'run'): GenerateAndSimulateCliConfig,
    ('sim', 'simulate'): SimulateCliConfig,
}
LLM_CONFIG_MODEL: dict[str, type[BaseModel]] = {'redteam': LLMConfig, 'sim': LLMCallConfig}


def _doc_files() -> list[Path]:
    files = [ROOT / 'README.md', *sorted((ROOT / 'docs').rglob('*.md'))]
    return [f for f in files if 'superpowers' not in f.parts]


def _commands(path: Path) -> list[tuple[int, list[str]]]:
    """Return (line number, argv) for each `eq ...` command in the file's shell fences."""
    found: list[tuple[int, list[str]]] = []
    lines = path.read_text(encoding='utf-8').splitlines()
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        i += 1
        if not m:
            continue
        block: list[tuple[int, str]] = []
        while i < len(lines) and not lines[i].strip().startswith(m['ticks']):
            block.append((i + 1, lines[i].strip()))
            i += 1
        buf, start = '', 0
        for lineno, text in block:
            if not buf and not text.startswith(('eq ', 'evaluatorq ', 'uv run eq ')):
                continue
            start = start or lineno
            buf += text.removesuffix('\\') + ' '
            if text.endswith('\\'):
                continue
            try:
                argv = shlex.split(buf, comments=True)
            except ValueError:  # an open quote: a multi-line JSON argument continues on the next line
                continue
            if argv[:2] == ['uv', 'run']:
                argv = argv[2:]
            if argv[0] in ('eq', 'evaluatorq') and not buf.startswith(SKIP):
                found.append((start, _before_shell_operator(argv[1:])))
            buf, start = '', 0
    return found


def _before_shell_operator(argv: list[str]) -> list[str]:
    for n, tok in enumerate(argv):
        if tok in ('|', '>', '>>', '&&', '||', ';', '2>&1') or tok.startswith(('>', '2>')):
            return argv[:n]
    return argv


def _cases() -> list[Any]:
    return [
        pytest.param(argv, id=f'{f.relative_to(ROOT)}:{line}')
        for f in _doc_files()
        for line, argv in _commands(f)
    ]


@pytest.fixture(scope='module')
def root_command() -> click.Command:
    if not app.registered_groups:
        _register_subapps(app)
    return typer.main.get_command(app)


def _resolve(root: click.Command, argv: list[str]) -> tuple[click.Command, tuple[str, ...], list[str], click.Context]:
    ctx = click.Context(root, info_name='eq', resilient_parsing=False)
    cmd, path, rest = root, [], list(argv)
    while isinstance(cmd, click.Group) and rest and not rest[0].startswith('-'):
        name, sub, rest = cmd.resolve_command(ctx, rest)
        if sub is None:
            raise click.UsageError(f'no such command {rest!r}')
        path.append(name)
        ctx = click.Context(sub, info_name=name, parent=ctx, resilient_parsing=False)
        cmd = sub
    return cmd, tuple(path), rest, ctx


@pytest.mark.parametrize('argv', _cases())
def test_documented_command_parses(root_command: click.Command, argv: list[str]) -> None:
    if any(a in ('--help', '-h') for a in argv):
        return
    cmd, path, rest, ctx = _resolve(root_command, argv)
    parser = cmd.make_parser(ctx)
    opts, _, _ = parser.parse_args(list(rest))

    for dest, hint, model in (
        ('config', '--config', CONFIG_MODEL.get(path)),
        ('llm_config_json', '--llm-config', LLM_CONFIG_MODEL.get(path[0]) if path else None),
    ):
        value = opts.get(dest)
        if model is not None and isinstance(value, str) and value.lstrip().startswith('{'):
            parse_json_config(value, model, flag=hint, origin='docs snippet')


@pytest.mark.parametrize(
    'argv',
    [
        pytest.param(['redteam', 'run', '--bogus', '1'], id='unknown-flag'),
        pytest.param(['sim', 'simulate', '--target-reasoning-effort', 'low'], id='flag-on-wrong-command'),
        pytest.param(
            ['redteam', 'run', '-t', 'a', '--llm-config', '{"extra_kwargs": {"top_p": 0.9}}'], id='wrong-llm-config-shape'
        ),
    ],
)
def test_checker_rejects_a_wrong_command(root_command: click.Command, argv: list[str]) -> None:
    with pytest.raises((click.UsageError, click.BadParameter)):
        test_documented_command_parses(root_command, argv)
