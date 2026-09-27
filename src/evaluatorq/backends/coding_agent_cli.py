"""``eq coding-agent``: build the image container-mode ``CodingAgentTarget`` runs in."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated

import typer

from evaluatorq.backends.container import DEFAULT_CODING_AGENT_IMAGE, cli_prefix

BUILD_DIR = Path(__file__).parent / 'docker'

app = typer.Typer(help='Coding-agent target utilities.', no_args_is_help=True)


@app.callback()
def coding_agent() -> None:
    """Utilities for container-mode coding-agent targets."""


@app.command('build-image')
def build_image(
    tag: Annotated[
        str, typer.Option(help='Image tag. The default is what DockerOptions() looks for.')
    ] = DEFAULT_CODING_AGENT_IMAGE,
    binary: Annotated[str, typer.Option(help='Container CLI, e.g. docker or podman.')] = 'docker',
    context: Annotated[str | None, typer.Option(help='Docker context, e.g. orbstack.')] = None,
    build_arg: Annotated[
        list[str] | None, typer.Option('--build-arg', help='KEY=VALUE, repeatable; pins an agent version.')
    ] = None,
) -> None:
    """Build the default coding-agent image from the Dockerfile shipped in the wheel."""
    args = [flag for pair in build_arg or [] for flag in ('--build-arg', pair)]
    argv = [*cli_prefix(binary, context), 'build', '-t', tag, *args, str(BUILD_DIR)]
    try:
        code = subprocess.call(argv)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f'{binary!r} not found on PATH', param_hint='--binary') from exc
    raise typer.Exit(code)
