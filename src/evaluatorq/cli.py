"""Top-level CLI for evaluatorq.

Usage:
    evaluatorq redteam run --target agent:my-agent
    evaluatorq dashboard .evaluatorq/runs
    evaluatorq dashboard
    evaluatorq dashboard /path/to/run.json
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Annotated

import typer

from evaluatorq.common import cli_width  # noqa: F401  — import for its non-TTY width side effect
from evaluatorq.common.cli_epilog import examples
from evaluatorq.common.cli_help import CONTEXT_SETTINGS

# ---------------------------------------------------------------------------
# Top-level application
# ---------------------------------------------------------------------------

# no_args_is_help is NOT set — on Click >=8.2 it raises NoArgsIsHelpError, which
# typer doesn't render, so bare `eq` exited 2 with no output. The callback below
# prints help explicitly instead (works across versions).
_ROOT_EPILOG = examples(
    '# red team an agent',
    'eq redteam run -t agent:my-agent',
    '# explore saved runs in the dashboard',
    'eq dashboard',
    '# docs: https://orq-ai.github.io/evaluatorq/',
    '# report issues: https://github.com/orq-ai/evaluatorq/issues',
)

app = typer.Typer(
    name='evaluatorq',
    help='Evaluation framework for AI systems.',
    rich_markup_mode='rich',
    context_settings=CONTEXT_SETTINGS,
    epilog=_ROOT_EPILOG,
)


def _version_callback(value: bool) -> None:  # noqa: FBT001
    if value:
        from importlib.metadata import version

        typer.echo(f'evaluatorq {version("evaluatorq")}')
        raise typer.Exit


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    version: Annotated[  # noqa: FBT002 — eager callback consumes it
        bool,
        typer.Option('--version', help='Show version and exit.', callback=_version_callback, is_eager=True),
    ] = False,
) -> None:
    """Evaluation framework for AI systems."""
    # Bare `eq` (no subcommand) prints help and exits — replaces the unreliable
    # no_args_is_help path (see the app definition above).
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit


# ---------------------------------------------------------------------------
# eq dashboard (FastHTML — preview, still in development)
# ---------------------------------------------------------------------------


@app.command(epilog=_ROOT_EPILOG)
def dashboard(
    paths: Annotated[
        list[Path] | None,
        typer.Argument(
            help=(
                'Optional paths to scan (repeatable). '
                'Omit to show all runs from both redteam and sim stores. '
                'Directories are scanned for reports; '
                "a file opens the dashboard scoped to that file's parent directory "
                'and prints the direct report URL so you can navigate straight to it.'
            )
        ),
    ] = None,
    host: Annotated[
        str,
        typer.Option(help='Host to bind the dashboard server to.'),
    ] = '127.0.0.1',
    port: Annotated[
        int,
        typer.Option(help='Port for the dashboard server.'),
    ] = 8080,
) -> None:
    """Launch the FastHTML dashboard (preview — still in development).

    With no PATH both the redteam run store (.evaluatorq/runs/) and the
    simulation run store (.evaluatorq/sim-runs/) are scanned.

    With one or more directory PATHs those directories are scanned together —
    e.g. sim runs from one repo next to red team runs from another.

    With a file PATH the file's parent directory is scanned so the report
    resolves.  The direct URL for that report is printed so you can open it
    immediately instead of landing on the index listing.
    """
    from evaluatorq.dashboard.launch import serve
    from evaluatorq.dashboard.library import report_id

    roots: list[Path] | None
    direct_rids: list[str] = []

    if not paths:
        roots = None  # library.py picks the defaults
    else:
        roots = []
        for path in paths:
            if path.is_dir():
                # Include direct stores and the conventional repository-local
                # `.evaluatorq/` stores. Non-existent subdirs are skipped by the scanner.
                roots += [
                    path,
                    path / 'runs',
                    path / 'sim-runs',
                    path / '.evaluatorq' / 'runs',
                    path / '.evaluatorq' / 'sim-runs',
                ]
            else:
                # Scan the parent so the report resolves, but surface the direct link.
                roots.append(path.parent)
                direct_rids.append(report_id(path))

    for rid in direct_rids:
        typer.echo(f'Direct report URL: http://{host}:{port}/r/{rid}')

    serve(roots, host=host, port=port)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _register_subapps(app: typer.Typer) -> None:
    """Register the redteam and sim sub-apps.

    Their deps are core, so a failing import here means a broken install — let it
    surface (via run_guarded) rather than silently dropping the subcommand
    (clig.dev: no silent failure).
    """
    from evaluatorq.redteam.cli import app as redteam_app
    from evaluatorq.simulation.cli import app as sim_app

    app.add_typer(redteam_app, name='redteam', help='Red teaming commands.')
    app.add_typer(sim_app, name='sim', help='Agent simulation pipeline.')


def main() -> None:
    """Entry point that assembles sub-commands and runs the CLI under a guard."""
    from evaluatorq.common.cli_errors import run_guarded

    def _run() -> None:
        _register_subapps(app)
        app()

    run_guarded(_run)


# Allow `python -m evaluatorq.cli` as well as the entry point.
if __name__ == '__main__':
    main()
