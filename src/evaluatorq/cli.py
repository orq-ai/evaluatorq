"""Top-level CLI for evaluatorq.

Usage:
    evaluatorq redteam run --target agent:my-agent
    evaluatorq redteam ui report.json
    evaluatorq dashboard
    evaluatorq dashboard /path/to/run.json
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Annotated

import typer

# ---------------------------------------------------------------------------
# Top-level application
# ---------------------------------------------------------------------------

# Two CLI-help incompatibilities are worked around here (see tests/test_cli_version.py):
#   1. rich_markup_mode=None — typer's Rich help renderer silently emits nothing
#      with our pinned Rich (14.x), so `eq --help` printed an empty page. Falling
#      back to Click's plain formatter fixes it; Rich is still used elsewhere
#      (tables/progress).
#   2. no_args_is_help is NOT set — on Click >=8.2 it raises NoArgsIsHelpError,
#      which typer doesn't render, so bare `eq` exited 2 with no output. The
#      callback below prints help explicitly instead (works across versions).
app = typer.Typer(
    name='evaluatorq',
    help='Evaluation framework for AI systems.',
    rich_markup_mode=None,
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


@app.command()
def dashboard(
    path: Annotated[
        Path | None,
        typer.Argument(
            help=(
                'Optional path to scan. '
                'Omit to show all runs from both redteam and sim stores. '
                'A directory is scanned for reports; '
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

    With a directory PATH only that directory is scanned.

    With a file PATH the file's parent directory is scanned so the report
    resolves.  The direct URL for that report is printed so you can open it
    immediately instead of landing on the index listing.
    """
    from evaluatorq.dashboard.launch import serve
    from evaluatorq.dashboard.library import report_id

    roots: list[Path] | None
    direct_rid: str | None = None

    if path is None:
        roots = None  # library.py picks the defaults
    elif path.is_dir():
        # Include direct stores and the conventional repository-local
        # `.evaluatorq/` stores. Non-existent subdirs are skipped by the scanner.
        roots = [
            path,
            path / 'runs',
            path / 'sim-runs',
            path / '.evaluatorq' / 'runs',
            path / '.evaluatorq' / 'sim-runs',
        ]
    else:
        # Scan the parent so the report resolves, but surface the direct link.
        roots = [path.parent]
        direct_rid = report_id(path)

    if direct_rid is not None:
        typer.echo(f'Direct report URL: http://{host}:{port}/r/{direct_rid}')

    serve(roots, host=host, port=port)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point that lazily assembles sub-commands and runs the CLI."""
    try:
        from evaluatorq.redteam.cli import app as redteam_app

        app.add_typer(redteam_app, name='redteam', help='Red teaming commands.')
    except ImportError:
        pass

    try:
        from evaluatorq.simulation.cli import app as sim_app

        app.add_typer(sim_app, name='sim', help='Agent simulation commands.')
    except ImportError:
        pass

    app()


# Allow `python -m evaluatorq.cli` as well as the entry point.
if __name__ == '__main__':
    main()
