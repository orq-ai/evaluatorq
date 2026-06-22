"""Top-level CLI for evaluatorq.

Usage:
    evaluatorq redteam run --target agent:my-agent
    evaluatorq redteam ui report.json
    evaluatorq ui
    evaluatorq ui /path/to/run.json
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Annotated

import typer

# ---------------------------------------------------------------------------
# Top-level application
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="evaluatorq",
    help="Evaluation framework for AI systems.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# eq ui
# ---------------------------------------------------------------------------


@app.command()
def ui(
    path: Annotated[
        Path | None,
        typer.Argument(
            help=(
                "Optional path to scan. "
                "Omit to show all runs from both redteam and sim stores. "
                "A directory is scanned for reports; "
                "a file's parent directory is used."
            )
        ),
    ] = None,
    host: Annotated[
        str,
        typer.Option(help="Host to bind the dashboard server to."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(help="Port for the dashboard server."),
    ] = 8080,
) -> None:
    """Launch the FastHTML dashboard.

    With no PATH both the redteam run store (.evaluatorq/runs/) and the
    simulation run store (.evaluatorq/sim-runs/) are scanned.

    With a directory PATH only that directory is scanned.

    With a file PATH the file's parent directory is scanned (the index
    will list all reports in that directory).
    """
    from evaluatorq.dashboard.launch import serve

    roots: list[Path] | None
    if path is None:
        roots = None  # library.py picks the defaults
    elif path.is_dir():
        roots = [path]
    else:
        roots = [path.parent]

    serve(roots, host=host, port=port)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point that lazily assembles sub-commands and runs the CLI."""
    try:
        from evaluatorq.redteam.cli import app as redteam_app

        app.add_typer(redteam_app, name="redteam", help="Red teaming commands.")
    except ImportError:
        pass

    try:
        from evaluatorq.simulation.cli import app as sim_app

        app.add_typer(sim_app, name="sim", help="Agent simulation commands.")
    except ImportError:
        pass

    app()


# Allow `python -m evaluatorq.cli` as well as the entry point.
if __name__ == "__main__":
    main()
