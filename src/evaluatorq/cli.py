"""Top-level CLI for evaluatorq.

Usage:

```bash
evaluatorq redteam run --target agent:my-agent
evaluatorq dashboard .evaluatorq/runs
evaluatorq dashboard
evaluatorq dashboard /path/to/run.json
```
"""

from __future__ import annotations

import os
from pathlib import Path  # noqa: TC003
from typing import Annotated

import typer

from evaluatorq.common import cli_width  # noqa: F401  — import for its non-TTY width side effect
from evaluatorq.common.cli_epilog import examples
from evaluatorq.common.cli_help import CONTEXT_SETTINGS
from evaluatorq.common.model_roles import (
    BUILTIN,
    TASKS,
    add_cli_models,
    export_cli_models,
    parse_overrides,
    set_cli_models,
)

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
        from evaluatorq import __version__

        typer.echo(f'evaluatorq {__version__}')
        raise typer.Exit


def _apply_model_flags(
    *,
    replace: bool = False,
    fast: str | None,
    smart: str | None,
    classifier: str | None,
    embedding: str | None,
    overrides: list[str] | None,
) -> None:
    """Record the global model flags for this process and for child processes."""
    try:
        parsed = parse_overrides(overrides or [])
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint='--model-override') from exc
    roles = {'fast': fast, 'smart': smart, 'classifier': classifier, 'embedding': embedding}
    # The root callback replaces state left by an earlier in-process invocation; a subcommand adds to the root's.
    (set_cli_models if replace else add_cli_models)(roles, parsed)  # ty: ignore[invalid-argument-type]
    # Subprocesses (the dashboard's Insights worker) inherit os.environ, not this state.
    export_cli_models()


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    version: Annotated[  # noqa: FBT002 — eager callback consumes it
        bool,
        typer.Option('--version', help='Show version and exit.', callback=_version_callback, is_eager=True),
    ] = False,
    fast_model: Annotated[
        str | None,
        typer.Option('--fast-model', help=f'Model for cheap, high-volume steps. Default: {BUILTIN["fast"]}.'),
    ] = None,
    smart_model: Annotated[
        str | None,
        typer.Option('--smart-model', help=f'Model for attacks, judges and summaries. Default: {BUILTIN["smart"]}.'),
    ] = None,
    classifier_model: Annotated[
        str | None,
        typer.Option('--classifier-model', help=f'Model for trace classification. Default: {BUILTIN["classifier"]}.'),
    ] = None,
    embedding_model: Annotated[
        str | None,
        typer.Option('--embedding-model', help=f'Model for Insights embeddings. Default: {BUILTIN["embedding"]}.'),
    ] = None,
    model_override: Annotated[
        list[str] | None,
        typer.Option(
            '--model-override',
            metavar='TASK=MODEL',
            help=f'Pin one task to a model, e.g. apply=openai/gpt-6-luna. Repeatable. Tasks: {", ".join(TASKS)}.',
        ),
    ] = None,
) -> None:
    """Evaluation framework for AI systems."""
    _apply_model_flags(
        replace=True,
        fast=fast_model,
        smart=smart_model,
        classifier=classifier_model,
        embedding=embedding_model,
        overrides=model_override,
    )
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
                'Omit to show all runs from the red-team, simulation and pairwise stores. '
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
        typer.Option('--port', '-p', help='Port for the dashboard server.'),
    ] = 8080,
    no_browser: Annotated[  # noqa: FBT002 — Typer exposes this as a named flag
        bool,
        typer.Option('--no-browser', help='Start the dashboard without opening a browser.'),
    ] = False,
    compiler_model: Annotated[
        str | None,
        typer.Option('--compiler-model', help='Finder compiler model (a finder.compiler override).'),
    ] = None,
    classifier_model: Annotated[
        str | None,
        typer.Option(
            '--classifier-model',
            help='Sets the classifier model role (every classifier task), as the global --classifier-model. '
            'A root --model-override for a classifier task wins over it.',
        ),
    ] = None,
    window_days: Annotated[
        int | None,
        typer.Option('--window-days', min=1, max=90, help='Finder search window in days.'),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option('--limit', min=1, max=5000, help='Finder trace limit.'),
    ] = None,
    parallelism: Annotated[
        int | None,
        typer.Option('--parallelism', min=1, max=200, help='Finder classifier concurrency.'),
    ] = None,
) -> None:
    """Launch the FastHTML dashboard (preview — still in development).

    With no PATH the red-team (.evaluatorq/runs/), simulation
    (.evaluatorq/sim-runs/) and pairwise (.evaluatorq/pairwise-runs/)
    run stores are scanned.

    With one or more directory PATHs those directories are scanned together —
    e.g. sim runs from one repo next to red team runs from another.

    With a file PATH the file's parent directory is scanned so the report
    resolves.  The direct URL for that report is printed so you can open it
    immediately instead of landing on the index listing.
    """
    from evaluatorq.dashboard.launch import serve
    from evaluatorq.dashboard.library import report_id

    _apply_model_flags(
        fast=None,
        smart=None,
        classifier=classifier_model,
        embedding=None,
        overrides=[f'finder.compiler={compiler_model}'] if compiler_model else None,
    )
    for value, env_name in (
        (window_days, 'EVALUATORQ_FINDER_WINDOW_DAYS'),
        (limit, 'EVALUATORQ_FINDER_LIMIT'),
        (parallelism, 'EVALUATORQ_FINDER_PARALLELISM'),
    ):
        if value is not None:
            os.environ[env_name] = str(value)

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
                    path / 'pairwise-runs',
                    path / '.evaluatorq' / 'runs',
                    path / '.evaluatorq' / 'sim-runs',
                    path / '.evaluatorq' / 'pairwise-runs',
                ]
            elif path.is_file():
                # Scan the parent so the report resolves, but surface the direct link.
                roots.append(path.parent)
                direct_rids.append(report_id(path))
            else:
                # A typo'd path would otherwise become a parent root and print a
                # confident Direct-report URL for a file that does not exist.
                raise typer.BadParameter(f'Path does not exist: {path}')

    for rid in direct_rids:
        typer.echo(f'Direct report URL: http://{host}:{port}/r/{rid}')

    serve(roots, host=host, port=port, open_browser=not no_browser)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _register_subapps(app: typer.Typer) -> None:
    """Register the redteam, sim, coding-agent, and Insights sub-apps.

    Their deps are core, so a failing import here means a broken install — let it
    surface (via run_guarded) rather than silently dropping the subcommand
    (clig.dev: no silent failure).
    """
    from evaluatorq.backends.coding_agent_cli import app as coding_agent_app
    from evaluatorq.insights.cli import insights_cmd
    from evaluatorq.local_sessions.cli import sessions_cmd
    from evaluatorq.redteam.cli import app as redteam_app
    from evaluatorq.simulation.cli import app as sim_app
    from evaluatorq.trace_finder.cli import _FIND_EPILOG, find

    app.add_typer(redteam_app, name='redteam', help='Red teaming commands.')
    app.add_typer(sim_app, name='sim', help='Agent simulation pipeline.')
    app.add_typer(coding_agent_app, name='coding-agent', help='Coding-agent target utilities.')
    app.command(
        'find',
        help='Find recent Orq traces with a natural-language classifier task.',
        epilog=_FIND_EPILOG,
    )(find)
    app.command('insights', help='Discover and label patterns in Orq or local traces.')(insights_cmd)
    app.command(
        'agent-sessions',
        help='Search local Claude Code, Claude desktop, Codex and omp sessions.',
    )(sessions_cmd)


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
