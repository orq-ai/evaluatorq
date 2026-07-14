"""CLI for evaluatorq agent simulation.

Three execution verbs:
    generate  — datapoints only (no simulation)
    simulate  — simulate pre-built datapoints
    run       — generate then simulate in one shot

``run`` is a convenience: it generates datapoints in-process and simulates them
under a single pipeline span (via ``generate_and_simulate``). It is not a literal
``generate`` + ``simulate`` pipe — it does not require an intermediate file. To
capture the exact generated inputs for reproducible re-runs, pass
``--datapoints PATH`` (then re-feed that file to ``sim simulate --input``).

Usage:
    evaluatorq sim generate --agent-description "..." --datapoints dp.jsonl
    evaluatorq sim simulate --input dp.jsonl --target my-agent
    evaluatorq sim run --agent-description "..." --openai-model gpt-4o-mini
    evaluatorq sim run --agent-description "..." --target my-agent --datapoints dp.jsonl
    evaluatorq sim export --input results.jsonl --output payload.json
    evaluatorq sim validate-dataset dp.jsonl
    evaluatorq sim runs
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NoReturn

import typer
from loguru import logger

from evaluatorq.common import cli_width  # noqa: F401  — import for its non-TTY width side effect
from evaluatorq.common.llm_client import resolve_llm_client
from evaluatorq.simulation.types import DEFAULT_MODEL
from evaluatorq.simulation.utils.run_store import auto_save_run as _auto_save_run
from evaluatorq.simulation.utils.run_store import get_sim_runs_dir as _get_sim_runs_dir
from evaluatorq.simulation.utils.run_store import sanitise_run_name as _sanitise_run_name
from evaluatorq.simulation.utils.run_store import write_report as _write_report

if TYPE_CHECKING:
    from evaluatorq.simulation.types import SimulationRun

app = typer.Typer(
    name='sim',
    help=(
        'Agent simulation pipeline.\n\n'
        '  generate  ->  simulate  ->  dashboard   freeze inputs, run them, explore\n'
        '  run                                  generate + simulate in one shot\n'
        '  upload-dataset / --dataset-id       round-trip through an Orq dataset\n\n'
        'Use `eq sim COMMAND --help` for command-specific options.'
    ),
    no_args_is_help=True,
    rich_markup_mode='rich',
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _configure_logging(verbosity: int, console: Any = None) -> None:
    if verbosity < 0:
        level = logging.ERROR
    elif verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    # When a Rich console is provided (a live Progress region is active), route
    # both log streams through it so writes are serialised with the live region
    # instead of racing it on raw stderr — that race is what caused the flicker.
    logger = logging.getLogger('evaluatorq')
    logger.setLevel(level)
    if not logger.handlers:
        if console is not None:
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(console=console, show_path=False, rich_tracebacks=True)
        else:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(handler)
    try:
        from loguru import logger as loguru_logger

        loguru_logger.remove()
        sink = (lambda m: console.print(m, end='')) if console is not None else sys.stderr
        loguru_logger.add(sink, level=level)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


def _resolve_target(
    *,
    target: str | None,
    vercel_url: str | None,
    openai_model: str | None,
) -> Any:
    """Resolve exactly one target flag to an AgentTarget.

    Raises typer.BadParameter when zero or more than one flag is set.
    """
    supplied = {
        '--target': target,
        '--vercel-url': vercel_url,
        '--openai-model': openai_model,
    }
    active = {k: v for k, v in supplied.items() if v is not None}

    if len(active) == 0:
        raise typer.BadParameter('Provide exactly one of: --target, --vercel-url, --openai-model')
    if len(active) > 1:
        raise typer.BadParameter(f'Only one target flag allowed; got: {", ".join(active)}')

    if target is not None:
        from evaluatorq.redteam.contracts import TargetKind

        kind, value = _parse_target_spec(target)
        if kind == TargetKind.AGENT:
            return _make_sim_agent_backend().create_target(value)
        if kind == TargetKind.DEPLOYMENT:
            _require_orq_api_key('--target deployment:<key>')
            from evaluatorq.simulation.adapters import from_orq_deployment

            return from_orq_deployment(value)
        raise typer.BadParameter(f'Unsupported target kind for sim: {kind}')

    if vercel_url is not None:
        from evaluatorq.integrations.vercel_ai_sdk_integration import VercelAISdkTarget

        return VercelAISdkTarget(vercel_url)

    # openai_model
    if not os.environ.get('OPENAI_API_KEY') and not os.environ.get('ORQ_API_KEY'):
        raise typer.BadParameter(
            '--openai-model requires OPENAI_API_KEY (or ORQ_API_KEY for the AI Router) to be set in the environment.'
        )
    from evaluatorq.redteam.backends.openai import OpenAIModelTarget

    return OpenAIModelTarget(model=openai_model)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def _require_orq_api_key(flag: str) -> None:
    if not os.environ.get('ORQ_API_KEY'):
        raise typer.BadParameter(f'{flag} requires ORQ_API_KEY to be set in the environment.')


def _parse_target_spec(target: str) -> tuple[Any, str]:
    from evaluatorq.redteam.runner import _parse_target

    return _parse_target(target)


def _make_sim_agent_backend() -> Any:
    from evaluatorq.redteam.contracts import LLMConfig, TargetConfig
    from evaluatorq.redteam.runner import _make_agent_backend

    return _make_agent_backend(
        target_config=TargetConfig(system_prompt=None),
        pipeline_config=LLMConfig(),
    )


async def _resolve_agent_description(
    *,
    agent_description: str | None,
    target: str | None,
) -> str:
    if agent_description:
        return agent_description
    if target is None:
        raise ValueError('This command requires --agent-description unless --target is an agent target')

    from evaluatorq.redteam.contracts import TargetKind

    kind, value = _parse_target_spec(target)
    if kind != TargetKind.AGENT:
        raise ValueError('This command requires --agent-description unless --target is an agent target')

    ctx = await _make_sim_agent_backend().resolve_context(value)
    if not ctx.description:
        raise ValueError(f'Agent {value!r} has no description; pass --agent-description explicitly.')
    return ctx.description


def _handle_cli_error(exc: Exception) -> NoReturn:
    typer.echo(f'Error: {exc}', err=True)
    raise typer.Exit(1) from None


def _clean_cli_error_types() -> tuple[type[Exception], ...]:
    # Guard the lazy imports: this runs while evaluating an `except` clause, so a
    # failed import here would replace the exception being handled with an
    # ImportError. Fall back to the base types instead of masking the real error.
    try:
        from evaluatorq.common.llm_client import MissingLLMCredentialsError
        from evaluatorq.redteam.exceptions import BackendError, CredentialError
    except ImportError:
        return (ValueError, RuntimeError)
    return (ValueError, RuntimeError, BackendError, CredentialError, MissingLLMCredentialsError)


# ---------------------------------------------------------------------------
# Evaluator resolution
# ---------------------------------------------------------------------------


def _resolve_evaluators(evaluators: list[str] | None) -> list[str] | None:
    """Validate evaluator names and raise BadParameter on unknown names."""
    if not evaluators:
        return None

    from evaluatorq.simulation.evaluators import SIMULATION_EVALUATORS, get_evaluator

    for name in evaluators:
        try:
            get_evaluator(name)
        except ValueError:  # noqa: PERF203
            known = ', '.join(SIMULATION_EVALUATORS)
            raise typer.BadParameter(f'Unknown evaluator: {name!r}. Known: {known}') from None

    return list(evaluators)


def _echo_using(model: str) -> None:
    """Show the provider branch selected by the environment for generations."""
    if os.environ.get('ORQ_API_KEY'):
        provider = 'Orq router'
    elif os.environ.get('OPENAI_API_KEY'):
        provider = 'OpenAI-compatible'
    else:
        provider = 'unresolved provider'
    typer.echo(f'Using for generations: {provider} · {model}', err=True)


def _shell_path(path: Path) -> str:
    """Render a path safely for a copyable shell command."""
    return shlex.quote(str(path))


def _dashboard_command(directory: Path) -> str:
    """Build the canonical multi-run dashboard command for *directory*."""
    return f'eq dashboard {_shell_path(directory)}'


def _simulate_command(datapoints_path: Path, target: str | None) -> str:
    """Build the follow-on ``sim simulate`` command for a generated datapoints file."""
    target_part = target or '<target>'
    return f'eq sim simulate -i {_shell_path(datapoints_path)} --target {target_part}'


def _render_rich(renderable: Any, *, soft_wrap: bool = False) -> str:
    """Render a Rich renderable to text at the current terminal width.

    Written to a buffer (not Rich's own handle) so Typer/Click redirection and
    test capture both see it. ``rich`` is a core runtime dep, so no fallback.
    """
    import io
    import shutil

    from rich.console import Console

    width = shutil.get_terminal_size(fallback=(100, 24)).columns
    buffer = io.StringIO()
    # highlight=False: only our explicit markup styles apply, not Rich's
    # repr auto-highlighter (which mis-colours things like "datapoint(s)").
    Console(file=buffer, width=width, soft_wrap=soft_wrap, highlight=False).print(renderable)
    return buffer.getvalue()


def _recho(renderable: Any, *, soft_wrap: bool = False, err: bool = True) -> None:
    """Echo a Rich renderable onto the Typer output stream (stderr by default)."""
    typer.echo(_render_rich(renderable, soft_wrap=soft_wrap), nl=False, err=err)


def _echo_next(command: str) -> None:
    """Print the signposted next-step command (the green seam between commands)."""
    from rich.markup import escape

    _recho(f'[bold cyan]next:[/] {escape(command)}', soft_wrap=True)


def _echo_saved(label: str, path: Path) -> None:
    """Print a styled ``✓ <label> saved → <path>`` line (green tick, bold label)."""
    from rich.markup import escape

    _recho(f'[green]✓[/] [bold]{label}[/] saved [dim]→[/] {escape(str(path))}', soft_wrap=True)


def _echo_generate_preview(datapoints: list[Any]) -> None:
    """Print a compact persona/scenario preview of freshly generated datapoints.

    Best-effort and cosmetic: silently skips anything that doesn't expose the
    expected persona/scenario shape, so it can never break ``generate``.
    """
    from rich.table import Table

    personas: dict[str, Any] = {}
    scenarios: dict[str, Any] = {}
    for dp in datapoints:
        persona = getattr(dp, 'persona', None)
        scenario = getattr(dp, 'scenario', None)
        if persona is not None:
            personas.setdefault(persona.name, persona)
        if scenario is not None:
            scenarios.setdefault(scenario.name, scenario)

    if personas:
        ptable = Table(title='Personas', title_style='bold', title_justify='left', header_style='dim', border_style='cyan')
        ptable.add_column('Name', style='cyan', no_wrap=True)
        ptable.add_column('Patience', justify='right')
        ptable.add_column('Assertive', justify='right')
        ptable.add_column('Style')
        for p in personas.values():
            ptable.add_row(p.name, f'{p.patience:g}', f'{p.assertiveness:g}', str(p.communication_style))
        _recho(ptable)

    if scenarios:
        stable = Table(title='Scenarios', title_style='bold', title_justify='left', header_style='dim', border_style='cyan')
        stable.add_column('Name', style='green')
        stable.add_column('Goal', style='dim')
        for s in scenarios.values():
            stable.add_row(s.name, str(getattr(s, 'goal', '') or ''))
        _recho(stable)


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------


@app.command(no_args_is_help=True)
def simulate(
    datapoints: Annotated[
        Path | None,
        typer.Option(
            '--input',
            '-i',
            help='Path to datapoints JSONL file (or use --dataset-id).',
        ),
    ] = None,
    dataset_id: Annotated[
        str | None,
        typer.Option(
            '--dataset-id',
            help='Fetch datapoints from this Orq dataset instead of a local file. Requires ORQ_API_KEY.',
        ),
    ] = None,
    target: Annotated[
        str | None,
        typer.Option(
            '--target',
            help=('Target to simulate: agent:<key> or deployment:<key>. Bare values default to agent:<key>.'),
        ),
    ] = None,
    vercel_url: Annotated[
        str | None,
        typer.Option('--vercel-url', help='Vercel AI SDK endpoint URL.'),
    ] = None,
    openai_model: Annotated[
        str | None,
        typer.Option(
            '--openai-model',
            help=(
                'OpenAI-compatible model name. Provider is resolved from env: '
                'ORQ_API_KEY routes via the Orq AI Router (ORQ_BASE_URL overrides '
                'the host), otherwise OPENAI_API_KEY (+ optional OPENAI_BASE_URL for '
                'vLLM/OpenRouter/Azure-compatible/local endpoints). ORQ wins if both '
                'keys are set; namespace the model accordingly '
                "(e.g. 'openai/gpt-4o-mini' for the Orq router, 'gpt-4o-mini' for OpenAI)."
            ),
        ),
    ] = None,
    results_path: Annotated[
        Path | None,
        typer.Option('--output', '-o', help='Path to write results JSONL.'),
    ] = None,
    report_path: Annotated[
        Path | None,
        typer.Option(
            '--report',
            help=(
                'Path to write the full SimulationRun report JSON (results + '
                'scorer averages + metadata) to an explicit location, instead of '
                'only the auto-named file under .evaluatorq/sim-runs/. The '
                'auto-save still happens unless --no-save.'
            ),
        ),
    ] = None,
    name: Annotated[
        str,
        typer.Option('--name', '-n', help='Run name for the run-store entry.'),
    ] = 'sim',
    sim_model: Annotated[
        str,
        typer.Option(
            '--sim-model',
            help=(
                'Model for the user-simulator and judge. Provider resolved from '
                'env: ORQ_API_KEY -> Orq router, else OPENAI_API_KEY '
                '(+ OPENAI_BASE_URL) -> OpenAI-compatible endpoint. The default '
                'targets the Orq router; for OpenAI-direct pass a bare model '
                "name (e.g. 'gpt-5.4-mini', no provider prefix)."
            ),
        ),
    ] = DEFAULT_MODEL,
    max_turns: Annotated[
        int,
        typer.Option('--max-turns', min=1, help='Maximum conversation turns.'),
    ] = 10,
    parallelism: Annotated[
        int,
        typer.Option('--parallelism', min=1, help='Concurrent simulations.'),
    ] = 5,
    evaluator: Annotated[
        list[str] | None,
        typer.Option('--evaluator', help='Evaluator name (repeatable). Defaults to API defaults.'),
    ] = None,
    no_save: Annotated[  # noqa: FBT002
        bool,
        typer.Option('--no-save', help='Skip writing to .evaluatorq/sim-runs/.'),
    ] = False,
    verbose: Annotated[
        int,
        typer.Option(
            '--verbose',
            '-v',
            count=True,
            help='Increase verbosity (-v info logs, -vv debug logs).',
        ),
    ] = 0,
    quiet: Annotated[  # noqa: FBT002
        bool,
        typer.Option('--quiet', '-q', help='Suppress non-error output.'),
    ] = False,
    yes: Annotated[  # noqa: FBT002
        bool,
        typer.Option('--yes', '-y', help='Skip interactive confirmation prompt.'),
    ] = False,
    report_md: Annotated[
        Path | None,
        typer.Option('--report-md', help='Directory for an auto-named Markdown report.'),
    ] = None,
    report_html: Annotated[
        Path | None,
        typer.Option('--report-html', help='Directory for an auto-named HTML report.'),
    ] = None,
    executive_summary: Annotated[  # noqa: FBT002
        bool,
        typer.Option(
            '--executive-summary/--no-executive-summary',
            help='Generate an LLM narrative executive summary at the top of the report (needs LLM creds).',
        ),
    ] = True,
) -> None:
    """Run simulations from a pre-built datapoints file.

    Targets (provide exactly one):

    - --target TARGET    agent:<key> (default for bare key) or deployment:<key>.
    - --vercel-url URL   Vercel AI SDK HTTP endpoint.
    - --openai-model M   OpenAI-compatible model. Provider from env:
      ORQ_API_KEY for the Orq AI Router, else OPENAI_API_KEY
      (+ optional OPENAI_BASE_URL for vLLM/OpenRouter/local). ORQ wins if both set.

    Note: --target agent:<key> invokes a hosted Orq agent through the Responses
    router. --target deployment:<key> uses the deployment callback path.
    """
    if quiet:
        verbose = -1

    hooks: Any = None
    log_console: Any = None
    if not quiet:
        from rich.console import Console

        from evaluatorq.simulation.hooks import RichHooks

        log_console = Console(stderr=True)
        hooks = RichHooks(
            console=log_console,
            skip_confirm=yes or not sys.stdin.isatty(),
            defer_summary=executive_summary,
            verbose=verbose,
        )
    _configure_logging(verbose, console=log_console)
    _echo_using(sim_model)

    if (datapoints is None) == (dataset_id is None):
        raise typer.BadParameter('Provide exactly one of --datapoints or --dataset-id.')
    if datapoints is not None and not datapoints.exists():
        raise typer.BadParameter(f'Datapoints file not found: {datapoints}')
    if dataset_id is not None:
        _require_orq_api_key('--dataset-id')

    try:
        resolved_target = _resolve_target(
            target=target,
            vercel_url=vercel_url,
            openai_model=openai_model,
        )
        evaluator_names = _resolve_evaluators(evaluator)
        run = asyncio.run(
            _simulate_impl(
                datapoints_path=datapoints,
                dataset_id=dataset_id,
                target=resolved_target,
                sim_model=sim_model,
                max_turns=max_turns,
                parallelism=parallelism,
                evaluator_names=evaluator_names,
                evaluation_name=name,
                hooks=hooks,
            )
        )
    except KeyboardInterrupt:
        typer.echo('^C aborted.', err=True)
        raise typer.Exit(130) from None
    except asyncio.CancelledError:
        typer.echo('^C aborted.', err=True)
        raise typer.Exit(130) from None
    except typer.BadParameter:
        raise
    except _clean_cli_error_types() as exc:
        # RuntimeError covers "no datapoints produced" (generation) and
        # SimulationDroppedError (dropped jobs) — surface as one line, not a
        # traceback. Exit 1 keeps the CI-gate behaviour (still non-zero).
        _handle_cli_error(exc)

    results = run.results

    if results_path:
        _write_results(results, results_path)

    _maybe_generate_executive_summary(run, enabled=executive_summary, model=sim_model)
    if hooks is not None and executive_summary:
        hooks.print_summary(results, executive_summary=run.executive_summary, experiment_url=run.experiment_url)

    if report_md is not None:
        _export_report(run, report_md, fmt='md')
    if report_html is not None:
        _export_report(run, report_html, fmt='html')

    if report_path is not None:
        _write_report(run, report_path)
        _echo_saved('Report', report_path)

    if not no_save:
        run_path = _auto_save_run(run=run, run_name=name)
        _echo_saved('Run', run_path)
        _echo_next(_dashboard_command(_get_sim_runs_dir()))
    elif report_path is None:
        _recho('[yellow]Tip:[/] run without --no-save to browse results in the dashboard.', soft_wrap=True)


async def _simulate_impl(
    *,
    datapoints_path: Path | None,
    dataset_id: str | None = None,
    target: Any,
    sim_model: str,
    max_turns: int,
    parallelism: int,
    evaluator_names: list[str] | None,
    evaluation_name: str,
    hooks: Any = None,
) -> SimulationRun:
    from evaluatorq.simulation.api import _simulate_run
    from evaluatorq.simulation.utils.dataset_export import load_datapoints_from_jsonl

    loaded = None
    if dataset_id is None:
        if datapoints_path is None:  # the command guarantees exactly one source
            raise ValueError('Either datapoints_path or dataset_id is required')
        loaded = load_datapoints_from_jsonl(str(datapoints_path))
        if not loaded:
            raise ValueError(f'No datapoints loaded from {datapoints_path}')

    return await _simulate_run(
        datapoints=loaded,
        dataset_id=dataset_id,
        target=target,
        sim_model=sim_model,
        max_turns=max_turns,
        parallelism=parallelism,
        evaluator_names=evaluator_names,
        evaluation_name=evaluation_name,
        hooks=hooks,
    )


# ---------------------------------------------------------------------------
# run  (generate + simulate)
# ---------------------------------------------------------------------------


@app.command(no_args_is_help=True)
def run(
    agent_description: Annotated[
        str | None,
        typer.Option('--agent-description', help='Free-text description of the agent under test.'),
    ] = None,
    target: Annotated[
        str | None,
        typer.Option(
            '--target',
            help=('Target to simulate: agent:<key> or deployment:<key>. Bare values default to agent:<key>.'),
        ),
    ] = None,
    vercel_url: Annotated[
        str | None,
        typer.Option('--vercel-url', help='Vercel AI SDK endpoint URL.'),
    ] = None,
    openai_model: Annotated[
        str | None,
        typer.Option(
            '--openai-model',
            help=(
                'OpenAI-compatible model name. Provider is resolved from env: '
                'ORQ_API_KEY routes via the Orq AI Router (ORQ_BASE_URL overrides '
                'the host), otherwise OPENAI_API_KEY (+ optional OPENAI_BASE_URL for '
                'vLLM/OpenRouter/Azure-compatible/local endpoints). ORQ wins if both '
                'keys are set; namespace the model accordingly '
                "(e.g. 'openai/gpt-4o-mini' for the Orq router, 'gpt-4o-mini' for OpenAI)."
            ),
        ),
    ] = None,
    results_path: Annotated[
        Path | None,
        typer.Option('--output', '-o', help='Path to write results JSONL.'),
    ] = None,
    report_path: Annotated[
        Path | None,
        typer.Option(
            '--report',
            help=(
                'Path to write the full SimulationRun report JSON (results + '
                'scorer averages + metadata) to an explicit location, instead of '
                'only the auto-named file under .evaluatorq/sim-runs/. The '
                'auto-save still happens unless --no-save.'
            ),
        ),
    ] = None,
    name: Annotated[
        str,
        typer.Option('--name', '-n', help='Run name for the run-store entry.'),
    ] = 'sim',
    sim_model: Annotated[
        str,
        typer.Option(
            '--sim-model',
            help=(
                'Model for the user-simulator, the judge, and persona/scenario/'
                'first-message generation. Provider resolved from env: '
                'ORQ_API_KEY -> Orq router, else OPENAI_API_KEY (+ OPENAI_BASE_URL) '
                '-> OpenAI-compatible endpoint. The default targets the Orq '
                'router; for OpenAI-direct pass a bare model name '
                "(e.g. 'gpt-5.4-mini', no provider prefix)."
            ),
        ),
    ] = DEFAULT_MODEL,
    max_turns: Annotated[
        int,
        typer.Option('--max-turns', min=1, help='Maximum conversation turns.'),
    ] = 10,
    parallelism: Annotated[
        int,
        typer.Option('--parallelism', min=1, help='Concurrent simulations.'),
    ] = 5,
    num_personas: Annotated[
        int,
        typer.Option('--num-personas', min=1, help='Number of personas to generate.'),
    ] = 5,
    num_scenarios: Annotated[
        int,
        typer.Option('--num-scenarios', min=1, help='Number of scenarios to generate.'),
    ] = 5,
    evaluator: Annotated[
        list[str] | None,
        typer.Option('--evaluator', help='Evaluator name (repeatable). Defaults to API defaults.'),
    ] = None,
    no_save: Annotated[  # noqa: FBT002
        bool,
        typer.Option('--no-save', help='Skip writing to .evaluatorq/sim-runs/.'),
    ] = False,
    datapoints_path: Annotated[
        Path | None,
        typer.Option(
            '--datapoints',
            help=(
                'Also write the generated datapoints (the simulate inputs) to this '
                'JSONL path, for reproducible re-runs via `sim simulate --input`.'
            ),
        ),
    ] = None,
    verbose: Annotated[
        int,
        typer.Option(
            '--verbose',
            '-v',
            count=True,
            help='Increase verbosity (-v info logs, -vv debug logs).',
        ),
    ] = 0,
    quiet: Annotated[  # noqa: FBT002
        bool,
        typer.Option('--quiet', '-q', help='Suppress non-error output.'),
    ] = False,
    yes: Annotated[  # noqa: FBT002
        bool,
        typer.Option('--yes', '-y', help='Skip interactive confirmation prompt.'),
    ] = False,
    report_md: Annotated[
        Path | None,
        typer.Option('--report-md', help='Directory for an auto-named Markdown report.'),
    ] = None,
    report_html: Annotated[
        Path | None,
        typer.Option('--report-html', help='Directory for an auto-named HTML report.'),
    ] = None,
    executive_summary: Annotated[  # noqa: FBT002
        bool,
        typer.Option(
            '--executive-summary/--no-executive-summary',
            help='Generate an LLM narrative executive summary at the top of the report (needs LLM creds).',
        ),
    ] = True,
) -> None:
    """Generate personas and scenarios, then run simulations (generate + simulate).

    Targets (provide exactly one):

    - --target TARGET    agent:<key> (default for bare key) or deployment:<key>.
    - --vercel-url URL   Vercel AI SDK HTTP endpoint.
    - --openai-model M   OpenAI-compatible model. Provider from env:
      ORQ_API_KEY for the Orq AI Router, else OPENAI_API_KEY
      (+ optional OPENAI_BASE_URL for vLLM/OpenRouter/local). ORQ wins if both set.

    If --target is an agent target, --agent-description may be omitted and is
    fetched from Orq agent context. Other targets require --agent-description.
    """
    if quiet:
        verbose = -1

    hooks: Any = None
    log_console: Any = None
    if not quiet:
        from rich.console import Console

        from evaluatorq.simulation.hooks import RichHooks

        log_console = Console(stderr=True)
        hooks = RichHooks(
            console=log_console,
            skip_confirm=yes or not sys.stdin.isatty(),
            defer_summary=executive_summary,
            verbose=verbose,
        )
    _configure_logging(verbose, console=log_console)
    _echo_using(sim_model)

    try:
        resolved_agent_description = asyncio.run(
            _resolve_agent_description(agent_description=agent_description, target=target)
        )
        resolved_target = _resolve_target(
            target=target,
            vercel_url=vercel_url,
            openai_model=openai_model,
        )
        evaluator_names = _resolve_evaluators(evaluator)
        run = asyncio.run(
            _run_impl(
                agent_description=resolved_agent_description,
                target=resolved_target,
                sim_model=sim_model,
                max_turns=max_turns,
                parallelism=parallelism,
                num_personas=num_personas,
                num_scenarios=num_scenarios,
                evaluator_names=evaluator_names,
                evaluation_name=name,
                save_datapoints=datapoints_path,
                hooks=hooks,
            )
        )
    except KeyboardInterrupt:
        typer.echo('^C aborted.', err=True)
        raise typer.Exit(130) from None
    except asyncio.CancelledError:
        typer.echo('^C aborted.', err=True)
        raise typer.Exit(130) from None
    except typer.BadParameter:
        raise
    except _clean_cli_error_types() as exc:
        # RuntimeError covers "no datapoints produced" (generation) and
        # SimulationDroppedError (dropped jobs) — surface as one line, not a
        # traceback. Exit 1 keeps the CI-gate behaviour (still non-zero).
        _handle_cli_error(exc)

    results = run.results

    if results_path:
        _write_results(results, results_path)

    _maybe_generate_executive_summary(run, enabled=executive_summary, model=sim_model)
    if hooks is not None and executive_summary:
        hooks.print_summary(results, executive_summary=run.executive_summary, experiment_url=run.experiment_url)

    if report_md is not None:
        _export_report(run, report_md, fmt='md')
    if report_html is not None:
        _export_report(run, report_html, fmt='html')

    if report_path is not None:
        _write_report(run, report_path)
        _echo_saved('Report', report_path)

    if not no_save:
        run_path = _auto_save_run(run=run, run_name=name)
        _echo_saved('Run', run_path)
        _echo_next(_dashboard_command(_get_sim_runs_dir()))
    elif report_path is None:
        _recho('[yellow]Tip:[/] run without --no-save to browse results in the dashboard.', soft_wrap=True)


async def _run_impl(
    *,
    agent_description: str,
    target: Any,
    sim_model: str,
    max_turns: int,
    parallelism: int,
    num_personas: int,
    num_scenarios: int,
    evaluator_names: list[str] | None,
    evaluation_name: str,
    save_datapoints: Path | None = None,
    hooks: Any = None,
) -> SimulationRun:
    from evaluatorq.simulation.api import _generate_and_simulate_run

    emit = None
    if save_datapoints is not None:
        save_path = save_datapoints

        def _emit(dps: list[Any]) -> None:
            # Echo at write time, not after the run succeeds: the file lands on
            # disk before simulation, so a later sim failure must not swallow the
            # confirmation — the saved datapoints are exactly what you re-feed.
            _write_datapoints(dps, save_path)
            typer.echo(f'Datapoints saved: {save_path}', err=True)

        emit = _emit

    return await _generate_and_simulate_run(
        agent_description=agent_description,
        target=target,
        sim_model=sim_model,
        max_turns=max_turns,
        parallelism=parallelism,
        num_personas=num_personas,
        num_scenarios=num_scenarios,
        evaluator_names=evaluator_names,
        evaluation_name=evaluation_name,
        emit_datapoints=emit,
        hooks=hooks,
    )


# ---------------------------------------------------------------------------
# generate  (datapoints only, no simulation)
# ---------------------------------------------------------------------------


@app.command(no_args_is_help=True)
def generate(
    datapoints_path: Annotated[
        Path,
        typer.Option('--output', '-o', help='Path to write the generated datapoints JSONL.'),
    ],
    agent_description: Annotated[
        str | None,
        typer.Option('--agent-description', help='Free-text description of the agent under test.'),
    ] = None,
    target: Annotated[
        str | None,
        typer.Option(
            '--target',
            help=(
                'Agent target used to fetch the description when --agent-description '
                'is omitted. Accepts agent:<key>; bare values default to agent:<key>.'
            ),
        ),
    ] = None,
    sim_model: Annotated[
        str,
        typer.Option(
            '--sim-model',
            help=(
                'Model for persona/scenario/first-message generation. Provider '
                'resolved from env: ORQ_API_KEY -> Orq router, else '
                'OPENAI_API_KEY (+ OPENAI_BASE_URL) -> OpenAI-compatible '
                'endpoint. The default targets the Orq router; for OpenAI-direct '
                "pass a bare model name (e.g. 'gpt-5.4-mini', no provider prefix)."
            ),
        ),
    ] = DEFAULT_MODEL,
    num_personas: Annotated[
        int,
        typer.Option('--num-personas', min=1, help='Number of personas to generate.'),
    ] = 5,
    num_scenarios: Annotated[
        int,
        typer.Option('--num-scenarios', min=1, help='Number of scenarios to generate.'),
    ] = 5,
    dataset_format: Annotated[  # noqa: FBT002
        bool,
        typer.Option(
            '--dataset-format',
            help=(
                'Write the orq.ai dataset-row envelope (inputs/expected_output, '
                'persona & scenario JSON-stringified) instead of raw datapoints. '
                'Feed the raw form to `sim simulate --input`; feed this form '
                'to `sim upload-dataset` (or upload straight from the raw file).'
            ),
        ),
    ] = False,
    verbose: Annotated[
        int,
        typer.Option(
            '--verbose',
            '-v',
            count=True,
            help='Increase verbosity (-v info logs, -vv debug logs).',
        ),
    ] = 0,
    quiet: Annotated[  # noqa: FBT002
        bool,
        typer.Option('--quiet', '-q', help='Suppress non-error output.'),
    ] = False,
) -> None:
    """Generate simulation datapoints from an agent description (no simulation).

    Builds personas x scenarios with generated first messages and writes them
    as a datapoints JSONL file. Feed that file to ``sim simulate --input``
    to run — splitting generation out keeps the datapoint set frozen across
    simulate runs. No execution target is contacted; ``--target agent:<key>``
    is only used to fetch the agent description when ``--agent-description`` is
    omitted.
    """
    if quiet:
        verbose = -1
    _configure_logging(verbose)
    _echo_using(sim_model)

    try:
        resolved_agent_description = asyncio.run(
            _resolve_agent_description(agent_description=agent_description, target=target)
        )
        datapoints = asyncio.run(
            _generate_impl(
                agent_description=resolved_agent_description,
                sim_model=sim_model,
                num_personas=num_personas,
                num_scenarios=num_scenarios,
            )
        )
    except KeyboardInterrupt:
        typer.echo('^C aborted.', err=True)
        raise typer.Exit(130) from None
    except asyncio.CancelledError:
        typer.echo('^C aborted.', err=True)
        raise typer.Exit(130) from None
    except typer.BadParameter:
        raise
    except _clean_cli_error_types() as exc:
        # RuntimeError covers "first-message generation produced no datapoints"
        # from _resolve_or_generate_datapoints — surface as one line, not a
        # traceback (per the spec error-handling contract).
        _handle_cli_error(exc)

    _write_datapoints(datapoints, datapoints_path, dataset_format=dataset_format)
    if not quiet:
        from rich.rule import Rule

        _recho(Rule('Generated inputs', style='cyan', align='left'))
        _echo_generate_preview(datapoints)
    from rich.markup import escape

    _recho(
        f'[green]✓[/] Generated [bold]{len(datapoints)}[/] datapoint(s) → {escape(str(datapoints_path))}',
        soft_wrap=True,
    )
    _echo_next(_simulate_command(datapoints_path, target))


async def _generate_impl(
    *,
    agent_description: str,
    sim_model: str,
    num_personas: int,
    num_scenarios: int,
) -> list[Any]:
    from evaluatorq.simulation.api import generate

    return await generate(
        agent_description=agent_description,
        num_personas=num_personas,
        num_scenarios=num_scenarios,
        sim_model=sim_model,
    )


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@app.command(no_args_is_help=True)
def export(
    input_path: Annotated[
        Path,
        typer.Option('--input', '-i', help='Path to results JSONL file.'),
    ],
    output: Annotated[
        Path,
        typer.Option('--output', '-o', help='Path to write OpenResponses payload JSON.'),
    ],
) -> None:
    """Convert simulation results JSONL to OpenResponses payload JSON."""
    if not input_path.exists():
        raise typer.BadParameter(f'Input file not found: {input_path}')

    from evaluatorq.simulation.convert import to_open_responses
    from evaluatorq.simulation.types import SimulationResult
    from evaluatorq.simulation.utils.dataset_export import parse_jsonl

    try:
        content = input_path.read_text(encoding='utf-8')
        results: list[SimulationResult] = parse_jsonl(content, cls=SimulationResult)  # pyright: ignore[reportAssignmentType]
    except Exception as exc:
        raise typer.BadParameter(f'Failed to read {input_path}: {exc}') from exc

    payloads = [to_open_responses(result) for result in results]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([p if isinstance(p, dict) else p.model_dump(mode='json') for p in payloads], indent=2),
        encoding='utf-8',
    )
    typer.echo(f'Exported {len(payloads)} result(s) to {output}')


# ---------------------------------------------------------------------------
# validate-dataset
# ---------------------------------------------------------------------------


def _validate_datapoints(path: Path) -> None:
    """Validate a simulation datapoints JSONL file."""
    if not path.exists():
        raise typer.BadParameter(f'File not found: {path}')

    from evaluatorq.simulation.types import SimulationDatapoint

    try:
        content = path.read_text(encoding='utf-8')
    except Exception as exc:
        raise typer.BadParameter(f'Cannot read {path}: {exc}') from exc

    bad_count = 0
    lines = [line for line in content.splitlines() if line.strip()]

    valid_datapoints: list[SimulationDatapoint] = []
    for i, line in enumerate(lines, start=1):
        try:
            dp = SimulationDatapoint.model_validate_json(line)
            valid_datapoints.append(dp)
        except Exception as exc:  # noqa: PERF203
            typer.echo(f'Line {i}: {exc}', err=True)
            bad_count += 1

    from rich.markup import escape

    if bad_count:
        _recho(f'[red]✗[/] [bold]{bad_count}[/] invalid line(s) in {escape(str(path))}', soft_wrap=True)
        raise typer.Exit(1)

    _recho(
        f'[green]✓[/] [bold]{len(valid_datapoints)}[/] valid datapoint(s) in {escape(str(path))}',
        soft_wrap=True,
        err=False,
    )
    _echo_next(_simulate_command(path, None))


@app.command('validate', no_args_is_help=True)
def validate(
    input_path: Annotated[
        Path | None,
        typer.Option('--input', '-i', help='Path to datapoints JSONL file to validate.'),
    ] = None,
    path: Annotated[
        Path | None,
        typer.Argument(help='Path to datapoints JSONL file (legacy positional form).'),
    ] = None,
) -> None:
    """Validate a simulation datapoints JSONL file before running it."""
    if input_path is not None and path is not None:
        raise typer.BadParameter('Provide the input path once, either as an argument or with --input.')
    selected_path = input_path or path
    if selected_path is None:
        raise typer.BadParameter('Provide a datapoints file with --input PATH.')
    _validate_datapoints(selected_path)


@app.command('validate-dataset', no_args_is_help=True)
def validate_dataset(
    path: Annotated[
        Path,
        typer.Argument(help='Path to datapoints JSONL file to validate.'),
    ],
) -> None:
    """Deprecated alias for :command:`validate`."""
    _validate_datapoints(path)


# ---------------------------------------------------------------------------
# upload-dataset
# ---------------------------------------------------------------------------


@app.command('upload-dataset', no_args_is_help=True)
def upload_dataset(
    input_path: Annotated[
        Path,
        typer.Option('--input', '-i', help='Datapoints JSONL to upload (raw or --dataset-format).'),
    ],
    name: Annotated[
        str | None,
        typer.Option('--name', '-n', help='Display name for a new dataset (required unless --dataset-id).'),
    ] = None,
    path: Annotated[
        str,
        typer.Option('--path', help='Orq folder path for a new dataset.'),
    ] = 'Default',
    dataset_id: Annotated[
        str | None,
        typer.Option('--dataset-id', help='Append to this existing dataset instead of creating one.'),
    ] = None,
) -> None:
    """Convert a simulation datapoints JSONL into an Orq dataset and upload it.

    Accepts either the raw ``sim generate`` output or a ``--dataset-format`` file;
    both are normalised, then persona/scenario are JSON-stringified into the
    scalar-only ``inputs`` shape the datasets API requires (see
    ``to_orq_dataset_rows``). Pass ``--dataset-id`` to *extend* an existing
    dataset rather than create a new one. Requires ``ORQ_API_KEY``.
    """
    if not input_path.exists():
        raise typer.BadParameter(f'File not found: {input_path}')
    if dataset_id is None and not name:
        raise typer.BadParameter('Provide --name to create a dataset, or --dataset-id to extend one.')

    _require_orq_api_key('upload-dataset')

    from evaluatorq.fetch_data import setup_orq_client
    from evaluatorq.simulation.utils.dataset_export import load_datapoints_from_jsonl, to_orq_dataset_rows

    try:
        datapoints = load_datapoints_from_jsonl(str(input_path))
    except Exception as exc:
        raise typer.BadParameter(f'Failed to read {input_path}: {exc}') from exc
    if not datapoints:
        raise typer.BadParameter(f'No datapoints found in {input_path}')

    rows = to_orq_dataset_rows(datapoints)
    client = setup_orq_client(os.environ['ORQ_API_KEY'])

    if dataset_id is None:
        # name is guaranteed non-None here (validated above); SDK expects a TypedDict.
        created = client.datasets.create(request={'display_name': name or '', 'path': path})
        dataset_id = created.id
        typer.echo(f'Created dataset {dataset_id} ("{name}")', err=True)

    # ponytail: single bulk create; chunk if a set ever exceeds the API's array cap.
    # rows are plain dicts matching the SDK's CreateDatasetItem TypedDict shape.
    client.datasets.create_datapoint(dataset_id=dataset_id, request_body=rows)  # pyright: ignore[reportArgumentType]

    base = os.environ.get('ORQ_BASE_URL', 'https://my.orq.ai').rstrip('/')
    typer.echo(f'Uploaded {len(rows)} datapoint(s) -> dataset {dataset_id} ({base})')
    typer.echo(f'next: eq sim simulate --dataset-id {dataset_id} --target <target>')


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------


@app.command()
def runs(
    directory: Annotated[
        Path | None,
        typer.Argument(help='Directory to scan (default: .evaluatorq/sim-runs/).'),
    ] = None,
    limit: Annotated[
        int,
        typer.Option('--limit', '-n', help='Maximum number of runs to show.'),
    ] = 20,
) -> None:
    """List recent simulation runs."""
    runs_dir = directory or _get_sim_runs_dir()

    if not runs_dir.exists():
        typer.echo(f'No sim-runs directory found at {runs_dir}')
        raise typer.Exit(0)

    run_files = sorted(
        runs_dir.glob('*.json'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]

    if not run_files:
        typer.echo(f'No runs found in {runs_dir}')
        raise typer.Exit(0)

    rows: list[dict[str, Any]] = []
    malformed = 0
    for run_file in run_files:
        try:
            data = json.loads(run_file.read_text(encoding='utf-8'))
            rows.append({
                'name': data.get('run_name', '—'),
                'date': data.get('created_at', '—')[:19].replace('T', ' '),
                'mode': data.get('mode', '—'),
                'target': data.get('target_kind', '—'),
                'n': str(data.get('total_results', '—')),
                'scores': _format_scorer_averages(data.get('scorer_averages', {})),
                'file': run_file.name,
            })
        except Exception:  # noqa: PERF203
            malformed += 1

    try:
        import io

        from rich.console import Console
        from rich.table import Table

        table = Table(title='Simulation Runs')
        for col in ('Name', 'Date', 'Mode', 'Target', 'N', 'Scores', 'File'):
            table.add_column(col)
        for row in rows:
            table.add_row(
                row['name'],
                row['date'],
                row['mode'],
                row['target'],
                row['n'],
                row['scores'],
                row['file'],
            )
        # Render through an explicit buffer + typer.echo rather than letting
        # Rich write straight to its own stdout handle: this keeps output on
        # the Click/Typer stream so redirection and test capture both see it.
        # Width is taken from the real terminal (falling back to 80 when there
        # is no tty) so layout still adapts instead of being pinned to a
        # constant.
        import shutil

        width = shutil.get_terminal_size(fallback=(80, 24)).columns
        buffer = io.StringIO()
        Console(file=buffer, width=width).print(table)
        typer.echo(buffer.getvalue(), nl=False)
    except ImportError:
        header = f'{"Name":<20} {"Date":<20} {"Mode":<10} {"Target":<16} {"N":>4}  {"Scores":<30} File'
        typer.echo(header)
        typer.echo('-' * len(header))
        for row in rows:
            typer.echo(
                f'{row["name"]:<20} {row["date"]:<20} {row["mode"]:<10} '
                f'{row["target"]:<16} {row["n"]:>4}  {row["scores"]:<30} {row["file"]}'
            )

    if malformed:
        typer.echo(f'Warning: {malformed} malformed run file(s) skipped.', err=True)
    if rows:
        typer.echo(f'open: {_dashboard_command(runs_dir)}')


def _format_scorer_averages(averages: dict[str, float]) -> str:
    if not averages:
        return '—'
    return '  '.join(f'{k}={v:.2f}' for k, v in averages.items())


# ---------------------------------------------------------------------------
# ui
# ---------------------------------------------------------------------------


@app.command()
def ui(
    run_path: Annotated[
        Path | None,
        typer.Argument(help='Path to a run JSON file. Omit to open the latest run.'),
    ] = None,
    latest: Annotated[  # noqa: FBT002
        bool,
        typer.Option('--latest', '-l', help='Open the most recent auto-saved run.'),
    ] = False,
    port: Annotated[
        int,
        typer.Option(help='Port for the Streamlit server.'),
    ] = 8501,
    host: Annotated[
        str,
        typer.Option(help='Host to bind the Streamlit server to.'),
    ] = 'localhost',
) -> None:
    """Launch the interactive Streamlit dashboard for a simulation run."""
    typer.echo('Warning: eq sim ui is deprecated; use eq dashboard instead.', err=True)
    from evaluatorq.common.ui.launch import launch_streamlit

    runs_dir = _get_sim_runs_dir()

    if run_path is None or latest:
        if runs_dir.exists():
            run_files = sorted(runs_dir.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
            if run_files:
                run_path = run_files[0]
                typer.echo(f'Opening latest run: {run_path.name}')
        if run_path is None:
            typer.echo(
                'No runs found. Run `eq sim run` first, or pass a run path.',
                err=True,
            )
            raise typer.Exit(code=1)

    run_path = run_path.resolve()
    if not run_path.exists():
        # Allow passing a bare filename from the runs directory.
        candidate = runs_dir / run_path.name
        if candidate.exists():
            run_path = candidate
        else:
            typer.echo(f'Error: {run_path} does not exist.', err=True)
            raise typer.Exit(code=1)

    dashboard_script = Path(__file__).parent / 'ui' / 'dashboard.py'
    launch_streamlit(dashboard_script, run_path, port=port, host=host, extra='simulation')


def _maybe_generate_executive_summary(run: Any, *, enabled: bool, model: str) -> None:
    """Populate ``run.executive_summary`` in place. Best-effort; never raises.

    Skips silently when disabled, when there are no results, or when no LLM
    credentials are configured, so a default-on run without creds still
    produces a report. Uses the module-level ``resolve_llm_client`` so tests can
    monkeypatch it.
    """
    if not enabled or not run.results:
        return

    from evaluatorq.common.llm_client import MissingLLMCredentialsError
    from evaluatorq.common.reports.executive_summary import generate_executive_summary
    from evaluatorq.simulation.reports.executive_summary import (
        SIM_EXECUTIVE_SUMMARY_SYSTEM_PROMPT,
        build_sim_facts,
    )

    try:
        resolved = resolve_llm_client()
    except MissingLLMCredentialsError:
        logger.warning('Skipping executive summary: no LLM credentials configured.')
        return

    try:
        run.executive_summary = asyncio.run(
            generate_executive_summary(
                build_sim_facts(run.results),
                llm_client=resolved.client,
                model=model,
                system_prompt=SIM_EXECUTIVE_SUMMARY_SYSTEM_PROMPT,
            )
        )
    except Exception:
        logger.warning('Failed to generate executive summary', exc_info=True)


def _export_report(run: Any, directory: Path, *, fmt: str) -> None:
    """Write an auto-named Markdown or HTML report for *run* into *directory*."""
    from evaluatorq.common.reports import write_text_report
    from evaluatorq.simulation.reports.export_html import export_html
    from evaluatorq.simulation.reports.export_md import export_markdown

    stem = f'sim-report-{_sanitise_run_name(run.run_name)}-{run.created_at:%Y%m%d-%H%M%S}'
    if fmt == 'md':
        content = export_markdown(run.results, run_date=run.created_at, executive_summary=run.executive_summary)
    else:
        content = export_html(run.results, run_date=run.created_at, executive_summary=run.executive_summary)
    write_text_report(directory, stem=stem, fmt=fmt, content=content)


def _write_results(results: list[Any], output: Path) -> None:
    from evaluatorq.simulation.utils.dataset_export import export_results_to_jsonl

    output.parent.mkdir(parents=True, exist_ok=True)
    export_results_to_jsonl(results, str(output))
    typer.echo(f'Results written to {output}')


def _write_datapoints(datapoints: list[Any], output: Path, *, dataset_format: bool = False) -> None:
    """Write datapoints as JSONL — one row per line.

    Default (raw) is the canonical local handoff format: it preserves the
    datapoint ``id``, round-trips through ``load_datapoints_from_jsonl`` (which
    ``simulate`` uses), and validates under ``validate-dataset``. With
    ``dataset_format`` it writes the Orq-dataset envelope instead
    (``to_orq_dataset_rows`` — inputs/expected_output, persona & scenario
    JSON-stringified) for hand-off to ``sim upload-dataset`` or manual upload.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if dataset_format:
        from evaluatorq.simulation.utils.dataset_export import to_orq_dataset_rows

        lines = [json.dumps(row) for row in to_orq_dataset_rows(datapoints)]
    else:
        lines = [dp.model_dump_json() for dp in datapoints]
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    app()
