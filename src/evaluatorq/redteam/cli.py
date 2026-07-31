"""CLI for evaluatorq red teaming."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import typer

from evaluatorq.common import cli_width  # noqa: F401  — import for its non-TTY width side effect
from evaluatorq.common.cli_epilog import examples
from evaluatorq.common.cli_help import CONTEXT_SETTINGS
from evaluatorq.common.cli_json import echo_json
from evaluatorq.common.cli_tty import should_skip_confirm
from evaluatorq.dashboard.library import report_id
from evaluatorq.redteam.contracts import DEFAULT_PIPELINE_MODEL, DeliveryMethod, Pipeline, SaveMode, Vulnerability

app = typer.Typer(
    name='redteam',
    help='Red teaming CLI for evaluatorq.',
    no_args_is_help=True,
    rich_markup_mode='rich',
    context_settings=CONTEXT_SETTINGS,
)

_RUN_EPILOG = examples(
    '# dynamic run against an orq agent',
    'eq redteam run -t agent:my-agent',
    '# static + generated attacks from an OWASP dataset',
    'eq redteam run -t agent:my-agent --mode hybrid',
    '# scope to one OWASP category, machine-readable later via `eq redteam runs --json`',
    'eq redteam run -t agent:my-agent -c ASI01',
    '# replay the last run against a new agent version (same attacks, new target)',
    'eq redteam run -t agent:my-agent-v2 --from-run latest',
)


def _dashboard_command(directory: Path) -> str:
    return f'eq dashboard {shlex.quote(str(directory))}'


def _split_csv(values: list[str] | None) -> list[str] | None:
    """Flatten comma-separated tokens within a repeatable CLI option.

    Lets users write ``-d a,b``, ``-d a -d b``, or a mix of both. Blank tokens
    (e.g. trailing commas or a bare ``-d ,``) are dropped; if nothing survives,
    returns ``None`` so a malformed/empty filter reads as "flag omitted" rather
    than the ambiguous empty-set, which would otherwise filter everything out.
    """
    if values is None:
        return None
    tokens = [item.strip() for value in values for item in value.split(',') if item.strip()]
    return tokens or None


def _configure_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level."""
    if verbosity < 0:
        level = logging.ERROR
    elif verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logger = logging.getLogger('evaluatorq')
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(handler)

    try:
        from loguru import logger as loguru_logger

        loguru_logger.remove()
        loguru_logger.add(sys.stderr, level=level)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Markdown export helpers (importable for tests)
# ---------------------------------------------------------------------------


def _generate_report_filename(target: str, timestamp: str, ext: str) -> str:
    """Generate a safe report filename with the given extension.

    Args:
        target:    Target identifier string.
        timestamp: Timestamp string.
        ext:       File extension including dot (e.g. ``".html"``).

    Returns:
        Filename string.
    """
    safe_target = re.sub(r'[/:|\s]+', '-', target).strip('-')
    safe_target = re.sub(r'-{2,}', '-', safe_target)
    return f'redteam-report-{safe_target}-{timestamp}{ext}'


def _generate_md_filename(target: str, timestamp: str) -> str:
    """Generate a safe markdown report filename.

    Sanitizes the target name by replacing characters that are unsafe in
    filenames (``/``, ``:``, whitespace) with hyphens and collapsing
    consecutive hyphens.

    Args:
        target:    Target identifier string (e.g. ``"agent:my/target"``).
        timestamp: Timestamp string (e.g. ``"20250615_103000"``).

    Returns:
        Filename string, e.g. ``"redteam-report-agent-my-target-20250615_103000.md"``.
    """
    safe_target = re.sub(r'[/:|\s]+', '-', target).strip('-')
    safe_target = re.sub(r'-{2,}', '-', safe_target)
    return f'redteam-report-{safe_target}-{timestamp}.md'


def write_markdown_report(
    report: Any,
    output_dir: Path,
    target: str,
) -> Path:
    """Export ``report`` to a Markdown file inside ``output_dir``.

    The filename is auto-generated from ``target`` and the current timestamp.
    The directory is created if it does not exist.

    Args:
        report:     The red team report to export.
        output_dir: Directory in which to write the file.
        target:     Target identifier used for filename generation.

    Returns:
        Path to the written Markdown file.
    """
    from evaluatorq.redteam.reports.export_md import export_markdown

    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        typer.echo(f'Error: failed to create report directory {output_dir}: {e}', err=True)
        raise typer.Exit(code=1) from e

    timestamp = datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')
    filename = _generate_md_filename(target=target, timestamp=timestamp)
    output_path = output_dir / filename

    md_content = export_markdown(report)
    try:
        output_path.write_text(md_content, encoding='utf-8')
    except OSError as e:
        typer.echo(f'Error: failed to write Markdown report to {output_path}: {e}', err=True)
        raise typer.Exit(code=1) from e
    return output_path


def write_html_report(
    report: Any,
    output_dir: Path,
    target: str,
) -> Path:
    """Export ``report`` to an HTML file inside ``output_dir``."""
    from evaluatorq.redteam.reports.export_html import export_html

    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        typer.echo(f'Error: failed to create report directory {output_dir}: {e}', err=True)
        raise typer.Exit(code=1) from e

    timestamp = datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')
    filename = _generate_report_filename(target=target, timestamp=timestamp, ext='.html')
    output_path = output_dir / filename

    html_content = export_html(report)
    try:
        output_path.write_text(html_content, encoding='utf-8')
    except OSError as e:
        typer.echo(f'Error: failed to write HTML report to {output_path}: {e}', err=True)
        raise typer.Exit(code=1) from e
    return output_path


@app.command(no_args_is_help=True, epilog=_RUN_EPILOG)
def run(
    target: Annotated[
        list[str],
        typer.Option(
            '--target',
            '-t',
            help='Target identifier(s), e.g. "agent:<key>" or "deployment:<key>". For OpenAI models use OpenAIModelTarget in the Python API. Repeatable.',
        ),
    ],
    name: Annotated[
        str | None,
        typer.Option('--name', '-n', help="Experiment name (defaults to 'red-team')."),
    ] = None,
    mode: Annotated[
        Pipeline,
        typer.Option(help='Execution mode.'),
    ] = Pipeline.DYNAMIC,
    categories: Annotated[
        list[str] | None,
        typer.Option(
            '--category',
            '-c',
            help='OWASP categories to test (e.g. ASI01). Repeatable and/or comma-separated. Defaults to all.',
        ),
    ] = None,
    vulnerabilities: Annotated[
        list[str] | None,
        typer.Option(
            '--vulnerability',
            '-V',
            help=(
                "Vulnerability IDs to test (e.g. 'goal_hijacking', 'prompt_injection'). "
                'Repeatable and/or comma-separated. Also accepts OWASP codes (ASI01, LLM01). '
                'Takes precedence over --category. '
                f'Available: {", ".join(v.value for v in Vulnerability)}.'
            ),
        ),
    ] = None,
    strategies: Annotated[
        list[str] | None,
        typer.Option(
            '--strategy',
            '-s',
            help=(
                'Restrict the run to attack strategies whose name matches one of '
                'the supplied values. Repeatable and/or comma-separated '
                '(-s a,b or -s a -s b). Filter applies to both registry and '
                'LLM-generated strategies. Unknown registry names are rejected.'
            ),
        ),
    ] = None,
    delivery_methods: Annotated[
        list[str] | None,
        typer.Option(
            '--delivery-method',
            '-d',
            help=(
                'Restrict the run to attacks using one of the listed delivery '
                'methods. Repeatable and/or comma-separated (-d a,b or -d a -d b). '
                'Combines with --strategy as AND. '
                f'Available: {", ".join(m.value for m in DeliveryMethod)}.'
            ),
        ),
    ] = None,
    max_turns: Annotated[
        int,
        typer.Option(help='Maximum conversation turns for multi-turn attacks.'),
    ] = 5,
    max_per_category: Annotated[
        int | None,
        typer.Option(help='Cap strategies per category.'),
    ] = None,
    attack_model: Annotated[
        str,
        typer.Option(help='Model for adversarial prompt generation.'),
    ] = DEFAULT_PIPELINE_MODEL,
    attacker_instructions: Annotated[
        str | None,
        typer.Option(
            '--attacker-instructions',
            help=(
                'Domain-specific context to steer attack generation '
                "(e.g. 'this agent handles financial transactions, try to get it to approve fraudulent ones')."
            ),
        ),
    ] = None,
    evaluator_model: Annotated[
        str,
        typer.Option(help='Model for OWASP evaluation scoring.'),
    ] = DEFAULT_PIPELINE_MODEL,
    parallelism: Annotated[
        int,
        typer.Option(help='Maximum concurrent evaluatorq jobs.'),
    ] = 10,
    generated_strategy_count: Annotated[
        int,
        typer.Option(help='Number of strategies to generate per category.'),
    ] = 2,
    no_generate_strategies: Annotated[  # noqa: FBT002
        bool,
        typer.Option('--no-generate-strategies', help='Disable LLM-based strategy generation.'),
    ] = False,
    max_dynamic_datapoints: Annotated[
        int | None,
        typer.Option(help='Cap dynamic (generated) datapoints.'),
    ] = None,
    max_static_datapoints: Annotated[
        int | None,
        typer.Option(help='Cap static (dataset) datapoints.'),
    ] = None,
    no_cleanup_memory: Annotated[  # noqa: FBT002
        bool,
        typer.Option('--no-cleanup-memory', help='Skip memory entity cleanup after dynamic runs.'),
    ] = False,
    dataset: Annotated[
        str | None,
        typer.Option(help='Dataset source: local path, "hf:org/repo", or "hf:org/repo/file.json".'),
    ] = None,
    from_run: Annotated[
        str | None,
        typer.Option(
            '--from-run',
            help=(
                'Replay a previous run instead of generating data: pass its file name, '
                'run id, path, or "latest". Re-runs the exact same attacks, so only the '
                'target and models may differ. Cannot be combined with --mode, --dataset, '
                '--category, --vulnerability, --strategy, --delivery-method, or the '
                '--max-*-datapoints caps.'
            ),
        ),
    ] = None,
    artifacts_dir: Annotated[
        Path | None,
        typer.Option('--artifacts-dir', help='Directory for saved JSON files. Required with --save detail.'),
    ] = None,
    save: Annotated[
        SaveMode,
        typer.Option(
            help="What to persist: 'none' (no files), 'final' (summary only), or 'detail' (all stage artifacts)."
        ),
    ] = SaveMode.FINAL,
    yes: Annotated[  # noqa: FBT002
        bool,
        typer.Option('--yes', '-y', help='Skip confirmation prompt.'),
    ] = False,
    verbose: Annotated[
        int,
        typer.Option(
            '--verbose',
            '-v',
            count=True,
            help='Increase verbosity (-v per-attack progress + info logs, -vv debug logs).',
        ),
    ] = 0,
    quiet: Annotated[  # noqa: FBT002
        bool,
        typer.Option('--quiet', '-q', help='Suppress progress bars and non-error output.'),
    ] = False,
    report_path: Annotated[
        Path | None,
        typer.Option('--report', help='Path to write the report JSON.'),
    ] = None,
    report_md: Annotated[
        Path | None,
        typer.Option(
            '--report-md',
            help='Directory path to write a Markdown report. Filename is auto-generated.',
        ),
    ] = None,
    report_html: Annotated[
        Path | None,
        typer.Option(
            '--report-html',
            help='Directory path to write an HTML report. Filename is auto-generated.',
        ),
    ] = None,
    executive_summary: Annotated[  # noqa: FBT002
        bool,
        typer.Option(
            '--executive-summary/--no-executive-summary',
            help='Generate an LLM narrative executive summary at the top of the report (needs LLM creds).',
        ),
    ] = True,
    system_prompt: Annotated[
        str | None,
        typer.Option('--system-prompt', help='System prompt for the target model/agent.'),
    ] = None,
) -> None:
    """Run red teaming against one or more targets."""
    if quiet:
        verbose = -1
    _configure_logging(verbose)

    from evaluatorq.common.replay import ReplayError
    from evaluatorq.redteam import red_team
    from evaluatorq.redteam.contracts import EvaluatorConfig, LLMCallConfig, LLMConfig, TargetConfig
    from evaluatorq.redteam.exceptions import CancelledError, RedTeamError
    from evaluatorq.redteam.hooks import RichHooks

    # Allow comma-separated values within repeatable flags (-s a,b == -s a -s b).
    # Done before validation so the checks below see individual tokens.
    strategies = _split_csv(strategies)
    delivery_tokens = _split_csv(delivery_methods)
    categories = _split_csv(categories)
    vulnerabilities = _split_csv(vulnerabilities)

    # Validate vulnerability IDs early for a clean error message
    if vulnerabilities:
        from evaluatorq.redteam.vulnerability_registry import CATEGORY_TO_VULNERABILITY

        valid_ids = {v.value for v in Vulnerability} | set(CATEGORY_TO_VULNERABILITY.keys())
        for v in vulnerabilities:
            if v not in valid_ids:
                raise typer.BadParameter(
                    f'Unknown vulnerability ID: {v!r}. Valid IDs: {sorted(vi.value for vi in Vulnerability)}'
                )

    # Validate strategy names early: must be a registered strategy or a
    # runtime-generated name (generated_* prefix). Mirrors --vulnerability.
    if strategies:
        from evaluatorq.redteam.adaptive.strategy_registry import known_strategy_names

        known = known_strategy_names()
        unknown = [s for s in strategies if s not in known and not s.startswith('generated_')]
        if unknown:
            raise typer.BadParameter(
                f'Unknown strategy name(s): {unknown}. '
                f"Valid names: {sorted(known)} (or a 'generated_*' name from a prior run)."
            )

    # Convert known delivery methods to the enum (parsed as plain strings to
    # support comma-separated input, which typer's enum binding cannot do).
    # Unknown values are an open set — kept as raw strings (so a dataset's custom
    # delivery method is still filterable) with a warning, not a hard error.
    resolved_delivery_methods: list[DeliveryMethod | str] | None = None
    if delivery_tokens:
        valid = {m.value for m in DeliveryMethod}
        unknown = [d for d in delivery_tokens if d not in valid]
        if unknown:
            typer.echo(
                f'Warning: delivery method(s) {unknown} are not in DeliveryMethod {sorted(valid)}; '
                'filtering by them literally (they will only match a dataset row spelled exactly the same).',
                err=True,
            )
        resolved_delivery_methods = [DeliveryMethod(d) if d in valid else d for d in delivery_tokens]

    target_config = TargetConfig(system_prompt=system_prompt) if system_prompt else None
    targets: list[str] | str = target if len(target) > 1 else target[0]

    # Build LLMConfig from CLI flags
    config = LLMConfig(
        attacker=LLMCallConfig(model=attack_model),
        evaluator=EvaluatorConfig(model=evaluator_model),
    )

    try:
        report = asyncio.run(
            red_team(
                target=targets,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
                llm_config=config,
                name=name,
                mode=mode,
                categories=categories,
                vulnerabilities=vulnerabilities,
                strategies=strategies,
                delivery_methods=resolved_delivery_methods,
                max_turns=max_turns,
                max_per_category=max_per_category,
                parallelism=parallelism,
                generate_strategies=not no_generate_strategies,
                generated_strategy_count=generated_strategy_count,
                max_dynamic_datapoints=max_dynamic_datapoints,
                max_static_datapoints=max_static_datapoints,
                cleanup_memory=not no_cleanup_memory,
                dataset=dataset,
                previous_run=from_run,
                hooks=RichHooks(skip_confirm=should_skip_confirm(yes)),
                artifacts_dir=artifacts_dir,
                save=save,
                target_config=target_config,
                generate_executive_summary=executive_summary,
                attacker_instructions=attacker_instructions,
                verbosity=verbose + 1,
            )
        )
    except CancelledError:
        typer.echo('Run cancelled.')
        raise typer.Exit(code=0)
    except KeyboardInterrupt:
        typer.echo('\nInterrupted.')
        raise typer.Exit(code=130)
    except (RedTeamError, ReplayError) as e:
        typer.echo(f'Error: {e}', err=True)
        raise typer.Exit(code=1)
    except ValueError as e:
        typer.echo(f'Error: {e}', err=True)
        raise typer.Exit(code=1)
    except ImportError as e:
        # e.g. static/hybrid run needs huggingface-hub — show the install hint
        # cleanly instead of a traceback.
        typer.echo(f'Error: {e}', err=True)
        raise typer.Exit(code=1)

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.model_dump(mode='json'), indent=2, default=str))
        typer.echo(f'Report saved to {report_path}')

    if report_md:
        target_label = targets if isinstance(targets, str) else ', '.join(targets)
        md_path = write_markdown_report(report, output_dir=report_md, target=target_label)
        typer.echo(f'Markdown report written to {md_path}')

    if report_html:
        target_label = targets if isinstance(targets, str) else ', '.join(targets)
        html_path = write_html_report(report, output_dir=report_html, target=target_label)
        typer.echo(f'HTML report written to {html_path}')


@app.command()
def ui(
    report_path: Annotated[
        Path | None,
        typer.Argument(help='Path to report JSON file or directory. Omit to use the latest run.'),
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
    """Launch the interactive Streamlit dashboard for a red team report."""
    typer.echo('Warning: eq redteam ui is deprecated; use eq dashboard instead.', err=True)
    from evaluatorq.redteam.runner import get_runs_dir

    if report_path is None or latest:
        # Find the most recent auto-saved run
        runs_dir = get_runs_dir()
        if runs_dir.exists():
            run_files = sorted(runs_dir.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
            if run_files:
                report_path = run_files[0]
                typer.echo(f'Opening latest run: {report_path.name}')
        if report_path is None:
            typer.echo(
                'No runs found. Run `eq redteam run` first, or pass a report path.',
                err=True,
            )
            raise typer.Exit(code=1)

    report_path = report_path.resolve()
    if not report_path.exists():
        # Check if it's a filename from the runs directory
        candidate = get_runs_dir() / report_path.name
        if candidate.exists():
            report_path = candidate
        else:
            typer.echo(f'Error: {report_path} does not exist.', err=True)
            raise typer.Exit(code=1)

    from evaluatorq.common.ui.launch import launch_streamlit

    dashboard_script = Path(__file__).parent / 'ui' / 'dashboard.py'
    launch_streamlit(dashboard_script, report_path, port=port, host=host, extra='redteam')


@app.command()
def validate_dataset(
    dataset: Annotated[
        str | None,
        typer.Argument(
            help='Dataset source: local path, "hf:org/repo", or "hf:org/repo/file.json". Default: HuggingFace orq/redteam-vulnerabilities.'
        ),
    ] = None,
) -> None:
    """Validate the shape of a red team dataset.

    Checks that every sample has the required fields and that messages
    are well-formed.  Does NOT enforce enum membership for open-set
    fields like attack_technique or delivery_method.
    """
    from pydantic import ValidationError as _ValidationError

    from evaluatorq.redteam.contracts import RedTeamSample

    # Load raw JSON via the unified dataset loader's internal helpers
    from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import (
        DEFAULT_HF_FILENAME,
        DEFAULT_HF_REPO,
        _parse_hf_source,
    )

    if dataset is None:
        # Default: download from HuggingFace
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            typer.echo(
                'huggingface-hub not installed. Install with: pip install evaluatorq[redteam]',
                err=True,
            )
            raise typer.Exit(code=1)
        typer.echo(f'Downloading from HuggingFace: {DEFAULT_HF_REPO}/{DEFAULT_HF_FILENAME}')
        try:
            local_path = hf_hub_download(repo_id=DEFAULT_HF_REPO, filename=DEFAULT_HF_FILENAME, repo_type='dataset')
        except Exception as e:
            typer.echo(
                f'Failed to download dataset from HuggingFace '
                f'({DEFAULT_HF_REPO}/{DEFAULT_HF_FILENAME}): {e}. Check your network '
                'connection, that the repository exists, and your access '
                '(set HF_TOKEN for gated/private datasets).',
                err=True,
            )
            raise typer.Exit(code=1)
        try:
            with open(local_path) as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            typer.echo(f'Failed to read dataset file: {e}', err=True)
            raise typer.Exit(code=1)
    elif dataset.startswith('hf:'):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            typer.echo(
                'huggingface-hub not installed. Install with: pip install evaluatorq[redteam]',
                err=True,
            )
            raise typer.Exit(code=1)
        repo, filename = _parse_hf_source(dataset.removeprefix('hf:'))
        typer.echo(f'Downloading from HuggingFace: {repo}/{filename}')
        try:
            local_path = hf_hub_download(repo_id=repo, filename=filename, repo_type='dataset')
        except Exception as e:
            typer.echo(
                f'Failed to download dataset from HuggingFace ({repo}/{filename}): {e}. '
                'Check your network connection, that the repository exists, and your '
                'access (set HF_TOKEN for gated/private datasets).',
                err=True,
            )
            raise typer.Exit(code=1)
        try:
            with open(local_path) as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            typer.echo(f'Failed to read dataset file: {e}', err=True)
            raise typer.Exit(code=1)
    else:
        path = Path(dataset)
        typer.echo(f'Validating local file: {path}')
        with path.open() as f:
            raw = json.load(f)

    # Validate top-level shape
    if not isinstance(raw, dict) or 'samples' not in raw:
        typer.echo("FAIL: Expected top-level object with 'samples' key.", err=True)
        raise typer.Exit(code=1)

    samples = raw['samples']
    typer.echo(f'Found {len(samples)} samples.')

    errors: list[str] = []
    for i, sample in enumerate(samples):
        try:
            RedTeamSample.model_validate(sample)
        except _ValidationError as e:  # noqa: PERF203
            for err in e.errors():
                loc = ' -> '.join(str(loc_part) for loc_part in err['loc'])
                errors.append(f'  sample[{i}].{loc}: {err["msg"]}')

    if errors:
        typer.echo(f'\nFAIL: {len(errors)} validation error(s):', err=True)
        for line in errors[:20]:
            typer.echo(line, err=True)
        if len(errors) > 20:
            typer.echo(f'  ... and {len(errors) - 20} more', err=True)
        raise typer.Exit(code=1)

    typer.echo(f'OK: All {len(samples)} samples are valid.')


@app.command()
def runs(
    path: Annotated[
        Path | None,
        typer.Argument(help='Directory containing run reports. Defaults to .evaluatorq/runs/'),
    ] = None,
    limit: Annotated[
        int,
        typer.Option('--limit', '-n', help='Maximum number of runs to show.'),
    ] = 20,
    json_output: Annotated[  # noqa: FBT002
        bool,
        typer.Option('--json', help='Emit runs as a JSON array on stdout (machine-readable).'),
    ] = False,
) -> None:
    """List previous red team runs saved locally."""
    from evaluatorq.redteam.runner import get_runs_dir

    runs_dir = Path(path) if path is not None else get_runs_dir()
    if not runs_dir.exists():
        if json_output:
            echo_json([])
            raise typer.Exit(code=0)
        typer.echo(f'No runs found (directory {runs_dir} does not exist).')
        raise typer.Exit(code=0)

    from evaluatorq.common.run_manifest import list_run_records
    from evaluatorq.dashboard.library import _manifest_card_id

    # Manifest-first: build rows from the tiny manifest sidecars (status + compact
    # summary), falling back to a full-report read only for legacy runs with no
    # manifest. A single table with a Status column shows running/error/cancelled/
    # completed runs together (no separate 'Active runs' block, no double-listing).
    records_src = list_run_records(runs_dir)[:limit]
    if not records_src:
        if json_output:
            echo_json([])
            raise typer.Exit(code=0)
        typer.echo(f'No runs found in {runs_dir}.')
        raise typer.Exit(code=0)

    def _row(manifest: Any, report_path: Path | None) -> dict[str, Any] | None:
        """Normalize a (manifest, report_path) record to the fields the table needs.

        Manifest rows read the compact ``summary`` (no full-report read); legacy
        rows (manifest is None) read the full report as before. Returns None when
        a legacy report can't be parsed.
        """
        if manifest is not None:
            summary = manifest.summary or {}
            rid = report_id(report_path) if report_path is not None else _manifest_card_id(manifest.run_id)
            return {
                'report_id': rid,
                'run_name': manifest.run_name,
                'created_at': manifest.started_at.isoformat(),
                'status': manifest.status.value,
                'pipeline': summary.get('pipeline'),
                'tested_agents': summary.get('tested_agents', []),
                'total_attacks': summary.get('total_attacks', summary.get('total_results')),
                'vulnerability_rate': summary.get('vulnerability_rate'),
                'file': report_path.name if report_path is not None else None,
            }
        # Legacy report with no manifest — read the full report for its stats.
        if report_path is None:
            return None
        try:
            data = json.loads(report_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, dict):
            return None
        summary = data.get('summary', {})
        if not isinstance(summary, dict):
            return None  # malformed report shape — skip (matches legacy behavior)
        return {
            'report_id': report_id(report_path),
            'run_name': data.get('run_name', report_path.stem),
            'created_at': data.get('created_at'),
            'status': 'completed',
            'pipeline': data.get('pipeline'),
            'tested_agents': data.get('tested_agents', []),
            'total_attacks': summary.get('total_attacks', data.get('total_results')),
            'vulnerability_rate': summary.get('vulnerability_rate'),
            'file': report_path.name,
        }

    rows: list[dict[str, Any]] = []
    skipped = 0
    for manifest, report_path in records_src:
        row = _row(manifest, report_path)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    if json_output:
        echo_json(rows)
        if skipped:
            typer.echo(f'Warning: {skipped} file(s) could not be parsed and were skipped.', err=True)
        raise typer.Exit(code=0)

    def _fmt_created(created: Any) -> str:
        return created[:16].replace('T', ' ') if isinstance(created, str) and len(created) >= 16 else str(created or '')

    def _fmt_asr(asr: Any) -> str:
        return f'{asr:.0%}' if isinstance(asr, (int, float)) else '—'

    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        table = Table(title=f'Red Team Runs ({runs_dir})', show_header=True, box=box.ROUNDED)
        table.add_column('Name', style='cyan')
        table.add_column('Date', style='white')
        table.add_column('Status', style='white')
        table.add_column('Mode', style='white')
        table.add_column('Targets', style='white')
        table.add_column('Attacks', style='white', justify='right')
        table.add_column('ASR', style='white', justify='right')
        table.add_column('File', style='dim')

        for row in rows:
            agents = row.get('tested_agents') or []
            table.add_row(
                str(row['run_name']),
                _fmt_created(row.get('created_at')),
                str(row.get('status', '—')),
                str(row.get('pipeline') or '—'),
                ', '.join(agents) if agents else '—',
                str(row.get('total_attacks') if row.get('total_attacks') is not None else '—'),
                _fmt_asr(row.get('vulnerability_rate')),
                str(row.get('file') or '—'),
            )

        console = Console()
        console.print(table)
        if skipped:
            console.print(f'[yellow]Warning: {skipped} file(s) could not be parsed and were skipped.[/yellow]')

    except ImportError:
        # Fallback without rich
        typer.echo(f'{"Name":<20} {"Date":<17} {"Status":<10} {"Mode":<8} {"Attacks":>7} {"ASR":>5}  File')
        typer.echo('-' * 88)
        for row in rows:
            name = str(row['run_name'])[:20]
            status = str(row.get('status') or '—')
            pipeline = str(row.get('pipeline') or '—')
            total = row.get('total_attacks')
            total_str = str(total) if total is not None else '—'
            typer.echo(
                f'{name:<20} {_fmt_created(row.get("created_at")):<17} '
                f'{status:<10} {pipeline:<8} '
                f'{total_str:>7} {_fmt_asr(row.get("vulnerability_rate")):>5}  '
                f'{row.get("file") or "—"}'
            )
        if skipped:
            typer.echo(
                f'Warning: {skipped} file(s) could not be parsed and were skipped.',
                err=True,
            )

    if rows:
        typer.echo(f'open: {_dashboard_command(runs_dir)}')
