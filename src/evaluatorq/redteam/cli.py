"""CLI for evaluatorq red teaming."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from evaluatorq.redteam.contracts import DEFAULT_PIPELINE_MODEL, Vulnerability

app = typer.Typer(
    name="redteam",
    help="Red teaming CLI for evaluatorq.",
    no_args_is_help=True,
)


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

    logger = logging.getLogger("evaluatorq")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
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
    safe_target = re.sub(r"[/:|\s]+", "-", target).strip("-")
    safe_target = re.sub(r"-{2,}", "-", safe_target)
    return f"redteam-report-{safe_target}-{timestamp}{ext}"


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
    safe_target = re.sub(r"[/:|\s]+", "-", target).strip("-")
    safe_target = re.sub(r"-{2,}", "-", safe_target)
    return f"redteam-report-{safe_target}-{timestamp}.md"


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
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = _generate_md_filename(target=target, timestamp=timestamp)
    output_path = output_dir / filename

    md_content = export_markdown(report)
    output_path.write_text(md_content, encoding="utf-8")
    return output_path


def write_html_report(
    report: Any,
    output_dir: Path,
    target: str,
) -> Path:
    """Export ``report`` to an HTML file inside ``output_dir``."""
    from evaluatorq.redteam.reports.export_html import export_html

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = _generate_report_filename(target=target, timestamp=timestamp, ext=".html")
    output_path = output_dir / filename

    html_content = export_html(report)
    output_path.write_text(html_content, encoding="utf-8")
    return output_path



@app.command(no_args_is_help=True)
def run(
    target: Annotated[
        list[str],
        typer.Option(
            "--target", "-t",
            help='Target identifier(s), e.g. "agent:<key>" or "llm:<model>". Repeatable.',
        ),
    ],
    name: Annotated[
        Optional[str],
        typer.Option("--name", "-n", help="Experiment name (defaults to 'red-team')."),
    ] = None,
    mode: Annotated[
        str,
        typer.Option(help='Execution mode: "dynamic", "static", or "hybrid".'),
    ] = "dynamic",
    categories: Annotated[
        Optional[list[str]],
        typer.Option(
            "--category", "-c",
            help="OWASP categories to test (e.g. ASI01). Repeatable. Defaults to all.",
        ),
    ] = None,
    vulnerabilities: Annotated[
        Optional[list[str]],
        typer.Option(
            "--vulnerability", "-V",
            help=(
                "Vulnerability IDs to test (e.g. 'goal_hijacking', 'prompt_injection'). "
                "Repeatable. Also accepts OWASP codes (ASI01, LLM01). "
                "Takes precedence over --category. "
                f"Available: {', '.join(v.value for v in Vulnerability)}."
            ),
        ),
    ] = None,
    max_turns: Annotated[
        int,
        typer.Option(help="Maximum conversation turns for multi-turn attacks."),
    ] = 5,
    max_per_category: Annotated[
        Optional[int],
        typer.Option(help="Cap strategies per category."),
    ] = None,
    attack_model: Annotated[
        str,
        typer.Option(help="Model for adversarial prompt generation."),
    ] = DEFAULT_PIPELINE_MODEL,
    attacker_instructions: Annotated[
        Optional[str],
        typer.Option(
            "--attacker-instructions",
            help=(
                "Domain-specific context to steer attack generation "
                "(e.g. 'this agent handles financial transactions, try to get it to approve fraudulent ones')."
            ),
        ),
    ] = None,
    evaluator_model: Annotated[
        str,
        typer.Option(help="Model for OWASP evaluation scoring."),
    ] = DEFAULT_PIPELINE_MODEL,
    parallelism: Annotated[
        int,
        typer.Option(help="Maximum concurrent evaluatorq jobs."),
    ] = 10,
    generated_strategy_count: Annotated[
        int,
        typer.Option(help="Number of strategies to generate per category."),
    ] = 2,
    no_generate_strategies: Annotated[
        bool,
        typer.Option("--no-generate-strategies", help="Disable LLM-based strategy generation."),
    ] = False,
    max_dynamic_datapoints: Annotated[
        Optional[int],
        typer.Option(help="Cap dynamic (generated) datapoints."),
    ] = None,
    max_static_datapoints: Annotated[
        Optional[int],
        typer.Option(help="Cap static (dataset) datapoints."),
    ] = None,
    no_cleanup_memory: Annotated[
        bool,
        typer.Option("--no-cleanup-memory", help="Skip memory entity cleanup after dynamic runs."),
    ] = False,
    backend: Annotated[
        str,
        typer.Option(help='Backend name ("openai" or "orq").'),
    ] = "openai",
    dataset: Annotated[
        Optional[str],
        typer.Option(help='Dataset source: local path, "hf:org/repo", or "hf:org/repo/file.json".'),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(help="Directory to save intermediate stage artifacts."),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt."),
    ] = False,
    verbose: Annotated[
        int,
        typer.Option("--verbose", "-v", count=True, help="Increase verbosity (-v per-attack progress + info logs, -vv debug logs)."),
    ] = 0,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress progress bars and non-error output."),
    ] = False,
    save_report: Annotated[
        Optional[Path],
        typer.Option(help="Path to write the report JSON."),
    ] = None,
    export_md: Annotated[
        Optional[Path],
        typer.Option(
            "--export-md",
            help="Directory path to write a Markdown report. Filename is auto-generated.",
        ),
    ] = None,
    export_html: Annotated[
        Optional[Path],
        typer.Option(
            "--export-html",
            help="Directory path to write an HTML report. Filename is auto-generated.",
        ),
    ] = None,
    system_prompt: Annotated[
        Optional[str],
        typer.Option("--system-prompt", help="System prompt for the target model/agent."),
    ] = None,
) -> None:
    """Run red teaming against one or more targets."""
    from dotenv import load_dotenv

    load_dotenv(override=False)

    if quiet:
        verbose = -1
    _configure_logging(verbose)

    from evaluatorq.redteam import red_team
    from evaluatorq.redteam.contracts import TargetConfig
    from evaluatorq.redteam.exceptions import CancelledError, RedTeamError
    from evaluatorq.redteam.hooks import RichHooks

    # Validate vulnerability IDs early for a clean error message
    if vulnerabilities:
        from evaluatorq.redteam.vulnerability_registry import CATEGORY_TO_VULNERABILITY

        valid_ids = {v.value for v in Vulnerability} | set(CATEGORY_TO_VULNERABILITY.keys())
        for v in vulnerabilities:
            if v not in valid_ids:
                raise typer.BadParameter(
                    f"Unknown vulnerability ID: {v!r}. "
                    f"Valid IDs: {sorted(vi.value for vi in Vulnerability)}"
                )


    target_config = TargetConfig(system_prompt=system_prompt) if system_prompt else None
    targets: list[str] | str = target if len(target) > 1 else target[0]

    try:
        report = asyncio.run(
            red_team(
                target=targets,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
                name=name,
                mode=mode,
                categories=categories,
                vulnerabilities=vulnerabilities,
                max_turns=max_turns,
                max_per_category=max_per_category,
                attack_model=attack_model,
                evaluator_model=evaluator_model,
                parallelism=parallelism,
                generate_strategies=not no_generate_strategies,
                generated_strategy_count=generated_strategy_count,
                max_dynamic_datapoints=max_dynamic_datapoints,
                max_static_datapoints=max_static_datapoints,
                cleanup_memory=not no_cleanup_memory,
                backend=backend,
                dataset=dataset,
                hooks=RichHooks(skip_confirm=yes),
                output_dir=output_dir,
                target_config=target_config,
                attacker_instructions=attacker_instructions,
                verbosity=verbose + 1,
            )
        )
    except CancelledError:
        typer.echo("Run cancelled.")
        raise typer.Exit(code=0)
    except KeyboardInterrupt:
        typer.echo("\nInterrupted.")
        raise typer.Exit(code=130)
    except RedTeamError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    if save_report:
        save_report.parent.mkdir(parents=True, exist_ok=True)
        save_report.write_text(json.dumps(report.model_dump(mode="json"), indent=2, default=str))
        typer.echo(f"Report saved to {save_report}")

    if export_md:
        target_label = targets if isinstance(targets, str) else ", ".join(targets)
        md_path = write_markdown_report(report, output_dir=export_md, target=target_label)
        typer.echo(f"Markdown report written to {md_path}")

    if export_html:
        target_label = targets if isinstance(targets, str) else ", ".join(targets)
        html_path = write_html_report(report, output_dir=export_html, target=target_label)
        typer.echo(f"HTML report written to {html_path}")


@app.command()
def ui(
    report_path: Annotated[
        Optional[Path],
        typer.Argument(help="Path to report JSON file or directory. Omit to use the latest run."),
    ] = None,
    latest: Annotated[
        bool,
        typer.Option("--latest", "-l", help="Open the most recent auto-saved run."),
    ] = False,
    port: Annotated[
        int,
        typer.Option(help="Port for the Streamlit server."),
    ] = 8501,
    host: Annotated[
        str,
        typer.Option(help="Host to bind the Streamlit server to."),
    ] = "localhost",
) -> None:
    """Launch the interactive Streamlit dashboard for a red team report."""
    from evaluatorq.redteam.runner import get_runs_dir

    if report_path is None or latest:
        # Find the most recent auto-saved run
        runs_dir = get_runs_dir()
        if runs_dir.exists():
            run_files = sorted(runs_dir.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
            if run_files:
                report_path = run_files[0]
                typer.echo(f"Opening latest run: {report_path.name}")
        if report_path is None:
            typer.echo("No runs found. Run `eq redteam run` first, or pass a report path.", err=True)
            raise typer.Exit(code=1)

    report_path = report_path.resolve()
    if not report_path.exists():
        # Check if it's a filename from the runs directory
        candidate = get_runs_dir() / report_path.name
        if candidate.exists():
            report_path = candidate
        else:
            typer.echo(f"Error: {report_path} does not exist.", err=True)
            raise typer.Exit(code=1)

    dashboard_script = Path(__file__).parent / "ui" / "dashboard.py"
    if not dashboard_script.exists():
        typer.echo("Error: Dashboard module not found.", err=True)
        raise typer.Exit(code=1)

    try:
        import streamlit  # noqa: F401
    except ImportError:
        typer.echo(
            'Streamlit is not installed. Install the ui extras:\n'
            '  pip install "evaluatorq[ui]"',
            err=True,
        )
        raise typer.Exit(code=1)

    result = subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_script),
            "--server.port", str(port),
            "--server.address", host,
            "--browser.gatherUsageStats", "false",
            "--", str(report_path),
        ],
    )
    raise typer.Exit(code=result.returncode)


@app.command()
def validate_dataset(
    dataset: Annotated[
        Optional[str],
        typer.Argument(help='Dataset source: local path, "hf:org/repo", or "hf:org/repo/file.json". Default: HuggingFace orq/redteam-vulnerabilities.'),
    ] = None,
) -> None:
    """Validate the shape of a red team dataset.

    Checks that every sample has the required fields and that messages
    are well-formed.  Does NOT enforce enum membership for open-set
    fields like attack_technique or delivery_method.
    """
    from pydantic import ValidationError as _VE

    from evaluatorq.redteam.contracts import RedTeamSample, StaticDataset

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
            typer.echo("huggingface-hub not installed. Install with: pip install evaluatorq[redteam]", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"Downloading from HuggingFace: {DEFAULT_HF_REPO}/{DEFAULT_HF_FILENAME}")
        local_path = hf_hub_download(repo_id=DEFAULT_HF_REPO, filename=DEFAULT_HF_FILENAME, repo_type='dataset')
        with open(local_path) as f:
            raw = json.load(f)
    elif dataset.startswith('hf:'):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            typer.echo("huggingface-hub not installed. Install with: pip install evaluatorq[redteam]", err=True)
            raise typer.Exit(code=1)
        repo, filename = _parse_hf_source(dataset.removeprefix('hf:'))
        typer.echo(f"Downloading from HuggingFace: {repo}/{filename}")
        local_path = hf_hub_download(repo_id=repo, filename=filename, repo_type='dataset')
        with open(local_path) as f:
            raw = json.load(f)
    else:
        path = Path(dataset)
        typer.echo(f"Validating local file: {path}")
        with path.open() as f:
            raw = json.load(f)

    # Validate top-level shape
    if not isinstance(raw, dict) or 'samples' not in raw:
        typer.echo("FAIL: Expected top-level object with 'samples' key.", err=True)
        raise typer.Exit(code=1)

    samples = raw['samples']
    typer.echo(f"Found {len(samples)} samples.")

    errors: list[str] = []
    for i, sample in enumerate(samples):
        try:
            RedTeamSample.model_validate(sample)
        except _VE as e:
            for err in e.errors():
                loc = ' -> '.join(str(l) for l in err['loc'])
                errors.append(f"  sample[{i}].{loc}: {err['msg']}")

    if errors:
        typer.echo(f"\nFAIL: {len(errors)} validation error(s):", err=True)
        for line in errors[:20]:
            typer.echo(line, err=True)
        if len(errors) > 20:
            typer.echo(f"  ... and {len(errors) - 20} more", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"OK: All {len(samples)} samples are valid.")


@app.command()
def runs(
    path: Annotated[
        Optional[Path],
        typer.Argument(help="Directory containing run reports. Defaults to .evaluatorq/runs/"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum number of runs to show."),
    ] = 20,
) -> None:
    """List previous red team runs saved locally."""
    from evaluatorq.redteam.runner import get_runs_dir

    runs_dir = Path(path) if path is not None else get_runs_dir()
    if not runs_dir.exists():
        typer.echo(f"No runs found (directory {runs_dir} does not exist).")
        raise typer.Exit(code=0)

    run_files = sorted(runs_dir.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_files:
        typer.echo(f"No runs found in {runs_dir}.")
        raise typer.Exit(code=0)

    run_files = run_files[:limit]

    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        table = Table(title=f"Red Team Runs ({runs_dir})", show_header=True, box=box.ROUNDED)
        table.add_column("Name", style="cyan")
        table.add_column("Date", style="white")
        table.add_column("Mode", style="white")
        table.add_column("Targets", style="white")
        table.add_column("Attacks", style="white", justify="right")
        table.add_column("ASR", style="white", justify="right")
        table.add_column("File", style="dim")

        skipped = 0
        for f in run_files:
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, OSError):
                skipped += 1
                continue

            run_name = data.get('run_name', f.stem)
            created = data.get('created_at', '')
            if isinstance(created, str) and len(created) >= 16:
                created = created[:16].replace('T', ' ')
            pipeline = data.get('pipeline', '?')
            agents = data.get('tested_agents', [])
            targets_str = ', '.join(agents) if agents else '?'
            summary = data.get('summary', {})
            total = summary.get('total_attacks', data.get('total_results', 0))
            asr = summary.get('vulnerability_rate', 0.0)
            asr_str = f"{asr:.0%}" if isinstance(asr, (int, float)) else '?'

            table.add_row(run_name, str(created), pipeline, targets_str, str(total), asr_str, f.name)

        console = Console()
        console.print(table)
        if skipped:
            console.print(f"[yellow]Warning: {skipped} file(s) could not be parsed and were skipped.[/yellow]")

    except ImportError:
        # Fallback without rich
        typer.echo(f"{'Name':<20} {'Date':<17} {'Mode':<8} {'Attacks':>7} {'ASR':>5}  File")
        typer.echo("-" * 80)
        skipped = 0
        for f in run_files:
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, OSError):
                skipped += 1
                continue
            run_name = data.get('run_name', f.stem)[:20]
            created = data.get('created_at', '')
            if isinstance(created, str) and len(created) >= 16:
                created = created[:16].replace('T', ' ')
            pipeline = data.get('pipeline', '?')
            summary = data.get('summary', {})
            total = summary.get('total_attacks', data.get('total_results', 0))
            asr = summary.get('vulnerability_rate', 0.0)
            asr_str = f"{asr:.0%}" if isinstance(asr, (int, float)) else '?'
            typer.echo(f"{run_name:<20} {str(created):<17} {pipeline:<8} {total:>7} {asr_str:>5}  {f.name}")
        if skipped:
            typer.echo(f"Warning: {skipped} file(s) could not be parsed and were skipped.", err=True)
