"""CLI for evaluatorq red teaming."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from evaluatorq.common import cli_width  # noqa: F401  — import for its non-TTY width side effect
from evaluatorq.common.cli_config import (
    JSON_OUTPUT_HELP,
    Flag,
    config_option_help,
    echo_schema,
    json_flag,
    model_kwargs,
    reserve_stdout_for_json,
    resolve_config,
)
from evaluatorq.common.cli_epilog import examples
from evaluatorq.common.cli_help import CONTEXT_SETTINGS, MODEL_OPTION_NOTE
from evaluatorq.common.cli_json import echo_json
from evaluatorq.common.cli_tty import shell_path, should_skip_confirm
from evaluatorq.common.llm_limit import DEFAULT_LLM_PARALLELISM, check_llm_parallelism_option
from evaluatorq.common.parallelism import DEFAULT_DATAPOINT_PARALLELISM
from evaluatorq.common.reports.html_helpers import pct
from evaluatorq.dashboard.library import _manifest_card_id, report_id
from evaluatorq.redteam.contracts import (
    DEFAULT_PIPELINE_MODEL,
    DeliveryMethod,
    EvaluatorConfig,
    LLMConfig,
    Pipeline,
    SaveMode,
    Vulnerability,
)
from evaluatorq.redteam.run_config import RedTeamCliConfig
from evaluatorq.redteam.runner import DEFAULT_MAX_TURNS

if TYPE_CHECKING:
    from evaluatorq.redteam.contracts import RedTeamReport

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
    '# every red_team() keyword argument from a JSON file, the report as JSON on stdout',
    'eq redteam run --config run.json --json > report.json',
)


def _dashboard_command(directory: Path) -> str:
    return f'eq dashboard {shell_path(directory)}'


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


_GENERIC_EVAL_HINT = (
    'Check evaluator model configuration, credentials, and any gateway guardrails that may be '
    'rejecting judge or target calls.'
)


def _evaluation_error_hint(report: RedTeamReport) -> str:
    """Name the dominant reason attacks could not be scored, with a sample.

    Covers both halves of "unscored": the judge failing (``evaluation_error``) and
    the target failing so there was nothing to judge (``error``). Restricting this
    to judge errors left a target outage — which fails the coverage gate just the
    same — with only the generic advice, which is not a diagnosis.

    The generic advice is a fallback, not the answer — when the results actually
    recorded why, say that instead, because "parse (47 attacks): <the guardrail's
    refusal text>" is diagnosable and "check three things" is not.
    """
    causes: list[tuple[str, str]] = []
    for r in report.results:
        if r.evaluation_error is not None:
            ev = r.evaluation_error
            causes.append((f'evaluation/{ev.code or ev.error_type or "unknown"}', ev.message or ''))
        elif r.vulnerable is None and r.error:
            causes.append((f'execution/{r.error_code or r.error_type or "unknown"}', r.error))
    if not causes:
        return _GENERIC_EVAL_HINT

    counts = Counter(code for code, _ in causes)
    top_code, top_n = counts.most_common(1)[0]
    sample = next((msg for code, msg in causes if code == top_code), '')
    detail = f': {sample[:200]}' if sample else ''
    return f'Dominant failure: {top_code} ({top_n}/{len(causes)} unscored attacks){detail}. {_GENERIC_EVAL_HINT}'


def _cli_default(field: str) -> Any:
    """`RedTeamCliConfig`'s default for ``field``, for a flag's help text."""
    return RedTeamCliConfig.model_fields[field].default


def _single_or_list(targets: list[str]) -> str | list[str]:
    return targets[0] if len(targets) == 1 else list(targets)


# Order matters: later rows win for a shared field, so the narrow LLM flags beat ``--llm-config`` and
# ``-q`` beats ``-v``. Each row names the `RedTeamCliConfig` field the click parameter sets.
_RUN_FLAGS: tuple[Flag, ...] = (
    Flag('target', 'target', to_value=_single_or_list),
    Flag('name', 'name'),
    Flag('mode', 'mode'),
    Flag('categories', 'categories'),
    Flag('vulnerabilities', 'vulnerabilities'),
    Flag('strategies', 'strategies'),
    Flag('delivery_methods', 'delivery_methods'),
    Flag('max_turns', 'max_turns'),
    Flag('max_per_category', 'max_per_category'),
    Flag('attacker_instructions', 'attacker_instructions'),
    Flag('datapoint_parallelism', 'datapoint_parallelism'),
    Flag('llm_parallelism', 'llm_parallelism'),
    Flag('generate_strategies', 'generate_strategies'),
    Flag('generated_strategy_count', 'generated_strategy_count'),
    Flag('max_dynamic_datapoints', 'max_dynamic_datapoints'),
    Flag('max_static_datapoints', 'max_static_datapoints'),
    Flag('cleanup_memory', 'cleanup_memory'),
    Flag('dataset', 'dataset'),
    Flag('from_run', 'previous_run'),
    Flag('artifacts_dir', 'artifacts_dir'),
    Flag('save', 'save'),
    Flag('executive_summary', 'generate_executive_summary'),
    Flag('recommendations', 'recommendations'),
    Flag('system_prompt', 'target_config.system_prompt'),
    json_flag('llm_config_json', 'llm_config', LLMConfig, flag='--llm-config'),
    Flag('attack_model', 'llm_config.attacker.model'),
    Flag('evaluator_model', 'llm_config.evaluator.model'),
    Flag('min_evaluation_coverage', 'llm_config.evaluator.min_evaluation_coverage'),
    Flag('target_timeout_ms', 'llm_config.target_agent_timeout_ms'),
    Flag('max_target_retries', 'llm_config.max_target_retries'),
    Flag('retry_count', 'llm_config.retry_count'),
    Flag('max_tool_continuations', 'llm_config.max_tool_continuations'),
    Flag('target_reasoning_effort', 'llm_config.target_reasoning_effort'),
    Flag('verbose', 'verbosity', to_value=lambda count: count + 1),
    Flag('quiet', 'verbosity', to_value=lambda _quiet: 0),
)


def _validate_run_config(cfg: RedTeamCliConfig) -> RedTeamCliConfig:
    """Flatten and validate the resolved config's data-selection fields.

    Requires a target. Raises ``typer.BadParameter`` on an unknown vulnerability ID or an unknown
    strategy name. An unknown delivery method is deliberately not fatal: it is
    kept as a literal string so a dataset's custom delivery method stays
    filterable, and a warning is echoed to stderr.
    """
    if not cfg.target:
        raise typer.BadParameter('provide --target, or set "target" in --config.', param_hint="'--target'")
    target = _single_or_list(cfg.target) if isinstance(cfg.target, list) else cfg.target

    # Allow comma-separated values within repeatable flags (-s a,b == -s a -s b).
    # Done before validation so the checks below see individual tokens.
    strategies = _split_csv(cfg.strategies)
    from evaluatorq.redteam.delivery_method_registry import (
        delivery_method_str,
        is_known_delivery_method,
        list_available_delivery_methods,
        resolve_delivery_methods,
    )

    delivery_tokens = _split_csv(
        [delivery_method_str(d) for d in cfg.delivery_methods] if cfg.delivery_methods else None
    )
    categories = _split_csv(cfg.categories)
    vulnerabilities = _split_csv(cfg.vulnerabilities)

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

    # Resolve delivery methods against the registry (enum plus registered), parsed
    # as plain strings to support comma-separated input which typer's enum
    # binding cannot do. Known values coerce to the DeliveryMethod object;
    # unknown ones are an open set — kept as raw strings (so a dataset's custom
    # delivery method is still filterable) with a warning, not a hard error.
    resolved_delivery_methods: list[DeliveryMethod | str] | None = None
    if delivery_tokens:
        unknown = [d for d in delivery_tokens if not is_known_delivery_method(d)]
        if unknown:
            # delivery_method_str, not str(): a member's str() is its repr on the
            # 3.10 StrEnum polyfill, which would print unusable 'DeliveryMethod.X'.
            known_repr = sorted(delivery_method_str(m) for m in list_available_delivery_methods())
            typer.echo(
                f'Warning: delivery method(s) {unknown} are not known delivery methods {known_repr}; '
                'filtering by them literally (they will only match a dataset row spelled exactly the same).',
                err=True,
            )
        resolved_delivery_methods = resolve_delivery_methods(list(delivery_tokens))

    return cfg.model_copy(
        update={
            'target': target,
            'strategies': strategies,
            'delivery_methods': resolved_delivery_methods,
            'categories': categories,
            'vulnerabilities': vulnerabilities,
            'llm_config': cfg.llm_config or LLMConfig(),
        }
    )


@app.command(no_args_is_help=True, epilog=_RUN_EPILOG)
def run(
    ctx: typer.Context,
    target: Annotated[
        list[str] | None,
        typer.Option(
            '--target',
            '-t',
            help='Target identifier(s), e.g. "agent:<key>" or "deployment:<key>" (deployments need --mode static). For OpenAI models use OpenAIModelTarget in the Python API. Repeatable. Required unless --config sets "target".',
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option('--name', '-n', help="Experiment name (defaults to 'red-team')."),
    ] = None,
    mode: Annotated[
        Pipeline | None,
        typer.Option(help='Execution mode. Defaults to dynamic. Cannot be combined with --from-run.'),
    ] = None,
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
        int | None,
        typer.Option(
            help=(
                'Maximum conversation turns for multi-turn attacks. '
                f"Defaults to {DEFAULT_MAX_TURNS}, or to the replayed run's turn budget with --from-run."
            )
        ),
    ] = None,
    max_per_category: Annotated[
        int | None,
        typer.Option(help='Cap strategies per category.'),
    ] = None,
    attack_model: Annotated[
        str | None,
        typer.Option(
            help=f'Model for adversarial prompt generation. Defaults to {DEFAULT_PIPELINE_MODEL}. {MODEL_OPTION_NOTE}'
        ),
    ] = None,
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
        str | None,
        typer.Option(
            help=f'Model for OWASP evaluation scoring. Defaults to {DEFAULT_PIPELINE_MODEL}. {MODEL_OPTION_NOTE}'
        ),
    ] = None,
    min_evaluation_coverage: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            max=1.0,
            help='Fraction of attacks that must produce a verdict, else exit non-zero. '
            'Pass 0 to warn instead of failing; a run where nothing could be scored '
            'still exits non-zero regardless. '
            f'Defaults to {EvaluatorConfig.model_fields["min_evaluation_coverage"].default}.',
        ),
    ] = None,
    target_timeout_ms: Annotated[
        int | None,
        typer.Option(
            '--target-timeout-ms',
            help='Per-call timeout (ms) for target invocations. '
            f'Defaults to {LLMConfig.model_fields["target_agent_timeout_ms"].default}.',
        ),
    ] = None,
    max_target_retries: Annotated[
        int | None,
        typer.Option(
            '--max-target-retries',
            min=0,
            max=10,
            help='Retries for a failed target transport call before abandoning its attacker turn. '
            f'Defaults to {LLMConfig.model_fields["max_target_retries"].default}.',
        ),
    ] = None,
    retry_count: Annotated[
        int | None,
        typer.Option(
            '--retry-count',
            min=0,
            max=10,
            help='Retries (after the initial call) for pipeline-owned LLM calls and ORQ '
            'context/enrichment/cleanup — distinct from --max-target-retries. '
            f'Defaults to {LLMConfig.model_fields["retry_count"].default}.',
        ),
    ] = None,
    max_tool_continuations: Annotated[
        int | None,
        typer.Option(
            '--max-tool-continuations',
            help='Max client-driven tool-result continuation rounds for ORQ agents that emit pending_tool_calls. '
            f'Defaults to {LLMConfig.model_fields["max_tool_continuations"].default}.',
        ),
    ] = None,
    target_reasoning_effort: Annotated[
        str | None,
        typer.Option(
            '--target-reasoning-effort',
            help='Reasoning effort for the target agent (Responses-capable targets only). '
            'Accepted values differ per model; an unsupported one is rejected by the provider.',
        ),
    ] = None,
    datapoint_parallelism: Annotated[
        int | None,
        typer.Option(
            '--datapoint-parallelism',
            '--parallelism',
            help='Maximum number of concurrent datapoints/jobs (tasks). --parallelism is a deprecated alias. '
            f'Defaults to {DEFAULT_DATAPOINT_PARALLELISM}.',
        ),
    ] = None,
    llm_parallelism: Annotated[
        int | None,
        typer.Option(
            '--llm-parallelism',
            callback=check_llm_parallelism_option,
            help=f'Ceiling on in-flight LLM requests for the whole run. Defaults to {DEFAULT_LLM_PARALLELISM}, '
            '-1 for no limit; '
            'size it against your provider concurrency limit.',
        ),
    ] = None,
    generated_strategy_count: Annotated[
        int | None,
        typer.Option(
            help='Number of strategies to generate per category. '
            f'Defaults to {_cli_default("generated_strategy_count")}.'
        ),
    ] = None,
    generate_strategies: Annotated[
        bool | None,
        typer.Option(
            '--generate-strategies/--no-generate-strategies',
            help=f'LLM-based strategy generation. Defaults to {_cli_default("generate_strategies")}.',
        ),
    ] = None,
    max_dynamic_datapoints: Annotated[
        int | None,
        typer.Option(help='Cap dynamic (generated) datapoints.'),
    ] = None,
    max_static_datapoints: Annotated[
        int | None,
        typer.Option(help='Cap static (dataset) datapoints.'),
    ] = None,
    cleanup_memory: Annotated[
        bool | None,
        typer.Option(
            '--cleanup-memory/--no-cleanup-memory',
            help=f'Clean up memory entities after dynamic runs. Defaults to {_cli_default("cleanup_memory")}.',
        ),
    ] = None,
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
        SaveMode | None,
        typer.Option(
            help="What to persist: 'none' (no files), 'final' (summary only), or 'detail' (all stage artifacts). "
            f'Defaults to {_cli_default("save").value}.'
        ),
    ] = None,
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
    executive_summary: Annotated[
        bool | None,
        typer.Option(
            '--executive-summary/--no-executive-summary',
            help='Generate an LLM narrative executive summary at the top of the report (needs LLM creds). '
            f'Defaults to {_cli_default("generate_executive_summary")}.',
        ),
    ] = None,
    recommendations: Annotated[
        bool | None,
        typer.Option(
            '--recommendations/--no-recommendations',
            help='Generate LLM remediation recommendations for the top focus areas (needs LLM creds). '
            f'Defaults to {_cli_default("recommendations")}.',
        ),
    ] = None,
    system_prompt: Annotated[
        str | None,
        typer.Option('--system-prompt', help='System prompt for the target model/agent.'),
    ] = None,
    config_source: Annotated[
        str | None,
        typer.Option(
            '--config',
            metavar='PATH|-',
            help=config_option_help('red_team()', 'eq redteam schema'),
        ),
    ] = None,
    llm_config_json: Annotated[
        str | None,
        typer.Option(
            '--llm-config',
            metavar='JSON',
            help=(
                'LLMConfig as a JSON object, e.g. \'{"attacker": {"temperature": 0.9}}\'. Merged field by field '
                'into "llm_config" from --config. --attack-model, --evaluator-model, --min-evaluation-coverage, '
                '--target-timeout-ms, --max-target-retries, --retry-count, --max-tool-continuations and '
                '--target-reasoning-effort win over it for their own field when passed.'
            ),
        ),
    ] = None,
    json_output: Annotated[  # noqa: FBT002
        bool,
        typer.Option('--json', help=JSON_OUTPUT_HELP),
    ] = False,
) -> None:
    """Run red teaming against one or more targets."""
    # Typer converts each value (str -> Path, str -> enum) only when it calls this function, so
    # ctx.params still holds click's raw strings; the arguments themselves are the converted values.
    cli_args = dict(locals())
    emit_json = reserve_stdout_for_json(ctx) if json_output else None
    cfg = resolve_config(ctx, RedTeamCliConfig, _RUN_FLAGS, cli_args, config_source=config_source)
    _configure_logging(cfg.verbosity - 1)
    cfg = _validate_run_config(cfg)
    targets: str = cfg.target if isinstance(cfg.target, str) else ', '.join(cfg.target or ())

    from evaluatorq.common.replay import ReplayError
    from evaluatorq.redteam import red_team
    from evaluatorq.redteam.exceptions import CancelledError, RedTeamError
    from evaluatorq.redteam.hooks import RichHooks

    kwargs = model_kwargs(cfg)
    kwargs['hooks'] = RichHooks(skip_confirm=should_skip_confirm(yes))

    try:
        report = asyncio.run(red_team(**kwargs))
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
        report_path.write_text(json.dumps(report.model_dump(mode='json'), indent=2, default=str), encoding='utf-8')
        typer.echo(f'Report saved to {report_path}')

    if report_md:
        md_path = write_markdown_report(report, output_dir=report_md, target=targets)
        typer.echo(f'Markdown report written to {md_path}')

    if report_html:
        html_path = write_html_report(report, output_dir=report_html, target=targets)
        typer.echo(f'HTML report written to {html_path}')

    if emit_json is not None:
        emit_json(report)

    # Coverage gates run last, after artifacts are written — those artifacts are exactly
    # what you need to diagnose the failure, so they must exist before we exit non-zero.
    # Neither case may exit 0: CI cannot read an unscored run as a clean bill of health.
    summary = report.summary
    if summary.no_verdict:
        typer.echo(
            f'Error: 0/{summary.total_attacks} attacks could be evaluated — the target was '
            f'not tested. {_evaluation_error_hint(report)}',
            err=True,
        )
        raise typer.Exit(code=1)

    if summary.coverage_below_minimum:
        typer.echo(
            f'Error: only {summary.evaluated_attacks}/{summary.total_attacks} attacks could be '
            f'evaluated ({pct(summary.evaluation_coverage)}), below the configured minimum of '
            f'{pct(summary.min_evaluation_coverage)}. The reported rates cover that subset only. '
            f'{_evaluation_error_hint(report)}',
            err=True,
        )
        raise typer.Exit(code=1)


@app.command()
def schema(
    input_schema: Annotated[  # noqa: FBT002
        bool,
        typer.Option(
            '--input/--output',
            help='--input (default): the shape `eq redteam run --config` accepts. '
            '--output: the RedTeamReport that `eq redteam run --json` prints.',
        ),
    ] = True,
) -> None:
    """Print the JSON schema of `eq redteam run`'s config file or of its JSON report."""
    from evaluatorq.redteam.contracts import RedTeamReport

    echo_schema(RedTeamCliConfig if input_schema else RedTeamReport, output=not input_schema)


def _import_hf_download() -> Any:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        typer.echo(
            'huggingface-hub not installed. Install with: uv add "evaluatorq[redteam]" '
            '(or: python -m pip install "evaluatorq[redteam]")',
            err=True,
        )
        raise typer.Exit(code=1)
    return hf_hub_download


def _download_and_read_hf_dataset(hf_hub_download: Any, repo: str, filename: str) -> Any:
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
        with open(local_path, encoding='utf-8') as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        typer.echo(f'Failed to read dataset file: {e}', err=True)
        raise typer.Exit(code=1)
    return raw


def _load_raw_dataset(dataset: str | None) -> Any:
    # Load raw JSON via the unified dataset loader's internal helpers
    from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import (
        DEFAULT_HF_FILENAME,
        DEFAULT_HF_REPO,
        _parse_hf_source,
    )

    if dataset is None:
        hf_hub_download = _import_hf_download()
        return _download_and_read_hf_dataset(hf_hub_download, DEFAULT_HF_REPO, DEFAULT_HF_FILENAME)
    if dataset.startswith('hf:'):
        hf_hub_download = _import_hf_download()
        repo, filename = _parse_hf_source(dataset.removeprefix('hf:'))
        return _download_and_read_hf_dataset(hf_hub_download, repo, filename)
    path = Path(dataset)
    typer.echo(f'Validating local file: {path}')
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def _collect_sample_errors(samples: Any) -> list[str]:
    from pydantic import ValidationError as _ValidationError

    from evaluatorq.redteam.contracts import RedTeamSample

    errors: list[str] = []
    for i, sample in enumerate(samples):
        try:
            RedTeamSample.model_validate(sample)
        except _ValidationError as e:  # noqa: PERF203
            for err in e.errors():
                loc = ' -> '.join(str(loc_part) for loc_part in err['loc'])
                errors.append(f'  sample[{i}].{loc}: {err["msg"]}')
    return errors


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
    raw = _load_raw_dataset(dataset)

    # Validate top-level shape
    if not isinstance(raw, dict) or 'samples' not in raw:
        typer.echo("FAIL: Expected top-level object with 'samples' key.", err=True)
        raise typer.Exit(code=1)

    samples = raw['samples']
    typer.echo(f'Found {len(samples)} samples.')

    errors = _collect_sample_errors(samples)

    if errors:
        typer.echo(f'\nFAIL: {len(errors)} validation error(s):', err=True)
        for line in errors[:20]:
            typer.echo(line, err=True)
        if len(errors) > 20:
            typer.echo(f'  ... and {len(errors) - 20} more', err=True)
        raise typer.Exit(code=1)

    typer.echo(f'OK: All {len(samples)} samples are valid.')


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
            # Which summary shape these stats came from; None on a legacy row
            # built by reading the report itself (see RUN_SUMMARY_VERSION).
            'summary_version': manifest.summary_version,
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
        'summary_version': None,  # read from the report, not from a stored summary
    }


def _fmt_created(created: Any) -> str:
    return created[:16].replace('T', ' ') if isinstance(created, str) and len(created) >= 16 else str(created or '')


def _fmt_asr(asr: Any) -> str:
    return f'{asr:.0%}' if isinstance(asr, (int, float)) else '—'


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
