"""Unified red teaming runner that dispatches to dynamic/static/hybrid pipelines."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import warnings
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

SaveMode = Literal['none', 'final', 'detail']
_SAVE_MODES: frozenset[str] = frozenset(get_args(SaveMode))

from loguru import logger

from evaluatorq import DataPoint, EvaluationResult, job
from evaluatorq.send_results import send_results_to_orq
from evaluatorq.tracing import capture_parent_context, init_tracing_if_needed
from evaluatorq.types import DataPointResult
from evaluatorq.redteam.adaptive.orchestrator import ProgressDisplay, _get_active_progress
from evaluatorq.redteam.adaptive.pipeline import (
    cleanup_memory_entities,
    create_dynamic_evaluator,
    create_dynamic_redteam_job,
    generate_dynamic_datapoints,
    generate_dynamic_datapoints_for_vulnerabilities,
)
from evaluatorq.redteam.adaptive.strategy_registry import (
    get_category_info,
    get_strategies_for_category,
    get_strategies_for_vulnerability,
    list_available_categories,
    select_applicable_strategies,
    select_applicable_strategies_for_vulnerability,
)
from evaluatorq.redteam.backends.base import (
    AgentTargetFactory,
    DefaultErrorMapper,
    DirectTargetFactory,
    ErrorMapper,
    MemoryCleanup,
    NoopMemoryCleanup,
    is_agent_target,
)
from evaluatorq.redteam.backends.registry import create_async_llm_client, resolve_backend
from evaluatorq.redteam.contracts import AgentContext, DEFAULT_PIPELINE_MODEL, PIPELINE_CONFIG, Pipeline, PipelineStage, RedTeamConfig, RedTeamReport, TargetConfig, TargetKind, Vulnerability, normalize_category
from evaluatorq.redteam.exceptions import CancelledError, CredentialError
from evaluatorq.redteam.hooks import ConfirmPayload, DefaultHooks, PipelineHooks
from evaluatorq.redteam.reports.recommendations import generate_focus_area_recommendations
from evaluatorq.redteam.runtime.jobs import _build_messages, create_model_job
from evaluatorq.redteam.tracing import set_span_attrs, with_redteam_span
from evaluatorq.redteam.vulnerability_registry import get_primary_category, resolve_vulnerabilities


def _save_stage(output_dir: Path | None, filename: str, content: str) -> None:
    """Write a stage artifact to *output_dir* when saving is enabled.

    Wraps the raw payload in an envelope with a ``saved_at`` ISO-8601
    timestamp so every artifact is self-describing.
    """
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(content)
    envelope = {
        "saved_at": datetime.now(tz=timezone.utc).isoformat(),
        "data": payload,
    }
    (output_dir / filename).write_text(json.dumps(envelope, indent=2, default=str), encoding="utf-8")
    logger.debug(f"Saved {filename} to {output_dir}")


def _save_report(output_dir: Path | None, filename: str, report: RedTeamReport) -> None:
    """Write a summary report to *output_dir* as flat JSON with ``saved_at``.

    Unlike :func:`_save_stage`, the report is written without an extra
    envelope layer — ``saved_at`` is injected directly into the top-level
    dict alongside the existing report fields.
    """
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    data = report.model_dump(mode="json")
    data["saved_at"] = datetime.now(tz=timezone.utc).isoformat()
    (output_dir / filename).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.debug(f"Saved {filename} to {output_dir}")


RUNS_DIR_NAME = Path('.evaluatorq') / 'runs'


def get_runs_dir() -> Path:
    """Return the runs directory resolved relative to the current working directory."""
    return Path.cwd() / RUNS_DIR_NAME


def _auto_save_run(report: RedTeamReport, name: str | None = None) -> Path | None:
    """Persist a report to ``.evaluatorq/runs/`` for later listing via ``runs`` CLI."""
    try:
        runs_dir = get_runs_dir()
        runs_dir.mkdir(parents=True, exist_ok=True)
        resolved_name = re.sub(r'[^a-zA-Z0-9_\-]', '-', name or 'red-team').strip('-') or 'red-team'
        ts = report.created_at.strftime('%Y%m%d_%H%M%S')
        filename = f'{resolved_name}_{ts}.json'
        path = runs_dir / filename
        data = report.model_dump(mode='json')
        data['run_name'] = resolved_name
        data['saved_at'] = datetime.now(tz=timezone.utc).isoformat()
        path.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
        logger.debug(f'Auto-saved run to {path}')
        return path
    except Exception:
        logger.warning('Failed to auto-save run report', exc_info=True)
        return None


async def _send_cleaned_results(
    results: list[Any],
    name: str,
    description: str,
    start_time: datetime,
) -> None:
    """Strip skipped job results and upload to the Orq platform.

    In multi-target mode each ``DataPointResult`` may contain job results
    from *all* target jobs, but only the matching target produces a real
    output — the rest return ``None``.  Before uploading we remove those
    empty job results so the experiment shows one clean row per datapoint
    without duplication.
    """
    api_key = os.environ.get("ORQ_API_KEY")
    if not api_key:
        logger.debug("Skipping result upload to Orq platform: ORQ_API_KEY not set")
        return

    cleaned: list[DataPointResult] = []
    for result in results:
        if not result.job_results:
            continue
        # Keep only job results with a non-None output
        real_jobs = [jr for jr in result.job_results if jr.output is not None]
        if not real_jobs:
            continue
        # Shallow-copy the result and replace job_results
        clean = copy.copy(result)
        clean.job_results = real_jobs
        cleaned.append(clean)

    if not cleaned:
        logger.debug("No cleaned results to send to Orq platform")
        return

    logger.debug(f"Sending {len(cleaned)} cleaned results to Orq platform (stripped from {len(results)} raw)")
    try:
        await send_results_to_orq(
            api_key=api_key,
            evaluation_name=name,
            evaluation_description=description,
            dataset_id=None,
            results=cleaned,
            start_time=start_time,
            end_time=datetime.now(tz=timezone.utc),
        )
    except Exception as e:
        logger.error(
            f'Failed to upload {len(cleaned)} results to Orq platform: {e}. '
            'Results have been saved locally.'
        )


def _datapoint_breakdown(datapoints: list[Any]) -> dict[str, int]:
    """Classify datapoints into static, template_dynamic, and generated_dynamic."""
    static = 0
    template_dynamic = 0
    generated_dynamic = 0
    for dp in datapoints:
        inputs = dp.inputs if hasattr(dp, "inputs") else dp
        source = inputs.get("hybrid_source", "")
        if source == "static":
            static += 1
        elif inputs.get("strategy", {}).get("is_generated", False):
            generated_dynamic += 1
        else:
            template_dynamic += 1
    return {
        "static": static,
        "template_dynamic": template_dynamic,
        "generated_dynamic": generated_dynamic,
    }

def _cap_datapoints_balanced(datapoints: list[Any], cap: int) -> list[Any]:
    """Cap datapoints using round-robin across vulnerabilities for balanced coverage.

    Instead of slicing the first N (which biases toward early vulnerabilities),
    this picks one datapoint per vulnerability in rotation until the cap is reached.
    """

    # Group by vulnerability
    by_vuln: dict[str, list[Any]] = defaultdict(list)
    vuln_order: list[str] = []
    for dp in datapoints:
        inputs = dp.inputs if hasattr(dp, 'inputs') else dp
        vuln = inputs.get('vulnerability', inputs.get('category', ''))
        if vuln not in by_vuln:
            vuln_order.append(vuln)
        by_vuln[vuln].append(dp)

    # Round-robin: pick one from each vulnerability per round
    result: list[Any] = []
    indices = {v: 0 for v in vuln_order}
    while len(result) < cap:
        added_this_round = False
        for vuln in vuln_order:
            if len(result) >= cap:
                break
            idx = indices[vuln]
            if idx < len(by_vuln[vuln]):
                result.append(by_vuln[vuln][idx])
                indices[vuln] = idx + 1
                added_this_round = True
        if not added_this_round:
            break

    return result


if TYPE_CHECKING:
    from collections.abc import Callable

    from openai import AsyncOpenAI

    from evaluatorq.redteam.backends.base import AgentTarget


# ---------------------------------------------------------------------------
# PreparedTarget dataclass
# ---------------------------------------------------------------------------

@dataclass
class PreparedTarget:
    """Typed container for all per-target state prepared before a run.

    Returned by :func:`_prepare_target` and consumed by
    :func:`_run_dynamic_or_hybrid`.
    """

    target: str
    target_kind: TargetKind
    target_value: str
    safe_target: str
    agent_context: AgentContext
    dynamic_datapoints: list[DataPoint]
    static_datapoints: list[DataPoint]  # empty list for dynamic mode
    all_datapoints: list[DataPoint]
    job: Callable[..., Any]  # the @job-decorated callable for this target
    dynamic_job: Callable[..., Any]  # raw inner dynamic job
    resolved_memory_cleanup: MemoryCleanup
    resolved_llm_client: AsyncOpenAI | None
    filtering_metadata: dict[str, Any]
    memory_entity_ids: list[str]  # runtime-accumulated entity IDs for cleanup


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def red_team(
    target: str | AgentTarget | list[str | AgentTarget],
    *,
    config: RedTeamConfig | None = None,
    mode: Pipeline | str = Pipeline.DYNAMIC,
    categories: list[str] | None = None,
    vulnerabilities: list[str] | None = None,
    max_turns: int = 5,
    max_per_category: int | None = None,
    attack_model: str = DEFAULT_PIPELINE_MODEL,
    evaluator_model: str = DEFAULT_PIPELINE_MODEL,
    parallelism: int = 10,
    generate_strategies: bool = True,
    generated_strategy_count: int = 2,
    max_dynamic_datapoints: int | None = None,
    max_static_datapoints: int | None = None,
    cleanup_memory: bool = True,
    backend: str = 'openai',
    target_factory: AgentTargetFactory | None = None,
    error_mapper: ErrorMapper | None = None,
    memory_cleanup: MemoryCleanup | None = None,
    llm_client: AsyncOpenAI | None = None,
    name: str | None = None,
    description: str | None = None,
    dataset: Path | str | None = None,
    hooks: PipelineHooks | None = None,
    output_dir: Path | str | None = None,
    target_config: TargetConfig | None = None,
    generate_recommendations: bool = False,
    attacker_instructions: str | None = None,
    verbosity: int = 0,
    llm_kwargs: dict[str, Any] | None = None,
    save: SaveMode = 'final',
) -> RedTeamReport:
    """Unified entry point for red teaming.

    Accepts a single target or a list of targets. When multiple targets are
    provided, each is run independently and the results are merged into a
    single report.

    Args:
        target: Target identifier(s). A single string like ``"agent:<key>"``
            or ``"llm:<model>"``, or a list of such strings for multi-target runs.
        mode: Execution mode — ``"dynamic"``, ``"static"``, or ``"hybrid"``.
        categories: OWASP categories to test (e.g., ``["ASI01", "ASI03"]``).
            Defaults to all available categories. Ignored if ``vulnerabilities`` is set.
        vulnerabilities: Vulnerability IDs to test (e.g., ``["goal_hijacking", "prompt_injection"]``).
            Takes precedence over ``categories``.
        max_turns: Maximum conversation turns for multi-turn attacks.
        max_per_category: Cap strategies per category (None = no cap).
        attack_model: Model for adversarial prompt generation.
        evaluator_model: Model for OWASP evaluation scoring.
        parallelism: Maximum concurrent evaluatorq jobs.
        generate_strategies: Whether to generate additional LLM-based strategies.
        generated_strategy_count: Number of strategies to generate per category.
        max_dynamic_datapoints: Cap dynamic (generated) datapoints (None = no cap).
        max_static_datapoints: Cap static (dataset) datapoints (None = no cap).
        cleanup_memory: Whether to clean up memory entities after dynamic runs.
        backend: Backend name (``"orq"`` or ``"openai"``).
        target_factory: Custom target factory (overrides backend default).
        error_mapper: Custom error mapper (overrides backend default).
        memory_cleanup: Custom memory cleanup (overrides backend default).
        llm_client: Pre-configured AsyncOpenAI client for attack/strategy generation.
        name: Optional experiment name for the run. Used as the evaluatorq experiment
            name and for the auto-saved run filename. Defaults to ``'red-team'``.
        description: Optional description for the report.
        dataset: Dataset source. Accepts: local file path, ``"hf:org/repo"``,
            ``"hf:org/repo/filename.json"``, or ``None`` for the default HuggingFace dataset.
        hooks: Optional ``PipelineHooks`` implementation. Defaults to
            ``DefaultHooks()`` (loguru output, auto-confirm).
        output_dir: Directory for saved JSON files. Ignored when
            ``save='none'``. For ``save='final'``, receives
            ``03_summary_report.json``; if ``None``, the summary is written
            to ``.evaluatorq/runs/<name>_<ts>.json`` instead so the run
            appears in ``evaluatorq redteam runs``. Required for
            ``save='detail'``, which writes ``01_all_datapoints.json``,
            ``02_attack_results.json``, and ``03_summary_report.json`` here.
        target_config: Optional backend-agnostic target configuration (e.g.
            system prompt for OpenAI targets).
        generate_recommendations: Whether to generate LLM-based actionable
            recommendations for the top focus areas by analyzing failed traces.
            Requires an LLM client (explicit or via environment credentials).
            Defaults to ``False``.
        attacker_instructions: Optional domain-specific context to steer attack
            generation (e.g. "this agent handles financial transactions, try to
            get it to approve fraudulent ones"). Appended to adversarial system
            prompts and objective generation prompts.
        verbosity: Verbosity level (0=silent, 1=summary progress bar,
            2=per-attack progress bars). Defaults to ``0``.
        save: What to persist to disk. ``'none'`` writes nothing. ``'final'``
            (default) writes only the summary report. ``'detail'`` writes all
            stage artifacts (datapoints, attack results, summary).

    Returns:
        RedTeamReport with results and summary statistics.

    Raises:
        ValueError: If mode is invalid, required arguments are missing, or
            ``save='detail'`` is passed without ``output_dir``.
        CancelledError: If hooks.on_confirm returns False.
    """
    if save not in _SAVE_MODES:
        msg = f"Invalid save mode {save!r}. Must be one of: {sorted(_SAVE_MODES)}."
        raise ValueError(msg)
    resolved_hooks: PipelineHooks = hooks or DefaultHooks()
    user_output_dir = Path(output_dir) if output_dir is not None else None
    # Inner pipelines (_run_dynamic, _run_static) only write 01/02/03 to their
    # output_dir parameter, so we gate it on save='detail'. For 'final' the
    # outer red_team() does the single summary write after inner returns.
    resolved_output_dir: Path | None = None
    if save == 'detail':
        if user_output_dir is None:
            msg = "save='detail' requires output_dir to be set."
            raise ValueError(msg)
        resolved_output_dir = user_output_dir

    if target_factory is not None:
        warnings.warn(
            'target_factory is deprecated. Pass an AgentTarget instance as the target parameter instead.',
            DeprecationWarning,
            stacklevel=2,
        )

    if isinstance(target, list):
        raw_targets: list[str | AgentTarget] = list(target)
    elif isinstance(target, str):
        raw_targets = [target]
    elif is_agent_target(target):
        raw_targets = [target]
    else:
        raise TypeError(f'Invalid target type: {type(target).__name__}. Expected str or AgentTarget.')

    if not raw_targets:
        msg = 'red_team() requires at least one target'
        raise ValueError(msg)

    # Separate string targets from AgentTarget objects
    string_targets: list[str] = []
    agent_targets: list[AgentTarget] = []
    for t in raw_targets:
        if isinstance(t, str):
            string_targets.append(t)
        elif is_agent_target(t):
            agent_targets.append(t)
        else:
            raise TypeError(f'Invalid target type: {type(t).__name__}. Expected str or AgentTarget.')

    # Deduplicate string targets (preserve order, warn on duplicates)
    seen_str: set[str] = set()
    deduped_str: list[str] = []
    for s in string_targets:
        if s in seen_str:
            logger.warning(f'Duplicate target {s!r} — ignoring repeated occurrence.')
        else:
            seen_str.add(s)
            deduped_str.append(s)
    targets = deduped_str  # existing code uses 'targets' as list[str]

    # Build or merge config -------------------------------------------------
    # When ``config`` is provided it is the source of truth for backend,
    # models, llm tuning, and llm_kwargs.  Individual params that were
    # *not* overridden by the caller still fall back to the config values.
    if config is None:
        config = RedTeamConfig()

    # Config fields are defaults — explicit caller params win.
    if attack_model == DEFAULT_PIPELINE_MODEL:
        attack_model = config.attack_model
    if evaluator_model == DEFAULT_PIPELINE_MODEL:
        evaluator_model = config.evaluator_model
    if not llm_kwargs and config.llm_kwargs:
        llm_kwargs = config.llm_kwargs

    # Resolve backend via config (handles auto-detection).
    # Only override the default 'openai' when config specifies a non-openai backend.
    backend = config.resolve_backend(targets) if backend == 'openai' and config.backend != 'openai' else backend

    # Auto-detect backend: agent:/deployment: targets require the orq backend.
    orq_prefixes = ('agent:', 'deployment:')
    has_orq_target = any(
        any(t.startswith(p) for p in orq_prefixes) or ':' not in t
        for t in string_targets
    )
    if has_orq_target and backend != 'orq':
        logger.debug(
            f'Auto-selected orq backend for agent/deployment target(s) (was {backend!r})',
        )
        backend = 'orq'

    resolved_mode = Pipeline(mode)

    # When using the orq router (no custom llm_client, no OPENAI_API_KEY),
    # model IDs need a provider prefix (e.g. "openai/gpt-5-mini").
    uses_orq_router = llm_client is None and config.uses_orq_router
    attack_model = config.resolve_model(attack_model, uses_orq_router=uses_orq_router)
    evaluator_model = config.resolve_model(evaluator_model, uses_orq_router=uses_orq_router)

    # Early credential validation — fail fast with a clear message
    if llm_client is None and not os.getenv('OPENAI_API_KEY') and not os.getenv('ORQ_API_KEY'):
        raise CredentialError(
            'Missing LLM credentials for attack/evaluation models. '
            'Set OPENAI_API_KEY for direct OpenAI access, or ORQ_API_KEY to use the ORQ router.'
        )

    resolved_vulns: list[Vulnerability] | None
    if vulnerabilities:
        resolved_vulns = resolve_vulnerabilities(vulnerabilities)
        resolved_categories = [get_primary_category(v) for v in resolved_vulns]
    elif categories:
        # Resolve category strings to vulnerabilities so the pipeline can use the
        # vulnerability-first path; fall back gracefully if any code is unknown.
        try:
            resolved_vulns = resolve_vulnerabilities(categories)
        except ValueError:
            resolved_vulns = None
        resolved_categories = categories
    else:
        resolved_categories = list_available_categories()
        try:
            resolved_vulns = resolve_vulnerabilities(resolved_categories)
        except ValueError:
            resolved_vulns = None

    if resolved_mode in (Pipeline.DYNAMIC, Pipeline.HYBRID):
        report = await _run_dynamic_or_hybrid(
            targets=targets,
            agent_targets=agent_targets,
            mode=resolved_mode,
            name=name,
            categories=resolved_categories,
            resolved_vulns=resolved_vulns,
            max_turns=max_turns,
            max_per_category=max_per_category,
            attack_model=attack_model,
            evaluator_model=evaluator_model,
            parallelism=parallelism,
            generate_strategies=generate_strategies,
            generated_strategy_count=generated_strategy_count,
            max_dynamic_datapoints=max_dynamic_datapoints,
            max_static_datapoints=max_static_datapoints,
            cleanup_memory=cleanup_memory,
            backend=backend,
            target_factory=target_factory,
            error_mapper=error_mapper,
            memory_cleanup=memory_cleanup,
            llm_client=llm_client,
            description=description,
            dataset=dataset,
            hooks=resolved_hooks,
            output_dir=resolved_output_dir,
            target_config=target_config,
            attacker_instructions=attacker_instructions,
            verbosity=verbosity,
            llm_kwargs=llm_kwargs,
        )
    elif resolved_mode == Pipeline.STATIC:
        if agent_targets:
            raise ValueError(
                'Static mode does not support direct AgentTarget objects. '
                'Use a string target (e.g., "agent:<key>") or switch to dynamic/hybrid mode.'
            )
        report = await _run_static(
            targets=targets,
            name=name,
            categories=resolved_categories,
            evaluator_model=evaluator_model,
            parallelism=parallelism,
            max_static_datapoints=max_static_datapoints,
            backend=backend,
            dataset=dataset,
            description=description,
            llm_client=llm_client,
            hooks=resolved_hooks,
            output_dir=resolved_output_dir,
            target_config=target_config,
            llm_kwargs=llm_kwargs,
        )
    else:
        msg = f'Invalid mode {mode!r}. Must be "dynamic", "static", or "hybrid".'
        raise ValueError(msg)

    # Generate LLM-based recommendations for focus areas (opt-in)
    if generate_recommendations:
        try:
            rec_client = llm_client
            if rec_client is None:
                rec_client = create_async_llm_client()

            report.focus_area_recommendations = await generate_focus_area_recommendations(
                report=report,
                llm_client=rec_client,
                model=evaluator_model or DEFAULT_PIPELINE_MODEL,
                llm_kwargs=llm_kwargs,
            )
        except (TypeError, AttributeError, ImportError, NameError, KeyError):
            raise
        except Exception:
            logger.warning('Failed to generate focus area recommendations', exc_info=True)
            report.pipeline_warnings.append(
                'Failed to generate focus area recommendations. Check LLM credentials and model configuration.'
            )

    # Persist report according to the save mode. 'detail' mode already had
    # the inner pipeline write 01/02/03 to resolved_output_dir, so here we
    # only handle 'final' (single summary file) and record the saved path.
    auto_save_path: Path | None = None
    if save == 'final':
        if user_output_dir is not None:
            _save_report(user_output_dir, '03_summary_report.json', report)
            auto_save_path = user_output_dir / '03_summary_report.json'
    elif save == 'detail':
        if resolved_output_dir is not None:
            auto_save_path = resolved_output_dir / '03_summary_report.json'

    # Index in .evaluatorq/runs/ so the run appears in `evaluatorq redteam runs`.
    if save != 'none':
        run_path = _auto_save_run(report, name=name)
        if auto_save_path is None:
            auto_save_path = run_path
        if run_path is None:
            report.pipeline_warnings.append(
                'Failed to auto-save run report. The run will not appear in `evaluatorq redteam runs`.'
            )

    resolved_hooks.on_complete(
        report,
        output_dir=str(user_output_dir) if user_output_dir and save != 'none' else None,
        auto_save_path=str(auto_save_path) if auto_save_path else None,
    )

    return report


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_target(target: str) -> tuple[TargetKind, str]:
    """Parse ``"kind:value"`` target string.

    Returns:
        Tuple of (TargetKind, value), e.g. (``TargetKind.AGENT``, ``"my-agent-key"``).
    """
    if ':' not in target:
        # Default to agent kind
        return TargetKind.AGENT, target
    kind, _, value = target.partition(':')
    if not value:
        msg = f'Target {target!r} is missing a value after the colon.'
        raise ValueError(msg)
    try:
        kind_enum = TargetKind(kind.lower())
    except ValueError:
        valid = ', '.join(f'"{k.value}"' for k in TargetKind if k is not TargetKind.DIRECT)
        msg = f'Unknown target kind {kind!r} in {target!r}. Valid kinds: {valid}.'
        raise ValueError(msg) from None
    if kind_enum is TargetKind.DIRECT:
        valid = ', '.join(f'"{k.value}"' for k in TargetKind if k is not TargetKind.DIRECT)
        msg = f'Target kind "direct" is not valid in string targets — pass an AgentTarget object directly instead. Valid string kinds: {valid}.'
        raise ValueError(msg) from None
    return kind_enum, value


def _safe_resolve_target_kind(at: Any) -> TargetKind:
    """Read ``target_kind`` from an AgentTarget, defaulting to DIRECT."""
    raw = getattr(at, 'target_kind', TargetKind.DIRECT)
    try:
        return TargetKind(raw)
    except ValueError:
        logger.warning(f'AgentTarget {type(at).__name__} has invalid target_kind={raw!r}; defaulting to DIRECT.')
        return TargetKind.DIRECT


def _make_safe_target(value: str) -> str:
    """Return a job-name-safe slug from a target value."""
    return ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '-' for ch in value).strip('-') or 'unknown'


def _deduplicate_target_labels(
    string_targets: Sequence[str],
    agent_targets: Sequence[Any],
) -> tuple[list[str], dict[int, str]]:
    """Build deduplicated labels for all targets.

    String targets keep their names as-is. Agent target labels are derived from
    ``.name`` (falling back to the class name) and suffixed with ``-1``, ``-2``
    etc. when duplicates are found.

    Args:
        string_targets (Sequence[str]): String-based target identifiers
            (e.g. ``"agent:my-bot"``).
        agent_targets (Sequence[AgentTarget]): Direct ``AgentTarget`` objects
            whose labels are derived from their ``.name`` attribute or class name.

    Returns:
        tuple[list[str], dict[int, str]]: A tuple of ``(all_labels, agent_label_map)``
            where ``agent_label_map`` maps ``id(agent_target)`` to its deduplicated label.
    """
    agent_label_map: dict[int, str] = {}
    seen: set[str] = set(string_targets)
    for at in agent_targets:
        label = getattr(at, 'name', None) or type(at).__name__
        if label in seen:
            suffix = 1
            while f'{label}-{suffix}' in seen:
                suffix += 1
            label = f'{label}-{suffix}'
        seen.add(label)
        agent_label_map[id(at)] = label

    all_labels = list(string_targets) + [agent_label_map[id(at)] for at in agent_targets]
    return all_labels, agent_label_map


def _create_job_for_target(
    target: str,
    llm_client: Any,
    system_prompt: str | None,
) -> Any:
    """Create a model job for the given target string.

    Dispatches on the target kind (``agent``, ``llm``, ``deployment``, or
    fallback to model) and returns the appropriate
    :func:`~evaluatorq.redteam.runtime.jobs.create_model_job` result.

    Uses ``PIPELINE_CONFIG.target_max_tokens`` for the response token limit.

    Args:
        target:        Full target string, e.g. ``"agent:my-key"`` or
                       ``"llm:openai/gpt-4o"``.
        llm_client:    Optional pre-configured :class:`openai.AsyncOpenAI`
                       client.
        system_prompt: Optional system prompt to pass to the job.

    Returns:
        A job callable as returned by ``create_model_job``.
    """
    kind, value = _parse_target(target)
    common = dict(llm_client=llm_client, system_prompt=system_prompt, max_tokens=PIPELINE_CONFIG.target_max_tokens)
    if kind == 'agent':
        return create_model_job(agent_key=value, **common)
    elif kind == 'deployment':
        return create_model_job(deployment_key=value, **common)
    elif kind in ('llm', 'openai'):
        has_orq = bool(os.environ.get('ORQ_API_KEY'))
        has_openai = bool(os.environ.get('OPENAI_API_KEY'))
        if has_orq:
            logger.info(f"Routing llm target '{value}' via Orq proxy (ORQ_API_KEY is set)")
        elif has_openai:
            logger.info(f"Routing llm target '{value}' via OpenAI directly (OPENAI_API_KEY is set)")
        else:
            logger.warning(
                f"No API key found for llm target '{value}'. "
                "Set ORQ_API_KEY (for Orq proxy) or OPENAI_API_KEY (for direct OpenAI)."
            )
        return create_model_job(model=value, **common)
    else:
        return create_model_job(model=value, **common)


# ---------------------------------------------------------------------------
# Per-target preparation (dynamic and hybrid)
# ---------------------------------------------------------------------------

async def _prepare_target(
    *,
    target: str,
    mode: Pipeline,
    categories: list[str] | None,
    resolved_vulns: list[Vulnerability] | None,
    max_turns: int,
    max_per_category: int | None,
    attack_model: str,
    parallelism: int,
    generate_strategies: bool,
    generated_strategy_count: int,
    max_dynamic_datapoints: int | None,
    max_static_datapoints: int | None,
    backend: str,
    target_factory: AgentTargetFactory | None,
    error_mapper: ErrorMapper | None,
    memory_cleanup: MemoryCleanup | None,
    llm_client: AsyncOpenAI | None,
    dataset: Any,
    hooks: PipelineHooks,
    output_dir: Path | None,
    target_config: TargetConfig | None,
    resolved_categories: list[str],
    shared_datapoints: list[Any] | None = None,
    attacker_instructions: str | None = None,
    prefetched_agent_context: AgentContext | None = None,
    prefetched_static_data: list[Any] | None = None,
    verbosity: int = 0,
    llm_kwargs: dict[str, Any] | None = None,
) -> PreparedTarget:
    """Prepare all per-target state for a dynamic or hybrid run.

    Always: parse target, resolve backend, retrieve agent context, and
    build a job closure for this target.

    When ``shared_datapoints`` is None (default): generate dynamic datapoints
    from this target's agent context (first-target behaviour).

    When ``shared_datapoints`` is provided: skip datapoint generation and
    reuse those datapoints directly (subsequent-target behaviour in
    multi-target runs).

    When ``mode == Pipeline.HYBRID`` and ``dataset`` is provided: also load the
    static dataset, create a static job, and build a hybrid dispatcher job
    that routes on ``hybrid_source``.

    When ``mode == 'dynamic'``: skip the static dataset and build a simpler
    dynamic job wrapper.

    Returns:
        A :class:`PreparedTarget` instance with all per-target state.
    """
    target_kind, target_value = _parse_target(target)
    safe_target = _make_safe_target(target_value)

    backend_bundle = resolve_backend(backend, llm_client=llm_client, target_config=target_config)
    resolved_factory = target_factory or backend_bundle.target_factory
    resolved_error_mapper = error_mapper or DefaultErrorMapper()
    resolved_memory_cleanup_t = memory_cleanup or backend_bundle.memory_cleanup

    # Context retrieval (skip if already fetched for the confirm step)
    if prefetched_agent_context is not None:
        agent_context = prefetched_agent_context
    else:
        hooks.on_stage_start(PipelineStage.CONTEXT_RETRIEVAL, {"target": target_value})
        agent_context = await backend_bundle.context_provider.get_agent_context(target_value)
        hooks.on_stage_end(PipelineStage.CONTEXT_RETRIEVAL, {
            "num_tools": len(agent_context.tools) if agent_context.tools else 0,
            "num_memory_stores": len(agent_context.memory_stores) if agent_context.memory_stores else 0,
            "num_knowledge_bases": len(agent_context.knowledge_bases) if agent_context.knowledge_bases else 0,
        })

    # LLM client
    resolved_llm_client = llm_client
    if resolved_llm_client is None and generate_strategies:
        resolved_llm_client = create_async_llm_client()

    if shared_datapoints is not None:
        # Reuse datapoints generated by the first target — skip generation (no banner)
        dynamic_datapoints = list(shared_datapoints)
        filtering_metadata: dict[str, Any] = {}
    else:
        # Datapoint generation
        hooks.on_stage_start(PipelineStage.DATAPOINT_GENERATION, {
            "num_categories": len(resolved_categories),
            "target": target,
        })

        if resolved_vulns is not None:
            # Primary vulnerability-first path
            dynamic_datapoints, filtering_metadata = await generate_dynamic_datapoints_for_vulnerabilities(
                agent_context=agent_context,
                vulnerabilities=resolved_vulns,
                max_per_category=max_per_category,
                max_turns=max_turns,
                generate_additional_strategies=generate_strategies,
                generated_strategy_count=generated_strategy_count,
                llm_client=resolved_llm_client,
                attack_model=attack_model,
                parallelism=parallelism,
                attacker_instructions=attacker_instructions,
                llm_kwargs=llm_kwargs,
            )
        else:
            # Fallback: categories that could not be resolved to vulnerabilities
            dynamic_datapoints, filtering_metadata = await generate_dynamic_datapoints(
                agent_context=agent_context,
                categories=resolved_categories,
                max_per_category=max_per_category,
                max_turns=max_turns,
                generate_additional_strategies=generate_strategies,
                generated_strategy_count=generated_strategy_count,
                llm_client=resolved_llm_client,
                attack_model=attack_model,
                parallelism=parallelism,
                attacker_instructions=attacker_instructions,
                llm_kwargs=llm_kwargs,
            )

        if max_dynamic_datapoints is not None and max_dynamic_datapoints > 0 and len(dynamic_datapoints) > max_dynamic_datapoints:
            total_before = len(dynamic_datapoints)
            dynamic_datapoints = _cap_datapoints_balanced(dynamic_datapoints, max_dynamic_datapoints)
            logger.info(
                f'[{target}] Capped dynamic datapoints from {total_before} to {len(dynamic_datapoints)}'
            )

    # Build the raw dynamic job for this target
    memory_entity_ids: list[str] = []
    dynamic_job = create_dynamic_redteam_job(
        agent_key=target_value,
        agent_context=agent_context,
        red_team_model=attack_model,
        max_turns=max_turns,
        target_factory=resolved_factory,
        error_mapper=resolved_error_mapper,
        attack_llm_client=resolved_llm_client,
        memory_entity_ids=memory_entity_ids,
        attacker_instructions=attacker_instructions,
        verbosity=verbosity,
        llm_kwargs=llm_kwargs,
    )

    # --- Mode-specific path ---------------------------------------------------

    if mode == Pipeline.HYBRID:
        if shared_datapoints is not None:
            # Shared datapoints already contain both dynamic and static (tagged).
            # Split them back out so PreparedTarget fields stay semantically correct.
            static_datapoints: list[Any] = [
                dp for dp in dynamic_datapoints
                if dp.inputs.get('hybrid_source') == 'static'
            ]
            dynamic_datapoints = [
                dp for dp in dynamic_datapoints
                if dp.inputs.get('hybrid_source') != 'static'
            ]
            all_datapoints = dynamic_datapoints + static_datapoints
        else:
            if prefetched_static_data is not None:
                static_datapoints = list(prefetched_static_data)
            else:
                from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import load_owasp_agentic_dataset
                # Load static datapoints
                static_datapoints = load_owasp_agentic_dataset(
                    dataset=dataset,
                    num_samples=max_static_datapoints,
                    categories=categories,
                )

            # Tag with hybrid_source only (no target_tag — datapoints are shared)
            for dp in dynamic_datapoints:
                dp.inputs['hybrid_source'] = 'dynamic'
            for dp in static_datapoints:
                dp.inputs['hybrid_source'] = 'static'

            all_datapoints = dynamic_datapoints + static_datapoints

        if shared_datapoints is None:
            hooks.on_stage_end(PipelineStage.DATAPOINT_GENERATION, {
                "num_datapoints": len(all_datapoints),
                "num_dynamic": len(dynamic_datapoints),
                "num_static": len(static_datapoints),
            })

        # Build the static job via shared helper
        sys_prompt = target_config.system_prompt if target_config else None
        static_job = _create_job_for_target(target, resolved_llm_client, sys_prompt)

        # Build the hybrid dispatcher job
        @job(f'redteam:hybrid:{safe_target}')
        async def target_job(
            data: DataPoint,
            row: int,
            _dyn: Any = dynamic_job,
            _sta: Any = static_job,
        ) -> Any:
            """Dispatch a hybrid datapoint to the dynamic or static inner job."""
            route = data.inputs.get('hybrid_source', 'static')
            inner = _dyn if route == 'dynamic' else _sta
            result = await inner(data, row)
            # Inner job is @job-decorated, so it returns {"name": ..., "output": ...}.
            # Unwrap to avoid double-wrapping since _target_job is also @job-decorated.
            return result.get('output', result) if isinstance(result, dict) else result

    else:
        # Dynamic mode — no target_tag tagging needed, datapoints are shared
        static_datapoints = []
        all_datapoints = dynamic_datapoints

        if shared_datapoints is None:
            hooks.on_stage_end(PipelineStage.DATAPOINT_GENERATION, {"num_datapoints": len(all_datapoints)})

        # Build the dynamic dispatcher job
        @job(f'redteam:dynamic:{safe_target}')
        async def target_job(
            data: DataPoint,
            row: int,
            _inner: Any = dynamic_job,
        ) -> Any:
            """Dispatch a dynamic datapoint to the inner job and unwrap the result."""
            result = await _inner(data, row)
            # Inner job is @job-decorated, so it returns {"name": ..., "output": ...}.
            # Unwrap to avoid double-wrapping since _target_job is also @job-decorated.
            return result.get('output', result) if isinstance(result, dict) else result

    return PreparedTarget(
        target=target,
        target_kind=target_kind,
        target_value=target_value,
        safe_target=safe_target,
        agent_context=agent_context,
        dynamic_datapoints=dynamic_datapoints,
        static_datapoints=static_datapoints,
        all_datapoints=all_datapoints,
        job=target_job,
        dynamic_job=dynamic_job,
        resolved_memory_cleanup=resolved_memory_cleanup_t,
        resolved_llm_client=resolved_llm_client,
        filtering_metadata=filtering_metadata,
        memory_entity_ids=memory_entity_ids,
    )


# ---------------------------------------------------------------------------
# Merged dynamic + hybrid runner
# ---------------------------------------------------------------------------

async def _run_dynamic_or_hybrid(
    *,
    targets: list[str],
    agent_targets: list[AgentTarget] | None = None,
    mode: Pipeline,
    name: str | None = None,
    categories: list[str] | None,
    resolved_vulns: list[Vulnerability] | None = None,
    max_turns: int,
    max_per_category: int | None,
    attack_model: str,
    evaluator_model: str,
    parallelism: int,
    generate_strategies: bool,
    generated_strategy_count: int,
    max_dynamic_datapoints: int | None,
    max_static_datapoints: int | None,
    cleanup_memory: bool,
    backend: str,
    target_factory: AgentTargetFactory | None,
    error_mapper: ErrorMapper | None,
    memory_cleanup: MemoryCleanup | None,
    llm_client: AsyncOpenAI | None,
    description: str | None,
    dataset: Any,
    hooks: PipelineHooks | None = None,
    output_dir: Path | None = None,
    target_config: TargetConfig | None = None,
    attacker_instructions: str | None = None,
    verbosity: int = 0,
    llm_kwargs: dict[str, Any] | None = None,
) -> RedTeamReport:
    """Run dynamic or hybrid red teaming for multiple targets in a single evaluatorq call.

    For each target, :func:`_prepare_target` retrieves agent context, generates
    agent-specific dynamic datapoints, and produces a job closure.  The first
    target generates shared datapoints; subsequent targets reuse them to avoid
    redundant LLM calls.  For hybrid mode an additional static dataset is loaded
    and a routing job is built.  All jobs are submitted in a single
    ``evaluatorq()`` call.

    After execution, results are split by ``job_name`` (matching each target's
    ``safe_target`` slug) and converted to per-target :class:`RedTeamReport`
    instances, which are merged into one unified report.

    Args:
        mode: ``"dynamic"`` or ``"hybrid"``.
    """
    from evaluatorq import evaluatorq
    from evaluatorq.redteam.reports.converters import (
        dynamic_evaluatorq_results_to_report,
        merge_reports,
        static_evaluatorq_results_to_reports,
        static_results_to_report,
    )

    resolved_name = name or 'red-team'

    resolved_hooks: PipelineHooks = hooks or DefaultHooks()
    pipeline_start = datetime.now(tz=timezone.utc).astimezone()

    resolved_categories = categories or list_available_categories()

    await init_tracing_if_needed()
    parent_context = await capture_parent_context()

    resolved_agent_targets = agent_targets or []
    all_target_labels, agent_target_labels = _deduplicate_target_labels(targets, resolved_agent_targets)
    async with with_redteam_span(
        "orq.redteam.pipeline",
        attributes={
            "orq.trace_type": "evaluatorq",
            "orq.redteam.targets": ", ".join(all_target_labels),
            "orq.redteam.mode": mode,
            "orq.redteam.backend": backend,
            "orq.redteam.max_turns": max_turns,
            "orq.redteam.parallelism": parallelism,
        },
        parent_context=parent_context,
    ) as pipeline_span:

        # Step 1: Retrieve agent context for all targets (cheap) so we
        # can show capabilities in the confirmation prompt before expensive generation.
        bundle = resolve_backend(backend, llm_client=llm_client, target_config=target_config)
        all_agent_contexts: dict[str, AgentContext] = {}
        resolved_hooks.on_stage_start(PipelineStage.CONTEXT_RETRIEVAL, {"targets": all_target_labels})
        for target_str in targets:
            kind, value = _parse_target(target_str)
            ctx = await bundle.context_provider.get_agent_context(value)
            all_agent_contexts[target_str] = ctx
            resolved_hooks.on_stage_end(PipelineStage.CONTEXT_RETRIEVAL, {
                "target": value,
                "num_tools": len(ctx.tools) if ctx.tools else 0,
                "num_memory_stores": len(ctx.memory_stores) if ctx.memory_stores else 0,
                "num_knowledge_bases": len(ctx.knowledge_bases) if ctx.knowledge_bases else 0,
            })
        # Pre-fetch contexts for AgentTarget objects (they may provide their own context)
        at_contexts: dict[int, AgentContext] = {}
        for at in resolved_agent_targets:
            get_ctx = getattr(at, 'get_agent_context', None)
            at_deduped_label = agent_target_labels[id(at)]
            if callable(get_ctx):
                try:
                    at_ctx = await cast("Any", get_ctx())
                except Exception as exc:
                    raise RuntimeError(
                        f'Failed to retrieve agent context from {type(at).__name__}.get_agent_context(): {exc}. '
                        f'Ensure the target implements get_agent_context() correctly.'
                    ) from exc
                if not isinstance(at_ctx, AgentContext):
                    raise TypeError(
                        f'{type(at).__name__}.get_agent_context() returned {type(at_ctx).__name__}, '
                        f'expected AgentContext.'
                    )
            else:
                logger.warning(f'AgentTarget {at_deduped_label!r} does not implement get_agent_context(); using minimal context.')
                at_ctx = AgentContext(key=at_deduped_label)
            at_contexts[id(at)] = at_ctx

        if targets:
            first_agent_context = all_agent_contexts[targets[0]]
        elif resolved_agent_targets:
            first_agent_context = at_contexts[id(resolved_agent_targets[0])]
        else:
            msg = 'red_team() requires at least one target'
            raise ValueError(msg)

        # Step 2: Estimate datapoint counts (cheap registry lookups, no LLM).
        est_dynamic = 0
        strategy_breakdown: dict[str, Any] = {}
        if resolved_vulns is not None:
            for vuln in resolved_vulns:
                all_strategies = get_strategies_for_vulnerability(vuln)
                applicable = select_applicable_strategies_for_vulnerability(
                    vuln, first_agent_context, agent_capabilities=None,
                )
                n_generated = generated_strategy_count if generate_strategies else 0
                n = len(applicable) + n_generated
                if max_per_category is not None:
                    n = min(n, max_per_category)
                est_dynamic += n
                strategy_breakdown[vuln.value] = {
                    'total_hardcoded': len(all_strategies),
                    'applicable': len(applicable),
                    'filtered': len(all_strategies) - len(applicable),
                    'generated': n_generated,
                    'selected': n,
                }
        else:
            for cat in resolved_categories:
                all_strategies = get_strategies_for_category(cat)
                applicable = select_applicable_strategies(
                    cat, first_agent_context, agent_capabilities=None,
                )
                n_generated = generated_strategy_count if generate_strategies else 0
                n = len(applicable) + n_generated
                if max_per_category is not None:
                    n = min(n, max_per_category)
                est_dynamic += n
                strategy_breakdown[cat] = {
                    'total_hardcoded': len(all_strategies),
                    'applicable': len(applicable),
                    'filtered': len(all_strategies) - len(applicable),
                    'generated': n_generated,
                    'selected': n,
                }
        if max_dynamic_datapoints is not None and est_dynamic > max_dynamic_datapoints:
            # Simulate round-robin allocation to show per-category capped counts
            categories_ordered = list(strategy_breakdown.keys())
            remaining = {k: v['selected'] for k, v in strategy_breakdown.items()}
            allocated = {k: 0 for k in categories_ordered}
            budget = max_dynamic_datapoints
            while budget > 0:
                added = False
                for cat in categories_ordered:
                    if budget <= 0:
                        break
                    if allocated[cat] < remaining[cat]:
                        allocated[cat] += 1
                        budget -= 1
                        added = True
                if not added:
                    break
            for cat in categories_ordered:
                strategy_breakdown[cat]['capped'] = allocated[cat]
            est_dynamic = max_dynamic_datapoints

        est_static: int | None = None
        static_data: list[Any] | None = None
        if mode == Pipeline.HYBRID:
            from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import load_owasp_agentic_dataset
            static_data = load_owasp_agentic_dataset(
                dataset=dataset,
                num_samples=max_static_datapoints,
                categories=categories,
            )
            est_static = len(static_data)

        est_total = est_dynamic + (est_static or 0)

        # Confirm before expensive datapoint generation.
        at_contexts_by_label = {
            agent_target_labels[id(at)]: at_contexts[id(at)]
            for at in resolved_agent_targets
        }
        all_contexts_for_confirm = {**all_agent_contexts, **at_contexts_by_label}
        confirm_payload: ConfirmPayload = {
            "agent_contexts": {
                t: ctx.model_dump(mode="json") for t, ctx in all_contexts_for_confirm.items()
            },
            "agent_context": first_agent_context.model_dump(mode="json"),
            "num_datapoints": est_total,
            "num_dynamic": est_dynamic,
            "num_static": est_static if mode == Pipeline.HYBRID else None,
            "categories": resolved_categories,
            "attack_model": attack_model,
            "evaluator_model": evaluator_model,
            "max_turns": max_turns,
            "parallelism": parallelism,
            "filtering_metadata": None,
            "strategy_breakdown": strategy_breakdown if strategy_breakdown else None,
            "mode": str(mode.value) if hasattr(mode, 'value') else str(mode),
            "target": ", ".join(all_target_labels),
            "dataset_path": str(dataset) if dataset else None,
            "vulnerabilities": [v.value for v in resolved_vulns] if resolved_vulns else None,
        }

        if not resolved_hooks.on_confirm(confirm_payload):
            msg = 'Execution cancelled by confirmation callback'
            raise CancelledError(msg)

        # Step 3: Prepare the first target fully — generates shared datapoints.
        common_prepare_kwargs: dict[str, Any] = dict(
            mode=mode,
            categories=categories,
            resolved_vulns=resolved_vulns,
            max_turns=max_turns,
            max_per_category=max_per_category,
            attack_model=attack_model,
            parallelism=parallelism,
            generate_strategies=generate_strategies,
            generated_strategy_count=generated_strategy_count,
            max_dynamic_datapoints=max_dynamic_datapoints,
            max_static_datapoints=max_static_datapoints,
            backend=backend,
            target_factory=target_factory,
            error_mapper=error_mapper,
            memory_cleanup=memory_cleanup,
            llm_client=llm_client,
            dataset=dataset,
            hooks=resolved_hooks,
            output_dir=output_dir,
            target_config=target_config,
            resolved_categories=resolved_categories,
            attacker_instructions=attacker_instructions,
            verbosity=verbosity,
            llm_kwargs=llm_kwargs or {},
        )

        prepared_targets: list[PreparedTarget]
        first_target: PreparedTarget | None = None

        if targets:
            first_target = await _prepare_target(
                target=targets[0],
                prefetched_agent_context=first_agent_context,
                prefetched_static_data=static_data,
                **common_prepare_kwargs,
            )

            # Step 4: Prepare remaining string targets with shared datapoints — skip generation.
            if len(targets) > 1:
                raw_results = await asyncio.gather(
                    *[
                        _prepare_target(
                            target=t,
                            shared_datapoints=first_target.all_datapoints,
                            prefetched_agent_context=all_agent_contexts.get(t),
                            **common_prepare_kwargs,
                        )
                        for t in targets[1:]
                    ],
                    return_exceptions=True,
                )
                failed_targets: list[str] = []
                for t, result in zip(targets[1:], raw_results):
                    if isinstance(result, BaseException):
                        logger.error(f"Failed to prepare target {t}: {result}")
                        failed_targets.append(f"{t}: {result}")
                if failed_targets:
                    failure_summary = "; ".join(failed_targets)
                    msg = f"Aborting multi-target run — failed to prepare target(s): {failure_summary}"
                    raise RuntimeError(msg)
                other_prepared: list[PreparedTarget] = [r for r in raw_results if not isinstance(r, BaseException)]
                prepared_targets = [first_target] + other_prepared
            else:
                prepared_targets = [first_target]
        else:
            # Pure AgentTarget run — no string targets at all.
            # We still need to generate datapoints; use the first AgentTarget's context.
            prepared_targets = []

        # Track memory entity info for all AgentTargets so cleanup can
        # reach them even if prepared_targets.append hasn't run yet.
        all_at_cleanup_info: list[tuple[AgentContext, list[str], Any]] = []

        # Step 4b: Prepare AgentTarget objects (direct targets)
        if resolved_agent_targets:
            at_llm_client = llm_client
            if at_llm_client is None:
                at_llm_client = create_async_llm_client()

            # If no string targets prepared yet, generate datapoints from first AgentTarget's context
            shared_at_dps: list[Any] | None = prepared_targets[0].all_datapoints if prepared_targets else None

            for at in resolved_agent_targets:
                at_label = agent_target_labels[id(at)]
                at_ctx = at_contexts[id(at)]

                # For direct targets, always use DirectTargetFactory (which uses clone()).
                # The target's create_target() expects an external agent_key string,
                # which doesn't apply when the key is embedded in the object.
                at_factory = DirectTargetFactory(at)
                at_mapper = at if callable(getattr(at, 'map_error', None)) else DefaultErrorMapper()
                at_cleanup = at if callable(getattr(at, 'cleanup_memory', None)) else NoopMemoryCleanup()

                at_mem_ids: list[str] = []
                all_at_cleanup_info.append((at_ctx, at_mem_ids, at_cleanup))
                at_dyn_job = create_dynamic_redteam_job(
                    agent_key=at_label,
                    agent_context=at_ctx,
                    red_team_model=attack_model,
                    max_turns=max_turns,
                    target_factory=cast(AgentTargetFactory, at_factory),
                    error_mapper=cast(ErrorMapper, at_mapper),
                    attack_llm_client=at_llm_client,
                    memory_entity_ids=at_mem_ids,
                    attacker_instructions=attacker_instructions,
                    verbosity=verbosity,
                    llm_kwargs=llm_kwargs,
                )

                at_safe = _make_safe_target(at_label)

                if shared_at_dps is None:
                    # This is the first target overall — generate datapoints
                    resolved_hooks.on_stage_start(PipelineStage.DATAPOINT_GENERATION, {
                        "num_categories": len(resolved_categories),
                        "target": at_label,
                    })
                    if resolved_vulns is not None:
                        at_dps, at_filter_meta = await generate_dynamic_datapoints_for_vulnerabilities(
                            agent_context=at_ctx,
                            vulnerabilities=resolved_vulns,
                            max_per_category=max_per_category,
                            max_turns=max_turns,
                            generate_additional_strategies=generate_strategies,
                            generated_strategy_count=generated_strategy_count,
                            llm_client=at_llm_client,
                            attack_model=attack_model,
                            parallelism=parallelism,
                            attacker_instructions=attacker_instructions,
                            llm_kwargs=llm_kwargs,
                        )
                    else:
                        at_dps, at_filter_meta = await generate_dynamic_datapoints(
                            agent_context=at_ctx,
                            categories=resolved_categories,
                            max_per_category=max_per_category,
                            max_turns=max_turns,
                            generate_additional_strategies=generate_strategies,
                            generated_strategy_count=generated_strategy_count,
                            llm_client=at_llm_client,
                            attack_model=attack_model,
                            parallelism=parallelism,
                            attacker_instructions=attacker_instructions,
                            llm_kwargs=llm_kwargs,
                        )
                    if max_dynamic_datapoints is not None and max_dynamic_datapoints > 0 and len(at_dps) > max_dynamic_datapoints:
                        at_dps = _cap_datapoints_balanced(at_dps, max_dynamic_datapoints)
                    resolved_hooks.on_stage_end(PipelineStage.DATAPOINT_GENERATION, {"num_datapoints": len(at_dps)})
                    shared_at_dps = at_dps
                else:
                    at_dps = list(shared_at_dps)
                    at_filter_meta = {}

                # Build the appropriate job based on mode (hybrid vs dynamic-only)
                at_static_dps: list[Any] = []
                if mode == Pipeline.HYBRID and static_data is not None:
                    at_static_dps = list(static_data)
                    # Tag datapoints with hybrid_source
                    for dp in at_dps:
                        dp.inputs['hybrid_source'] = 'dynamic'
                    for dp in at_static_dps:
                        dp.inputs['hybrid_source'] = 'static'

                    # Build a static job that invokes the AgentTarget directly
                    # Reuse the same DirectTargetFactory created for the dynamic job
                    @job(f'redteam:static:{at_safe}')
                    async def at_static_job(
                        data: DataPoint,
                        _row: int,
                        _factory: Any = at_factory,
                        _label: str = at_label,
                    ) -> Any:
                        """Send a static datapoint to the AgentTarget via send_prompt."""
                        messages = _build_messages(data)
                        prompt = '\n'.join(
                            content for m in messages
                            if m.get('role') == 'user' and (content := m.get('content'))
                        )
                        if not prompt:
                            sample_id = data.inputs.get('id', 'unknown')
                            raise ValueError(
                                f'Static datapoint {sample_id!r} for target {_label!r} '
                                f'produced an empty prompt ({len(messages)} messages, none with user content).'
                            )
                        target_instance = _factory.create_target(_label)
                        response = await target_instance.send_prompt(prompt)
                        _active_progress = _get_active_progress()
                        if _active_progress is not None:
                            await _active_progress.finish_attack(None)
                        return {'response': response}

                    @job(f'redteam:hybrid:{at_safe}')
                    async def at_target_job(
                        data: DataPoint,
                        row: int,
                        _dyn: Any = at_dyn_job,
                        _sta: Any = at_static_job,
                    ) -> Any:
                        route = data.inputs.get('hybrid_source', 'static')
                        inner = _dyn if route == 'dynamic' else _sta
                        result = await inner(data, row)
                        return result.get('output', result) if isinstance(result, dict) else result

                    at_all_dps = list(at_dps) + at_static_dps
                else:
                    at_static_dps = []
                    at_all_dps = list(at_dps)

                    @job(f'redteam:dynamic:{at_safe}')
                    async def at_target_job(
                        data: DataPoint,
                        row: int,
                        _inner: Any = at_dyn_job,
                    ) -> Any:
                        result = await _inner(data, row)
                        return result.get('output', result) if isinstance(result, dict) else result

                prepared_targets.append(PreparedTarget(
                    target=at_label,
                    target_kind=_safe_resolve_target_kind(at),
                    target_value=at_label,
                    safe_target=at_safe,
                    agent_context=at_ctx,
                    dynamic_datapoints=list(at_dps),
                    static_datapoints=at_static_dps,
                    all_datapoints=at_all_dps,
                    job=at_target_job,
                    dynamic_job=at_dyn_job,
                    resolved_memory_cleanup=cast(MemoryCleanup, at_cleanup),
                    resolved_llm_client=at_llm_client,
                    filtering_metadata=at_filter_meta,
                    memory_entity_ids=at_mem_ids,
                ))

        # Step 5: Use the first target's datapoints as THE shared datapoints.
        all_datapoints: list[Any] = prepared_targets[0].all_datapoints if prepared_targets else []
        all_jobs: list[Any] = [pt.job for pt in prepared_targets]

        if not all_datapoints:
            msg = f'No datapoints generated for any target in {mode} multi-target mode.'
            raise ValueError(msg)

        _save_stage(
            output_dir,
            "01_all_datapoints.json",
            json.dumps([dp.inputs for dp in all_datapoints], indent=2, default=str),
        )

        # Stage: attack_execution
        resolved_hooks.on_stage_start(PipelineStage.ATTACK_EXECUTION, {
            "num_datapoints": len(all_datapoints),
            "targets": all_target_labels,
        })

        resolved_llm_client = llm_client

        # Build evaluator — hybrid routes on hybrid_source; dynamic uses the
        # dynamic evaluator directly.
        has_static = any(pt.static_datapoints for pt in prepared_targets)
        if mode == Pipeline.HYBRID and has_static:
            from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import create_owasp_evaluator
            dynamic_evaluator = create_dynamic_evaluator(evaluator_model=evaluator_model, llm_client=resolved_llm_client)
            static_evaluator = create_owasp_evaluator(evaluator_model=evaluator_model, llm_client=resolved_llm_client)

            async def hybrid_scorer(params: Any) -> EvaluationResult:
                """Route evaluation to the dynamic or static OWASP scorer based on datapoint source."""
                data = params['data']
                route = data.inputs.get('hybrid_source', 'static')
                if route == 'dynamic':
                    raw_output = params.get('output')
                    if isinstance(raw_output, str):
                        try:
                            raw_output = json.loads(raw_output)
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f'Failed to deserialize dynamic output as JSON: {e}')
                    dynamic_params = dict(params, output=raw_output)
                    return await dynamic_evaluator['scorer'](dynamic_params)
                return await static_evaluator['scorer'](params)

            evaluators: list[Any] = [{'name': 'hybrid-owasp-security', 'scorer': hybrid_scorer}]
            first = first_target if first_target is not None else prepared_targets[0]
            log_label = (
                f'{len(first.dynamic_datapoints)} dynamic + '
                f'{len(first.static_datapoints)} static datapoints'
            )
        else:
            evaluator = create_dynamic_evaluator(evaluator_model=evaluator_model, llm_client=resolved_llm_client)
            evaluators = [evaluator]
            log_label = f'{len(all_datapoints)} datapoints'

        async with ProgressDisplay(est_total * len(prepared_targets), verbosity):
            try:
                results = await evaluatorq(
                    resolved_name,
                    data=all_datapoints,
                    jobs=all_jobs,
                    evaluators=evaluators,
                    parallelism=parallelism,
                    print_results=False,
                    _exit_on_failure=False,
                    _send_results=False,
                    _trace_type="evaluatorq",
                    description=description or f'{mode.capitalize()} red teaming ({len(all_target_labels)} targets)',
                )
            except (asyncio.CancelledError, KeyboardInterrupt):
                logger.warning(f'Multi-target {mode} run cancelled — attempting memory cleanup')
                if cleanup_memory:
                    for pt in prepared_targets:
                        entity_ids = pt.memory_entity_ids
                        if entity_ids:
                            await cleanup_memory_entities(
                                pt.agent_context, entity_ids, memory_cleanup=pt.resolved_memory_cleanup
                            )
                    # Also clean up AgentTarget memory entities not yet in prepared_targets
                    prepared_mem_id_lists = {id(pt.memory_entity_ids) for pt in prepared_targets}
                    for at_ctx_c, at_mem_c, at_cleanup_c in all_at_cleanup_info:
                        if id(at_mem_c) not in prepared_mem_id_lists and at_mem_c:
                            await cleanup_memory_entities(at_ctx_c, at_mem_c, memory_cleanup=at_cleanup_c)
                raise

        resolved_hooks.on_stage_end(PipelineStage.ATTACK_EXECUTION, {"num_results": len(results)})

        _save_stage(
            output_dir,
            "02_attack_results.json",
            json.dumps([r.model_dump(mode='json') for r in results], indent=2, default=str),
        )

        pipeline_duration = (datetime.now(tz=timezone.utc).astimezone() - pipeline_start).total_seconds()

        # Stage: report_generation — split by job_name, convert, merge
        resolved_hooks.on_stage_start(PipelineStage.REPORT_GENERATION, {"num_results": len(results)})

        # Group raw evaluatorq results by target safe_target slug.
        # Build a direct job_name → safe_target lookup to avoid substring matching.
        job_name_to_safe: dict[str, str] = {}
        for pt in prepared_targets:
            for prefix in ('redteam:dynamic:', 'redteam:hybrid:', 'redteam:static:'):
                job_name_to_safe[f'{prefix}{pt.safe_target}'] = pt.safe_target

        # With shared datapoints each DataPointResult has job_results from ALL targets,
        # so we create synthetic per-target copies containing only that target's job result.
        results_by_target: dict[str, list[Any]] = {}
        for result in results:
            if not result.job_results:
                # No job results — assign to all targets so reports can show the gap
                for pt in prepared_targets:
                    results_by_target.setdefault(pt.safe_target, []).append(result)
                continue
            for jr in result.job_results:
                matched_safe = job_name_to_safe.get(jr.job_name or '')
                if matched_safe is not None:
                    target_result = copy.copy(result)
                    target_result.job_results = [jr]
                    results_by_target.setdefault(matched_safe, []).append(target_result)
                else:
                    logger.warning(f'Job result with name {jr.job_name!r} did not match any target — excluded from reports')

        per_target_reports: list[RedTeamReport] = []
        for pt in prepared_targets:
            safe = pt.safe_target
            target_results = results_by_target.get(safe, [])

            if mode == Pipeline.HYBRID and has_static:
                # Split by hybrid_source and convert separately, then merge
                dynamic_results = [
                    r for r in target_results
                    if getattr(r, 'data_point', None) is not None
                    and r.data_point.inputs.get('hybrid_source') == 'dynamic'
                ]
                static_results_for_target = [
                    r for r in target_results
                    if getattr(r, 'data_point', None) is not None
                    and r.data_point.inputs.get('hybrid_source') != 'dynamic'
                ]

                target_sub_reports: list[RedTeamReport] = []

                if dynamic_results:
                    dyn_report = dynamic_evaluatorq_results_to_report(
                        agent_context=pt.agent_context,
                        results=dynamic_results,
                        duration_seconds=pipeline_duration,
                        description=f'{description or "Hybrid"} ({pt.target}) (dynamic)',
                    )
                    target_sub_reports.append(dyn_report)

                if static_results_for_target:
                    static_reports = static_evaluatorq_results_to_reports(
                        results=static_results_for_target,
                        agent_model=pt.target_value if pt.target_kind.is_model else None,
                        agent_key=pt.target_value if not pt.target_kind.is_model else None,
                        description=f'{description or "Hybrid"} ({pt.target}) (static)',
                    )
                    target_sub_reports.extend(static_reports.values())

                if target_sub_reports:
                    t_report = merge_reports(
                        *target_sub_reports,
                        description=f'{description or "Hybrid red teaming"} ({pt.target})',
                    )
                else:
                    t_report = static_results_to_report(
                        [],
                        description=f'{description or "Hybrid red teaming"} ({pt.target})',
                    )

                t_report.summary.datapoint_breakdown = _datapoint_breakdown(pt.all_datapoints)

            else:
                # Dynamic mode: convert all target results as dynamic
                if target_results:
                    t_report = dynamic_evaluatorq_results_to_report(
                        agent_context=pt.agent_context,
                        results=target_results,
                        duration_seconds=pipeline_duration,
                        description=f'{description or "Dynamic red teaming"} ({pt.target})',
                    )
                else:
                    t_report = static_results_to_report(
                        [],
                        description=f'{description or "Dynamic red teaming"} ({pt.target})',
                    )

            per_target_reports.append(t_report)

        if not per_target_reports:
            merged = static_results_to_report(
                [],
                description=description or f'{mode.capitalize()} red teaming ({len(all_target_labels)} targets)',
            )
        else:
            merged = merge_reports(
                *per_target_reports,
                description=description or f'{mode.capitalize()} red teaming ({len(all_target_labels)} targets)',
            )

        merged.duration_seconds = pipeline_duration
        merged.agent_contexts = {pt.target_value: pt.agent_context for pt in prepared_targets}
        # Canonical tested_agents: use the same keys as agent_contexts so
        # dashboard/report lookups resolve correctly.
        merged.tested_agents = [pt.target_value for pt in prepared_targets]
        if mode == Pipeline.HYBRID:
            merged.summary.datapoint_breakdown = _datapoint_breakdown(all_datapoints)

        if len(all_datapoints) == 0:
            merged.pipeline_warnings.append(
                'Zero datapoints generated for requested vulnerabilities. '
                'Strategy generation may have failed — check logs for details.'
            )
        elif merged.summary.total_attacks == 0:
            merged.pipeline_warnings.append(
                'Zero attacks executed. The resistance rate of 100% does not reflect actual security posture — '
                'no attacks were run. Check strategy generation logs and LLM credentials.'
            )

        for pt in prepared_targets:
            fm = pt.filtering_metadata
            if not fm:
                continue
            unresolved = fm.get('_unresolved_categories', [])
            for cat in unresolved:
                merged.pipeline_warnings.append(
                    f'Category {cat!r}: zero strategies selected — category could not be resolved. Check for typos or unsupported category names.'
                )
            for cat_key, cat_meta in fm.items():
                if cat_key.startswith('_') or not isinstance(cat_meta, dict):
                    continue
                if cat_meta.get('total_selected', 0) == 0:
                    gen_error = cat_meta.get('generation_error')
                    if gen_error:
                        merged.pipeline_warnings.append(
                            f'Category {cat_key!r}: zero strategies selected (generation error: {gen_error})'
                        )
                    else:
                        merged.pipeline_warnings.append(
                            f'Category {cat_key!r}: zero strategies selected — no applicable strategies found for this agent.'
                        )

        total_attacks = merged.summary.total_attacks
        unevaluated_attacks = merged.summary.unevaluated_attacks
        if total_attacks > 0 and unevaluated_attacks / total_attacks > 0.5:
            merged.pipeline_warnings.append(
                f'High evaluation failure rate: {unevaluated_attacks}/{total_attacks} attacks could not be evaluated. '
                'Check evaluator model configuration and credentials.'
            )
            logger.warning(
                f'High evaluation failure rate: {unevaluated_attacks}/{total_attacks} attacks returned inconclusive results.'
            )

        resolved_hooks.on_stage_end(PipelineStage.REPORT_GENERATION, {
            "resistance_rate": merged.summary.resistance_rate,
            "elapsed_s": pipeline_duration,
        })

        _save_report(output_dir, "03_summary_report.json", merged)

        # Upload cleaned results to Orq platform — strip skipped job results
        # (jobs return None for datapoints belonging to a different target).
        await _send_cleaned_results(
            results=results,
            name=resolved_name,
            description=description or f'{mode.capitalize()} red teaming ({len(all_target_labels)} targets)',
            start_time=pipeline_start,
        )

        set_span_attrs(pipeline_span, {
            "orq.redteam.num_datapoints": len(all_datapoints),
            "orq.redteam.num_categories": len(resolved_categories),
            "orq.redteam.duration_seconds": pipeline_duration,
        })

        # Memory cleanup for all targets — use runtime-accumulated entity IDs
        for pt in prepared_targets:
            if cleanup_memory:
                entity_ids = pt.memory_entity_ids
                if entity_ids:
                    resolved_hooks.on_stage_start(PipelineStage.CLEANUP, {"num_entities": len(entity_ids), "target": pt.target})
                    async with with_redteam_span(
                        "orq.redteam.memory_cleanup",
                        {"orq.redteam.num_entities": len(entity_ids)},
                    ) as cleanup_span:
                        cleanup_error = await cleanup_memory_entities(pt.agent_context, entity_ids, memory_cleanup=pt.resolved_memory_cleanup)
                        set_span_attrs(cleanup_span, {
                            "orq.redteam.num_stores": len(pt.agent_context.memory_stores) if pt.agent_context.memory_stores else 0,
                        })
                    if cleanup_error:
                        merged.pipeline_warnings.append(
                            f'Memory cleanup failed: {cleanup_error}. '
                            'Red-team attack data may persist in the agent\'s memory stores. '
                            'Manual cleanup may be required.'
                        )
                    resolved_hooks.on_stage_end(PipelineStage.CLEANUP, {"num_entities_cleaned": len(entity_ids)})

        return merged


# ---------------------------------------------------------------------------
# Static runner
# ---------------------------------------------------------------------------

async def _run_static(
    *,
    targets: list[str],
    name: str | None = None,
    categories: list[str] | None,
    evaluator_model: str,
    parallelism: int,
    max_static_datapoints: int | None,
    backend: str,
    dataset: Any,
    description: str | None,
    llm_client: AsyncOpenAI | None = None,
    hooks: PipelineHooks | None = None,
    output_dir: Path | None = None,
    target_config: TargetConfig | None = None,
    llm_kwargs: dict[str, Any] | None = None,
) -> RedTeamReport:
    """Run static red teaming for multiple targets in a single ``evaluatorq()`` call.

    Each target becomes its own ``job`` within the single evaluatorq run.
    The shared dataset is used as-is; each job processes every datapoint.
    Results are split by ``job_name`` into per-target reports, which are
    then merged into one unified ``RedTeamReport``.
    """
    from evaluatorq import evaluatorq
    from evaluatorq.redteam.reports.converters import (
        merge_reports,
        static_evaluatorq_results_to_reports,
        static_results_to_report,
    )

    resolved_name = name or 'red-team'

    resolved_hooks: PipelineHooks = hooks or DefaultHooks()
    pipeline_start = datetime.now(tz=timezone.utc).astimezone()

    # Load the shared dataset once for all targets
    from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import load_owasp_agentic_dataset
    data = load_owasp_agentic_dataset(
        dataset=dataset,
        num_samples=max_static_datapoints,
        categories=categories,
    )

    # Filter out datapoints whose category has no registered evaluator
    if data:
        from evaluatorq.redteam.frameworks.owasp.evaluators import get_evaluator_for_category

        skipped: Counter[str] = Counter()
        filtered_data: list[Any] = []
        for dp in data:
            cat = dp.inputs.get('category', '')
            norm_cat = normalize_category(cat)
            if get_evaluator_for_category(norm_cat) is None:
                skipped[norm_cat] += 1
            else:
                filtered_data.append(dp)
        for cat, count in sorted(skipped.items()):
            logger.warning(f'Skipped {count} datapoints for {cat}: no evaluator registered')
        if not filtered_data and data:
            msg = 'All datapoints were filtered out — no evaluator registered for any category.'
            raise ValueError(msg)
        data = filtered_data  # type: ignore[assignment]

    if isinstance(data, list):
        _save_stage(output_dir, "01_datapoints.json", json.dumps([dp.inputs for dp in data], indent=2, default=str))

    # Build one job per target using the shared helper
    sys_prompt = target_config.system_prompt if target_config else None
    jobs: list[Any] = [
        _create_job_for_target(t, llm_client, sys_prompt)
        for t in targets
    ]

    from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import create_owasp_evaluator
    evaluator = create_owasp_evaluator(evaluator_model=evaluator_model, llm_client=llm_client, llm_kwargs=llm_kwargs)

    # Confirm hook — report aggregate counts
    vulnerabilities: list[str] = list({
        dp.inputs.get("category", "") for dp in data  # pyright: ignore[reportAttributeAccessIssue]
        if dp.inputs.get("category")
    } if isinstance(data, list) else [])
    confirm_payload: ConfirmPayload = {
        "agent_context": None,
        "num_datapoints": len(data) if isinstance(data, list) else 0,  # type: ignore[arg-type]
        "num_dynamic": None,
        "num_static": len(data) if isinstance(data, list) else 0,  # type: ignore[arg-type]
        "categories": categories or vulnerabilities,
        "attack_model": "",
        "evaluator_model": evaluator_model,
        "max_turns": 1,
        "parallelism": parallelism,
        "filtering_metadata": None,
        "mode": "static",
        "target": ", ".join(targets),
        "dataset_path": str(dataset) if dataset else None,
        "vulnerabilities": vulnerabilities,
    }
    if not resolved_hooks.on_confirm(confirm_payload):
        msg = 'Execution cancelled by confirmation callback'
        raise CancelledError(msg)

    resolved_hooks.on_stage_start(PipelineStage.ATTACK_EXECUTION, {
        "num_datapoints": len(data) if isinstance(data, list) else 0,  # type: ignore[arg-type]
        "targets": targets,
    })

    results = await evaluatorq(
        resolved_name,
        data=data,
        jobs=jobs,
        evaluators=[evaluator],
        parallelism=parallelism,
        print_results=False,
        _exit_on_failure=False,
        _send_results=False,
        _trace_type="evaluatorq",
        description=description or f'Static red teaming ({len(targets)} targets)',
    )

    resolved_hooks.on_stage_end(PipelineStage.ATTACK_EXECUTION, {"num_results": len(results)})

    _save_stage(output_dir, "02_attack_results.json", json.dumps([r.model_dump(mode='json') for r in results], indent=2, default=str))

    pipeline_duration = (datetime.now(tz=timezone.utc).astimezone() - pipeline_start).total_seconds()

    resolved_hooks.on_stage_start(PipelineStage.REPORT_GENERATION, {"num_results": len(results)})

    # Build a job_name → (target_kind, target_value) lookup from the actual
    # job objects so that we can populate agent_key / agent_model correctly.
    job_name_to_target: dict[str, tuple[TargetKind, str]] = {}
    for t in targets:
        t_kind, t_value = _parse_target(t)
        safe = _make_safe_target(t_value)
        # create_model_job names follow "redteam:static:<safe_target>" convention;
        # use the safe slug for the lookup key to handle collisions gracefully.
        job_name_to_target[safe] = (t_kind, t_value)

    # static_evaluatorq_results_to_reports groups by job_name already.
    # We call it once with all results; it returns a dict keyed by job_name.
    per_job_reports: dict[str, RedTeamReport] = static_evaluatorq_results_to_reports(
        results=results,
        description=description or 'Static red teaming',
    )

    # Patch each per-job report with the correct agent identity, since
    # static_evaluatorq_results_to_reports does not know about target mapping.
    for job_name, job_report in per_job_reports.items():
        # Match by looking for the target slug as a suffix of the job_name.
        for slug, (t_kind, t_value) in job_name_to_target.items():
            if job_name.endswith(f':{slug}'):
                for result in job_report.results:
                    result.agent.key = t_value if not t_kind.is_model else result.agent.key
                    result.agent.model = t_value if t_kind.is_model else result.agent.model
                job_report.tested_agents = [t_value]
                break

    # Fetch agent contexts for all targets (best-effort)
    agent_contexts: dict[str, AgentContext] = {}
    try:
        backend_bundle = resolve_backend(backend, llm_client=llm_client, target_config=target_config)
        for t in targets:
            kind, value = _parse_target(t)
            try:
                ctx = await backend_bundle.context_provider.get_agent_context(value)
                agent_contexts[value] = ctx
            except Exception:
                logger.debug(f'Could not retrieve agent context for {value} — skipping')
    except Exception:
        logger.debug('Could not resolve backend for agent context retrieval — skipping')

    all_job_reports = list(per_job_reports.values())
    if not all_job_reports:
        merged = static_results_to_report(
            [],
            description=description or f'Static red teaming ({len(targets)} targets)',
        )
    else:
        merged = merge_reports(
            *all_job_reports,
            description=description or f'Static red teaming ({len(targets)} targets)',
        )

    merged.duration_seconds = pipeline_duration
    if agent_contexts:
        merged.agent_contexts = agent_contexts
    resolved_hooks.on_stage_end(PipelineStage.REPORT_GENERATION, {
        "resistance_rate": merged.summary.resistance_rate,
        "elapsed_s": pipeline_duration,
    })

    _save_report(output_dir, "03_summary_report.json", merged)

    # Upload results to Orq platform — in static mode all jobs produce real
    # output for every datapoint, so we send results as-is (no stripping).
    await _send_cleaned_results(
        results=results,
        name=resolved_name,
        description=description or f'Static red teaming ({len(targets)} targets)',
        start_time=pipeline_start,
    )

    return merged
