"""Typer command for discovering dimensions and labels in recent Orq traces."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any, cast

import httpx
import typer
from openai import OpenAIError
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from evaluatorq.common import cli_width  # noqa: F401 — import for its non-TTY width side effect
from evaluatorq.common.cli_errors import emit_error
from evaluatorq.common.llm_client import resolve_llm_client
from evaluatorq.common.orq_client import DEFAULT_ORQ_BASE_URL, OrqProfile, close_orq_client, resolve_orq_client
from evaluatorq.trace_finder.cli import _facets, resolve_cli_profile
from evaluatorq.trace_finder.export import RunExport
from evaluatorq.trace_finder.models import NumericFilters
from evaluatorq.trace_finder.settings import effective_settings

from . import presets
from .models import DimensionName, InsightsPopulation, InsightsRun, LabelSpec

_DIMENSIONS: tuple[DimensionName, ...] = ('intent', 'failure', 'sentiment')


async def insights(*args: Any, **kwargs: Any) -> InsightsRun:
    """Load the optional analysis pipeline only when the command is executed."""
    from .pipeline import insights as run_insights

    return await run_insights(*args, **kwargs)


async def _run_insights_with_profile(
    population: InsightsPopulation, profile: OrqProfile | None, **kwargs: Any
) -> InsightsRun:
    if profile is None:
        return await insights(population, **kwargs)
    host = profile.server or DEFAULT_ORQ_BASE_URL
    orq = resolve_orq_client(profile.api_key, base_url=host)
    resolved = None
    try:
        resolved = resolve_llm_client(extra_api_key=profile.api_key, orq_host=host, require_orq=True, max_retries=0)
        return await insights(population, orq_client=orq, llm_client=resolved.client, **kwargs)
    finally:
        try:
            await close_orq_client(orq)
        finally:
            if resolved is not None and resolved.owned:
                await resolved.client.close()


def _resolve_labels(values: list[str] | None) -> list[LabelSpec]:
    labels: list[LabelSpec] = []
    for value in values or ():
        preset = presets.LABEL_PRESETS.get(value)
        if preset is not None:
            labels.append(preset)
            continue
        path = Path(value)
        if not path.is_file():
            raise ValueError(f'unknown Insights label preset or JSON file: {value!r}')
        try:
            raw: Any = json.loads(path.read_text(encoding='utf-8'))
            payloads = raw if isinstance(raw, list) else [raw]
            if not payloads:
                raise ValueError(f'label file {path} must contain one LabelSpec or a non-empty list')
            labels.extend(LabelSpec.model_validate(payload) for payload in payloads)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f'could not read label spec(s) from {path}: {exc}') from exc
    return labels


def _validate_finder_export(path: Path) -> RunExport:
    """Reject unreadable or invalid finder exports as CLI input errors before running Insights."""
    try:
        return RunExport.model_validate_json(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'could not read a valid finder export from {path}: {exc}') from exc


def _duplicate_dimensions(names: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    return sorted(duplicates)


def _print_run(run: InsightsRun, run_path: Path | None) -> None:
    console = Console()
    for name, dimension in run.dimensions.items():
        table = Table(title=f'{name.title()} clusters (top 5)')
        table.add_column('Cluster')
        table.add_column('Traces', justify='right')
        clusters = sorted(
            (cluster for cluster in dimension.clusters if cluster.level == 'top'),
            key=lambda cluster: cluster.size,
            reverse=True,
        )[:5]
        for cluster in clusters:
            table.add_row(cluster.name, f'{cluster.size:,}')
        if clusters:
            console.print(table)
        else:
            console.print(f'{name.title()}: no top-level clusters.')

    for name, label in run.labels.items():
        title = f'Label: {name}'
        table = Table(title=title, min_width=len(title) + 4)
        table.add_column('Value')
        table.add_column('Traces', justify='right')
        for value, count in sorted(label.counts.items(), key=lambda item: (-item[1], item[0])):
            display_value = value
            if label.spec.kind == 'score' and isinstance(label.spec.criteria, list) and value.isdigit():
                index = int(value)
                if index < len(label.spec.criteria):
                    criterion = label.spec.criteria[index].split(':', 1)[0]
                    display_value = f'{value} · {criterion.removeprefix(f"{value} ")}'
            table.add_row(display_value, f'{count:,}')
        if label.n_failed:
            table.add_row('Failed / unclassified', f'{label.n_failed:,}')
        console.print(table)

    failed = run.counts.get('n_failed_traces', 0)
    console.print(f'Summary: {len(run.traces):,} traces; {failed:,} failed traces.')
    if run.stage_failures:
        table = Table(title='Stage failures')
        table.add_column('Stage')
        table.add_column('Message')
        for failure in run.stage_failures:
            table.add_row(failure.stage, failure.message)
        console.print(table)
    console.print(f'Run file: {run_path}' if run_path else 'Run file: unavailable.')


def insights_cmd(
    query: Annotated[str | None, typer.Option('--query', help='Optional semantic question to select traces.')] = None,
    profile: Annotated[
        str | None,
        typer.Option('--profile', help='Orq CLI profile for traces and model calls; overrides saved settings.'),
    ] = None,
    from_finder: Annotated[
        Path | None,
        typer.Option('--from-finder', help='Hydrate the population from an eq find JSON export.'),
    ] = None,
    label: Annotated[
        list[str] | None,
        typer.Option('--label', help='Label preset name or path to a JSON LabelSpec (repeatable).'),
    ] = None,
    dimension: Annotated[
        list[str] | None,
        typer.Option('--dimension', help='Discovered dimension to include; repeatable.'),
    ] = None,
    max_clusters: Annotated[int, typer.Option('--max-clusters', min=1, help='Maximum top-level clusters.')] = 15,
    max_subclusters: Annotated[
        int,
        typer.Option('--max-subclusters', min=1, help='Maximum base clusters per top-level cluster.'),
    ] = 15,
    outlier_zscore: Annotated[
        float | None,
        typer.Option('--outlier-zscore', min=0, help='Move far-away traces to noise using this z-score.'),
    ] = None,
    summary_model: Annotated[
        str,
        typer.Option('--summary-model', help='Model used to summarize each trace.'),
    ] = 'openai/gpt-6-luna',
    classifier_model: Annotated[
        str | None,
        typer.Option(
            '--classifier-model',
            help='Model used to answer label questions. Defaults to EVALUATORQ_CLASSIFIER_MODEL or the dashboard setting.',
        ),
    ] = None,
    embedding_model: Annotated[
        str,
        typer.Option('--embedding-model', help='Model used to embed discovered-dimension text.'),
    ] = 'openai/text-embedding-3-small',
    priority_dimension: Annotated[
        str | None,
        typer.Option(
            '--priority-dimension',
            help='Priority matrix dimension; defaults to intent or the first selected dimension.',
        ),
    ] = None,
    no_cache: Annotated[bool, typer.Option('--no-cache', help='Disable the local Insights cache.')] = False,  # noqa: FBT002
    json_path: Annotated[
        Path | None, typer.Option('--json', help='Write a copy of the completed run JSON to PATH.')
    ] = None,
    window_days: Annotated[
        int | None, typer.Option('--window-days', min=1, max=90, help='Recent days to scan.')
    ] = None,
    limit: Annotated[int | None, typer.Option('--limit', min=1, max=5000, help='Maximum traces to scan.')] = None,
    parallelism: Annotated[
        int | None,
        typer.Option('--parallelism', min=1, max=200, help='Concurrent model calls.'),
    ] = None,
    project: Annotated[list[str] | None, typer.Option('--project', help='Project facet; repeatable.')] = None,
    model: Annotated[list[str] | None, typer.Option('--model', help='Model facet; repeatable.')] = None,
    provider: Annotated[list[str] | None, typer.Option('--provider', help='Provider facet; repeatable.')] = None,
    status: Annotated[list[str] | None, typer.Option('--status', help='Status facet; repeatable.')] = None,
    product: Annotated[list[str] | None, typer.Option('--product', help='Product facet; repeatable.')] = None,
    trace_type: Annotated[list[str] | None, typer.Option('--trace-type', help='Trace type facet; repeatable.')] = None,
    agent: Annotated[list[str] | None, typer.Option('--agent', help='Agent name facet; repeatable.')] = None,
    tool: Annotated[list[str] | None, typer.Option('--tool', help='Tool name facet; repeatable.')] = None,
    tokens_min: Annotated[int | None, typer.Option('--tokens-min', min=0, help='Minimum total tokens.')] = None,
    tokens_max: Annotated[int | None, typer.Option('--tokens-max', min=0, help='Maximum total tokens.')] = None,
    duration_ms_min: Annotated[
        int | None,
        typer.Option('--duration-ms-min', min=0, help='Minimum trace duration in milliseconds.'),
    ] = None,
    duration_ms_max: Annotated[
        int | None,
        typer.Option('--duration-ms-max', min=0, help='Maximum trace duration in milliseconds.'),
    ] = None,
) -> None:
    """Discover trace dimensions and answer fixed label questions."""
    if from_finder is not None and query is not None:
        emit_error('--from-finder cannot be combined with --query')
        raise typer.Exit(code=2)
    if from_finder is not None:
        population_options = (
            ('--window-days', window_days is not None),
            ('--limit', limit is not None),
            ('--project', bool(project)),
            ('--model', bool(model)),
            ('--provider', bool(provider)),
            ('--status', bool(status)),
            ('--product', bool(product)),
            ('--trace-type', bool(trace_type)),
            ('--agent', bool(agent)),
            ('--tool', bool(tool)),
            ('--tokens-min', tokens_min is not None),
            ('--tokens-max', tokens_max is not None),
            ('--duration-ms-min', duration_ms_min is not None),
            ('--duration-ms-max', duration_ms_max is not None),
        )
        conflicting = [name for name, used in population_options if used]
        if conflicting:
            emit_error(f'--from-finder cannot be combined with population options: {", ".join(conflicting)}')
            raise typer.Exit(code=2)
    if query is not None and not query.strip():
        emit_error('--query must not be empty')
        raise typer.Exit(code=2)
    raw_dimensions = tuple(dimension or _DIMENSIONS)
    invalid_dimensions = sorted(set(raw_dimensions) - set(_DIMENSIONS))
    if invalid_dimensions:
        emit_error(f'unknown dimension(s): {", ".join(invalid_dimensions)}')
        raise typer.Exit(code=2)
    duplicate_dimensions = _duplicate_dimensions(raw_dimensions)
    if duplicate_dimensions:
        emit_error(f'--dimension cannot be repeated: {", ".join(sorted(duplicate_dimensions))}')
        raise typer.Exit(code=2)
    dimensions = cast('tuple[DimensionName, ...]', raw_dimensions)
    if priority_dimension is not None and priority_dimension not in _DIMENSIONS:
        emit_error(f'unknown priority dimension: {priority_dimension!r}')
        raise typer.Exit(code=2)
    if priority_dimension is not None and priority_dimension not in dimensions:
        emit_error(f'--priority-dimension {priority_dimension!r} must also be included with --dimension')
        raise typer.Exit(code=2)
    selected_priority_dimension = priority_dimension or ('intent' if 'intent' in dimensions else dimensions[0])
    try:
        labels = _resolve_labels(label)
        numeric = NumericFilters(
            tokens_min=tokens_min,
            tokens_max=tokens_max,
            duration_ms_min=duration_ms_min,
            duration_ms_max=duration_ms_max,
        )
        settings = effective_settings({'window_days': window_days, 'limit': limit, 'parallelism': parallelism})
        selected_profile = resolve_cli_profile(profile if profile is not None else settings.orq_profile)
        population = (
            InsightsPopulation.from_finder_export(from_finder, export=_validate_finder_export(from_finder))
            if from_finder is not None
            else InsightsPopulation(
                query=query,
                facets=_facets(
                    project=project,
                    model=model,
                    provider=provider,
                    status=status,
                    product=product,
                    trace_type=trace_type,
                    agent=agent,
                    tool=tool,
                ),
                numeric=numeric,
                window_days=settings.window_days,
                limit=settings.limit,
            )
        )
    except (OSError, ValueError, ValidationError) as exc:
        emit_error(exc)
        raise typer.Exit(code=2) from None

    run_path: Path | None = None

    def remember_run_path(path: Path) -> None:
        nonlocal run_path
        run_path = path

    try:
        run = asyncio.run(
            _run_insights_with_profile(
                population,
                selected_profile,
                labels=labels,
                dimensions=dimensions,
                max_clusters=max_clusters,
                max_subclusters=max_subclusters,
                outlier_zscore=outlier_zscore,
                summary_model=summary_model,
                classifier_model=classifier_model,
                embedding_model=embedding_model,
                priority_dimension=selected_priority_dimension,
                parallelism=settings.parallelism,
                cache=not no_cache,
                _finder_export_source=from_finder.resolve() if from_finder is not None else None,
                _on_saved=remember_run_path,
            )
        )
    except (ImportError, OSError, OpenAIError, RuntimeError, TimeoutError, ValueError, httpx.HTTPError) as exc:
        emit_error(exc)
        raise typer.Exit(code=1) from None
    if json_path is not None:
        try:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(run.model_dump_json(indent=2) + '\n', encoding='utf-8')
        except OSError as exc:
            emit_error(f'Could not write JSON export to {json_path}: {exc}')
            raise typer.Exit(code=1) from None
    _print_run(run, run_path)
    if run.status == 'error':
        raise typer.Exit(code=1)


__all__ = ['insights_cmd']
