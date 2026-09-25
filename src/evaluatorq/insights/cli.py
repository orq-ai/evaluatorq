"""Typer command for discovering dimensions and labels in recent Orq traces."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from evaluatorq.common import cli_width  # noqa: F401 — import for its non-TTY width side effect
from evaluatorq.common.cli_errors import emit_error
from evaluatorq.trace_finder.cli import _facets
from evaluatorq.trace_finder.export import RunExport
from evaluatorq.trace_finder.models import NumericFilters
from evaluatorq.trace_finder.settings import effective_settings

from . import presets
from .models import DimensionName, InsightsPopulation, InsightsRun, LabelSpec
from .pipeline import insights
from .store import list_runs

_DIMENSIONS: tuple[DimensionName, ...] = ('intent', 'failure', 'sentiment')


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


def _stored_run_path(run: InsightsRun) -> Path | None:
    for path, stored in list_runs():
        if isinstance(stored, InsightsRun) and stored.run_id == run.run_id:
            return path
    return None


def _validate_finder_export(path: Path) -> None:
    """Reject unreadable or invalid finder exports as CLI input errors before running Insights."""
    try:
        RunExport.model_validate_json(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'could not read a valid finder export from {path}: {exc}') from exc


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
        table = Table(title=f'Label: {name}')
        table.add_column('Value')
        table.add_column('Traces', justify='right')
        for value, count in sorted(label.counts.items(), key=lambda item: (-item[1], item[0])):
            table.add_row(value, f'{count:,}')
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
        str,
        typer.Option('--classifier-model', help='Model used to answer label questions.'),
    ] = 'typesafe/jev-latest',
    embedding_model: Annotated[
        str,
        typer.Option('--embedding-model', help='Model used to embed discovered-dimension text.'),
    ] = 'openai/text-embedding-3-small',
    priority_dimension: Annotated[
        str,
        typer.Option('--priority-dimension', help='Dimension used for the optional priority matrix.'),
    ] = 'intent',
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
    if query is not None and not query.strip():
        emit_error('--query must not be empty')
        raise typer.Exit(code=2)
    raw_dimensions = tuple(dimension or _DIMENSIONS)
    invalid_dimensions = sorted(set(raw_dimensions) - set(_DIMENSIONS))
    if invalid_dimensions:
        emit_error(f'unknown dimension(s): {", ".join(invalid_dimensions)}')
        raise typer.Exit(code=2)
    dimensions = cast('tuple[DimensionName, ...]', raw_dimensions)
    if priority_dimension not in _DIMENSIONS:
        emit_error(f'unknown priority dimension: {priority_dimension!r}')
        raise typer.Exit(code=2)
    try:
        labels = _resolve_labels(label)
        numeric = NumericFilters(
            tokens_min=tokens_min,
            tokens_max=tokens_max,
            duration_ms_min=duration_ms_min,
            duration_ms_max=duration_ms_max,
        )
        settings = effective_settings({'window_days': window_days, 'limit': limit, 'parallelism': parallelism})
        if from_finder is not None:
            _validate_finder_export(from_finder)
        population = (
            InsightsPopulation.from_finder_export(from_finder)
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

    try:
        run = asyncio.run(
            insights(
                population,
                labels=labels,
                dimensions=dimensions,
                max_clusters=max_clusters,
                max_subclusters=max_subclusters,
                outlier_zscore=outlier_zscore,
                summary_model=summary_model,
                classifier_model=classifier_model,
                embedding_model=embedding_model,
                priority_dimension=priority_dimension,
                parallelism=settings.parallelism,
                cache=not no_cache,
            )
        )
    except ValueError as exc:
        emit_error(exc)
        raise typer.Exit(code=2) from None
    run_path = _stored_run_path(run)
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
