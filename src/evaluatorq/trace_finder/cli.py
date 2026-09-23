"""CLI command for finding recent traces with a natural-language query."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path  # noqa: TC003 — Typer resolves this annotation at runtime
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from evaluatorq.common import cli_width  # noqa: F401 — import for its non-TTY width side effect
from evaluatorq.common.cli_epilog import examples
from evaluatorq.common.cli_errors import emit_error
from evaluatorq.common.llm_client import resolve_llm_client
from evaluatorq.common.orq_client import resolve_orq_client

from .compiler import CompileError
from .export import export_json
from .filter_selector import FilterSelectionError
from .models import (
    FacetSelection,
    NumericFilters,
    PopulationRequest,
    RunRequest,
    RunSnapshot,
    TraceClassification,
)
from .orq_source import OrqSourceError
from .pipeline import build_run_store
from .settings import effective_settings

_FIND_EPILOG = examples(
    '# find traces whose conversations match a semantic question',
    'eq find "customers asking for a refund"',
    '# scope the live population with repeatable metadata facets',
    'eq find "mentions a refund" --project support-agent --model gpt-5.6-luna',
    '# save the completed run without printing trace content',
    'eq find "contains a frustrated customer" --json finder.json',
)


def _values(values: list[str] | None) -> frozenset[str]:
    return frozenset(value.strip() for value in values or () if value.strip())


def _facets(
    *,
    project: list[str] | None,
    model: list[str] | None,
    provider: list[str] | None,
    status: list[str] | None,
    product: list[str] | None,
    trace_type: list[str] | None,
    agent: list[str] | None,
    tool: list[str] | None,
) -> FacetSelection:
    return FacetSelection(
        project=_values(project),
        model=_values(model),
        provider=_values(provider),
        status=_values(status),
        product=_values(product),
        trace_type=_values(trace_type),
        agent_name=_values(agent),
        tool_name=_values(tool),
    )


def _request(
    *,
    query: str,
    settings: Any,
    project: list[str] | None,
    model: list[str] | None,
    provider: list[str] | None,
    status: list[str] | None,
    product: list[str] | None,
    trace_type: list[str] | None,
    agent: list[str] | None,
    tool: list[str] | None,
    tokens_min: int | None,
    tokens_max: int | None,
    duration_ms_min: int | None,
    duration_ms_max: int | None,
) -> RunRequest:
    end = datetime.now(timezone.utc)
    return RunRequest(
        query=query,
        mode='immediate',
        population=PopulationRequest(
            start=end - timedelta(days=settings.window_days),
            end=end,
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
            numeric=NumericFilters(
                tokens_min=tokens_min,
                tokens_max=tokens_max,
                duration_ms_min=duration_ms_min,
                duration_ms_max=duration_ms_max,
            ),
            limit=settings.limit,
        ),
        parallelism=settings.parallelism,
    )


def _print_progress(console: Console, snapshot: RunSnapshot) -> None:
    line = Text(
        f'{snapshot.state.upper():>11}  {snapshot.completed:,}/{snapshot.total:,} judged  '
        f'{snapshot.matched:,} matched  {snapshot.failed:,} failed'
    )
    line.stylize('cyan' if snapshot.state in {'compiling', 'classifying'} else 'green')
    console.print(line)


def _print_matches(console: Console, snapshot: RunSnapshot) -> None:
    table = Table(title='Matched traces')
    for column in ('Trace ID', 'Verdict', 'Confidence', 'Project', 'Model', 'Time'):
        table.add_column(column)
    traces = {trace.trace_id: trace for trace in snapshot.traces}
    matches: list[tuple[Any, TraceClassification]] = [
        (traces[trace_id], result)
        for trace_id, result in snapshot.results.items()
        if result.matched and result.error is None and trace_id in traces
    ]
    for trace, result in sorted(matches, key=lambda item: item[0].timestamp, reverse=True):
        confidence = '—' if result.confidence is None else f'{result.confidence:.2f}'
        verdict = '—' if result.value is None else str(result.value)
        table.add_row(
            trace.trace_id,
            verdict,
            confidence,
            trace.project,
            trace.model,
            trace.timestamp.isoformat(timespec='seconds'),
        )
    if matches:
        console.print(table)
    else:
        console.print('No matching traces.')
    console.print(f'Summary: {snapshot.matched:,} matched of {snapshot.total:,}; {snapshot.failed:,} failed.')


async def _run(store: Any, request: RunRequest, console: Console) -> RunSnapshot:
    snapshot = await store.compile(request, wait=False)
    _print_progress(console, snapshot)
    while snapshot.state not in {'completed', 'failed', 'cancelled'}:
        await asyncio.sleep(0.5)
        snapshot = await store.snapshot()
        _print_progress(console, snapshot)
    return snapshot


def find(
    query: Annotated[str, typer.Argument(help='Natural-language question to classify against recent traces.')],
    window_days: Annotated[
        int | None,
        typer.Option('--window-days', min=1, max=90, help='How many recent days to search.'),
    ] = None,
    limit: Annotated[int | None, typer.Option('--limit', min=1, max=5000, help='Maximum traces to classify.')] = None,
    parallelism: Annotated[
        int | None,
        typer.Option('--parallelism', min=1, max=200, help='Concurrent JEV classifications.'),
    ] = None,
    compiler_model: Annotated[
        str | None,
        typer.Option(
            '--compiler-model',
            help='Model that compiles the search question via the Orq router. Default: openai/gpt-5.6-luna. Requires ORQ_API_KEY.',
        ),
    ] = None,
    jev_model: Annotated[
        str | None,
        typer.Option(
            '--jev-model',
            help='JEV model that classifies each trace via the Orq router. Default: typesafe/jev-latest. Requires ORQ_API_KEY.',
        ),
    ] = None,
    json_path: Annotated[Path | None, typer.Option('--json', help='Write the completed run export to PATH.')] = None,
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
    """Find recent Orq traces that satisfy a natural-language JEV task."""
    settings = effective_settings({
        'window_days': window_days,
        'limit': limit,
        'parallelism': parallelism,
        'compiler_model': compiler_model,
        'jev_model': jev_model,
    })
    try:
        orq = resolve_orq_client()
        resolved = resolve_llm_client(require_orq=True, max_retries=0)
    except ValueError as exc:
        emit_error(exc)
        raise typer.Exit(code=2) from None

    client = resolved.client
    try:
        store = build_run_store(settings, client=client, orq=orq)
        request = _request(
            query=query,
            settings=settings,
            project=project,
            model=model,
            provider=provider,
            status=status,
            product=product,
            trace_type=trace_type,
            agent=agent,
            tool=tool,
            tokens_min=tokens_min,
            tokens_max=tokens_max,
            duration_ms_min=duration_ms_min,
            duration_ms_max=duration_ms_max,
        )
        snapshot = asyncio.run(_run(store, request, Console()))
    except (CompileError, FilterSelectionError, OrqSourceError, ValueError) as exc:
        emit_error(exc)
        raise typer.Exit(code=1) from None
    except Exception as exc:  # noqa: BLE001 — a CLI command must render unexpected runtime failures
        emit_error(exc)
        raise typer.Exit(code=1) from None

    if snapshot.state != 'completed' or snapshot.failed:
        detail = snapshot.error or (
            f'Find run completed with {snapshot.failed} failed classifications.'
            if snapshot.failed
            else f'Find run ended in {snapshot.state}.'
        )
        emit_error(detail)
        raise typer.Exit(code=1)
    if json_path is not None:
        try:
            json_path.write_text(export_json(snapshot), encoding='utf-8')
        except (OSError, ValueError) as exc:
            emit_error(f'Could not write JSON export to {json_path}: {exc}')
            raise typer.Exit(code=1) from None
    _print_matches(Console(), snapshot)


__all__ = ['find']
