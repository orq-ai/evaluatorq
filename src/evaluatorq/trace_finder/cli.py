"""CLI command for finding recent traces with a natural-language query."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path  # noqa: TC003 — Typer resolves this annotation at runtime
from typing import Annotated, Any

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.text import Text

from evaluatorq.common import cli_width  # noqa: F401 — import for its non-TTY width side effect
from evaluatorq.common.cli_epilog import examples
from evaluatorq.common.cli_errors import emit_error
from evaluatorq.common.llm_client import resolve_llm_client
from evaluatorq.common.orq_client import (
    DEFAULT_ORQ_BASE_URL,
    close_orq_client,
    list_orq_profiles,
    resolve_orq_client,
)

from .compiler import CompileError
from .debug import cli_debug
from .debug import enabled as debug_enabled
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
from .run_store import RunStore
from .settings import credential_fingerprint, effective_settings

MAX_FIND_WAIT_SECONDS = 2 * 60 * 60

_FIND_EPILOG = examples(
    '# find traces whose conversations match a semantic question',
    'eq find "customers asking for a refund"',
    '# scope the live population with repeatable metadata facets',
    'eq find "mentions a refund" --project support-agent --model gpt-5.6-luna',
    '# use credentials from an orq CLI profile',
    'eq find "mentions a refund" --profile research --limit 20',
    '# inspect compiler and classifier requests and responses',
    'eq find "mentions a refund" --limit 10 --debug',
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
            ).model_copy(update={'project_id': settings.orq_project_id if not project else None}),
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


def _progress_key(snapshot: RunSnapshot) -> tuple[str, int, int, int, int]:
    return snapshot.state, snapshot.completed, snapshot.total, snapshot.matched, snapshot.failed


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
    last_progress = _progress_key(snapshot)
    if debug_enabled():
        _print_progress(console, snapshot)

    async def poll() -> RunSnapshot:
        nonlocal snapshot, last_progress
        while snapshot.state not in {'completed', 'failed', 'cancelled'}:
            await asyncio.sleep(0.5)
            snapshot = await store.snapshot_for_render() if isinstance(store, RunStore) else await store.snapshot()
            progress = _progress_key(snapshot)
            if debug_enabled() and progress != last_progress:
                _print_progress(console, snapshot)
                last_progress = progress
        return await store.snapshot() if isinstance(store, RunStore) else snapshot

    try:
        return await asyncio.wait_for(poll(), timeout=MAX_FIND_WAIT_SECONDS)
    except asyncio.TimeoutError as error:
        try:
            await asyncio.wait_for(store.cancel(), timeout=10)
        except asyncio.TimeoutError:
            logger.warning('Trace finder cancellation did not complete within 10 seconds after the CLI wait limit')
        raise TimeoutError(f'Find run exceeded the {MAX_FIND_WAIT_SECONDS}-second wait limit.') from error


async def _run_with_cleanup(store: Any, request: RunRequest, console: Console, resolved: Any, orq: Any) -> RunSnapshot:
    try:
        return await _run(store, request, console)
    finally:
        try:
            await store.close()
        finally:
            await _close_clients(resolved, orq)


async def _close_clients(resolved: Any, orq: Any) -> None:
    try:
        await close_orq_client(orq)
    finally:
        if getattr(resolved, 'owned', False):
            await resolved.client.close()


def find(
    query: Annotated[str, typer.Argument(help='Natural-language question to classify against recent traces.')],
    debug: Annotated[  # noqa: FBT002 — named CLI flag
        bool,
        typer.Option('--debug', help='Show progress and model requests and responses, including trace content.'),
    ] = False,
    profile: Annotated[
        str | None,
        typer.Option(
            '--profile', help='Orq CLI profile for traces and model calls; overrides ORQ_API_KEY and ORQ_BASE_URL.'
        ),
    ] = None,
    window_days: Annotated[
        int | None,
        typer.Option('--window-days', min=1, max=90, help='How many recent days to search.'),
    ] = None,
    limit: Annotated[int | None, typer.Option('--limit', min=1, max=5000, help='Maximum traces to classify.')] = None,
    parallelism: Annotated[
        int | None,
        typer.Option('--parallelism', min=1, max=200, help='Concurrent classify calls.'),
    ] = None,
    compiler_model: Annotated[
        str | None,
        typer.Option(
            '--compiler-model',
            help='Model that compiles the search question via the Orq router. Default: openai/gpt-5.6-luna. Requires Orq credentials.',
        ),
    ] = None,
    classifier_model: Annotated[
        str | None,
        typer.Option(
            '--classifier-model',
            help='Model that classifies each trace via the Orq router. Default: typesafe/jev-latest. Requires Orq credentials.',
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
    """Find recent Orq traces that satisfy a natural-language classifier task."""
    settings = effective_settings({
        'window_days': window_days,
        'limit': limit,
        'parallelism': parallelism,
        'compiler_model': compiler_model,
        'classifier_model': classifier_model,
    })
    orq = None
    try:
        selected = None
        profile_name = profile if profile is not None else settings.orq_profile
        if profile_name is not None:
            selected = next((candidate for candidate in list_orq_profiles() if candidate.name == profile_name), None)
            if selected is None:
                raise ValueError(f'Orq profile {profile_name!r} is unavailable. Check `orq auth profile list`.')
            if '*' in selected.api_key:
                raise ValueError(
                    f'Orq profile {profile_name!r} has a masked key that evaluatorq cannot read. '
                    'Run `orq doctor --fix` or use ORQ_API_KEY.'
                )
        if selected is None:
            fingerprint = credential_fingerprint(os.environ.get('ORQ_API_KEY'), os.environ.get('ORQ_BASE_URL'))
            orq = resolve_orq_client()
            resolved = resolve_llm_client(require_orq=True, max_retries=0)
        else:
            host = selected.server or DEFAULT_ORQ_BASE_URL
            fingerprint = credential_fingerprint(selected.api_key, host)
            orq = resolve_orq_client(selected.api_key, base_url=host)
            resolved = resolve_llm_client(
                extra_api_key=selected.api_key, orq_host=host, require_orq=True, max_retries=0
            )
    except (ImportError, ValueError) as exc:
        if orq is not None:
            asyncio.run(close_orq_client(orq))
        emit_error(exc)
        raise typer.Exit(code=2) from None

    if profile_name != settings.orq_profile or fingerprint != settings.orq_credential_fingerprint:
        settings = settings.model_copy(update={'orq_project_id': None, 'orq_project_name': None})

    client = resolved.client
    runner_entered = False
    try:
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
        store = build_run_store(settings, client=client, orq=orq)
        runner_entered = True
        console = Console()
        with cli_debug(active=debug):
            if debug_enabled():
                snapshot = asyncio.run(_run_with_cleanup(store, request, console, resolved, orq))
            else:
                with console.status('Finding traces…'):
                    snapshot = asyncio.run(_run_with_cleanup(store, request, console, resolved, orq))
    except (CompileError, FilterSelectionError, OrqSourceError, TimeoutError, ValueError) as exc:
        emit_error(exc)
        raise typer.Exit(code=1) from None
    finally:
        if not runner_entered:
            asyncio.run(_close_clients(resolved, orq))

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
