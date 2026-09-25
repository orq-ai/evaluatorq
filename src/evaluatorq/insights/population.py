"""Resolve an `InsightsPopulation` into traces by reusing `trace_finder` end to end.

Three mutually exclusive paths, matching `InsightsPopulation`'s own fields:

- **Query path** (`pop.query` set): compile the semantic query with `compile_query`
  and select generated facet/numeric filters with `select_filters_with_response`
  concurrently (mirrors `trace_finder/run_store.py`'s `_plan`), merge them under the
  caller's explicit facets/numeric (explicit wins — reuses `run_store._merge_facets`/
  `_merge_numeric` directly; they're module-private but not duplicated here), then
  load with `OrqTraceSource`.
- **Finder export path** (`pop.finder_export` set): read a `RunExport` JSON, reload
  its window and filters through `OrqTraceSource`, and keep only the traces whose
  ids are in `matched_trace_ids`. No compile, no match question — the export already
  pins the matched population.
- **Filter-only path** (neither set): load `pop.facets`/`pop.numeric` directly. No
  compile, no match question.

No retry layer here: `compile_query`, `select_filters_with_response` and
`OrqTraceSource.load_async` each own their one retry layer already (see their
docstrings); this module does not wrap them a second time.

Filter *selection* failing (facet catalogue or classify unavailable) is not fatal —
mirroring `trace_finder/pipeline.py`'s `filter_selector`, it degrades to no generated
filters with a `logger.warning` naming the cause, recorded in `ResolvedPopulation.echo`.
Everything else that can go wrong here (compiling the query, reading/parsing a finder
export, loading traces) raises `PopulationError`, which the insights pipeline turns
into a `StageFailure` for the `'population'` stage.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.trace_finder.compiler import compile_query
from evaluatorq.trace_finder.export import RunExport
from evaluatorq.trace_finder.facets import load_facet_catalogue
from evaluatorq.trace_finder.filter_selector import select_filters_with_response
from evaluatorq.trace_finder.models import FacetSelection, NumericFilters
from evaluatorq.trace_finder.orq_source import OrqTraceSource
from evaluatorq.trace_finder.run_store import _merge_facets, _merge_numeric  # reportPrivateUsage is off; see docstring

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from orq_ai_sdk import Orq

    from evaluatorq.insights.models import InsightsPopulation
    from evaluatorq.trace_finder.export import ExportFilters
    from evaluatorq.trace_finder.models import CompiledQuery, TraceRecord


class PopulationError(RuntimeError):
    """Resolving the population failed: a bad finder export, a failed compile, or a failed trace load."""


@dataclass
class ResolvedPopulation:
    """The traces an insights run will label and cluster, plus provenance for the run JSON's `population` echo."""

    traces: list[TraceRecord]
    compiled: CompiledQuery | None
    echo: dict[str, Any]
    n_scanned: int


def _window(pop: InsightsPopulation) -> tuple[datetime, datetime]:
    """Mirror `trace_finder/pipeline.py`'s window resolution: explicit bounds win, else `window_days` back from now."""
    end = pop.end or datetime.now(timezone.utc)
    start = pop.start or end - timedelta(days=pop.window_days)
    return start, end


def _facets_from_export(filters: ExportFilters) -> FacetSelection:
    """Rebuild the `FacetSelection` an export's filters were loaded with (`tool_names` -> `tool_name`)."""
    return FacetSelection(
        project=frozenset(filters.project),
        model=frozenset(filters.model),
        provider=frozenset(filters.provider),
        status=frozenset(filters.status),
        product=frozenset(filters.product),
        trace_type=frozenset(filters.trace_type),
        agent_name=frozenset(filters.agent_name),
        tool_name=frozenset(filters.tool_names),
    )


async def _load_traces(
    orq: Orq,
    *,
    start: datetime | None,
    end: datetime | None,
    limit: int,
    facets: FacetSelection,
    numeric: NumericFilters,
) -> tuple[TraceRecord, ...]:
    """Load a population's traces through `OrqTraceSource`; any failure becomes a `PopulationError`."""
    source = OrqTraceSource(orq)
    try:
        snapshot = await source.load_async(start, end, limit, facets=facets, numeric=numeric)
    except Exception as error:
        raise PopulationError(f'loading the trace population failed: {error}') from error
    finally:
        source.close()
    return snapshot.traces


async def _select_generated_filters(
    query: str, *, orq: Orq, client: AsyncOpenAI, classifier_model: str, start: datetime, end: datetime
) -> tuple[FacetSelection, str | None]:
    """Select generated facet filters for `query`, degrading to none on failure (mirrors `trace_finder/pipeline.py`)."""
    try:
        catalogue = await load_facet_catalogue(orq, start=start, end=end, limit=50)
        result = await select_filters_with_response(client, classifier_model, catalogue, query)
    except Exception as error:  # noqa: BLE001 - filter selection is best-effort; the query itself still resolves
        logger.warning(
            'Insights population filter selection unavailable or incomplete: {}; skipping generated filters', error
        )
        return FacetSelection(), str(error)
    return result.selection, result.error


async def _resolve_from_query(
    pop: InsightsPopulation,
    *,
    orq: Orq,
    client: AsyncOpenAI,
    compiler_model: str,
    classifier_model: str,
) -> ResolvedPopulation:
    assert pop.query is not None  # noqa: S101 - guarded by the caller's dispatch before this is reached
    start, end = _window(pop)

    # Run the compile and the filter selection concurrently, mirroring
    # `trace_finder/run_store.py`'s `_plan`. Only the compile task can raise
    # (`_select_generated_filters` already catches its own failures and
    # degrades instead); either way, clean up the other task before returning.
    compile_task = asyncio.create_task(compile_query(client, compiler_model, pop.query))
    filters_task = asyncio.create_task(
        _select_generated_filters(
            pop.query, orq=orq, client=client, classifier_model=classifier_model, start=start, end=end
        )
    )
    tasks = (compile_task, filters_task)
    try:
        plan, (generated_facets, filter_error) = await asyncio.gather(*tasks)
    except Exception as error:
        raise PopulationError(f'compiling the population query failed: {error}') from error
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    merged_facets = _merge_facets(pop.facets, generated_facets)
    merged_numeric = _merge_numeric(pop.numeric, plan.numeric)

    traces = await _load_traces(
        orq, start=start, end=end, limit=pop.limit, facets=merged_facets, numeric=merged_numeric
    )

    echo: dict[str, Any] = {
        'mode': 'query',
        'query': pop.query,
        'compiled_task': plan.compiled.task.instructions,
        'facets': merged_facets.model_dump(mode='json'),
        'generated_facets': generated_facets.model_dump(mode='json'),
        'numeric': merged_numeric.model_dump(mode='json'),
        'generated_numeric': plan.numeric.model_dump(mode='json'),
        'start': start.isoformat(),
        'end': end.isoformat(),
        'limit': pop.limit,
        'filter_selection_error': filter_error,
    }
    return ResolvedPopulation(traces=list(traces), compiled=plan.compiled, echo=echo, n_scanned=len(traces))


async def _resolve_from_export(pop: InsightsPopulation, *, orq: Orq) -> ResolvedPopulation:
    assert pop.finder_export is not None  # noqa: S101 - guarded by the caller's dispatch before this is reached

    try:
        export = RunExport.model_validate_json(pop.finder_export.read_text())
    except Exception as error:
        raise PopulationError(f'loading finder export {pop.finder_export} failed: {error}') from error

    facets = _facets_from_export(export.filters)
    numeric = NumericFilters(**export.numeric.model_dump())

    traces = await _load_traces(
        orq, start=export.start, end=export.end, limit=export.limit, facets=facets, numeric=numeric
    )

    by_id = {trace.trace_id: trace for trace in traces}
    resolved: list[TraceRecord] = []
    missing = 0
    for trace_id in export.matched_trace_ids:
        trace = by_id.get(trace_id)
        if trace is None:
            missing += 1
            continue
        resolved.append(trace)
    if missing:
        logger.warning(
            'Insights population from finder export {} could not find {} of {} matched traces in the reloaded window',
            pop.finder_export,
            missing,
            len(export.matched_trace_ids),
        )

    echo: dict[str, Any] = {
        'mode': 'export',
        'query': export.query,
        'finder_export': str(pop.finder_export),
        'facets': facets.model_dump(mode='json'),
        'numeric': numeric.model_dump(mode='json'),
        'start': export.start.isoformat() if export.start else None,
        'end': export.end.isoformat() if export.end else None,
        'limit': export.limit,
        'n_matched_ids': len(export.matched_trace_ids),
        'n_missing_export_ids': missing,
    }
    return ResolvedPopulation(traces=resolved, compiled=None, echo=echo, n_scanned=len(traces))


async def _resolve_from_filters(pop: InsightsPopulation, *, orq: Orq) -> ResolvedPopulation:
    start, end = _window(pop)
    traces = await _load_traces(orq, start=start, end=end, limit=pop.limit, facets=pop.facets, numeric=pop.numeric)

    echo: dict[str, Any] = {
        'mode': 'filter',
        'query': None,
        'facets': pop.facets.model_dump(mode='json'),
        'numeric': pop.numeric.model_dump(mode='json'),
        'start': start.isoformat(),
        'end': end.isoformat(),
        'limit': pop.limit,
    }
    return ResolvedPopulation(traces=list(traces), compiled=None, echo=echo, n_scanned=len(traces))


async def resolve_population(
    pop: InsightsPopulation,
    *,
    orq: Orq,
    client: AsyncOpenAI,
    compiler_model: str,
    classifier_model: str,
) -> ResolvedPopulation:
    """Resolve `pop` into traces via the query, finder-export, or filter-only path — see module docstring.

    `compiler_model`/`classifier_model` are only used on the query path
    (`compile_query` and `select_filters_with_response`, respectively); the
    other two paths never call the LLM.
    """
    if pop.finder_export is not None:
        return await _resolve_from_export(pop, orq=orq)
    if pop.query is not None:
        return await _resolve_from_query(
            pop, orq=orq, client=client, compiler_model=compiler_model, classifier_model=classifier_model
        )
    return await _resolve_from_filters(pop, orq=orq)
