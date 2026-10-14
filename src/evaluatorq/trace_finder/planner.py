"""Shared planner for semantic dimensions and generated trace-population filters."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from .compiler import canonical_generated_end, compile_query, normalize_generated_time
from .facets import facet_catalogue_cache_hit, load_facet_catalogue_cached
from .filter_selector import FilterSelectionResult, select_filters_with_response, select_llm_filters_with_response
from .models import FACET_NAMES, FacetCatalogue, FacetName, FacetSelection, NumericFilters, PopulationRequest
from .orq_source import OrqTraceSource

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from orq_ai_sdk import Orq

    from evaluatorq.common.judge import ClassifyResponse
    from evaluatorq.trace_finder.compiler import CompiledQuery


@dataclass(frozen=True)
class PlanningResult:
    """Complete planner output, retaining generated values separately from caller precedence."""

    dimensions: tuple[CompiledQuery, ...]
    generated_facets: FacetSelection
    generated_numeric: NumericFilters
    generated_time: tuple[datetime | None, datetime | None] | None
    effective_population: PopulationRequest
    selector_response: ClassifyResponse | None
    selector_error: str | None
    provenance: Mapping[str, str]
    usage: Mapping[str, Any]
    unsupported_reason: str | None = None
    empty_match: bool = False


async def plan_query(
    query: str,
    population: PopulationRequest,
    *,
    client: AsyncOpenAI,
    orq: Orq,
    compiler_model: str,
    filter_selection: Literal['llm', 'llm+jev'] = 'llm+jev',
    classifier_model: str,
    fast_filter_model: str,
    within_results: bool = False,
    now: datetime | None = None,
    catalogue_loader: Callable[..., Awaitable[FacetCatalogue]] | None = None,
    candidate_confirmer: Callable[..., Awaitable[bool]] | None = None,
) -> PlanningResult:
    """Compile semantic dimensions and select facet/numeric/time filters using configured models."""
    anchor = now or datetime.now(timezone.utc)
    compiled_task = asyncio.create_task(compile_query(client, compiler_model, query, now=anchor))
    try:
        compiled = await compiled_task
    except BaseException:
        if not compiled_task.done():
            compiled_task.cancel()
        await asyncio.gather(compiled_task, return_exceptions=True)
        raise
    default_window = population.time_origin == 'default'
    initial_end = population.end or anchor
    initial_start = population.start or initial_end - timedelta(days=30)
    default_bounds = (initial_start, initial_end)
    cache_events: list[dict[str, Any]] = []
    selector_usage_events: list[dict[str, Any]] = []
    catalogue = FacetCatalogue()

    async def load_catalogue(start: datetime, end: datetime) -> FacetCatalogue:
        # Only the caller's own default window may share the duration-keyed rolling entry; a
        # query-generated range keeps its exact dates in the key.
        rolling = default_window and (start, end) == default_bounds
        started = time.monotonic()
        cache_hit = not catalogue_loader and facet_catalogue_cache_hit(orq, start, end, 50, default_window=rolling)
        try:
            return await load_facet_catalogue_cached(
                orq,
                start,
                end,
                50,
                default_window=rolling,
                now=anchor,
                loader=catalogue_loader,
            )
        finally:
            cache_events.append({
                'start': start.isoformat(),
                'end': end.isoformat(),
                'hit': cache_hit,
                'elapsed_seconds': time.monotonic() - started,
                'source': 'injected' if catalogue_loader else 'production',
            })

    time_filter = compiled.time if filter_selection == 'llm+jev' else None
    if filter_selection == 'llm+jev':
        initial_time = normalize_generated_time(
            compiled.time,
            population=population,
            within_results=within_results,
            reference=anchor,
        )
        if initial_time and not (within_results or population.time_origin == 'within_results'):
            initial_start, initial_end = initial_time
    initial_catalogue_error: str | None = None
    try:
        catalogue = (
            await load_catalogue(initial_start, initial_end) if initial_start < initial_end else FacetCatalogue()
        )
    except Exception as exc:  # noqa: BLE001 - fail closed while retaining the compiled semantic plan
        logger.warning('Trace filter catalogue load failed: {}', exc)
        catalogue = FacetCatalogue()
        initial_catalogue_error = str(exc)
    try:
        if initial_catalogue_error is not None:
            raise RuntimeError(initial_catalogue_error)
        selected = await _select_from_catalogue(
            query,
            catalogue,
            client=client,
            filter_selection=filter_selection,
            classifier_model=classifier_model,
            fast_filter_model=fast_filter_model,
            now=anchor,
        )
        usage_value = getattr(selected, 'usage', None) or getattr(selected.response, 'usage', None)
        if usage_value is not None:
            selector_usage_events.append(
                usage_value.model_dump(mode='json', exclude_none=True)
                if hasattr(usage_value, 'model_dump')
                else dict(usage_value)
                if isinstance(usage_value, Mapping)
                else {'selector': usage_value}
            )
        if filter_selection == 'llm':
            time_filter = selected.time
    except Exception as exc:  # noqa: BLE001 - selector failures intentionally degrade to semantic-only planning
        selected = FilterSelectionResult(FacetSelection(), error=str(exc))
        time_filter = compiled.time if filter_selection == 'llm+jev' else None

    catalogue, selected, time_filter, effective_time, catalogue_valid = await _resolve_final_selection(
        query,
        population,
        catalogue,
        selected,
        time_filter,
        initial_start,
        initial_end,
        client=client,
        filter_selection=filter_selection,
        default_window=default_window,
        within_results=within_results,
        anchor=anchor,
        classifier_model=classifier_model,
        fast_filter_model=fast_filter_model,
        selector_usage_events=selector_usage_events,
        loader=load_catalogue,
    )
    generated_exclusive_end = (
        canonical_generated_end(time_filter, selector_output=filter_selection == 'llm')
        if effective_time is not None and time_filter is not None and time_filter.calendar_day
        else None
    )
    if initial_catalogue_error is not None:
        catalogue_valid = False

    generated_numeric = getattr(selected, 'numeric', None)
    if filter_selection == 'llm+jev':
        generated_numeric = compiled.numeric
    elif generated_numeric is None:
        generated_numeric = NumericFilters()
    generated_time = effective_time
    candidates = _literal_candidates(query, catalogue) if catalogue_valid else ()
    final_start, final_end = _candidate_window(
        population,
        effective_time,
        within_results=within_results,
        fallback_start=anchor - timedelta(days=30),
        fallback_end=anchor,
    )
    selected = await _confirm_literal_candidates(
        orq,
        selected,
        catalogue,
        candidates,
        start=final_start,
        end=final_end,
        facets=population.facets,
        numeric=population.numeric,
        confirmer=candidate_confirmer,
    )
    generated_facets = selected.selection
    facets = _merge_facets(population.facets, generated_facets)
    numeric = _merge_numeric(population.numeric, generated_numeric)
    start, end, origin, time_source, empty_match = _effective_time_scope(
        population,
        generated_time,
        within_results=within_results,
        force_empty=not catalogue_valid,
    )
    effective = population.model_copy(
        update={
            'start': start,
            'end': end,
            'time_origin': origin,
            'end_exclusive': bool(
                (
                    generated_exclusive_end is not None
                    and generated_exclusive_end.time() == datetime.min.time()
                    and end == generated_exclusive_end
                )
                or (population.end_exclusive and end == population.end)
            ),
            'facets': facets,
            'numeric': numeric,
        },
        deep=True,
    )
    response = selected.response
    usage = _aggregate_selector_usage(selector_usage_events, selected)
    last_event = cache_events[-1] if cache_events else {}
    usage['catalogue_cache'] = {
        'hit': last_event.get('hit', False),
        'elapsed_seconds': sum(event['elapsed_seconds'] for event in cache_events),
        'source': 'injected' if catalogue_loader else 'production',
        'events': cache_events,
    }
    provenance = {
        'filter_mode': filter_selection,
        'dimensions': 'compiler',
        'facets': 'explicit'
        if population.facets != FacetSelection()
        else ('llm' if filter_selection == 'llm' else 'jev'),
        'numeric': 'explicit'
        if population.numeric != NumericFilters()
        else ('compiler' if filter_selection == 'llm+jev' else 'llm'),
        'time': time_source,
    }
    return PlanningResult(
        compiled.dimensions,
        generated_facets,
        generated_numeric,
        generated_time,
        effective,
        response,
        selected.error,
        provenance,
        usage,
        compiled.unsupported_reason,
        empty_match,
    )


def _merge_facets(explicit: FacetSelection, generated: FacetSelection) -> FacetSelection:
    fields: dict[str, Any] = {name: (getattr(explicit, name) or getattr(generated, name)) for name in FACET_NAMES}
    fields['project_id'] = explicit.project_id or generated.project_id
    return FacetSelection.model_validate(fields)


def _merge_numeric(explicit: NumericFilters, generated: NumericFilters) -> NumericFilters:
    """Use each caller-provided numeric bound in preference to the generated bound."""
    fields: dict[str, int | None] = {
        name: getattr(explicit, name) if getattr(explicit, name) is not None else getattr(generated, name)
        for name in NumericFilters.model_fields
    }
    return NumericFilters.model_validate(fields)


def _effective_time_scope(
    population: PopulationRequest,
    generated_time: tuple[datetime, datetime] | None,
    *,
    within_results: bool,
    force_empty: bool = False,
) -> tuple[datetime | None, datetime | None, str, str, bool]:
    """Apply generated bounds under caller precedence and identify an empty intersection."""
    empty = force_empty or bool(generated_time and generated_time[0] >= generated_time[1])
    if within_results or population.time_origin == 'within_results':
        start = _later(population.start, generated_time[0] if generated_time else None)
        end = _earlier(population.end, generated_time[1] if generated_time else None)
        empty = empty or (start is not None and end is not None and start > end)
        if empty:
            start, end = population.start, population.end
        return start, end, 'within_results', 'within-results', empty
    if population.time_origin == 'explicit':
        time_source = 'explicit' if population.start is not None or population.end is not None else 'default'
        return population.start, population.end, 'explicit', time_source, empty
    if generated_time and not empty:
        return generated_time[0], generated_time[1], 'explicit', 'query', False
    if empty:
        return population.start, population.end, 'default', 'query', True
    return population.start, population.end, 'default', 'default', False


def _selector_query(query: str, candidates: tuple[tuple[FacetName, str], ...], catalogue: FacetCatalogue) -> str:
    """Make namespace-suffix model matches explicit in the selector's exact catalogue vocabulary."""
    available = set(catalogue.model)
    annotations = [value for name, value in candidates if name == 'model' and value in available and value not in query]
    return query if not annotations else f'{query}\nExplicit model value: {", ".join(annotations)}'


def _later(left: datetime | None, right: datetime | None) -> datetime | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _earlier(left: datetime | None, right: datetime | None) -> datetime | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _candidate_window(
    population: PopulationRequest,
    generated_time: tuple[datetime, datetime] | None,
    *,
    within_results: bool,
    fallback_start: datetime,
    fallback_end: datetime,
) -> tuple[datetime, datetime]:
    """Bound confirmation by caller bounds and the normalized (retention-clipped) generated bounds."""
    start = population.start or fallback_start
    end = population.end or fallback_end
    if generated_time is None:
        return start, end
    generated_start, generated_end = generated_time
    if within_results or population.time_origin == 'within_results':
        return max(start, generated_start), min(end, generated_end)
    if population.time_origin == 'explicit':
        return start, end
    return generated_start, generated_end


def _literal_candidates(query: str, catalogue: FacetCatalogue) -> tuple[tuple[FacetName, str], ...]:
    """Extract bounded facet-specific names, preferring longest exact available names."""
    result: list[tuple[FacetName, str]] = []
    patterns: dict[FacetName, str] = {
        'model': r'\b(?:model\s+([A-Za-z0-9_.:/-]+)|([A-Za-z0-9_.-]+/[A-Za-z0-9_.:/-]+))\b',
        'provider': r'\bprovider\s+([A-Za-z0-9_.-]+)\b',
        'agent_name': r'\bagent\s+([A-Za-z0-9_. -]+?)(?=\s+(?:with|on|from|and|where)\b|["“]|[,.;]|$)',
        'tool_name': r'\btool\s+([A-Za-z0-9_. -]+?)(?=\s+(?:with|on|from|and|where)\b|["“]|[,.;]|$)',
        'project': r'\bproject\s+([A-Za-z0-9_. -]+?)(?=\s+(?:with|on|from|and|where)\b|["“]|[,.;]|$)',
    }
    for name, pattern in patterns.items():
        for match in re.finditer(pattern, query, re.IGNORECASE):
            value = next((group for group in match.groups() if group is not None), '').strip()
            values: tuple[str, ...] = getattr(catalogue, name)
            exact = next((available for available in values if available.casefold() == value.casefold()), None)
            if name == 'model' and exact is None and '/' in value:
                suffix = value.rsplit('/', 1)[-1].casefold()
                matches = [available for available in values if available.casefold().endswith('/' + suffix)]
                if matches:
                    exact = max(matches, key=len)
            candidate = exact or value
            if candidate and (name, candidate) not in result:
                result.append((name, candidate))
    for match in re.findall(r'["“]([^"”]{1,100})["”]', query):
        candidate = match.strip()
        if candidate:
            for name in FACET_NAMES:
                pair = (name, candidate)
                if pair not in result:
                    result.append(pair)
                if len(result) >= 8:
                    break
        if len(result) >= 8:
            break
    return tuple(result[:8])


async def _offer_confirmed_candidates(
    orq: Orq,
    catalogue: FacetCatalogue,
    candidates: tuple[tuple[FacetName, str], ...],
    *,
    start: datetime,
    end: datetime,
    facets: FacetSelection | None = None,
    numeric: NumericFilters | None = None,
    confirmer: Callable[..., Awaitable[bool]] | None = None,
) -> FacetCatalogue:
    """Offer literal out-of-catalogue values only after exact scoped OQL finds a trace."""
    if start > end:
        return catalogue
    facets = facets if facets is not None else FacetSelection()
    numeric = numeric if numeric is not None else NumericFilters()
    offered: dict[str, list[str]] = {name: list(getattr(catalogue, name)) for name in FACET_NAMES}
    pending = tuple((name, value) for name, value in candidates if value not in offered[name])
    confirmed: dict[str, list[str]] = {name: [] for name in FACET_NAMES}
    source = OrqTraceSource(orq) if confirmer is None else None
    try:
        for name, value in pending:
            try:
                if confirmer is not None:
                    found = await confirmer(orq, name, value, start, end)
                elif source is not None:
                    rows = await source.search(
                        start,
                        end,
                        1,
                        facets=_merge_facets(facets, FacetSelection.model_validate({name: frozenset({value})})),
                        numeric=numeric,
                    )
                    found = bool(rows)
                else:
                    raise RuntimeError('candidate confirmation source was not initialized')
            except Exception as exc:  # noqa: BLE001 - individual confirmation errors are reported and skipped
                logger.warning('Could not confirm explicitly named {} facet value {!r}: {}', name, value, exc)
                continue
            if found:
                confirmed[name].append(value)
            else:
                logger.info('Ignoring unconfirmed explicitly named {} facet value {!r}', name, value)
    finally:
        if source is not None:
            source.close()
    return catalogue.model_copy(
        update={name: tuple((*offered[name], *confirmed[name])) for name in FACET_NAMES}, deep=True
    )


def _selection_with_candidate(selected: FilterSelectionResult, name: FacetName, value: str) -> FilterSelectionResult:
    fields: dict[FacetName, Any] = {facet: getattr(selected.selection, facet) for facet in FACET_NAMES}
    fields[name] = frozenset((*fields[name], value))
    return FilterSelectionResult(
        FacetSelection.model_validate(fields),
        selected.response,
        selected.error,
        selected.numeric,
        selected.time,
        selected.usage,
    )


async def _select_from_catalogue(
    query: str,
    catalogue: FacetCatalogue,
    *,
    client: AsyncOpenAI,
    filter_selection: Literal['llm', 'llm+jev'],
    classifier_model: str,
    fast_filter_model: str,
    now: datetime,
) -> FilterSelectionResult:
    candidates = _literal_candidates(query, catalogue)
    selector_query = _selector_query(query, candidates, catalogue)
    if filter_selection == 'llm+jev':
        return await select_filters_with_response(client, classifier_model, catalogue, selector_query)
    return await select_llm_filters_with_response(
        client,
        fast_filter_model,
        catalogue,
        selector_query,
        now=now,
    )


async def _resolve_final_selection(
    query: str,
    population: PopulationRequest,
    catalogue: FacetCatalogue,
    selected: FilterSelectionResult,
    time_filter: Any,
    initial_start: datetime,
    initial_end: datetime,
    *,
    client: AsyncOpenAI,
    filter_selection: Literal['llm', 'llm+jev'],
    default_window: bool,
    within_results: bool,
    anchor: datetime,
    classifier_model: str,
    fast_filter_model: str,
    selector_usage_events: list[dict[str, Any]],
    loader: Callable[[datetime, datetime], Awaitable[FacetCatalogue]],
) -> tuple[FacetCatalogue, FilterSelectionResult, Any, tuple[datetime, datetime] | None, bool]:
    """Normalize time and reselect only against the final generated-time catalogue window."""
    generated_time = normalize_generated_time(
        selected.time if filter_selection == 'llm' else time_filter,
        population=population,
        within_results=within_results,
        reference=anchor,
        selector_output=filter_selection == 'llm',
    )
    if generated_time and generated_time[0] >= generated_time[1]:
        return (
            FacetCatalogue(),
            FilterSelectionResult(FacetSelection(), error='generated time window has no retained overlap'),
            time_filter,
            generated_time,
            False,
        )
    reselect = (
        filter_selection == 'llm'
        and bool(generated_time)
        and default_window
        and not (within_results or population.time_origin == 'within_results')
        and generated_time != (initial_start, initial_end)
    )
    if not reselect:
        return catalogue, selected, time_filter, generated_time, True
    try:
        if generated_time is None:
            return (
                FacetCatalogue(),
                FilterSelectionResult(FacetSelection(), error='selector time disappeared before catalogue reload'),
                None,
                None,
                False,
            )
        catalogue = await loader(*generated_time)
        selected = await _select_from_catalogue(
            query,
            catalogue,
            client=client,
            filter_selection='llm',
            classifier_model=classifier_model,
            fast_filter_model=fast_filter_model,
            now=anchor,
        )
        if selected.usage is not None:
            selector_usage_events.append(selected.usage.model_dump(mode='json', exclude_none=True))
        second_time = normalize_generated_time(
            selected.time,
            population=population,
            within_results=within_results,
            reference=anchor,
            selector_output=True,
        )
        if second_time != generated_time:
            return (
                FacetCatalogue(),
                FilterSelectionResult(
                    FacetSelection(), error='selector changed its time window after catalogue reselection'
                ),
                None,
                None,
                False,
            )
        return catalogue, selected, selected.time, second_time, True
    except Exception as exc:  # noqa: BLE001 - never return facets from the stale default-window catalogue
        logger.warning('Trace filter catalogue reselection failed: {}', exc)
        return (
            FacetCatalogue(),
            FilterSelectionResult(FacetSelection(), error=f'catalogue reselection failed: {exc}'),
            None,
            None,
            False,
        )


async def _confirm_literal_candidates(
    orq: Orq,
    selected: FilterSelectionResult,
    catalogue: FacetCatalogue,
    candidates: tuple[tuple[FacetName, str], ...],
    *,
    start: datetime,
    end: datetime,
    facets: FacetSelection,
    numeric: NumericFilters,
    confirmer: Callable[..., Awaitable[bool]] | None,
) -> FilterSelectionResult:
    confirmed = await _offer_confirmed_candidates(
        orq,
        catalogue,
        candidates,
        start=start,
        end=end,
        facets=facets,
        numeric=numeric,
        confirmer=confirmer,
    )
    for name, value in candidates:
        if value not in getattr(catalogue, name) and value in getattr(confirmed, name):
            selected = _selection_with_candidate(selected, name, value)
    return selected


def _aggregate_selector_usage(events: list[dict[str, Any]], selected: FilterSelectionResult) -> dict[str, Any]:
    """Sum usage across all successful selector requests and internal retries."""
    usage: dict[str, Any] = {}
    for event in events:
        for key, value in event.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage[key] = usage.get(key, 0) + value
            else:
                usage[key] = value
    if events:
        return usage
    value = getattr(selected, 'usage', None)
    if value is None:
        return usage
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json', exclude_none=True)
    return dict(value) if isinstance(value, Mapping) else {'selector': value}
