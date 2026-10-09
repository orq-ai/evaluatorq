"""Resolve an `InsightsPopulation` into local traces or Orq traces via `trace_finder`.

Four mutually exclusive paths, matching `InsightsPopulation`'s own fields:

- **Query path** (`pop.query` set): compile the semantic query with `compile_query`
  and select generated facet/numeric filters with `select_filters_with_response`
  concurrently (mirrors `trace_finder/run_store.py`'s `_plan`), merge them under the
  caller's explicit facets/numeric (explicit wins through `run_store.merge_facets`/
  `merge_numeric`), then
  load with `OrqTraceSource`.
- **Finder export path** (`pop.finder_export` set): read a `RunExport` JSON, reload
  its merged filters and the pinned matches' timestamp range through `OrqTraceSource`,
  and keep only the traces whose ids are in `matched_trace_ids`. No compile, no match
  question — the export already pins the matched population.
- **Local snapshot path** (`pop.snapshot_path` set): validate a `Snapshot` JSON and
  use its embedded trace messages and ATIF session documents directly. No Orq trace fetch or match question.
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
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.common.model_input import (
    fit_classifier_input,
    has_truncated_source_text,
    is_jev_model,
    serialized_chars,
    serialized_question_chars,
)
from evaluatorq.common.model_roles import role_model
from evaluatorq.common.trace_document import TraceDocument, ensure_trace_document, prompt_messages
from evaluatorq.insights.labeling import MATCH_KEY, prepare_classifier_state
from evaluatorq.insights.models import ensure_unique_trace_ids
from evaluatorq.insights.summarize import _build_prompt_with_coverage
from evaluatorq.trace_finder.compiler import compile_query
from evaluatorq.trace_finder.export import RunExport
from evaluatorq.trace_finder.facets import load_facet_catalogue
from evaluatorq.trace_finder.filter_selector import select_filters_with_response
from evaluatorq.trace_finder.models import FacetSelection, NumericFilters, Snapshot
from evaluatorq.trace_finder.orq_source import OrqTraceSource
from evaluatorq.trace_finder.projection import serialize_projection
from evaluatorq.trace_finder.run_store import merge_facets, merge_numeric
from evaluatorq.trace_finder.settings import effective_settings

if TYPE_CHECKING:
    from collections.abc import Sequence

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

    traces: list[TraceDocument]
    compiled: tuple[CompiledQuery, ...] | None
    echo: dict[str, Any]
    n_scanned: int


def projection_coverage(
    traces: Sequence[TraceRecord | TraceDocument],
    *,
    trace_input_chars: int | None = None,
    classifier_model: str | None = None,
    classifier_questions: Sequence[dict[str, Any]] | Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Measure the summary prompt and the same classifier state the Insights run prepares."""
    settings = effective_settings()
    cap = trace_input_chars if trace_input_chars is not None else settings.trace_input_chars
    model = classifier_model or role_model('classifier', task='insights.labels')
    state_format = 'jev' if is_jev_model(model) else 'text'
    reserve = 'applied' if classifier_questions is not None else 'not_available'
    coverage: dict[str, Any] = {
        'n_traces': len(traces),
        'n_projection_truncated': 0,
        'n_summary_truncated': 0,
        'n_classifier_truncated': 0,
        'n_source_messages': 0,
        'n_omitted_messages': 0,
        'source_chars': 0,
        'classifier_state_chars': 0,
        'classifier_question_chars': 0,
        'classifier_request_chars': 0,
        'trace_input_chars': cap,
        'summary_input_chars': 0,
        'classifier_state_format': state_format,
        'classifier_question_reserve': reserve,
    }
    questions = classifier_questions
    question_chars = serialized_question_chars(questions) if questions is not None else 0
    for trace in traces:
        messages = prompt_messages(trace)
        prompt, summary_truncated = _build_prompt_with_coverage(trace, input_char_cap=cap)
        state = prepare_classifier_state(trace, model=model, char_cap=cap)
        if questions is not None:
            state = fit_classifier_input(
                state,
                model=model,
                questions=questions,
                global_char_cap=cap,
            )
        state_chars = serialized_chars(state) if isinstance(state, dict) else len(state)
        request_chars = state_chars + question_chars
        omitted = 0
        if isinstance(state, dict):
            represented: set[int] = set()
            for entry in state.get('messages', []):
                index = entry.get('index') if isinstance(entry, dict) else None
                if isinstance(index, int):
                    represented.add(index)
                if isinstance(entry, dict):
                    represented.update(
                        result['index']
                        for result in entry.get('results', [])
                        if isinstance(result, dict) and isinstance(result.get('index'), int)
                    )
            omitted = max(0, len(messages) - len(represented))
        state_truncated = omitted > 0 or has_truncated_source_text(state)
        coverage['n_summary_truncated'] += int(summary_truncated)
        coverage['n_classifier_truncated'] += int(state_truncated)
        coverage['n_projection_truncated'] += int(summary_truncated or state_truncated)
        coverage['n_source_messages'] += len(messages)
        coverage['n_omitted_messages'] += omitted
        coverage['source_chars'] += len(serialize_projection({'trace_status': trace.status, 'messages': messages}))
        coverage['classifier_state_chars'] += state_chars
        coverage['classifier_question_chars'] += question_chars
        coverage['classifier_request_chars'] += request_chars
        coverage['summary_input_chars'] += len(prompt)
    return coverage


def describe_projection_coverage(coverage: dict[str, Any]) -> str:
    """Describe measured rendered character sizes without presenting estimates as tokenizer counts."""
    traces = int(coverage['n_traces'])
    affected = int(coverage['n_projection_truncated'])
    input_cap = int(coverage.get('trace_input_chars', effective_settings().trace_input_chars))
    avg_summary = int(coverage.get('summary_input_chars', 0)) // max(traces, 1)
    avg_classifier = int(coverage.get('classifier_state_chars', 0)) // max(traces, 1)
    avg_classifier_questions = int(coverage.get('classifier_question_chars', 0)) // max(traces, 1)
    avg_classifier_request = int(coverage.get('classifier_request_chars', 0)) // max(traces, 1)
    detail = (
        f'{int(coverage.get("n_omitted_messages", 0)):,} of {int(coverage.get("n_source_messages", 0)):,} '
        'source messages were omitted from the Jev classifier state.'
        if coverage.get('classifier_state_format') == 'jev'
        else 'The non-Jev classifier receives a readable transcript shortened as needed to fit its questions.'
    )
    return (
        f'{affected:,} of {traces:,} traces hit a summary or classifier input cap of {input_cap:,} characters. '
        f'Average rendered inputs: summary {avg_summary:,} characters; classifier request {avg_classifier_request:,} '
        f'(state {avg_classifier:,}, questions {avg_classifier_questions:,}). {detail}'
    )


def snapshot_documents(snapshot: Snapshot) -> list[TraceDocument]:
    """Every trace of a snapshot as a `TraceDocument`: its trace records first, then its ATIF documents."""
    return [*(ensure_trace_document(trace) for trace in snapshot.traces), *snapshot.documents]


def format_size(n_bytes: int) -> str:
    """Render a byte count with one decimal in KB or MB."""
    if n_bytes >= 1024 * 1024:
        return f'{n_bytes / (1024 * 1024):.1f} MB'
    return f'{n_bytes / 1024:.1f} KB'


def describe_local_send(
    *, n_traces: int, n_bytes: int, file_name: str, models: Mapping[str, str], preview: bool = False
) -> str:
    """State what a local-snapshot run sends to which models; `models` maps summary/classifier/embedding to ids.

    `preview` words it as what a run would send, for output printed without starting one.
    """
    noun = 'trace' if n_traces == 1 else 'traces'
    verb = 'A run would send' if preview else 'Sending'
    return (
        f'{verb} {n_traces:,} {noun} ({format_size(n_bytes)}) from {file_name} to models: '
        f'summary {models["summary"]}, classifier {models["classifier"]}, embedding {models["embedding"]}.'
    )


def preview_snapshot(
    raw_path: str,
    *,
    trace_input_chars: int | None = None,
    classifier_model: str | None = None,
    classifier_questions: Sequence[dict[str, Any]] | Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return coverage from the same summary and classifier renderers used by a run."""
    path = Path(raw_path).expanduser()
    try:
        snapshot = Snapshot.model_validate_json(path.read_bytes())
        return projection_coverage(
            snapshot_documents(snapshot),
            trace_input_chars=trace_input_chars,
            classifier_model=classifier_model,
            classifier_questions=classifier_questions,
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise PopulationError(f'loading local trace snapshot {path} failed: {error}') from error


def _resolve_from_snapshot(pop: InsightsPopulation) -> ResolvedPopulation:
    """Read a complete local snapshot; this path makes no Orq or model calls."""
    assert pop.snapshot_path is not None  # noqa: S101 - guarded by the caller's dispatch
    try:
        raw = pop.snapshot_path.read_bytes()
        snapshot = Snapshot.model_validate_json(raw)
    except (OSError, ValueError) as error:
        raise PopulationError(f'loading local trace snapshot {pop.snapshot_path} failed: {error}') from error
    traces = snapshot_documents(snapshot)
    return ResolvedPopulation(
        traces=traces,
        compiled=None,
        echo={
            'mode': 'snapshot',
            'snapshot_path': str(pop.snapshot_path),
            'snapshot_sha256': hashlib.sha256(raw).hexdigest(),
            'limit': len(traces),
        },
        n_scanned=len(traces),
    )


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
    target_trace_ids: set[str] | None = None,
) -> tuple[TraceDocument, ...]:
    """Load a population's traces through `OrqTraceSource`; any failure becomes a `PopulationError`."""
    source = OrqTraceSource(orq)
    try:
        if target_trace_ids is None:
            snapshot = await source.load_async(start, end, limit, facets=facets, numeric=numeric)
        else:
            snapshot = await source.load_async(
                start,
                end,
                limit,
                facets=facets,
                numeric=numeric,
                target_trace_ids=target_trace_ids,
            )
        traces = await source.enrich_selected_signal_spans(snapshot.traces)
    except Exception as error:
        raise PopulationError(f'loading the trace population failed: {error}') from error
    finally:
        try:
            source.close()
        except Exception as close_error:  # noqa: BLE001 - cleanup must not mask the load result or error
            logger.warning('Insights population trace source cleanup failed: {}', close_error)
    if target_trace_ids is not None and snapshot.capture_metadata.get('incomplete_reason'):
        raise PopulationError(
            f'reloading Finder export traces was incomplete: {snapshot.capture_metadata["incomplete_reason"]}'
        )
    documents = tuple(ensure_trace_document(trace) for trace in traces)
    try:
        ensure_unique_trace_ids(documents)
    except ValueError as error:
        raise PopulationError(str(error)) from error
    return documents


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

    merged_facets = merge_facets(pop.facets, generated_facets)
    merged_numeric = merge_numeric(pop.numeric, plan.numeric)

    traces = await _load_traces(
        orq, start=start, end=end, limit=pop.limit, facets=merged_facets, numeric=merged_numeric
    )

    echo: dict[str, Any] = {
        'mode': 'query',
        'query': pop.query,
        'compiled_dimensions': [
            {
                'name': dimension.name,
                'task': dimension.task.instructions,
                'selection': dimension.selection.model_dump(mode='json'),
            }
            for dimension in plan.dimensions
        ],
        'facets': merged_facets.model_dump(mode='json'),
        'generated_facets': generated_facets.model_dump(mode='json'),
        'numeric': merged_numeric.model_dump(mode='json'),
        'generated_numeric': plan.numeric.model_dump(mode='json'),
        'start': start.isoformat(),
        'end': end.isoformat(),
        'limit': pop.limit,
        'filter_selection_error': filter_error,
    }
    return ResolvedPopulation(
        traces=[ensure_trace_document(trace) for trace in traces],
        compiled=plan.dimensions,
        echo=echo,
        n_scanned=len(traces),
    )


async def _resolve_from_export(pop: InsightsPopulation, *, orq: Orq) -> ResolvedPopulation:
    assert pop.finder_export is not None  # noqa: S101 - guarded by the caller's dispatch before this is reached

    export = pop.finder_export_snapshot()
    if export is None:
        try:
            export = RunExport.model_validate_json(pop.finder_export.read_text())
        except Exception as error:
            raise PopulationError(f'loading finder export {pop.finder_export} failed: {error}') from error

    facets = merge_facets(_facets_from_export(export.filters), _facets_from_export(export.generated_filters))
    numeric = merge_numeric(
        NumericFilters(**export.numeric.model_dump()), NumericFilters(**export.generated_numeric.model_dump())
    )

    # Restrict the reload to the exported matches' timestamps. A rolling window
    # or later arrivals must not push those pinned IDs past the source's limit.
    matched_ids = set(export.matched_trace_ids)
    matched_times = [trace.timestamp for trace in export.traces if trace.trace_id in matched_ids]
    start, end = export.start, export.end
    if matched_times:
        first = min(matched_times) - timedelta(seconds=1)
        last = max(matched_times) + timedelta(seconds=1)
        start = max(start, first) if start is not None else first
        end = min(end, last) if end is not None else last

    traces = (
        await _load_traces(
            orq,
            start=start,
            end=end,
            limit=max(export.limit, len(matched_ids)),
            facets=facets,
            numeric=numeric,
            target_trace_ids=matched_ids,
        )
        if matched_ids
        else ()
    )

    by_id = {trace.trace_id: trace for trace in traces}
    resolved: list[TraceDocument] = []
    missing = 0
    for trace_id in export.matched_trace_ids:
        trace = by_id.get(trace_id)
        if trace is None:
            missing += 1
            continue
        resolved.append(ensure_trace_document(trace))
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
        'finder_export': str(pop.finder_export.resolve()),
        'facets': facets.model_dump(mode='json'),
        'numeric': numeric.model_dump(mode='json'),
        'start': start.isoformat() if start else None,
        'end': end.isoformat() if end else None,
        'limit': max(export.limit, len(matched_ids)),
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
    return ResolvedPopulation(
        traces=[ensure_trace_document(trace) for trace in traces], compiled=None, echo=echo, n_scanned=len(traces)
    )


async def resolve_population(
    pop: InsightsPopulation,
    *,
    orq: Orq | None,
    client: AsyncOpenAI | None,
    compiler_model: str,
    classifier_model: str,
    trace_input_chars: int | None = None,
    label_classifier_model: str | None = None,
    classifier_questions: Sequence[dict[str, Any]] | Mapping[str, dict[str, Any]] | None = None,
) -> ResolvedPopulation:
    """Resolve `pop` into traces via a local file or the Orq-backed paths — see module docstring.

    `client` and the models are only used on the query path
    (`compile_query` and `select_filters_with_response`, respectively); the
    other paths never call the LLM during population selection.
    """
    if pop.snapshot_path is not None:
        resolved = _resolve_from_snapshot(pop)
    elif pop.finder_export is not None:
        if orq is None:
            raise PopulationError('a Finder export population requires an Orq client')
        resolved = await _resolve_from_export(pop, orq=orq)
    elif pop.query is not None:
        if orq is None:
            raise PopulationError('a query population requires an Orq client')
        if client is None:
            raise PopulationError('a query population requires an LLM client')
        resolved = await _resolve_from_query(
            pop, orq=orq, client=client, compiler_model=compiler_model, classifier_model=classifier_model
        )
    else:
        if orq is None:
            raise PopulationError('a filter population requires an Orq client')
        resolved = await _resolve_from_filters(pop, orq=orq)
    try:
        ensure_unique_trace_ids(resolved.traces)
    except ValueError as error:
        raise PopulationError(str(error)) from error
    questions = (
        dict(classifier_questions)
        if isinstance(classifier_questions, Mapping)
        else (
            {f'question_{index}': question for index, question in enumerate(classifier_questions)}
            if classifier_questions is not None
            else None
        )
    )
    if questions is not None and resolved.compiled:
        dimension_count = len(resolved.compiled)
        for index, dimension in enumerate(resolved.compiled):
            key = MATCH_KEY if dimension_count == 1 else f'{MATCH_KEY[:-2]}_{index}__'
            questions[key] = dimension.task.model_dump(mode='json')
    try:
        resolved.echo.update(
            projection_coverage(
                resolved.traces,
                trace_input_chars=trace_input_chars,
                classifier_model=label_classifier_model,
                classifier_questions=questions,
            )
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PopulationError(f'projecting loaded traces failed: {error}') from error
    truncated = resolved.echo['n_projection_truncated']
    if truncated:
        trace_noun = 'trace' if len(resolved.traces) == 1 else 'traces'
        logger.warning(
            'Insights {} population has {} of {} {} with shortened summary or classifier input',
            resolved.echo['mode'],
            truncated,
            len(resolved.traces),
            trace_noun,
        )
    return resolved
