"""Shared runtime wiring for dashboard and CLI trace-finder runs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import partial
from typing import TYPE_CHECKING

from evaluatorq.common.model_roles import role_model

from .classifier import run_classifier
from .explorer import ExplorerStore
from .orq_source import OrqTraceSource
from .planner import PlanningResult, plan_query
from .projection import project_trace
from .run_store import RunStore

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from openai import AsyncOpenAI
    from orq_ai_sdk import Orq

    from .models import PopulationRequest, Snapshot
    from .settings import DashboardSettings


def build_run_store(
    settings: DashboardSettings,
    *,
    client: AsyncOpenAI,
    orq: Orq,
    cleanup: Callable[[], Awaitable[None]] | None = None,
) -> RunStore:
    """Build the shared trace-finder pipeline for one application runtime."""
    source = OrqTraceSource(orq)
    explorer = ExplorerStore(
        search=source.search,
        hydrate=source.hydrate_rows,
        load_spans=source.list_spans,
        load_first_error_message=source.first_error_message,
    )

    async def close() -> None:
        try:
            source.close()
        finally:
            if cleanup is not None:
                await cleanup()

    # The shared planner and classifier intentionally use separate task roles.
    classifier_model = role_model('classifier', task='finder.classifier')
    compiler_model = role_model('fast', task='finder.compiler')
    fast_filter_model = role_model('fast', task='finder.filters')

    async def planner(query: str, population: PopulationRequest, *, within_results: bool) -> PlanningResult:
        end = population.end or datetime.now(timezone.utc)
        start = population.start or end - timedelta(days=settings.window_days)
        planning_population = population.model_copy(update={'start': start, 'end': end})
        return await plan_query(
            query,
            planning_population,
            client=client,
            orq=orq,
            compiler_model=compiler_model,
            filter_selection=settings.filter_selection,
            classifier_model=classifier_model,
            fast_filter_model=fast_filter_model,
            within_results=within_results,
        )

    async def population_loader(request: PopulationRequest) -> Snapshot:
        end = request.end or datetime.now(timezone.utc)
        start = request.start or end - timedelta(days=settings.window_days)
        return await source.load_async(
            start,
            end,
            request.limit,
            facets=request.facets,
            numeric=request.numeric,
        )

    return RunStore(
        planner=planner,
        population_loader=population_loader,
        run_classifier=partial(
            run_classifier,
            model=classifier_model,
            client=client,
            trace_input_chars=settings.trace_input_chars,
        ),
        project_trace=partial(
            project_trace,
            model=classifier_model,
            trace_input_chars=settings.trace_input_chars,
        ),
        close=close,
        explorer=explorer,
        classifier_model=classifier_model,
    )
