"""Shared runtime wiring for dashboard and CLI trace-finder runs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import partial
from typing import TYPE_CHECKING

from .compiler import compile_query
from .facets import load_facet_catalogue
from .filter_selector import select_filters
from .jev import run_jev
from .orq_source import OrqTraceSource
from .run_store import RunStore

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from orq_ai_sdk import Orq

    from .models import FacetSelection, PopulationRequest, Snapshot
    from .settings import DashboardSettings


def build_run_store(settings: DashboardSettings, *, client: AsyncOpenAI, orq: Orq) -> RunStore:
    """Build the shared trace-finder pipeline for one application runtime."""
    source = OrqTraceSource(orq)

    async def filter_selector(query: str, request: PopulationRequest) -> FacetSelection:
        end = request.end or datetime.now(timezone.utc)
        start = request.start or end - timedelta(days=settings.window_days)
        catalogue = await load_facet_catalogue(orq, start=start, end=end, limit=50)
        return await select_filters(client, settings.jev_model, catalogue, query)

    async def population_loader(request: PopulationRequest) -> Snapshot:
        return await source.load_async(
            request.start,
            request.end,
            request.limit,
            facets=request.facets,
            numeric=request.numeric,
        )

    return RunStore(
        compiler=partial(compile_query, client, settings.compiler_model),
        filter_selector=filter_selector,
        population_loader=population_loader,
        run_jev=partial(run_jev, model=settings.jev_model, client=client),
    )
