"""Shared runtime wiring for dashboard and CLI trace-finder runs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import partial
from typing import TYPE_CHECKING

from loguru import logger

from .compiler import compile_query
from .facets import load_facet_catalogue
from .filter_selector import select_filters
from .jev import run_jev
from .models import FacetSelection
from .orq_source import OrqTraceSource
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

    async def close() -> None:
        try:
            source.close()
        finally:
            if cleanup is not None:
                await cleanup()

    async def filter_selector(query: str, request: PopulationRequest) -> FacetSelection:
        end = request.end or datetime.now(timezone.utc)
        start = request.start or end - timedelta(days=settings.window_days)
        try:
            catalogue = await load_facet_catalogue(orq, start=start, end=end, limit=50)
            return await select_filters(client, settings.jev_model, catalogue, query)
        except Exception as error:  # noqa: BLE001 - explicit filters and semantic classification remain available
            logger.warning('Trace filter selection unavailable or incomplete: {}; skipping generated filters', error)
            return FacetSelection()

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
        compiler=partial(compile_query, client, settings.compiler_model),
        filter_selector=filter_selector,
        population_loader=population_loader,
        run_jev=partial(run_jev, model=settings.jev_model, client=client),
        close=close,
    )
