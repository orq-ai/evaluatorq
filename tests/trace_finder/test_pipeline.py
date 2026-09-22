from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, cast

import pytest

from evaluatorq.trace_finder import FacetCatalogue, FacetSelection, PopulationRequest, pipeline
from evaluatorq.trace_finder.settings import DashboardSettings


@pytest.mark.asyncio
async def test_filter_selector_falls_back_to_the_settings_window_when_bounds_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, datetime] = {}

    async def fake_catalogue(orq: Any, *, start: datetime, end: datetime, limit: int) -> FacetCatalogue:
        seen['start'] = start
        seen['end'] = end
        return FacetCatalogue()

    async def fake_select(client: Any, model: str, catalogue: FacetCatalogue, query: str) -> FacetSelection:
        return FacetSelection()

    monkeypatch.setattr(pipeline, 'load_facet_catalogue', fake_catalogue)
    monkeypatch.setattr(pipeline, 'select_filters', fake_select)
    settings = DashboardSettings(window_days=3, limit=500, parallelism=100)
    store = pipeline.build_run_store(settings, client=cast(Any, object()), orq=cast(Any, object()))

    await store._filter_selector('refunds', PopulationRequest())

    assert seen['end'] - seen['start'] == timedelta(days=3)
    assert seen['end'] <= datetime.now(timezone.utc)
