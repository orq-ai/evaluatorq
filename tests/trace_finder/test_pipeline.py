from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, cast

import pytest

from evaluatorq.trace_finder import FacetCatalogue, FacetSelection, PopulationRequest, Snapshot, pipeline
from evaluatorq.trace_finder.filter_selector import FilterSelectionResult
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

    async def fake_select(client: Any, model: str, catalogue: FacetCatalogue, query: str) -> FilterSelectionResult:
        return FilterSelectionResult(FacetSelection())

    monkeypatch.setattr(pipeline, 'load_facet_catalogue', fake_catalogue)
    monkeypatch.setattr(pipeline, 'select_filters_with_response', fake_select)
    settings = DashboardSettings(window_days=3, limit=500, parallelism=100)
    store = pipeline.build_run_store(settings, client=cast(Any, object()), orq=cast(Any, object()))

    await store._filter_selector('refunds', PopulationRequest())

    assert seen['end'] - seen['start'] == timedelta(days=3)
    assert seen['end'] <= datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_population_loader_uses_the_configured_window_when_bounds_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, datetime] = {}

    class FakeSource:
        def __init__(self, orq: Any) -> None:
            del orq

        async def load_async(self, start: datetime, end: datetime, *args: Any, **kwargs: Any) -> Snapshot:
            seen['start'] = start
            seen['end'] = end
            return Snapshot(traces=())

        def close(self) -> None:
            pass

    monkeypatch.setattr(pipeline, 'OrqTraceSource', FakeSource)
    store = pipeline.build_run_store(
        DashboardSettings(window_days=3, limit=500, parallelism=100),
        client=cast(Any, object()),
        orq=cast(Any, object()),
    )

    await store._population_loader(PopulationRequest())

    assert seen['end'] - seen['start'] == timedelta(days=3)


@pytest.mark.asyncio
async def test_filter_selector_announces_degradation_and_keeps_semantic_run_available(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def unavailable(*args: Any, **kwargs: Any) -> FacetCatalogue:
        raise RuntimeError('facet service unavailable')

    monkeypatch.setattr(pipeline, 'load_facet_catalogue', unavailable)
    store = pipeline.build_run_store(
        DashboardSettings(window_days=7, limit=500, parallelism=100),
        client=cast(Any, object()),
        orq=cast(Any, object()),
    )

    with caplog.at_level('WARNING'):
        selected = await store._filter_selector('refunds', PopulationRequest())

    assert selected == FilterSelectionResult(FacetSelection(), error='facet service unavailable')
    assert 'facet service unavailable' in caplog.text
