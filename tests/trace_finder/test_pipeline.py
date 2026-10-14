from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, cast

import pytest

from evaluatorq.trace_finder import PopulationRequest, Snapshot, pipeline
from evaluatorq.trace_finder.settings import DashboardSettings




@pytest.mark.asyncio
async def test_factory_keeps_classifier_model_bound_to_the_store_after_role_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected: list[str] = []

    async def fake_plan(query: str, population: PopulationRequest, **kwargs: Any) -> object:
        selected.append(kwargs['classifier_model'])
        return object()

    monkeypatch.setattr(pipeline, 'plan_query', fake_plan)
    monkeypatch.setenv('EVALUATORQ_CLASSIFIER_MODEL', 'acme/built-with')
    store = pipeline.build_run_store(DashboardSettings(), client=cast(Any, object()), orq=cast(Any, object()))
    monkeypatch.setenv('EVALUATORQ_CLASSIFIER_MODEL', 'acme/changed-later')

    planner = store._planner
    assert planner is not None
    await planner('refunds', PopulationRequest(), within_results=False)

    assert selected == ['acme/built-with']
    assert (await store.snapshot()).classifier_model == 'acme/built-with'


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

        async def search(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
            del args, kwargs
            return ()

        async def hydrate_rows(self, rows: Any) -> dict[str, Any]:
            del rows
            return {}

        async def list_spans(self, trace_id: str) -> list[Any]:
            del trace_id
            return []

        async def first_error_message(self, trace_id: str, spans: Any) -> str | None:
            del trace_id, spans
            return None

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
