import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from evaluatorq.trace_finder.compiler import TimeFilters
from evaluatorq.trace_finder.facets import invalidate_facet_catalogue_cache, load_facet_catalogue_cached
from evaluatorq.trace_finder.filter_selector import FilterSelectionResult
from evaluatorq.trace_finder.models import FacetCatalogue, FacetSelection, NumericFilters, PopulationRequest
from evaluatorq.trace_finder.planner import (
    _earlier,
    _later,
    _literal_candidates,
)


def _client(key='secret', workspace='one'):
    return SimpleNamespace(sdk_configuration=SimpleNamespace(
        security=SimpleNamespace(api_key=key),
        async_client=SimpleNamespace(headers={'X-Orq-Workspace-Id': workspace}),
        get_server_details=lambda: ('https://example.test', {}),
    ))


@pytest.mark.asyncio
async def test_catalogue_cache_reuses_same_scope_and_bounds(monkeypatch):
    from evaluatorq.trace_finder import facets

    client = _client()
    load = AsyncMock(return_value=FacetCatalogue(model=('provider/model',)))
    monkeypatch.setattr(facets, 'load_facet_catalogue', load)
    await invalidate_facet_catalogue_cache()
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 2, tzinfo=timezone.utc)
    first = await load_facet_catalogue_cached(client, start, end)
    second = await load_facet_catalogue_cached(client, start, end)
    assert first == second
    assert load.await_count == 1
    await invalidate_facet_catalogue_cache()


@pytest.mark.asyncio
async def test_catalogue_cache_isolates_scope_and_expires(monkeypatch):
    from evaluatorq.trace_finder import facets

    first, other_scope, other_credential = _client(), _client(workspace='two'), _client(key='rotated')
    load = AsyncMock(return_value=FacetCatalogue())
    monkeypatch.setattr(facets, 'load_facet_catalogue', load)
    await invalidate_facet_catalogue_cache()
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 2, tzinfo=timezone.utc)
    await load_facet_catalogue_cached(first, start, end)
    await load_facet_catalogue_cached(other_scope, start, end)
    await load_facet_catalogue_cached(other_credential, start, end)
    assert load.await_count == 3
    await invalidate_facet_catalogue_cache()


@pytest.mark.asyncio
async def test_catalogue_cache_expires_after_five_minutes(monkeypatch):
    from evaluatorq.trace_finder import facets

    client = _client()
    load = AsyncMock(return_value=FacetCatalogue())
    monkeypatch.setattr(facets, 'load_facet_catalogue', load)
    clock = [10.0]

    def monotonic():
        return clock[0]

    monkeypatch.setattr(facets.time, 'monotonic', monotonic)
    await invalidate_facet_catalogue_cache()
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 2, tzinfo=timezone.utc)
    await load_facet_catalogue_cached(client, start, end)
    clock[0] += 1.0
    await load_facet_catalogue_cached(client, start, end)
    clock[0] += 300.0
    await load_facet_catalogue_cached(client, start, end)
    assert load.await_count == 2
    await invalidate_facet_catalogue_cache()


@pytest.mark.asyncio
async def test_default_window_cache_key_is_stable_across_now_values(monkeypatch):
    from evaluatorq.trace_finder import facets

    client = _client()
    load = AsyncMock(return_value=FacetCatalogue())
    monkeypatch.setattr(facets, 'load_facet_catalogue', load)
    await invalidate_facet_catalogue_cache()
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 2, tzinfo=timezone.utc)
    await load_facet_catalogue_cached(
        client, start, end, default_window=True, now=datetime(2026, 1, 2, tzinfo=timezone.utc)
    )
    await load_facet_catalogue_cached(
        client, start, end, default_window=True, now=datetime(2026, 1, 2, 0, 1, tzinfo=timezone.utc)
    )
    assert load.await_count == 1
    await invalidate_facet_catalogue_cache()


@pytest.mark.asyncio
async def test_default_window_cache_isolates_different_window_lengths(monkeypatch):
    from evaluatorq.trace_finder import facets

    client = _client()
    load = AsyncMock(return_value=FacetCatalogue())
    monkeypatch.setattr(facets, 'load_facet_catalogue', load)
    await invalidate_facet_catalogue_cache()
    anchor = datetime(2026, 1, 2, tzinfo=timezone.utc)
    await load_facet_catalogue_cached(
        client, anchor - timedelta(days=1), anchor, default_window=True, now=anchor
    )
    await load_facet_catalogue_cached(
        client, anchor - timedelta(days=2), anchor, default_window=True, now=anchor
    )
    assert load.await_count == 2
    await invalidate_facet_catalogue_cache()


def test_time_bounds_intersect_and_named_value_matching_is_longest_first():
    lower = datetime(2026, 1, 2, tzinfo=timezone.utc)
    upper = datetime(2026, 1, 10, tzinfo=timezone.utc)
    assert _later(lower, datetime(2026, 1, 3, tzinfo=timezone.utc)) == datetime(2026, 1, 3, tzinfo=timezone.utc)
    assert _earlier(upper, datetime(2026, 1, 9, tzinfo=timezone.utc)) == datetime(2026, 1, 9, tzinfo=timezone.utc)
    catalogue = FacetCatalogue(project=('My Project', 'My Project West'))
    candidates = _literal_candidates('find project My Project West', catalogue)
    assert ('project', 'My Project West') in candidates


@pytest.mark.asyncio
async def test_candidate_confirmation_preserves_explicit_facets_and_numeric_scope(monkeypatch):
    from evaluatorq.trace_finder import planner

    searches: list[dict[str, Any]] = []

    class Source:
        def __init__(self, _orq):
            pass

        async def search(self, start, end, limit, *, facets, numeric):
            searches.append({'start': start, 'end': end, 'limit': limit, 'facets': facets, 'numeric': numeric})
            return ('trace',) if 'yes' in facets.model else ()

        def close(self):
            pass

    monkeypatch.setattr(planner, 'OrqTraceSource', Source)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 2, tzinfo=timezone.utc)
    scope = FacetSelection(provider=frozenset({'openai'}), project_id='project-1')
    numeric = NumericFilters(tokens_min=50)
    result = await planner._offer_confirmed_candidates(
        cast(Any, object()),
        FacetCatalogue(model=('popular/model',)),
        (('model', 'yes'), ('model', 'no')),
        start=start,
        end=end,
        facets=scope,
        numeric=numeric,
    )
    assert result.model == ('popular/model', 'yes')
    assert searches[0]['facets'] == FacetSelection(
        provider=frozenset({'openai'}), project_id='project-1', model=frozenset({'yes'})
    )
    assert searches[0]['numeric'] == numeric
    assert searches[0]['start'] == start
    assert searches[0]['end'] == end


def test_quoted_candidates_do_not_displace_named_candidates():
    candidates = _literal_candidates(
        'model preferred provider desired agent chosen "quoted one"',
        FacetCatalogue(),
    )
    assert ('model', 'preferred') in candidates
    assert ('provider', 'desired') in candidates
    assert ('agent_name', 'chosen') in candidates


@pytest.mark.asyncio
async def test_disjoint_within_results_plan_is_explicitly_empty(monkeypatch):
    from evaluatorq.trace_finder import planner

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 2, tzinfo=timezone.utc)
    later_start = datetime(2026, 1, 3, tzinfo=timezone.utc)
    later_end = datetime(2026, 1, 4, tzinfo=timezone.utc)

    async def compile_stub(*_args, **_kwargs):
        return SimpleNamespace(
            dimensions=(),
            numeric=NumericFilters(),
            time=TimeFilters(start=later_start, end=later_end),
            unsupported_reason=None,
        )

    async def catalogue_loader(*_args, **_kwargs):
        return FacetCatalogue()

    async def selector(*_args, **_kwargs):
        return FilterSelectionResult(FacetSelection())

    monkeypatch.setattr(planner, 'compile_query', compile_stub)
    monkeypatch.setattr(planner, 'select_filters_with_response', selector)
    await invalidate_facet_catalogue_cache()
    result = await planner.plan_query(
        'find matching traces',
        PopulationRequest(start=start, end=end, time_origin='within_results'),
        client=cast(Any, object()),
        orq=cast(Any, _client()),
        compiler_model='compiler',
        classifier_model='classifier',
        fast_filter_model='fast',
        catalogue_loader=catalogue_loader,
        within_results=True,
    )
    assert result.empty_match is True
    assert result.effective_population.start == start
    assert result.effective_population.end == end
    await invalidate_facet_catalogue_cache()


@pytest.mark.asyncio
async def test_cancelled_planner_awaits_compiler_task(monkeypatch):
    from evaluatorq.trace_finder import planner

    entered = asyncio.Event()
    finished = asyncio.Event()

    async def compile_stub(*_args, **_kwargs):
        entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await asyncio.sleep(0)
            finished.set()
            raise

    async def loader(*_args, **_kwargs):
        await asyncio.Future()

    monkeypatch.setattr(planner, 'compile_query', compile_stub)
    task = asyncio.create_task(planner.plan_query(
        'find traces', PopulationRequest(), client=cast(Any, object()), orq=cast(Any, _client()),
        compiler_model='compiler', classifier_model='classifier', fast_filter_model='fast',
        catalogue_loader=loader,
    ))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()



@pytest.mark.asyncio
async def test_llm_selector_time_drives_catalogue_reselection_and_candidate_confirmation(monkeypatch):
    from evaluatorq.trace_finder import planner

    anchor = datetime(2026, 5, 31, tzinfo=timezone.utc)
    generated_start = datetime(2026, 5, 20, tzinfo=timezone.utc)
    generated_end = datetime(2026, 5, 21, tzinfo=timezone.utc)
    loaded: list[tuple[datetime, datetime]] = []
    confirmations: list[dict[str, Any]] = []

    async def compile_stub(*_args, **_kwargs):
        return SimpleNamespace(
            dimensions=(), numeric=NumericFilters(),
            time=TimeFilters(start=datetime(2026, 5, 25, tzinfo=timezone.utc), end=anchor),
            unsupported_reason=None,
        )

    async def loader(_orq, *, start, end, **_kwargs):
        loaded.append((start, end))
        return FacetCatalogue(model=('outside-window/model',) if start == generated_start else ())

    async def selector(*_args, **_kwargs):
        return FilterSelectionResult(FacetSelection(model=frozenset({'outside-window/model'})),
                                     time=TimeFilters(start=generated_start, end=generated_end))

    async def confirm(_orq, name, value, start, end):
        confirmations.append({'name': name, 'value': value, 'start': start, 'end': end})
        return True

    monkeypatch.setattr(planner, 'compile_query', compile_stub)
    monkeypatch.setattr(planner, 'select_llm_filters_with_response', selector)
    await invalidate_facet_catalogue_cache()
    result = await planner.plan_query(
        'model outside-window/model', PopulationRequest(time_origin='default'), client=cast(Any, object()),
        orq=cast(Any, _client()), compiler_model='compiler', filter_selection='llm',
        classifier_model='classifier', fast_filter_model='fast', now=anchor, catalogue_loader=loader,
        candidate_confirmer=confirm,
    )
    assert len(loaded) == 2
    assert result.effective_population.start == generated_start
    assert result.effective_population.end == generated_end
    assert confirmations == []  # exact values in the final catalogue need no OQL confirmation
    assert result.usage['catalogue_cache']['events'][1]['start'] == generated_start.isoformat()
    await invalidate_facet_catalogue_cache()


@pytest.mark.asyncio
async def test_final_selector_time_confirms_skipped_candidate_in_explicit_scope(monkeypatch):
    from evaluatorq.trace_finder import planner

    anchor = datetime(2026, 5, 31, tzinfo=timezone.utc)
    selector_start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    selector_end = datetime(2026, 5, 2, tzinfo=timezone.utc)
    confirmation: list[dict[str, Any]] = []

    async def compile_stub(*_args, **_kwargs):
        return SimpleNamespace(
            dimensions=(), numeric=NumericFilters(),
            time=TimeFilters(start=datetime(2026, 5, 25, tzinfo=timezone.utc),
                             end=datetime(2026, 5, 31, tzinfo=timezone.utc)),
            unsupported_reason=None,
        )
    async def loader(*_args, **_kwargs):
        return FacetCatalogue()

    async def selector(*_args, **_kwargs):
        return FilterSelectionResult(FacetSelection(), time=TimeFilters(start=selector_start, end=selector_end))

    async def confirm(_orq, name, value, start, end):
        confirmation.append({'name': name, 'value': value, 'start': start, 'end': end})
        return True

    monkeypatch.setattr(planner, 'compile_query', compile_stub)
    monkeypatch.setattr(planner, 'select_llm_filters_with_response', selector)
    result = await planner.plan_query(
        'model unusual/model', PopulationRequest(
            time_origin='default', facets=FacetSelection(provider=frozenset({'openai'})),
            numeric=NumericFilters(tokens_min=20),
        ), client=cast(Any, object()), orq=cast(Any, _client()),
        compiler_model='compiler', filter_selection='llm', classifier_model='classifier', fast_filter_model='fast',
        now=anchor, catalogue_loader=loader, candidate_confirmer=confirm,
    )
    assert confirmation == [{'name': 'model', 'value': 'unusual/model', 'start': selector_start, 'end': selector_end}]
    assert result.generated_facets.model == frozenset({'unusual/model'})




@pytest.mark.asyncio
async def test_reselection_accepts_equivalent_whole_calendar_day_bounds(monkeypatch):
    from evaluatorq.trace_finder import planner

    anchor = datetime(2026, 5, 31, tzinfo=timezone.utc)
    start = datetime(2026, 5, 20, tzinfo=timezone.utc)
    midnight_end = datetime(2026, 5, 21, tzinfo=timezone.utc)
    end_of_day = datetime(2026, 5, 20, 23, 59, 59, 999999, tzinfo=timezone.utc)
    calls = 0

    async def compile_stub(*_args, **_kwargs):
        return SimpleNamespace(dimensions=(), numeric=NumericFilters(), time=None, unsupported_reason=None)

    async def loader(*_args, **_kwargs):
        return FacetCatalogue(model=('some/model',))

    async def selector(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        end = midnight_end if calls == 1 else end_of_day
        return FilterSelectionResult(
            FacetSelection(model=frozenset({'some/model'})),
            time=TimeFilters(start=start, end=end, calendar_day=True),
        )

    monkeypatch.setattr(planner, 'compile_query', compile_stub)
    monkeypatch.setattr(planner, 'select_llm_filters_with_response', selector)
    await invalidate_facet_catalogue_cache()
    result = await planner.plan_query(
        'model some/model from yesterday', PopulationRequest(time_origin='default'),
        client=cast(Any, object()), orq=cast(Any, _client()), compiler_model='compiler',
        filter_selection='llm', classifier_model='classifier', fast_filter_model='fast',
        now=anchor, catalogue_loader=loader,
    )
    assert calls == 2
    assert result.selector_error is None
    assert result.generated_facets.model == frozenset({'some/model'})
    assert result.generated_time == (start, midnight_end)
    await invalidate_facet_catalogue_cache()
@pytest.mark.asyncio
async def test_reselection_time_flip_fails_closed_with_visible_error(monkeypatch):
    from evaluatorq.trace_finder import planner

    anchor = datetime(2026, 5, 31, tzinfo=timezone.utc)
    first_time = TimeFilters(start=datetime(2026, 5, 20, tzinfo=timezone.utc),
                             end=datetime(2026, 5, 21, tzinfo=timezone.utc))
    second_time = TimeFilters(start=datetime(2026, 5, 18, tzinfo=timezone.utc),
                              end=datetime(2026, 5, 19, tzinfo=timezone.utc))
    calls = 0

    async def compile_stub(*_args, **_kwargs):
        return SimpleNamespace(dimensions=(), numeric=NumericFilters(), time=None, unsupported_reason=None)

    async def loader(*_args, **_kwargs):
        return FacetCatalogue(model=('some/model',))

    async def selector(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return FilterSelectionResult(
            FacetSelection(model=frozenset({'some/model'})), time=first_time if calls == 1 else second_time,
        )

    monkeypatch.setattr(planner, 'compile_query', compile_stub)
    monkeypatch.setattr(planner, 'select_llm_filters_with_response', selector)
    await invalidate_facet_catalogue_cache()
    result = await planner.plan_query(
        'model some/model', PopulationRequest(time_origin='default'), client=cast(Any, object()),
        orq=cast(Any, _client()), compiler_model='compiler', filter_selection='llm',
        classifier_model='classifier', fast_filter_model='fast', now=anchor, catalogue_loader=loader,
    )
    assert calls == 2
    assert result.selector_error == 'selector changed its time window after catalogue reselection'
    assert result.generated_facets == FacetSelection()
    assert result.empty_match is True
    await invalidate_facet_catalogue_cache()


@pytest.mark.asyncio
async def test_generated_time_outside_retention_returns_empty_without_reloading_catalogue(monkeypatch):
    from evaluatorq.trace_finder import planner

    anchor = datetime(2026, 5, 31, tzinfo=timezone.utc)
    population_start = anchor - timedelta(days=80)
    population_end = anchor - timedelta(days=70)
    old_start = anchor - timedelta(days=90)
    old_end = anchor - timedelta(days=89)
    loaded: list[tuple[datetime, datetime]] = []

    async def compile_stub(*_args, **_kwargs):
        return SimpleNamespace(dimensions=(), numeric=NumericFilters(), time=None, unsupported_reason=None)

    async def loader(_orq, *, start, end, **_kwargs):
        loaded.append((start, end))
        return FacetCatalogue()

    async def selector(*_args, **_kwargs):
        return FilterSelectionResult(FacetSelection(), time=TimeFilters(start=old_start, end=old_end))

    monkeypatch.setattr(planner, 'compile_query', compile_stub)
    monkeypatch.setattr(planner, 'select_llm_filters_with_response', selector)
    await invalidate_facet_catalogue_cache()
    result = await planner.plan_query(
        'find traces', PopulationRequest(
            start=population_start, end=population_end, time_origin='default',
        ), client=cast(Any, object()), orq=cast(Any, _client()), compiler_model='compiler',
        filter_selection='llm', classifier_model='classifier', fast_filter_model='fast',
        now=anchor, catalogue_loader=loader,
    )
    assert len(loaded) == 1
    assert result.empty_match is True
    assert result.effective_population.start == population_start
    assert result.effective_population.end == population_end
    await invalidate_facet_catalogue_cache()
