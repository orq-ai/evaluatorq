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


@pytest.mark.asyncio
async def test_injected_catalogue_loaders_do_not_share_cached_results():
    client = _client()
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 2, tzinfo=timezone.utc)

    async def first_loader(*_args, **_kwargs):
        return FacetCatalogue(model=('first-loader',))

    async def second_loader(*_args, **_kwargs):
        return FacetCatalogue(model=('second-loader',))

    first = await load_facet_catalogue_cached(client, start, end, loader=first_loader)
    second = await load_facet_catalogue_cached(client, start, end, loader=second_loader)

    assert first.model == ('first-loader',)
    assert second.model == ('second-loader',)


@pytest.mark.asyncio
async def test_calendar_end_is_exclusive_only_when_it_is_the_effective_upper_bound(monkeypatch):
    from evaluatorq.trace_finder import planner

    start = datetime(2026, 10, 12, tzinfo=timezone.utc)
    generated_end = datetime(2026, 10, 13, tzinfo=timezone.utc)

    async def compile_stub(*_args, **_kwargs):
        return SimpleNamespace(
            dimensions=(),
            numeric=NumericFilters(),
            time=TimeFilters(start=start, end=generated_end, calendar_day=True),
            unsupported_reason=None,
        )

    async def loader(*_args, **_kwargs):
        return FacetCatalogue()

    async def selector(*_args, **_kwargs):
        return FilterSelectionResult(FacetSelection())

    monkeypatch.setattr(planner, 'compile_query', compile_stub)
    monkeypatch.setattr(planner, 'select_filters_with_response', selector)
    await invalidate_facet_catalogue_cache()
    base: dict[str, Any] = dict(
        query='find traces',
        client=cast(Any, object()),
        orq=cast(Any, _client()),
        compiler_model='compiler',
        classifier_model='classifier',
        fast_filter_model='fast',
        catalogue_loader=loader,
        within_results=True,
        now=generated_end,
    )
    earlier = await planner.plan_query(
        population=PopulationRequest(
            start=start - timedelta(days=1),
            end=generated_end - timedelta(hours=1),
            time_origin='within_results',
        ),
        **base,
    )
    equal = await planner.plan_query(
        population=PopulationRequest(start=start, end=generated_end, time_origin='within_results'),
        **base,
    )

    assert earlier.effective_population.end == generated_end - timedelta(hours=1)
    assert earlier.effective_population.end_exclusive is False
    assert equal.effective_population.end == generated_end
    assert equal.effective_population.end_exclusive is True
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


@pytest.mark.asyncio
async def test_initial_catalogue_failure_keeps_compiled_semantics_and_fails_closed(monkeypatch):
    from evaluatorq.trace_finder import planner

    semantic_dimension = cast(Any, object())
    async def compile_stub(*_args, **_kwargs):
        return SimpleNamespace(
            dimensions=(semantic_dimension,),
            numeric=NumericFilters(),
            time=None,
            unsupported_reason=None,
        )

    async def broken_loader(*_args, **_kwargs):
        raise RuntimeError('catalogue unavailable')

    monkeypatch.setattr(planner, 'compile_query', compile_stub)
    await invalidate_facet_catalogue_cache()
    result = await planner.plan_query(
        'find semantic traces',
        PopulationRequest(time_origin='default'),
        client=cast(Any, object()),
        orq=cast(Any, _client()),
        compiler_model='compiler',
        classifier_model='classifier',
        fast_filter_model='fast',
        catalogue_loader=broken_loader,
    )

    assert result.dimensions == (semantic_dimension,)
    assert result.selector_error == 'catalogue unavailable'
    assert result.empty_match is True


@pytest.mark.asyncio
@pytest.mark.parametrize('supplied_bounds', [False, True])
async def test_inferred_population_origin_preserves_only_supplied_bounds(monkeypatch, supplied_bounds):
    from evaluatorq.trace_finder import planner

    anchor = datetime(2026, 5, 31, 12, tzinfo=timezone.utc)
    start, end = anchor - timedelta(days=7), anchor
    generated = TimeFilters(start=anchor - timedelta(days=2), end=anchor - timedelta(days=1))
    population = PopulationRequest(start=start, end=end) if supplied_bounds else PopulationRequest()
    population = PopulationRequest.model_validate_json(population.model_dump_json())

    async def compile_stub(*_args, **_kwargs):
        return SimpleNamespace(dimensions=(), numeric=NumericFilters(), time=generated, unsupported_reason=None)

    async def loader(*_args, **_kwargs):
        return FacetCatalogue()

    async def selector(*_args, **_kwargs):
        return FilterSelectionResult(FacetSelection())

    monkeypatch.setattr(planner, 'compile_query', compile_stub)
    monkeypatch.setattr(planner, 'select_filters_with_response', selector)
    result = await planner.plan_query(
        'from yesterday', population, client=cast(Any, object()), orq=cast(Any, _client()),
        compiler_model='compiler', classifier_model='classifier', fast_filter_model='fast',
        now=anchor, catalogue_loader=loader,
    )
    expected = (start, end) if supplied_bounds else (generated.start, generated.end)
    assert (result.effective_population.start, result.effective_population.end) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['inherited', 'clipped', 'date_only', 'caller_inclusive'])
async def test_effective_end_retains_actual_boundary_semantics(monkeypatch, case):
    from evaluatorq.trace_finder import planner

    anchor = datetime(2026, 5, 31, 12, tzinfo=timezone.utc)
    start = anchor - timedelta(days=3)
    end = datetime(2026, 5, 30, tzinfo=timezone.utc)
    if case == 'inherited':
        population = PopulationRequest(start=start, end=end, end_exclusive=True, time_origin='within_results')
        generated = None
        expected_kept = False
    elif case == 'clipped':
        population = PopulationRequest()
        generated = TimeFilters(start=start, end=anchor + timedelta(days=1), calendar_day=True)
        expected_kept = True
    elif case == 'date_only':
        population = PopulationRequest(start=start, end=anchor, time_origin='within_results')
        generated = TimeFilters.model_validate({'start': start, 'end': '2026-05-29'})
        expected_kept = False
    else:
        population = PopulationRequest(start=start, end=end)
        generated = TimeFilters(start=start, end=end, calendar_day=True)
        expected_kept = True

    async def compile_stub(*_args, **_kwargs):
        return SimpleNamespace(dimensions=(), numeric=NumericFilters(), time=generated, unsupported_reason=None)

    async def loader(*_args, **_kwargs):
        return FacetCatalogue()

    async def selector(*_args, **_kwargs):
        return FilterSelectionResult(FacetSelection())

    monkeypatch.setattr(planner, 'compile_query', compile_stub)
    monkeypatch.setattr(planner, 'select_filters_with_response', selector)
    result = await planner.plan_query(
        'filter loaded traces', population, client=cast(Any, object()), orq=cast(Any, _client()),
        compiler_model='compiler', classifier_model='classifier', fast_filter_model='fast',
        now=anchor, catalogue_loader=loader,
    )
    boundary = result.effective_population.end
    assert boundary is not None
    assert result.effective_population.contains_time(boundary) is expected_kept


def _time_compiler(*windows: tuple[datetime, datetime]):
    pending = iter(windows)

    async def compile_stub(*_args, **_kwargs):
        start, end = next(pending)
        return SimpleNamespace(
            dimensions=(), numeric=NumericFilters(), time=TimeFilters(start=start, end=end), unsupported_reason=None
        )

    return compile_stub


@pytest.mark.asyncio
async def test_generated_days_of_equal_length_load_their_own_facet_catalogues(monkeypatch):
    from evaluatorq.trace_finder import facets, planner

    anchor = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)
    first = datetime(2026, 10, 12, tzinfo=timezone.utc)
    second = datetime(2026, 10, 13, tzinfo=timezone.utc)
    offered: list[tuple[str, ...]] = []

    async def load(_client, *, start, end, limit):
        return FacetCatalogue(model=(f'model-{start.date()}',))

    async def selector(_client, _model, catalogue, _query):
        offered.append(catalogue.model)
        return FilterSelectionResult(FacetSelection())

    monkeypatch.setattr(facets, 'load_facet_catalogue', load)
    monkeypatch.setattr(planner, 'compile_query', _time_compiler((first, second), (second, second + timedelta(days=1))))
    monkeypatch.setattr(planner, 'select_filters_with_response', selector)
    await invalidate_facet_catalogue_cache()
    try:
        for _ in range(2):
            await planner.plan_query(
                'find traces', PopulationRequest(), client=cast(Any, object()), orq=cast(Any, _client()),
                compiler_model='compiler', classifier_model='classifier', fast_filter_model='fast', now=anchor,
            )
    finally:
        await invalidate_facet_catalogue_cache()

    assert offered == [('model-2026-10-12',), ('model-2026-10-13',)]


@pytest.mark.asyncio
async def test_literal_confirmation_searches_only_the_retained_generated_window(monkeypatch):
    from evaluatorq.trace_finder import planner

    anchor = datetime(2026, 10, 20, tzinfo=timezone.utc)
    confirmations: list[tuple[str, str, datetime, datetime]] = []

    async def loader(*_args, **_kwargs):
        return FacetCatalogue()

    async def selector(*_args, **_kwargs):
        return FilterSelectionResult(FacetSelection())

    async def confirmer(_orq, name, value, start, end):
        confirmations.append((name, value, start, end))
        return False

    monkeypatch.setattr(
        planner, 'compile_query', _time_compiler((anchor - timedelta(days=60), anchor - timedelta(days=20)))
    )
    monkeypatch.setattr(planner, 'select_filters_with_response', selector)
    await planner.plan_query(
        'traces from model acme/foo', PopulationRequest(), client=cast(Any, object()), orq=cast(Any, _client()),
        compiler_model='compiler', classifier_model='classifier', fast_filter_model='fast', now=anchor,
        catalogue_loader=loader, candidate_confirmer=confirmer,
    )

    assert confirmations == [('model', 'acme/foo', anchor - timedelta(days=30), anchor - timedelta(days=20))]
