"""Route wiring and per-application runtime ownership for ``/find``."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger
from pydantic import ValidationError
from starlette.requests import Request  # noqa: TC002 — FastHTML inspects this annotation at runtime
from starlette.responses import Response

from evaluatorq.common.llm_client import resolve_llm_client
from evaluatorq.common.orq_client import resolve_orq_client
from evaluatorq.dashboard.finder_views import drawer, facet_menu, fragment, page_html
from evaluatorq.trace_finder import (
    CompiledQuery,
    FacetCatalogue,
    FacetSelection,
    NumericFilters,
    PopulationRequest,
    RunRequest,
    RunSnapshot,
    RunStore,
    build_run_store,
    effective_settings,
    export_json,
    load_facet_catalogue,
)
from evaluatorq.trace_finder.models import FACET_NAMES


def _settings(app: Any) -> Any:
    settings = getattr(app.state, 'finder_settings', None)
    if settings is None:
        settings = effective_settings()
        app.state.finder_settings = settings
    return settings


def _build_store(app: Any) -> RunStore | None:
    """Build the app-owned store, returning ``None`` when Orq is unavailable."""
    try:
        resolved = resolve_llm_client(require_orq=True, max_retries=0)
        orq = resolve_orq_client()
    except ValueError as exc:
        logger.warning('Find surface is unavailable because an Orq client could not be resolved: {}', exc)
        return None

    settings = _settings(app)
    return build_run_store(settings, client=resolved.client, orq=orq)


def _store(app: Any) -> RunStore | None:
    store = getattr(app.state, 'finder_store', None)
    if store is None:
        store = _build_store(app)
        if store is not None:
            app.state.finder_store = store
    return store


async def _load_catalogue(app: Any) -> FacetCatalogue | None:
    """Load and cache facet values for five minutes; failures render an empty menu."""
    now = datetime.now(timezone.utc)
    cached = getattr(app.state, 'finder_catalogue_cache', None)
    if cached is not None:
        expires, catalogue = cached
        if expires > now:
            return catalogue
    try:
        orq = resolve_orq_client()
        settings = _settings(app)
        catalogue = await load_facet_catalogue(
            orq,
            start=now - timedelta(days=settings.window_days),
            end=now,
            limit=50,
        )
    except ValueError as exc:
        logger.warning('Find facet menu is empty because the Orq client could not be resolved: {}', exc)
        return None
    app.state.finder_catalogue_cache = (now + timedelta(minutes=5), catalogue)
    return catalogue


def _form_values(form: Any, name: str) -> list[str]:
    values = form.getlist(name) if hasattr(form, 'getlist') else [form.get(name)]
    return [str(value) for value in values if value not in (None, '')]


def _optional_int(form: Any, name: str) -> int | None:
    raw = form.get(name)
    if raw in (None, ''):
        return None
    return int(str(raw))


def _run_request(form: Any, settings: Any) -> RunRequest:
    query = str(form.get('query') or '').strip()
    mode = str(form.get('mode') or 'immediate')
    window_days = int(str(form.get('window_days') or settings.window_days))
    limit = int(str(form.get('limit') or settings.limit))
    parallelism = int(str(form.get('parallelism') or settings.parallelism))
    facet_values = {name: frozenset(_form_values(form, f'facet_{name}')) for name in FACET_NAMES}
    numeric = NumericFilters(
        tokens_min=_optional_int(form, 'tokens_min'),
        tokens_max=_optional_int(form, 'tokens_max'),
        duration_ms_min=_optional_int(form, 'duration_ms_min'),
        duration_ms_max=_optional_int(form, 'duration_ms_max'),
    )
    end = datetime.now(timezone.utc)
    return RunRequest(
        query=query,
        mode=mode,
        population=PopulationRequest(
            start=end - timedelta(days=window_days),
            end=end,
            facets=FacetSelection(**facet_values),
            numeric=numeric,
            limit=limit,
        ),
        parallelism=parallelism,
    )


def _compiled_from_form(current: CompiledQuery, form: Any) -> CompiledQuery:
    from evaluatorq.common.judge import ClassifyQuestion
    from evaluatorq.trace_finder import ThresholdSelection, ValueSelection

    current_task = current.task
    kind = str(form.get('kind') or form.get('task_kind') or current_task.kind)
    instructions = str(form.get('instructions') or current_task.instructions).strip()

    if kind == 'choice':
        existing = current_task.criteria.items() if isinstance(current_task.criteria, dict) else ()
        existing_pairs = tuple(existing)
        pairs: list[tuple[str, str]] = []
        for index, (old_label, old_description) in enumerate(existing_pairs):
            label = str(form.get(f'criteria_label_{index}') or old_label).strip()
            description = str(form.get(f'criteria_description_{index}') or old_description or '').strip()
            pairs.append((label, description))
        if not pairs:
            old_descriptions = current_task.criteria if isinstance(current_task.criteria, list) else ()
            pairs = [(f'option_{index + 1}', str(description)) for index, description in enumerate(old_descriptions)]
        if not pairs:
            pairs = [('yes', 'The request is satisfied.'), ('no', 'The request is not satisfied.')]
        criteria: dict[str, str] | list[str] | None = dict(pairs)
    elif kind == 'score':
        existing_descriptions = current_task.criteria if isinstance(current_task.criteria, list) else ()
        if not existing_descriptions and isinstance(current_task.criteria, dict):
            existing_descriptions = tuple(current_task.criteria.values())
        criteria_values = [
            str(form.get(f'score_criteria_{index}') or form.get(f'criteria_{index}') or description).strip()
            for index, description in enumerate(existing_descriptions)
        ]
        if not criteria_values:
            criteria_values = ['Low match', 'High match']
        criteria = criteria_values
    elif kind == 'noul':
        criteria = None
    else:
        raise ValueError('kind must be one of choice, noul, or score')

    threshold = float(form.get('noul_threshold') or current_task.noul_threshold)
    task = ClassifyQuestion(
        kind=kind,
        instructions=instructions,
        criteria=criteria,
        noul_threshold=threshold,
        state={},
    )

    values = _form_values(form, 'selection_value') or _form_values(form, 'selection_values')
    if kind == 'choice':
        labels = tuple(criteria) if isinstance(criteria, dict) else ()
        selected = tuple(value for value in values if value in labels)
        if not selected:
            current_values = getattr(current.selection, 'values', ())
            selected = tuple(value for value in current_values if type(value) is str and value in labels)
        selection = ValueSelection(kind='values', values=selected or (labels[0],))
    elif kind == 'noul':
        selected_bool = tuple(value.casefold() == 'true' for value in values if value.casefold() in {'true', 'false'})
        if not selected_bool:
            selected_bool = tuple(value for value in getattr(current.selection, 'values', ()) if type(value) is bool)
        selection = ValueSelection(kind='values', values=selected_bool or (False,))
    else:
        rule = str(form.get('selection_rule') or '')
        if ':' in rule:
            operator, raw_threshold = rule.split(':', 1)
            score_threshold = float(raw_threshold)
        else:
            operator = str(form.get('selection_operator') or getattr(current.selection, 'operator', 'gte'))
            raw_threshold = form.get('selection_threshold') or form.get('selection_value')
            score_threshold = float(raw_threshold or getattr(current.selection, 'value', 0.5))
        selection = ThresholdSelection(operator=operator, value=score_threshold, kind='threshold')
    return CompiledQuery(task=task, selection=selection)


def _html(content: str, *, status_code: int = 200) -> Response:
    return Response(content, status_code=status_code, media_type='text/html')


def register_finder_routes(app: Any, roots: list[Any] | None = None) -> None:  # noqa: C901
    """Register the Find page and its HTMX fragments on *app*."""

    @app.get('/find')
    async def find_page(req: Request) -> Response:
        api_available = bool(os.environ.get('ORQ_API_KEY', '').strip())
        settings = _settings(req.app)
        store = _store(req.app) if api_available else None
        snapshot = await store.snapshot() if store is not None else RunSnapshot()
        return _html(page_html(snapshot, settings, api_available=api_available))

    @app.post('/find/run')
    async def find_run(req: Request) -> Response:
        settings = _settings(req.app)
        store = _store(req.app)
        if store is None:
            return _html(fragment(RunSnapshot(), settings, error='Set ORQ_API_KEY to load traces', api_available=False))
        form = await req.form()
        try:
            request = _run_request(form, settings)
            snapshot = await store.compile(request)
        except (ValidationError, ValueError, TypeError) as exc:
            return _html(fragment(RunSnapshot(), settings, error=str(exc)), status_code=422)
        return _html(fragment(snapshot, settings))

    @app.get('/find/poll')
    async def find_poll(req: Request) -> Response:
        settings = _settings(req.app)
        store = _store(req.app)
        snapshot = await store.snapshot() if store is not None else RunSnapshot()
        return _html(fragment(snapshot, settings))

    @app.post('/find/start')
    async def find_start(req: Request) -> Response:
        settings = _settings(req.app)
        store = _store(req.app)
        if store is None:
            return _html(fragment(RunSnapshot(), settings, error='Set ORQ_API_KEY to load traces', api_available=False))
        form = await req.form()
        current = await store.snapshot()
        if current.request is None or current.compiled is None:
            return _html(fragment(current, settings, error='There is no plan waiting for review.'), status_code=409)
        try:
            compiled = _compiled_from_form(current.compiled, form)
            snapshot = await store.start(current.request, compiled)
        except (ValidationError, ValueError, TypeError) as exc:
            return _html(fragment(current, settings, error=str(exc)), status_code=422)
        return _html(fragment(snapshot, settings))

    @app.post('/find/cancel')
    async def find_cancel(req: Request) -> Response:
        settings = _settings(req.app)
        store = _store(req.app)
        snapshot = await store.cancel() if store is not None else RunSnapshot()
        return _html(fragment(snapshot, settings))

    @app.post('/find/reset')
    async def find_reset(req: Request) -> Response:
        settings = _settings(req.app)
        store = _store(req.app)
        snapshot = await store.reset() if store is not None else RunSnapshot()
        return _html(fragment(snapshot, settings))

    @app.get('/find/trace/{trace_id:path}')
    async def find_trace(trace_id: str, req: Request) -> Response:
        store = _store(req.app)
        if store is None:
            return _html('<p class="finder-empty">Trace finding is unavailable.</p>', status_code=404)
        detail = await store.trace_detail(trace_id)
        if detail is None:
            return _html('<p class="finder-empty">Trace not found.</p>', status_code=404)
        return _html(drawer(detail))

    @app.get('/find/export.json')
    async def find_export(req: Request) -> Response:
        store = _store(req.app)
        if store is None:
            return Response('Not found', status_code=404, media_type='text/plain')
        snapshot = await store.snapshot()
        if snapshot.state != 'completed' or snapshot.request is None or snapshot.compiled is None:
            return Response('Not found', status_code=404, media_type='text/plain')
        return Response(
            export_json(snapshot),
            media_type='application/json',
            headers={'Content-Disposition': f'attachment; filename="trace-finder-{snapshot.generation}.json"'},
        )

    @app.get('/find/facets')
    async def find_facets(req: Request) -> Response:
        catalogue = await _load_catalogue(req.app)
        return _html(facet_menu(catalogue, open_=True))

    @app.get('/find/dismiss')
    def find_dismiss() -> Response:
        return Response('', media_type='text/html')
