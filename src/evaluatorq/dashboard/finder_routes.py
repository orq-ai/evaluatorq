"""Route wiring and per-application runtime ownership for ``/find``."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.requests import Request  # noqa: TC002 — FastHTML inspects this annotation at runtime
from starlette.responses import Response

from evaluatorq.common.llm_client import resolve_llm_client
from evaluatorq.common.orq_client import DEFAULT_ORQ_BASE_URL, list_orq_profiles, resolve_orq_client
from evaluatorq.dashboard.finder_views import drawer, facet_menu, fragment, page_html
from evaluatorq.dashboard.security import request_rejected
from evaluatorq.trace_finder import (
    CompiledQuery,
    FacetSelection,
    NumericFilters,
    PopulationRequest,
    RunRequest,
    RunSnapshot,
    build_run_store,
    effective_settings,
    export_json,
    load_facet_catalogue,
)

if TYPE_CHECKING:
    from evaluatorq.trace_finder import FacetCatalogue, RunStore
from evaluatorq.trace_finder.models import FACET_NAMES, NUMERIC_FACET_NAMES
from evaluatorq.trace_finder.settings import (
    MAX_LIMIT,
    MAX_PARALLELISM,
    MAX_WINDOW_DAYS,
    MIN_LIMIT,
    MIN_PARALLELISM,
    MIN_WINDOW_DAYS,
)

_NUMERIC_FIELDS = tuple(f'{name}_{bound}' for name in NUMERIC_FACET_NAMES for bound in ('min', 'max'))


class FinderRunForm(BaseModel):
    """Validated values accepted by finder run and reviewed-start forms."""

    model_config = ConfigDict(extra='ignore')

    query: str = Field(default='', min_length=1)
    mode: Literal['immediate', 'review'] = 'immediate'
    window_days: int = Field(ge=MIN_WINDOW_DAYS, le=MAX_WINDOW_DAYS)
    limit: int = Field(ge=MIN_LIMIT, le=MAX_LIMIT)
    parallelism: int = Field(ge=MIN_PARALLELISM, le=MAX_PARALLELISM)
    tokens_min: int | None = Field(default=None, ge=0)
    tokens_max: int | None = Field(default=None, ge=0)
    duration_ms_min: int | None = Field(default=None, ge=0)
    duration_ms_max: int | None = Field(default=None, ge=0)


def _settings(app: Any) -> Any:
    settings = getattr(app.state, 'finder_settings', None)
    if settings is None:
        settings = effective_settings()
        profiles = list_orq_profiles() if settings.orq_profile is not None else ()
        app.state.finder_profile = next((p for p in profiles if p.name == settings.orq_profile), None)
        if settings.orq_profile is not None and app.state.finder_profile is None:
            logger.warning(
                'Saved Orq profile {} is unavailable; finder uses environment credentials', settings.orq_profile
            )
        app.state.finder_settings = settings
    return settings


def _profile(app: Any) -> Any:
    _settings(app)
    return getattr(app.state, 'finder_profile', None)


def _api_available(app: Any) -> bool:
    return _profile(app) is not None or bool(os.environ.get('ORQ_API_KEY', '').strip())


def _build_store(app: Any) -> RunStore | None:
    """Build the app-owned store, returning ``None`` when Orq is unavailable."""
    settings = _settings(app)
    profile = _profile(app)
    try:
        resolved = resolve_llm_client(
            extra_api_key=profile.api_key if profile else None,
            orq_host=(profile.server or DEFAULT_ORQ_BASE_URL) if profile else None,
            require_orq=True,
            max_retries=0,
        )
        orq = resolve_orq_client(
            profile.api_key if profile else None,
            server_url=(profile.server or DEFAULT_ORQ_BASE_URL) if profile else None,
        )
    except (ImportError, ValueError) as exc:
        logger.warning('Find surface is unavailable because an Orq client could not be resolved: {}', exc)
        return None

    return build_run_store(settings, client=resolved.client, orq=orq)


def _store(app: Any) -> RunStore | None:
    store = getattr(app.state, 'finder_store', None)
    if store is None:
        store = _build_store(app)
        if store is not None:
            app.state.finder_store = store
    return store


async def _load_catalogue(app: Any, window_days: int | None = None) -> FacetCatalogue | None:
    """Load and cache facet values for five minutes; failures render an empty menu."""
    now = datetime.now(timezone.utc)
    settings = _settings(app)
    window = window_days if window_days is not None else settings.window_days
    if not MIN_WINDOW_DAYS <= window <= MAX_WINDOW_DAYS:
        logger.warning(
            'Find facet menu uses the configured window because {} days is outside {}..{}',
            window,
            MIN_WINDOW_DAYS,
            MAX_WINDOW_DAYS,
        )
        window = settings.window_days
    cached = getattr(app.state, 'finder_catalogue_cache', None)
    if cached is not None:
        expires, cached_window, catalogue = cached
        if expires > now and cached_window == window:
            return catalogue
    try:
        profile = _profile(app)
        orq = resolve_orq_client(
            profile.api_key if profile else None,
            server_url=(profile.server or DEFAULT_ORQ_BASE_URL) if profile else None,
        )
        catalogue = await load_facet_catalogue(
            orq,
            start=now - timedelta(days=window),
            end=now,
            limit=50,
        )
    except (ImportError, ValueError) as exc:
        logger.warning('Find facet menu is empty because the Orq client could not be resolved: {}', exc)
        # Cached too, briefly: a menu that never resolves would otherwise retry on every poll swap.
        app.state.finder_catalogue_cache = (now + timedelta(minutes=1), window, None)
        return None
    app.state.finder_catalogue_cache = (now + timedelta(minutes=5), window, catalogue)
    return catalogue


def _catalogue_kwargs(app: Any) -> dict[str, Any]:
    """The cached facet catalogue for a page render, or ``pending`` so the menu fetches it itself."""
    cached = getattr(app.state, 'finder_catalogue_cache', None)
    if cached is not None:
        expires, cached_window, catalogue = cached
        if expires > datetime.now(timezone.utc) and cached_window == _settings(app).window_days:
            return {'catalogue': catalogue}
    return {'pending': True}


def _form_values(form: Any, name: str) -> list[str]:
    values = form.getlist(name) if hasattr(form, 'getlist') else [form.get(name)]
    return [str(value) for value in values if value not in (None, '')]


def _optional_value(form: Any, name: str) -> object | None:
    raw = form.get(name)
    if raw in (None, ''):
        return None
    return raw


def _run_request(
    form: Any,
    settings: Any,
    *,
    anchor: PopulationRequest | None = None,
    query_fallback: str | None = None,
) -> RunRequest:
    """Build the run request; *anchor* pins ``end`` to a reviewed population so an unchanged review reuses its traces."""
    values = {
        'query': str(form.get('query') or query_fallback or '').strip(),
        'mode': str(form.get('mode') or 'immediate'),
        'window_days': form.get('window_days') or settings.window_days,
        'limit': form.get('limit') or settings.limit,
        'parallelism': form.get('parallelism') or settings.parallelism,
        'tokens_min': _optional_value(form, 'tokens_min'),
        'tokens_max': _optional_value(form, 'tokens_max'),
        'duration_ms_min': _optional_value(form, 'duration_ms_min'),
        'duration_ms_max': _optional_value(form, 'duration_ms_max'),
    }
    parsed = FinderRunForm.model_validate(values)
    facet_values = {name: frozenset(_form_values(form, f'facet_{name}')) for name in FACET_NAMES}
    numeric = NumericFilters(
        tokens_min=parsed.tokens_min,
        tokens_max=parsed.tokens_max,
        duration_ms_min=parsed.duration_ms_min,
        duration_ms_max=parsed.duration_ms_max,
    )
    end = anchor.end if anchor is not None and anchor.end is not None else datetime.now(timezone.utc)
    return RunRequest(
        query=parsed.query,
        mode=parsed.mode,
        population=PopulationRequest(
            start=end - timedelta(days=parsed.window_days),
            end=end,
            facets=FacetSelection(**facet_values),
            numeric=numeric,
            limit=parsed.limit,
        ),
        parallelism=parsed.parallelism,
    )


def _compiled_from_form(current: CompiledQuery, form: Any) -> CompiledQuery:
    from evaluatorq.common.judge import ClassifyQuestion
    from evaluatorq.trace_finder import ThresholdSelection, ValueSelection

    current_task = current.task
    kind = current_task.kind
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


def register_finder_routes(app: Any) -> None:  # noqa: C901
    """Register the Find page and its HTMX fragments on *app*."""

    @app.get('/find')
    async def find_page(req: Request) -> Response:
        api_available = _api_available(req.app)
        settings = _settings(req.app)
        store = _store(req.app) if api_available else None
        snapshot = await store.snapshot() if store is not None else RunSnapshot()
        return _html(page_html(snapshot, settings, api_available=api_available, **_catalogue_kwargs(req.app)))

    @app.post('/find/run')
    async def find_run(req: Request) -> Response:
        form = await req.form()
        rejected = request_rejected(req, form)
        if rejected:
            settings = _settings(req.app)
            return _html(
                fragment(
                    RunSnapshot(),
                    settings,
                    error=rejected,
                    api_available=_api_available(req.app),
                    **_catalogue_kwargs(req.app),
                ),
                status_code=403,
            )
        settings = _settings(req.app)
        store = _store(req.app)
        if store is None:
            return _html(
                fragment(
                    RunSnapshot(),
                    settings,
                    **_catalogue_kwargs(req.app),
                    error='Set ORQ_API_KEY to load traces',
                    api_available=False,
                )
            )
        try:
            request = _run_request(form, settings)
            snapshot = await store.compile(request, wait=False)
        except (ValidationError, ValueError, TypeError) as exc:
            return _html(
                fragment(RunSnapshot(), settings, **_catalogue_kwargs(req.app), error=str(exc)), status_code=422
            )
        return _html(fragment(snapshot, settings, **_catalogue_kwargs(req.app)))

    @app.get('/find/poll')
    async def find_poll(req: Request) -> Response:
        settings = _settings(req.app)
        store = _store(req.app)
        snapshot = await store.snapshot() if store is not None else RunSnapshot()
        return _html(fragment(snapshot, settings, **_catalogue_kwargs(req.app)))

    @app.post('/find/start')
    async def find_start(req: Request) -> Response:
        form = await req.form()
        rejected = request_rejected(req, form)
        if rejected:
            settings = _settings(req.app)
            return _html(
                fragment(RunSnapshot(), settings, **_catalogue_kwargs(req.app), error=rejected), status_code=403
            )
        settings = _settings(req.app)
        store = _store(req.app)
        if store is None:
            return _html(
                fragment(
                    RunSnapshot(),
                    settings,
                    **_catalogue_kwargs(req.app),
                    error='Set ORQ_API_KEY to load traces',
                    api_available=False,
                )
            )
        current = await store.snapshot()
        if current.state != 'awaiting_review' or current.request is None or current.compiled is None:
            return _html(
                fragment(current, settings, error='There is no plan waiting for review.', **_catalogue_kwargs(req.app)),
                status_code=409,
            )
        try:
            request = _run_request(
                form, settings, anchor=current.request.population, query_fallback=current.request.query
            )
            compiled = _compiled_from_form(current.compiled, form)
            snapshot = await store.start(request, compiled)
        except (ValidationError, ValueError, TypeError) as exc:
            return _html(fragment(current, settings, error=str(exc), **_catalogue_kwargs(req.app)), status_code=422)
        return _html(fragment(snapshot, settings, **_catalogue_kwargs(req.app)))

    @app.post('/find/cancel')
    async def find_cancel(req: Request) -> Response:
        form = await req.form()
        rejected = request_rejected(req, form)
        if rejected:
            settings = _settings(req.app)
            return _html(
                fragment(RunSnapshot(), settings, **_catalogue_kwargs(req.app), error=rejected), status_code=403
            )
        settings = _settings(req.app)
        store = _store(req.app)
        snapshot = await store.cancel() if store is not None else RunSnapshot()
        return _html(fragment(snapshot, settings, **_catalogue_kwargs(req.app)))

    @app.post('/find/reset')
    async def find_reset(req: Request) -> Response:
        form = await req.form()
        rejected = request_rejected(req, form)
        if rejected:
            settings = _settings(req.app)
            return _html(
                fragment(RunSnapshot(), settings, **_catalogue_kwargs(req.app), error=rejected), status_code=403
            )
        settings = _settings(req.app)
        store = _store(req.app)
        snapshot = await store.reset() if store is not None else RunSnapshot()
        return _html(fragment(snapshot, settings, **_catalogue_kwargs(req.app)))

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
        """Re-render the filter menu from the controls the client sends, so it never lags the page."""
        settings = _settings(req.app)
        params = req.query_params
        try:
            parsed = FinderRunForm.model_validate({
                'window_days': params.get('window_days') or settings.window_days,
                'limit': settings.limit,
                'parallelism': settings.parallelism,
            })
        except ValidationError as exc:
            logger.warning('Find facet menu uses the configured window because the request was invalid: {}', exc)
            parsed = FinderRunForm(
                window_days=settings.window_days, limit=settings.limit, parallelism=settings.parallelism
            )
        catalogue = await _load_catalogue(req.app, parsed.window_days)
        form_id = params.get('form_id')
        if form_id not in {'finder-query-form', 'finder-start-form'}:
            form_id = 'finder-query-form'
        selection = FacetSelection(**{name: frozenset(_form_values(params, f'facet_{name}')) for name in FACET_NAMES})
        numeric_values: dict[str, int | None] = {}
        for name in _NUMERIC_FIELDS:
            raw = _optional_value(params, name)
            if raw is None:
                numeric_values[name] = None
                continue
            try:
                value = int(str(raw))
                if value < 0:
                    raise ValueError('must be nonnegative')
            except (TypeError, ValueError):
                logger.warning('Find facet menu ignores invalid {}={}', name, raw)
                numeric_values[name] = None
            else:
                numeric_values[name] = value
        for facet in NUMERIC_FACET_NAMES:
            minimum, maximum = f'{facet}_min', f'{facet}_max'
            lower, upper = numeric_values[minimum], numeric_values[maximum]
            if lower is not None and upper is not None and lower > upper:
                logger.warning('Find facet menu ignores {} because it is below {}', maximum, minimum)
                numeric_values[maximum] = None
        numeric = NumericFilters(**numeric_values)
        return _html(facet_menu(catalogue, numeric=numeric, form_id=form_id, selection=selection))

    @app.get('/find/dismiss')
    def find_dismiss() -> Response:
        return Response('', media_type='text/html')
