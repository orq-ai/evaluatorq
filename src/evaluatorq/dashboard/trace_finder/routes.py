"""Route wiring and per-application runtime ownership for ``/find``."""

from __future__ import annotations

import asyncio
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.requests import Request  # noqa: TC002 — FastHTML inspects this annotation at runtime
from starlette.responses import Response

from evaluatorq.common.llm_client import resolve_llm_client
from evaluatorq.common.orq_client import (
    DEFAULT_ORQ_BASE_URL,
    OrqProfile,
    close_orq_client,
    list_orq_profiles,
    resolve_orq_client,
)
from evaluatorq.dashboard.security import request_rejected
from evaluatorq.dashboard.trace_finder import explorer_views
from evaluatorq.dashboard.trace_finder.search_views import search_fragment, search_page_html
from evaluatorq.dashboard.trace_finder.sessions import TraceSessionRegistry
from evaluatorq.dashboard.trace_finder.views import (
    drawer,
    facet_menu,
    fragment,
    missing_trace_drawer,
    page_html,
    scope_toggle,
)
from evaluatorq.trace_finder import (
    CompiledQuery,
    FacetSelection,
    NumericFilters,
    PopulationRequest,
    RunRequest,
    RunSnapshot,
    RunStore,
    TraceDetail,
    build_run_store,
    effective_settings,
    export_json,
    load_facet_catalogue,
)

if TYPE_CHECKING:
    from evaluatorq.trace_finder import FacetCatalogue
from evaluatorq.trace_finder.columns import COLUMNS, resolve_columns
from evaluatorq.trace_finder.explorer import QUICK_VIEWS, ExplorerView
from evaluatorq.trace_finder.models import FACET_NAMES, NUMERIC_FACET_NAMES, TraceRecord
from evaluatorq.trace_finder.orq_source import MAX_LIVE_TRACES
from evaluatorq.trace_finder.settings import (
    MAX_LIMIT,
    MAX_PARALLELISM,
    MAX_WINDOW_DAYS,
    MIN_LIMIT,
    MIN_PARALLELISM,
    MIN_WINDOW_DAYS,
    load_settings,
    save_settings,
)

_NUMERIC_FIELDS = tuple(f'{name}_{bound}' for name in NUMERIC_FACET_NAMES for bound in ('min', 'max'))
CONFIRM_ROWS = 500


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


class _FinderSettingsChangedError(Exception):
    """A session store build raced with the settings generation transition."""


def initialize_finder_settings(app: Any) -> None:
    """Resolve the saved profile while constructing the app, outside request handlers."""
    if getattr(app.state, 'finder_settings', None) is not None:
        return
    settings = effective_settings()
    profiles = list_orq_profiles() if settings.orq_profile is not None else ()
    app.state.finder_profile = next((p for p in profiles if p.name == settings.orq_profile), None)
    if settings.orq_profile is not None and app.state.finder_profile is None:
        logger.warning(
            'Saved Orq profile {} is unavailable; select another profile or Environment', settings.orq_profile
        )
    app.state.finder_settings = settings
    app.state.finder_generation = 0
    app.state.finder_store_lock = asyncio.Lock()
    app.state.dashboard_session_signing_key = secrets.token_bytes(32)
    app.state.trace_sessions = TraceSessionRegistry()


def _settings(app: Any) -> Any:
    settings = getattr(app.state, 'finder_settings', None)
    if settings is None:
        raise RuntimeError('Finder settings must be initialized before handling requests.')
    return settings


def _profile(app: Any) -> OrqProfile | None:
    settings = _settings(app)
    profile: OrqProfile | None = getattr(app.state, 'finder_profile', None)
    if settings.orq_profile is not None and profile is None:
        raise ValueError(
            f'Orq profile {settings.orq_profile} is unavailable. Choose another profile or Environment in Settings.'
        )
    if profile is not None and '*' in profile.api_key:
        raise ValueError(f'Orq profile {profile.name} has no usable API key. Choose another profile or Environment.')
    return profile


def _api_available(app: Any) -> bool:
    try:
        return _profile(app) is not None or bool(os.environ.get('ORQ_API_KEY', '').strip())
    except ValueError:
        return False


def _unavailable_reason(app: Any) -> str:
    try:
        _profile(app)
    except ValueError as exc:
        return str(exc)
    failure = getattr(app.state, 'finder_unavailable_reason', None)
    if failure:
        return str(failure)
    return 'Set ORQ_API_KEY to load traces'


async def _build_store(app: Any) -> RunStore | None:
    """Build the app-owned store, returning ``None`` when Orq is unavailable."""
    settings = _settings(app)
    resolved = None
    try:
        profile = _profile(app)
        resolved = resolve_llm_client(
            extra_api_key=profile.api_key if profile else None,
            orq_host=(profile.server or DEFAULT_ORQ_BASE_URL) if profile else None,
            require_orq=True,
            max_retries=0,
        )
        orq = resolve_orq_client(
            profile.api_key if profile else None,
            base_url=(profile.server or DEFAULT_ORQ_BASE_URL) if profile else None,
        )
    except (ImportError, ValueError) as exc:
        if resolved is not None and resolved.owned:
            await resolved.client.close()
        app.state.finder_unavailable_reason = str(exc)
        logger.warning('Find surface is unavailable because an Orq client could not be resolved: {}', exc)
        return None
    app.state.finder_unavailable_reason = None

    async def cleanup() -> None:
        try:
            await close_orq_client(orq)
        finally:
            if resolved.owned:
                await resolved.client.close()

    return build_run_store(settings, client=resolved.client, orq=orq, cleanup=cleanup)


async def _store(
    app: Any,
    *,
    surface: Literal['search', 'traces'] = 'traces',
    session_id: str | None = None,
    request_state: Any | None = None,
) -> RunStore | None:
    if surface == 'traces':
        if session_id is None:
            raise RuntimeError('A verified browser session is required for the Traces store.')

        async def build_session_store() -> RunStore | None:
            async with app.state.finder_store_lock:
                generation = app.state.finder_generation
            store = await _build_store(app)
            async with app.state.finder_store_lock:
                current_generation = app.state.finder_generation
            if current_generation != generation:
                if store is not None:
                    try:
                        await store.close()
                    except Exception as exc:  # noqa: BLE001 — stale stores must never survive a settings change.
                        logger.opt(exception=True).warning('Could not close a stale Traces store: {}', exc)
                raise _FinderSettingsChangedError
            return store

        async def get_session_store() -> tuple[bool, RunStore | None]:
            try:
                store = await app.state.trace_sessions.get(session_id, build_session_store, request_state=request_state)
            except _FinderSettingsChangedError:
                return False, None
            return True, store

        while True:
            is_current, store = await get_session_store()
            if is_current:
                return store

    state_name = 'finder_search_store'
    store = getattr(app.state, state_name, None)
    if store is not None:
        return store
    async with app.state.finder_store_lock:
        store = getattr(app.state, state_name, None)
        if store is None:
            store = await _build_store(app)
            if store is not None:
                setattr(app.state, state_name, store)
        return store


async def _load_catalogue(app: Any, window_days: int | None = None) -> FacetCatalogue | None:
    """Load and cache facet values for five minutes; failures render an empty menu."""
    now = datetime.now(timezone.utc)
    settings = _settings(app)
    generation = app.state.finder_generation
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
    orq = None
    try:
        profile = _profile(app)
        orq = resolve_orq_client(
            profile.api_key if profile else None,
            base_url=(profile.server or DEFAULT_ORQ_BASE_URL) if profile else None,
        )
        catalogue = await load_facet_catalogue(
            orq,
            start=now - timedelta(days=window),
            end=now,
            limit=50,
        )
        if settings.orq_project_id and settings.orq_project_name:
            catalogue = catalogue.model_copy(update={'project': (settings.orq_project_name,)})
    except Exception as exc:  # noqa: BLE001 — provider and SDK failures vary; this UI boundary must degrade visibly.
        logger.opt(exception=True).warning(
            'Find facet menu is unavailable because loading facet values failed: {}', exc
        )
        # Cached too, briefly: a menu that never resolves would otherwise retry on every poll swap.
        if generation == app.state.finder_generation:
            app.state.finder_catalogue_cache = (now + timedelta(minutes=1), window, None)
        return None
    finally:
        if orq is not None:
            try:
                await close_orq_client(orq)
            except Exception as exc:  # noqa: BLE001 — cleanup failure must not replace the menu response.
                logger.opt(exception=True).warning('Could not close the facet catalogue Orq client: {}', exc)
    if generation != app.state.finder_generation:
        logger.debug('Discarding facet values loaded for a retired finder configuration')
        return None
    app.state.finder_catalogue_cache = (now + timedelta(minutes=5), window, catalogue)
    return catalogue


async def _warm_catalogue(app: Any, window_days: int) -> None:
    """Start one background fetch so opening the filter menu can reuse it."""
    async with app.state.finder_store_lock:
        generation = app.state.finder_generation
        cached = getattr(app.state, 'finder_catalogue_cache', None)
        if cached is not None and cached[0] > datetime.now(timezone.utc) and cached[1] == window_days:
            return
        active = getattr(app.state, 'finder_catalogue_warmup', None)
        if active is not None and active[0:2] == (generation, window_days) and not active[2].done():
            return
        if active is not None and not active[2].done():
            active[2].cancel()
        task = asyncio.create_task(_load_catalogue(app, window_days))
        app.state.finder_catalogue_warmup = (generation, window_days, task)


async def _catalogue_for_window(app: Any, window_days: int) -> FacetCatalogue | None:
    """Share an in-flight warmup with the menu request for the same window."""
    active = getattr(app.state, 'finder_catalogue_warmup', None)
    if active is not None and active[0:2] == (app.state.finder_generation, window_days) and not active[2].done():
        try:
            return await asyncio.shield(active[2])
        except asyncio.CancelledError:
            if not active[2].cancelled():
                raise
    return await _load_catalogue(app, window_days)


def _catalogue_window(app: Any, snapshot: RunSnapshot | None, explorer_view: ExplorerView | None) -> int:
    """Use the range shown in the controls for the facet catalogue."""
    window_days = _settings(app).window_days
    if snapshot is not None and snapshot.request is not None:
        population = snapshot.request.population
        if population.start is not None and population.end is not None:
            window_days = max(1, round((population.end - population.start).total_seconds() / 86400))
    elif explorer_view is not None and explorer_view.start is not None and explorer_view.end is not None:
        window_days = max(1, round((explorer_view.end - explorer_view.start).total_seconds() / 86400))
    return window_days


def _catalogue_kwargs(
    app: Any, snapshot: RunSnapshot | None = None, *, explorer_view: ExplorerView | None = None
) -> dict[str, Any]:
    """The cached facet catalogue for a page render, or ``pending`` so the menu fetches it itself."""
    window_days = _catalogue_window(app, snapshot, explorer_view)
    cached = getattr(app.state, 'finder_catalogue_cache', None)
    if cached is not None:
        expires, cached_window, catalogue = cached
        if expires > datetime.now(timezone.utc) and cached_window == window_days:
            return {'catalogue': catalogue}
    return {'pending': True}


async def warm_initial_finder(app: Any, session_id: str, request_state: Any) -> RunStore | None:
    """Start the default rows and facets without waiting for either Orq query."""
    if not _api_available(app):
        return None
    store = await _store(app, session_id=session_id, request_state=request_state)
    if store is None:
        return None
    explorer = store.explorer
    if explorer is not None:
        async with app.state.finder_store_lock:
            if (await explorer.view()).state == 'idle':
                end = datetime.now(timezone.utc)
                await explorer.load(
                    end - timedelta(days=7),
                    end,
                    explorer_views.DEFAULT_EXPLORER_ROWS,
                    facets=FacetSelection(project_id=_settings(app).orq_project_id),
                    numeric=NumericFilters(),
                )
        explorer_view = await explorer.view()
        snapshot = await store.snapshot()
        await _warm_catalogue(app, _catalogue_window(app, snapshot, explorer_view))
    return store


def _form_values(form: Any, name: str) -> list[str]:
    values = form.getlist(name) if hasattr(form, 'getlist') else [form.get(name)]
    return [str(value) for value in values if value not in (None, '')]


def _optional_value(form: Any, name: str) -> object | None:
    raw = form.get(name)
    if raw in (None, ''):
        return None
    return raw


MAX_RANGE = timedelta(days=30)


def _range_values(form: Any) -> tuple[str, str]:
    """Combine the calendar and time controls, accepting older datetime-local submissions too."""
    values: list[str] = []
    for name in ('from', 'to'):
        date = str(form.get(name) or '')
        time = form.get(f'{name}_time')
        values.append(f'{date}T{time}' if time is not None and 'T' not in date else date)
    return values[0], values[1]


def parse_range(
    from_value: str,
    to_value: str,
    tz_offset: str | None,
    from_tz_offset: str | None = None,
    to_tz_offset: str | None = None,
) -> tuple[datetime, datetime]:
    """Turn local timestamps and their browser offsets (getTimezoneOffset) into UTC bounds."""

    def zone_for(value: str | None) -> timezone:
        try:
            return timezone(timedelta(minutes=-int(value or '')))
        except ValueError:
            logger.warning('Explorer time range has no usable browser offset {!r}; reading it as UTC', value)
            return timezone.utc

    # Older clients submit one offset for both endpoints. Keep that fallback for
    # direct callers and already-open dashboard pages.
    start_zone = zone_for(from_tz_offset if from_tz_offset is not None else tz_offset)
    end_zone = zone_for(to_tz_offset if to_tz_offset is not None else tz_offset)
    try:
        if 'T' not in from_value or 'T' not in to_value:
            raise ValueError('A time is required at each end.')
        start = datetime.fromisoformat(from_value).replace(tzinfo=start_zone).astimezone(timezone.utc)
        end = datetime.fromisoformat(to_value).replace(tzinfo=end_zone).astimezone(timezone.utc)
    except ValueError as exc:
        raise ValueError('From and To must be full dates and times.') from exc
    if start >= end:
        raise ValueError('From must be before To.')
    if end - start > MAX_RANGE:
        raise ValueError('The time range can be at most 30 days, the API retention.')
    return start, end


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
    if anchor is not None and anchor.end is not None:
        end = anchor.end
        start = anchor.start if anchor.start is not None else end - timedelta(days=parsed.window_days)
    elif form.get('from') and form.get('to'):
        start, end = parse_range(*_range_values(form), str(form.get('tz_offset') or '0'))
    else:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=parsed.window_days)
    return RunRequest(
        query=parsed.query,
        mode=parsed.mode,
        population=PopulationRequest(
            start=start,
            end=end,
            facets=FacetSelection(project_id=settings.orq_project_id, **facet_values),
            numeric=numeric,
            limit=parsed.limit,
        ),
        parallelism=parsed.parallelism,
    )


def _compiled_from_form(current: CompiledQuery, form: Any, *, prefix: str) -> CompiledQuery:
    """Rebuild one reviewed dimension from its ``prefix``-named form fields, keeping current values for blanks."""
    from evaluatorq.common.judge import ClassifyQuestion
    from evaluatorq.trace_finder import ThresholdSelection, ValueSelection

    current_task = current.task
    kind = current_task.kind
    instructions = str(form.get(f'{prefix}instructions') or current_task.instructions).strip()

    if kind == 'choice':
        existing = current_task.criteria.items() if isinstance(current_task.criteria, dict) else ()
        existing_pairs = tuple(existing)
        pairs: list[tuple[str, str]] = []
        for index, (old_label, old_description) in enumerate(existing_pairs):
            label = str(form.get(f'{prefix}criteria_label_{index}') or old_label).strip()
            description = str(form.get(f'{prefix}criteria_description_{index}') or old_description or '').strip()
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
            str(
                form.get(f'{prefix}score_criteria_{index}') or form.get(f'{prefix}criteria_{index}') or description
            ).strip()
            for index, description in enumerate(existing_descriptions)
        ]
        if not criteria_values:
            criteria_values = ['Low match', 'High match']
        criteria = criteria_values
    elif kind == 'noul':
        criteria = None
    else:
        raise ValueError('kind must be one of choice, noul, or score')

    raw_noul_threshold = form.get(f'{prefix}noul_threshold')
    threshold = float(current_task.noul_threshold if raw_noul_threshold in (None, '') else raw_noul_threshold)
    task = ClassifyQuestion(
        kind=kind,
        instructions=instructions,
        criteria=criteria,
        noul_threshold=threshold,
        state={},
    )

    values = _form_values(form, f'{prefix}selection_value') or _form_values(form, f'{prefix}selection_values')
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
        rule = str(form.get(f'{prefix}selection_rule') or '')
        if ':' in rule:
            operator, raw_threshold = rule.split(':', 1)
            score_threshold = float(raw_threshold)
        else:
            operator = str(form.get(f'{prefix}selection_operator') or getattr(current.selection, 'operator', 'gte'))
            raw_threshold = form.get(f'{prefix}selection_threshold')
            if raw_threshold in (None, ''):
                raw_threshold = form.get(f'{prefix}selection_value')
            score_threshold = float(
                getattr(current.selection, 'value', 0.5) if raw_threshold in (None, '') else raw_threshold
            )
        selection = ThresholdSelection(operator=operator, value=score_threshold, kind='threshold')
    name = str(form.get(f'{prefix}name') or current.name).strip() or current.name
    return CompiledQuery(name=name, task=task, selection=selection)


def _html(content: str, *, status_code: int = 200) -> Response:
    return Response(content, status_code=status_code, media_type='text/html')


def register_finder_routes(app: Any) -> None:  # noqa: C901
    """Register the Trace search and Traces pages with shared HTMX endpoints."""
    initialize_finder_settings(app)

    def is_search(req: Request) -> bool:
        return req.query_params.get('surface') == 'search' or getattr(req.state, 'finder_surface', None) == 'search'

    async def store_for(req: Request) -> RunStore | None:
        surface: Literal['search', 'traces'] = 'search' if is_search(req) else 'traces'
        session_id = None if surface == 'search' else req.state.dashboard_session_id
        return await _store(req.app, surface=surface, session_id=session_id, request_state=req.scope['state'])

    def render_fragment(req: Request, snapshot: RunSnapshot, settings: Any, **kwargs: Any) -> str:
        if is_search(req):
            return search_fragment(
                snapshot,
                settings,
                error=kwargs.get('error'),
                api_available=kwargs.get('api_available', True),
                catalogue=kwargs.get('catalogue'),
                pending=kwargs.get('pending', False),
            )
        return fragment(snapshot, settings, **kwargs)

    async def _explorer_html(req: Request, *, oob: bool = False, error: str | None = None) -> str:
        store = await _store(req.app, session_id=req.state.dashboard_session_id, request_state=req.scope['state'])
        explorer = store.explorer if store is not None else None
        if store is None or explorer is None:
            return explorer_views.results(
                ExplorerView(),
                resolve_columns(None),
                records=None,
                snapshot=None,
                oob=oob,
                error=error,
                window_days=_settings(req.app).window_days,
                traces_layout=not is_search(req),
            )
        view = await explorer.view()
        snapshot = await store.snapshot_for_render()
        results = snapshot.results if snapshot.within_results else None
        page_ids = [row.trace_id for row in view.page_rows(results)]
        records = None
        if view.view == 'trajectories' and view.rows:
            records = explorer.cached_records(page_ids) if view.state == 'loading' else await explorer.records(page_ids)
        return explorer_views.results(
            view,
            resolve_columns(_settings(req.app).explorer_columns),
            records=records,
            snapshot=snapshot,
            oob=oob,
            error=error,
            window_days=_settings(req.app).window_days,
            traces_layout=not is_search(req),
        )

    @app.get('/find')
    async def find_page(req: Request) -> Response:
        settings = _settings(req.app)
        store = await _store(req.app, surface='search')
        snapshot = await store.snapshot() if store is not None else RunSnapshot()
        return _html(
            search_page_html(
                snapshot,
                settings,
                api_available=store is not None,
                error=_unavailable_reason(req.app) if store is None else None,
                **_catalogue_kwargs(req.app, snapshot),
            )
        )

    @app.get('/traces')
    async def traces_page(req: Request) -> Response:
        settings = _settings(req.app)
        store = await warm_initial_finder(req.app, req.state.dashboard_session_id, req.scope['state'])
        api_available = store is not None
        explorer = store.explorer if store is not None else None
        explorer_view = await explorer.view() if explorer is not None else None
        snapshot = await store.snapshot() if store is not None else RunSnapshot()
        return _html(
            page_html(
                snapshot,
                settings,
                api_available=api_available,
                error=_unavailable_reason(req.app) if not api_available else None,
                explorer_html=await _explorer_html(req) if store is not None else '',
                explorer_view=explorer_view,
                has_rows=bool(explorer_view and explorer_view.rows),
                **_catalogue_kwargs(req.app, snapshot, explorer_view=explorer_view),
            )
        )

    @app.post('/find/run')
    async def find_run(req: Request) -> Response:
        form = await req.form()
        req.state.finder_surface = form.get('surface')
        rejected = request_rejected(req, form)
        if rejected:
            settings = _settings(req.app)
            return _html(
                render_fragment(
                    req,
                    RunSnapshot(),
                    settings,
                    error=rejected,
                    api_available=_api_available(req.app),
                    **_catalogue_kwargs(req.app),
                ),
                status_code=403,
            )
        settings = _settings(req.app)
        store = await store_for(req)
        if store is None:
            return _html(
                render_fragment(
                    req,
                    RunSnapshot(),
                    settings,
                    **_catalogue_kwargs(req.app),
                    error=_unavailable_reason(req.app),
                    api_available=False,
                )
            )
        try:

            async def hydrate_explorer_rows(ids: list[str]) -> tuple[TraceRecord, ...]:
                explorer_store = store.explorer
                if explorer_store is None:
                    raise RuntimeError('The trace table is unavailable.')
                records = await explorer_store.records(ids)
                missing = sum(record is None for record in records.values())
                if missing:
                    logger.warning('Skipping {} loaded trace(s) without usable messages in the classification', missing)
                return tuple(record for record in records.values() if record is not None)

            if str(form.get('scope') or 'new') == 'within':
                explorer = store.explorer
                explorer_view = await explorer.view() if explorer is not None else None
                if explorer is None or explorer_view is None or not explorer_view.rows:
                    return _html(
                        render_fragment(
                            req,
                            await store.snapshot(),
                            settings,
                            error='Load traces first, then ask within the results.',
                            **_catalogue_kwargs(req.app),
                        )
                    )
                base = _run_request(form, settings)
                # The AI trace limit caps how many rows are judged, not how many the table loaded.
                limit = min(len(explorer_view.rows), base.population.limit)
                request = base.model_copy(
                    update={
                        'mode': 'review' if limit > CONFIRM_ROWS else base.mode,
                        'population': PopulationRequest(
                            start=explorer_view.start,
                            end=explorer_view.end,
                            facets=explorer_view.facets,
                            numeric=explorer_view.numeric,
                            limit=limit,
                        ),
                    }
                )
                ids = [row.trace_id for row in explorer_view.rows]

                async def loaded_traces() -> tuple[TraceRecord, ...]:
                    return await hydrate_explorer_rows(ids)

                snapshot = await store.compile(request, wait=False, traces=loaded_traces)
            else:
                request = _run_request(form, settings)
                explorer = store.explorer if req.state.finder_surface != 'search' else None

                async def load_table(population: PopulationRequest) -> tuple[TraceRecord, ...]:
                    if explorer is None:
                        raise RuntimeError('The trace table is unavailable.')
                    end = population.end or datetime.now(timezone.utc)
                    start = population.start or end - timedelta(days=settings.window_days)
                    view = await explorer.load(
                        start,
                        end,
                        population.limit,
                        facets=population.facets,
                        numeric=population.numeric,
                        wait=True,
                        warm_trajectories=False,
                    )
                    if view.state == 'failed':
                        raise RuntimeError(view.error or 'Trace table loading failed.')
                    return await hydrate_explorer_rows([row.trace_id for row in view.rows])

                snapshot = await store.compile(request, wait=False, table=load_table if explorer is not None else None)
        except (ValidationError, ValueError, TypeError) as exc:
            return _html(
                render_fragment(req, RunSnapshot(), settings, **_catalogue_kwargs(req.app), error=str(exc)),
                status_code=422,
            )
        return _html(render_fragment(req, snapshot, settings, **_catalogue_kwargs(req.app, snapshot)))

    @app.get('/find/poll')
    async def find_poll(req: Request) -> Response:
        settings = _settings(req.app)
        store = await store_for(req)
        if isinstance(store, RunStore):
            snapshot = await store.snapshot_for_render()
        else:
            snapshot = await store.snapshot() if store is not None else RunSnapshot()
        explorer = getattr(store, 'explorer', None) if not is_search(req) else None
        explorer_view = await explorer.view() if explorer is not None else None
        fragment_kwargs = _catalogue_kwargs(req.app, snapshot, explorer_view=explorer_view)
        if explorer_view is not None:
            fragment_kwargs['explorer_view'] = explorer_view
        body = render_fragment(
            req,
            snapshot,
            settings,
            api_available=store is not None,
            error=_unavailable_reason(req.app) if store is None else None,
            **fragment_kwargs,
        )
        if explorer is not None and explorer_view is not None:
            # A within-results run may narrow the table to zero rows, so it still needs the refresh.
            if (explorer_view.rows or snapshot.within_results) and (
                snapshot.state in {'classifying', 'completed'} or snapshot.results
            ):
                body += await _explorer_html(req, oob=True)
        return _html(body)

    @app.post('/find/load')
    async def find_load(req: Request) -> Response:
        form = await req.form()
        rejected = request_rejected(req, form)
        if rejected:
            return _html(await _explorer_html(req, error=rejected))
        store = await _store(req.app, session_id=req.state.dashboard_session_id, request_state=req.scope['state'])
        explorer = store.explorer if store is not None else None
        if store is None or explorer is None:
            return _html(await _explorer_html(req, error=_unavailable_reason(req.app)))
        settings = _settings(req.app)
        try:
            start, end = parse_range(
                *_range_values(form), form.get('tz_offset'), form.get('from_tz_offset'), form.get('to_tz_offset')
            )
            rows = int(str(form.get('rows') or explorer_views.DEFAULT_EXPLORER_ROWS))
            if not 1 <= rows <= MAX_LIVE_TRACES:
                raise ValueError(f'Rows must be between 1 and {MAX_LIVE_TRACES}.')
            numeric = NumericFilters(**{
                name: int(str(raw)) for name in _NUMERIC_FIELDS if (raw := _optional_value(form, name)) is not None
            })
        except (ValueError, ValidationError) as exc:
            logger.warning('Explorer load rejected: {}', exc)
            return _html(await _explorer_html(req, error=str(exc)))
        facets = FacetSelection(
            project_id=settings.orq_project_id,
            **{name: frozenset(_form_values(form, f'facet_{name}')) for name in FACET_NAMES},
        )
        await explorer.load(start, end, rows, facets=facets, numeric=numeric)
        snapshot = await store.snapshot_for_render()
        explorer_view = await explorer.view()
        await _warm_catalogue(req.app, _catalogue_window(req.app, snapshot, explorer_view))
        load_kwargs = _catalogue_kwargs(req.app, snapshot, explorer_view=explorer_view)
        load_kwargs.update(explorer_facets=facets, explorer_numeric=numeric, explorer_view=explorer_view)
        body_oob = (
            f'<div id="finder-body" hx-swap-oob="innerHTML">'
            f'{render_fragment(req, snapshot, settings, **load_kwargs)}</div>'
            f'<div id="finder-scope" hx-swap-oob="innerHTML">{scope_toggle(has_rows=bool((await explorer.view()).rows), selected=str(form.get("scope") or ""))}</div>'
        )
        return _html(await _explorer_html(req) + body_oob)

    @app.get('/find/rows')
    async def find_rows(req: Request) -> Response:
        store = await _store(req.app, session_id=req.state.dashboard_session_id, request_state=req.scope['state'])
        explorer = store.explorer if store is not None else None
        if explorer is not None:
            params = req.query_params
            direction = params.get('dir')
            page = params.get('page')
            await explorer.set_view(
                sort=params.get('sort') if params.get('sort') in COLUMNS else None,
                descending=None if direction not in {'asc', 'desc'} else direction == 'desc',
                page=int(page) if page and page.isdigit() else (0 if params.get('sort') else None),
                view=params.get('view') if params.get('view') in {'table', 'trajectories'} else None,
                matched_only={'1': True, '0': False}.get(params.get('matched_only') or ''),
                quick_view=params.get('quick_view') if params.get('quick_view') in QUICK_VIEWS else None,
            )
            view = await explorer.view()
            if view.state == 'loaded' and view.quick_view == 'conv_longest' and view.rows:
                try:
                    await explorer.message_counts()
                except Exception as exc:  # noqa: BLE001 - hydration must not jam the table
                    logger.warning('Counting conversation messages failed: {}', exc)
                    return _html(await _explorer_html(req, error=f'Could not count messages: {exc}'))
        body = await _explorer_html(req)
        if explorer is not None:
            body += f'<div id="finder-scope" hx-swap-oob="innerHTML">{scope_toggle(has_rows=bool((await explorer.view()).rows), selected=req.query_params.get("scope"))}</div>'
        return _html(body)

    @app.post('/find/columns')
    async def find_columns(req: Request) -> Response:
        form = await req.form()
        rejected = request_rejected(req, form)
        if rejected:
            return _html(await _explorer_html(req, error=rejected))
        keys = tuple(key for key in _form_values(form, 'columns') if key in COLUMNS)
        saved = load_settings()
        await asyncio.to_thread(save_settings, saved.model_copy(update={'explorer_columns': keys}))
        req.app.state.finder_settings = _settings(req.app).model_copy(update={'explorer_columns': keys})
        return _html(await _explorer_html(req))

    @app.post('/find/start')
    async def find_start(req: Request) -> Response:
        form = await req.form()
        req.state.finder_surface = form.get('surface')
        rejected = request_rejected(req, form)
        if rejected:
            settings = _settings(req.app)
            return _html(
                render_fragment(req, RunSnapshot(), settings, **_catalogue_kwargs(req.app), error=rejected),
                status_code=403,
            )
        settings = _settings(req.app)
        store = await store_for(req)
        if store is None:
            return _html(
                render_fragment(
                    req,
                    RunSnapshot(),
                    settings,
                    **_catalogue_kwargs(req.app),
                    error=_unavailable_reason(req.app),
                    api_available=False,
                )
            )
        current = await store.snapshot()
        if current.state != 'awaiting_review' or current.request is None or current.dimensions is None:
            return _html(
                render_fragment(
                    req,
                    current,
                    settings,
                    error='There is no plan waiting for review.',
                    **_catalogue_kwargs(req.app, current),
                ),
                status_code=409,
            )
        try:
            request = _run_request(
                form, settings, anchor=current.request.population, query_fallback=current.request.query
            )
            dimensions = tuple(
                _compiled_from_form(dimension, form, prefix=f'd{index}_')
                for index, dimension in enumerate(current.dimensions)
            )
            snapshot = await store.start(request, dimensions, wait=False)
        except (ValidationError, ValueError, TypeError) as exc:
            return _html(
                render_fragment(req, current, settings, error=str(exc), **_catalogue_kwargs(req.app, current)),
                status_code=422,
            )
        return _html(render_fragment(req, snapshot, settings, **_catalogue_kwargs(req.app, snapshot)))

    @app.post('/find/cancel')
    async def find_cancel(req: Request) -> Response:
        form = await req.form()
        req.state.finder_surface = form.get('surface')
        rejected = request_rejected(req, form)
        if rejected:
            settings = _settings(req.app)
            return _html(
                render_fragment(req, RunSnapshot(), settings, **_catalogue_kwargs(req.app), error=rejected),
                status_code=403,
            )
        settings = _settings(req.app)
        store = await store_for(req)
        snapshot = await store.cancel() if store is not None else RunSnapshot()
        return _html(
            render_fragment(
                req, snapshot, settings, api_available=store is not None, **_catalogue_kwargs(req.app, snapshot)
            )
        )

    @app.post('/find/reset')
    async def find_reset(req: Request) -> Response:
        form = await req.form()
        req.state.finder_surface = form.get('surface')
        rejected = request_rejected(req, form)
        if rejected:
            settings = _settings(req.app)
            return _html(
                render_fragment(req, RunSnapshot(), settings, **_catalogue_kwargs(req.app), error=rejected),
                status_code=403,
            )
        settings = _settings(req.app)
        store = await store_for(req)
        snapshot = await store.reset() if store is not None else RunSnapshot()
        if store is not None and store.explorer is not None:
            await store.explorer.set_view(quick_view='all')
        return _html(
            render_fragment(
                req, snapshot, settings, api_available=store is not None, **_catalogue_kwargs(req.app, snapshot)
            )
            + (await _explorer_html(req, oob=True) if not is_search(req) else '')
        )

    @app.get('/find/trace-spans')
    async def find_trace_spans(req: Request) -> Response:
        trace_id = req.query_params.get('trace_id', '')
        store = await store_for(req)
        if store is None or store.explorer is None or not trace_id:
            return _html('<p class="finder-empty">Span loading is unavailable.</p>', status_code=404)
        if await store.explorer.row(trace_id) is None:
            return _html('<p class="finder-empty">This trace is not in the loaded table.</p>', status_code=404)
        try:
            spans = await store.explorer.spans(trace_id)
        except Exception as error:  # noqa: BLE001 - the conversation drawer remains usable if span lookup fails
            logger.warning('Find span lookup failed for trace {}: {}', trace_id, error)
            return _html('<p class="finder-empty" role="status">Could not load spans. Try again.</p>', status_code=200)
        try:
            error_message = await store.explorer.first_error_message(trace_id, spans)
        except Exception as error:  # noqa: BLE001 - raw status text is optional and must not block summaries
            logger.warning('Find span status lookup failed for trace {}: {}', trace_id, type(error).__name__)
            error_message = None
        from evaluatorq.dashboard.trace_finder.views import span_tree

        return _html(span_tree(trace_id, spans, first_error_message=error_message))

    @app.get('/find/trace/{trace_id:path}')
    async def find_trace(trace_id: str, req: Request) -> Response:
        store = await store_for(req)
        if store is None:
            return _html('<p class="finder-empty">Trace finding is unavailable.</p>', status_code=404)
        raw_msg = req.query_params.get('msg', '')
        msg = int(raw_msg) if raw_msg.isdigit() else None
        row = await store.explorer.row(trace_id) if store.explorer is not None else None
        snapshot = await store.snapshot()
        detail = await store.trace_detail(trace_id) if row is None or snapshot.within_results else None
        # A within-results run can narrow the visible explorer rows while its drawer
        # request is in flight. The explorer keeps the hydrated records by trace ID,
        # so recover from that cache even when the row has since left the visible set.
        if detail is None and store.explorer is not None and (row is not None or snapshot.within_results):
            record = (await store.explorer.records([trace_id])).get(trace_id)
            if record is not None:
                detail = TraceDetail(trace=record, projection=None, classification=None)
            elif row is not None or await store.explorer.has_cached_record(trace_id):
                logger.warning('Find drawer could not load messages for explorer trace {}', trace_id)
                return _html(
                    missing_trace_drawer(
                        trace_id,
                        reason='The messages could not be loaded for this trace. Open it in Orq instead.',
                        traces_layout=req.query_params.get('surface') == 'traces',
                    )
                )
        if detail is None:
            # htmx does not swap a 4xx body, so a 404 here would leave the click silently doing nothing.
            logger.warning('Find drawer requested trace {} that is not in the current run', trace_id)
            return _html(missing_trace_drawer(trace_id))
        return _html(drawer(detail, msg=msg, row=row, traces_layout=req.query_params.get('surface') == 'traces'))

    @app.get('/find/export.json')
    async def find_export(req: Request) -> Response:
        store = await store_for(req)
        if store is None:
            return Response('Not found', status_code=404, media_type='text/plain')
        snapshot = await store.snapshot()
        if snapshot.state != 'completed' or snapshot.request is None or snapshot.dimensions is None:
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
        catalogue = await _catalogue_for_window(req.app, parsed.window_days)
        form_id = params.get('form_id')
        allowed_form_ids = {'finder-query-form', 'finder-start-form'}
        if params.get('counts') == 'loaded':
            # Only the Traces surface requests row-scoped counts, and its
            # toolbar checkboxes must submit with the table reload form.
            allowed_form_ids.add('explorer-load-form')
        if form_id not in allowed_form_ids:
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
        loaded_rows = None
        if params.get('counts') == 'loaded':
            store = await _store(req.app, session_id=req.state.dashboard_session_id, request_state=req.scope['state'])
            explorer = store.explorer if store is not None else None
            loaded_rows = (await explorer.view()).rows if explorer is not None else None
        return _html(
            facet_menu(
                catalogue,
                numeric=numeric,
                open_=params.get('open') == '1',
                form_id=form_id,
                selection=selection,
                loaded_rows=loaded_rows,
            )
        )

    @app.get('/find/dismiss')
    def find_dismiss() -> Response:
        return Response('', media_type='text/html')
