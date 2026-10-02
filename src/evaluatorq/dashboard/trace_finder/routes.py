"""Route wiring and per-application runtime ownership for ``/find``."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
import secrets
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.requests import Request  # noqa: TC002 — FastHTML inspects this annotation at runtime
from starlette.responses import Response

from evaluatorq.common.llm_client import resolve_llm_client
from evaluatorq.common.orq_client import (
    OrqProfile,
    close_orq_client,
    list_orq_profiles,
    resolve_orq_client,
)
from evaluatorq.common.run_store_dir import get_store_dir
from evaluatorq.dashboard.auth import DashboardAuth, build_auth_clients, resolve_dashboard_auth
from evaluatorq.dashboard.insights_launch import (
    FINDER_EXPORT_REFERENCE_DIR,
    read_private_finder_reference,
    validate_private_finder_reference_dir,
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
    run_status,
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
from evaluatorq.trace_finder.export import export_filename

if TYPE_CHECKING:
    from evaluatorq.trace_finder import FacetCatalogue
from evaluatorq.trace_finder.columns import COLUMNS, resolve_columns
from evaluatorq.trace_finder.explorer import QUICK_VIEWS, ExplorerView, matches_first_view
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
from evaluatorq.trace_finder.table_csv import export_table_csv

_NUMERIC_FIELDS = tuple(f'{name}_{bound}' for name in NUMERIC_FACET_NAMES for bound in ('min', 'max'))
CONFIRM_ROWS = 500
_POLL_RENDER_SEQUENCE = itertools.count(1)
_FINDER_EXPORT_RETENTION = 50
_FINDER_EXPORT_HANDOFF_GRACE = timedelta(hours=1)
_FINDER_EXPORT_REFERENCE_MAX_AGE = timedelta(days=30)
_FINDER_EXPORT_THREAD_LOCK = threading.Lock()


@contextlib.contextmanager
def _finder_export_lock(export_dir: Path):
    """Serialize Finder export replacement and pruning across threads and processes."""
    export_dir.mkdir(parents=True, exist_ok=True)
    with _FINDER_EXPORT_THREAD_LOCK:
        lock_path = export_dir / '.finder-export.lock'
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, 'O_NOFOLLOW'):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            if hasattr(os, 'fchmod'):
                os.fchmod(descriptor, 0o600)
            try:
                import fcntl
            except ImportError:
                try:
                    import msvcrt
                except ImportError as exc:
                    raise OSError('Finder export inter-process locking is unavailable') from exc
                # msvcrt locks byte ranges, so make byte zero available first.
                # Competing initializers write the same byte at offset zero.
                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b'\0')
                os.lseek(descriptor, 0, os.SEEK_SET)
                # LK_LOCK waits for contention and raises if it cannot acquire
                # the lock. Never continue with only the process-local lock.
                msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _referenced_finder_exports(export_dir: Path) -> set[Path] | None:
    """Find exports in saved and active runs, returning None if references are uncertain."""
    referenced: set[Path] = set()
    try:
        run_paths = list(get_store_dir('insights-runs').glob('insights_*.json'))
    except OSError as exc:
        logger.warning('Could not inspect Insights runs for Finder export references: {}', exc)
        return None
    export_root = export_dir.resolve()
    for run_path in run_paths:
        try:
            data = json.loads(run_path.read_text(encoding='utf-8'))
            population = data.get('population') if isinstance(data, dict) else None
            value = population.get('finder_export') if isinstance(population, dict) else None
            if not isinstance(value, str) or not value:
                continue
            candidate = Path(value).expanduser()
            resolved = (candidate if candidate.is_absolute() else export_root / candidate).resolve()
            if (
                resolved.parent == export_root
                and resolved.name.startswith('trace-finder-')
                and resolved.suffix == '.json'
            ):
                referenced.add(resolved)
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            logger.warning('Could not inspect Insights run {} for Finder export references: {}', run_path, exc)
            return None
    runs_dir = get_store_dir('insights-runs')
    manifests_dir = runs_dir / '.manifests'
    references_dir = runs_dir / FINDER_EXPORT_REFERENCE_DIR
    now = datetime.now(timezone.utc).timestamp()
    try:
        validate_private_finder_reference_dir(references_dir)
    except FileNotFoundError:
        return referenced
    except OSError as exc:
        logger.warning('Could not inspect Finder export lease directory {}: {}', references_dir, exc)
        return None
    for marker in references_dir.glob('*.json'):
        candidate, trusted = _inflight_finder_export(marker, manifests_dir, export_root, now)
        if not trusted:
            return None
        if candidate is not None:
            referenced.add(candidate)
    return referenced


def _inflight_finder_export(
    marker: Path, manifests_dir: Path, export_root: Path, now: float
) -> tuple[Path | None, bool]:
    """Read one lease and say whether its export reference was trustworthy."""
    try:
        # An unreadable marker can block pruning while it might still belong to
        # a live run, but the lease itself expires after the maximum run age.
        if now - marker.lstat().st_mtime > _FINDER_EXPORT_REFERENCE_MAX_AGE.total_seconds():
            try:
                marker.unlink(missing_ok=True)
            except IsADirectoryError:
                logger.warning('Ignoring expired non-file Finder export lease {}', marker)
            else:
                logger.warning('Removed expired Finder export lease {} before pruning', marker)
            return None, True
        if not marker.stem.replace('-', '').replace('_', '').isalnum():
            logger.warning('Could not inspect in-flight Finder export reference with invalid run ID: {}', marker)
            return None, False
        data = read_private_finder_reference(marker)
        manifest_path = manifests_dir / f'{marker.stem}.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if not isinstance(manifest, dict):
            return None, False
        age = now - manifest_path.stat().st_mtime
        if manifest.get('status') != 'running' or age > _FINDER_EXPORT_REFERENCE_MAX_AGE.total_seconds():
            marker.unlink(missing_ok=True)
            return None, True
        value = data.get('finder_export')
        if not isinstance(value, str) or not value:
            return None, False
        candidate_path = Path(value).expanduser()
        candidate = (candidate_path if candidate_path.is_absolute() else export_root / candidate_path).resolve()
        if (
            candidate.parent == export_root
            and candidate.name.startswith('trace-finder-')
            and candidate.suffix == '.json'
        ):
            return candidate, True
        return None, True
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        logger.warning('Could not inspect in-flight Finder export reference {}: {}', marker, exc)
        return None, False


def _prune_finder_exports(export_dir: Path) -> None:
    """Keep recent handoff exports, the newest older exports, and saved Insights sources."""
    try:
        with _finder_export_lock(export_dir):
            _prune_finder_exports_locked(export_dir)
    except OSError as exc:
        logger.warning('Could not prune saved Finder exports in {}: {}', export_dir, exc)


def _prune_finder_exports_locked(export_dir: Path) -> None:
    """Prune exports while the cross-process export lock is held."""
    try:
        files = [path for path in export_dir.glob('trace-finder-*.json') if path.is_file() and not path.is_symlink()]
        files.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
        # With at most the retention limit of exports, every unreferenced file
        # is retained anyway. Avoid scanning all Insights manifests and leases
        # on the common download path until pruning could actually remove a file.
        if len(files) <= _FINDER_EXPORT_RETENTION:
            return
        retained = _referenced_finder_exports(export_dir)
        if retained is None:
            logger.warning('Skipping Finder export pruning because Insights references could not be verified')
            return
        handoff_cutoff = datetime.now(timezone.utc).timestamp() - _FINDER_EXPORT_HANDOFF_GRACE.total_seconds()
        retained.update(path.resolve() for path in files if path.stat().st_mtime >= handoff_cutoff)
        unreferenced = [path for path in files if path.resolve() not in retained]
        for path in unreferenced[_FINDER_EXPORT_RETENTION:]:
            try:
                path.unlink()
            except OSError as exc:  # noqa: PERF203 - one failed deletion must not stop retention cleanup
                logger.warning('Could not remove expired Finder export {}: {}', path.name, exc)
    except (OSError, RuntimeError) as exc:
        logger.warning('Could not prune saved Finder exports in {}: {}', export_dir, exc)


def _save_finder_export(export_dir: Path, export_name: str, payload: str) -> None:
    """Persist one completed export and prune older unreferenced files off-loop."""
    with _finder_export_lock(export_dir):
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=export_dir, prefix='.finder-', suffix='.tmp', delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(export_dir / export_name)
        finally:
            if temporary is not None:
                with contextlib.suppress(OSError):
                    temporary.unlink(missing_ok=True)
        _prune_finder_exports_locked(export_dir)


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
    profiles = list_orq_profiles() if settings.orq_auth_method == 'cli_profile' else ()
    app.state.finder_profile = next((p for p in profiles if p.name == settings.orq_profile), None)
    if settings.orq_auth_method == 'cli_profile' and app.state.finder_profile is None:
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


def selected_orq_profile(app: Any) -> OrqProfile | None:
    settings = _settings(app)
    if settings.orq_auth_method != 'cli_profile':
        return None
    profile: OrqProfile | None = getattr(app.state, 'finder_profile', None)
    if settings.orq_profile is not None and profile is None:
        raise ValueError(
            f'Orq profile {settings.orq_profile} is unavailable. Choose another profile or Environment in Settings.'
        )
    if profile is not None and '*' in profile.api_key:
        raise ValueError(f'Orq profile {profile.name} has no usable API key. Choose another profile or Environment.')
    return profile


def selected_dashboard_auth(app: Any) -> DashboardAuth:
    """Resolve the dashboard's saved method, including API keys outside CLI profiles."""

    settings = _settings(app)
    profile = selected_orq_profile(app)
    return resolve_dashboard_auth(settings, profiles=(profile,) if profile is not None else ())


def _api_available(app: Any) -> bool:
    try:
        auth = selected_dashboard_auth(app)
        return auth.method == 'cli_oauth' or bool(auth.api_key)
    except ValueError:
        return False


def _unavailable_reason(app: Any) -> str:
    try:
        auth = selected_dashboard_auth(app)
    except ValueError as exc:
        return str(exc)
    failure = getattr(app.state, 'finder_unavailable_reason', None)
    if failure:
        return str(failure)
    if auth.method == 'environment' and not auth.api_key:
        return 'ORQ_API_KEY is not set. Choose an authentication method in Settings.'
    return 'Choose a working authentication method in Settings to load traces'


async def _build_store(app: Any) -> RunStore | None:
    """Build the app-owned store, returning ``None`` when Orq is unavailable."""
    settings = _settings(app)
    resolved = None
    try:
        auth = selected_dashboard_auth(app)
        if auth.method == 'cli_oauth':
            orq, llm = build_auth_clients(auth, workspace=settings.orq_workspace, project=settings.orq_project_id)
        else:
            resolved = resolve_llm_client(
                extra_api_key=auth.api_key,
                orq_host=auth.base_url,
                require_orq=True,
                max_retries=0,
            )
            llm = resolved.client
            orq = resolve_orq_client(auth.api_key, base_url=auth.base_url)
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
            if resolved is not None and resolved.owned:
                await resolved.client.close()

    return build_run_store(settings, client=llm, orq=orq, cleanup=cleanup)


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
    llm = None
    try:
        auth = selected_dashboard_auth(app)
        if auth.method == 'cli_oauth':
            orq, llm = build_auth_clients(auth, workspace=settings.orq_workspace, project=settings.orq_project_id)
        else:
            orq = resolve_orq_client(auth.api_key, base_url=auth.base_url)
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
        if llm is not None:
            try:
                await llm.close()
            except Exception as exc:  # noqa: BLE001 — cleanup failure must not replace the menu response.
                logger.opt(exception=True).warning('Could not close the facet catalogue LLM client: {}', exc)
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

        def finish_warmup(done: asyncio.Task[FacetCatalogue | None]) -> None:
            current = getattr(app.state, 'finder_catalogue_warmup', None)
            if current is not None and current[2] is done:
                app.state.finder_catalogue_warmup = None
            if done.cancelled():
                return
            try:
                done.result()
            except Exception as exc:  # noqa: BLE001 — background failures must be observed and visible.
                logger.opt(exception=True).warning('Could not warm the facet catalogue: {}', exc)

        task.add_done_callback(finish_warmup)


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
                    initial_load=True,
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
        except (OverflowError, ValueError):
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
    elif (form.get('new_from') and form.get('new_to')) or (form.get('from') and form.get('to')):
        if form.get('new_from') and form.get('new_to'):
            start, end = parse_range(
                str(form.get('new_from')),
                str(form.get('new_to')),
                '0',
                str(form.get('new_from_tz_offset') or '0'),
                str(form.get('new_to_tz_offset') or '0'),
            )
        else:
            start, end = parse_range(
                *_range_values(form),
                str(form.get('tz_offset') or '0'),
                str(form.get('from_tz_offset') or '') or None,
                str(form.get('to_tz_offset') or '') or None,
            )
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

    async def _current_fragment_kwargs(req: Request, snapshot: RunSnapshot) -> dict[str, Any]:
        """Carry the active table filters into every /traces controls render."""
        if is_search(req):
            return _catalogue_kwargs(req.app, snapshot)
        store = await _store(req.app, session_id=req.state.dashboard_session_id, request_state=req.scope['state'])
        explorer = store.explorer if store is not None else None
        explorer_view = await explorer.view() if explorer is not None else None
        return {
            **_catalogue_kwargs(req.app, snapshot, explorer_view=explorer_view),
            'explorer_view': explorer_view,
            'explorer_facets': explorer_view.facets if explorer_view is not None else None,
            'explorer_numeric': explorer_view.numeric if explorer_view is not None else None,
        }

    async def _explorer_html(
        req: Request,
        *,
        oob: bool = False,
        error: str | None = None,
        poll: bool = False,
        poll_sequence: int | None = None,
    ) -> str:
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
                poll=poll,
                poll_sequence=poll_sequence,
            )
        view = await explorer.view()
        snapshot = await store.snapshot_for_render()
        results = snapshot.results if snapshot.within_results else None
        page_ids = [row.trace_id for row in view.page_rows(results)]
        records = None
        if view.view == 'trajectories' and view.rows:
            # A poll never fetches messages: it would wait on the hydration lock a run can hold for minutes.
            records = (
                explorer.cached_records(page_ids)
                if poll or view.state == 'loading' or view.trajectory_warming
                else await explorer.records(page_ids)
            )
        elif view.state == 'loaded' and not view.trajectory_warming:
            # Agent root spans often have no model of their own; the model is on a child
            # chat-completion span. Hydrate only visible agent rows missing that summary field.
            page_rows = view.page_rows(results)
            missing_agent_models = [
                row.trace_id
                for row in page_rows
                if not row.models
                and not row.is_error
                and (
                    row.agent_name
                    or 'agent' in (row.name or '').casefold()
                    or 'agent' in (row.operation or '').casefold()
                    or (row.product or '').casefold() == 'agents'
                )
            ]
            if missing_agent_models:
                await explorer.records(missing_agent_models)
                view = await explorer.view()
        return explorer_views.results(
            view,
            resolve_columns(_settings(req.app).explorer_columns),
            records=records,
            snapshot=snapshot,
            oob=oob,
            error=error,
            window_days=_settings(req.app).window_days,
            traces_layout=not is_search(req),
            poll=poll,
            poll_sequence=poll_sequence,
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
                auto_scope_pending=bool(
                    explorer_view and explorer_view.initial_load and explorer_view.state == 'loading'
                ),
                **_catalogue_kwargs(req.app, snapshot, explorer_view=explorer_view),
            )
        )

    @app.get('/traces/export.csv')
    async def traces_export_csv(req: Request) -> Response:
        """Download every row in the active traces table view, across all pages."""
        store = await _store(req.app, session_id=req.state.dashboard_session_id, request_state=req.scope['state'])
        explorer = store.explorer if store is not None else None
        if store is None or explorer is None:
            return Response('Trace table is unavailable.', status_code=503, media_type='text/plain')
        # Read the explorer on both sides of the run snapshot. A load/reset during
        # the snapshot read must not pair different table generations in this export.
        for _ in range(3):
            view = await explorer.view()
            snapshot = await store.snapshot_for_render()
            confirmed_view = await explorer.view()
            confirmed_snapshot = await store.snapshot_for_render()
            if (view.generation, view.version) == (
                confirmed_view.generation,
                confirmed_view.version,
            ) and snapshot.generation == confirmed_snapshot.generation:
                snapshot = confirmed_snapshot
                break
        else:
            return Response('Trace results changed while preparing the export. Retry the download.', status_code=409)
        source_ids = set(snapshot.trace_ids)
        snapshot_matches_view = (
            not (snapshot.results or snapshot.dimensions) or snapshot.explorer_generation == view.generation
        ) and (not source_ids or all(row.trace_id in source_ids for row in view.rows))
        active_snapshot = snapshot if snapshot.within_results and snapshot_matches_view else None
        results = active_snapshot.results if active_snapshot is not None else None
        dimensions = (
            tuple(dimension.name for dimension in (active_snapshot.dimensions or ()))
            if active_snapshot is not None and active_snapshot.results
            else ()
        )
        content = export_table_csv(
            matches_first_view(view, active_snapshot).visible_rows(results),
            _settings(req.app).explorer_columns,
            dimensions=dimensions,
            results=results,
        )
        return Response(
            content,
            media_type='text/csv',
            headers={'Content-Disposition': 'attachment; filename="traces.csv"'},
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
                    current_snapshot = await store.snapshot()
                    return _html(
                        render_fragment(
                            req,
                            current_snapshot,
                            settings,
                            error='Load traces first, then ask within the results.',
                            **await _current_fragment_kwargs(req, current_snapshot),
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

                async def loaded_traces_limited(limit: int) -> tuple[TraceRecord, ...]:
                    records, missing = await explorer.records_until_usable(
                        ids, generation=explorer_view.generation, limit=limit
                    )
                    if missing:
                        logger.warning(
                            'Skipping {} loaded trace(s) without usable messages in the classification', missing
                        )
                    return records

                snapshot = await store.compile(
                    request,
                    wait=False,
                    traces=loaded_traces,
                    traces_limited=loaded_traces_limited,
                    source_generation=lambda: explorer_view.generation,
                )
            else:
                request = _run_request(form, settings)
                explorer = store.explorer if req.state.finder_surface != 'search' else None
                loaded_table_generation: int | None = None

                async def load_table(population: PopulationRequest) -> tuple[TraceRecord, ...]:
                    nonlocal loaded_table_generation
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
                    loaded_table_generation = view.generation
                    return await hydrate_explorer_rows([row.trace_id for row in view.rows])

                snapshot = await store.compile(
                    request,
                    wait=False,
                    table=load_table if explorer is not None else None,
                    source_generation=lambda: loaded_table_generation,
                )
        except (ValidationError, ValueError, TypeError) as exc:
            current_snapshot = await store.snapshot()
            return _html(
                render_fragment(
                    req,
                    current_snapshot,
                    settings,
                    **await _current_fragment_kwargs(req, current_snapshot),
                    error=str(exc),
                ),
                status_code=422,
            )
        return _html(
            render_fragment(req, snapshot, settings, **await _current_fragment_kwargs(req, snapshot))
            + (await _explorer_html(req, oob=True) if not is_search(req) else '')
        )

    # How /traces polls without redrawing its controls. Two timers run: the Ask AI band polls
    # /find/poll while a run is compiling or classifying, and #explorer-results polls /find/rows
    # (HX-Trigger: explorer-results) while rows load or trajectories warm. Neither re-renders
    # #finder-controls, so the filter menu, its unsaved ticks and filtersDirty survive. /find/poll
    # swaps only #finder-run-status and answers 286 (htmx: stop polling) once the run settles; a
    # state that needs other controls (review, idle) retargets #finder-body instead. Explorer
    # renders from either timer are out of band and marked data-poll, and every explorer render
    # carries data-view-version (ExplorerView.version, bumped by each load and each visible view
    # change) and data-render-key. dashboard.js drops a poll render older than the table on screen
    # or identical to it, and after a real swap restores open menus, scroll, .sel and focus.
    @app.get('/find/poll')
    async def find_poll(req: Request) -> Response:
        poll_sequence = next(_POLL_RENDER_SEQUENCE)
        settings = _settings(req.app)
        store = await store_for(req)
        if isinstance(store, RunStore):
            snapshot = await store.snapshot_for_render()
        else:
            snapshot = await store.snapshot() if store is not None else RunSnapshot()
        explorer = getattr(store, 'explorer', None) if not is_search(req) else None
        explorer_view = await explorer.view() if explorer is not None else None
        explorer_oob = ''
        if explorer is not None and explorer_view is not None:
            # A within-results run may narrow the table to zero rows, so it still needs the refresh.
            if (explorer_view.rows or snapshot.within_results) and (
                snapshot.state in {'classifying', 'completed'} or snapshot.results
            ):
                explorer_oob = await _explorer_html(req, oob=True, poll=True, poll_sequence=poll_sequence)
        if not is_search(req) and store is not None and snapshot.state not in {'idle', 'awaiting_review'}:
            running = snapshot.state in {'compiling', 'classifying'}
            status = run_status(snapshot, settings, has_explorer=explorer_view is not None)
            return _html(status + explorer_oob, status_code=200 if running else 286)
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
        response = _html(body + explorer_oob)
        if not is_search(req):
            response.headers['HX-Retarget'] = '#finder-body'
            response.headers['HX-Reswap'] = 'innerHTML'
        return response

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
            f'<div id="finder-scope" hx-swap-oob="innerHTML">{scope_toggle(has_rows=True, selected="within")}</div>'
        )
        return _html(await _explorer_html(req) + body_oob)

    @app.get('/find/rows')
    async def find_rows(req: Request) -> Response:
        polling = req.headers.get('HX-Trigger') == 'explorer-results'
        poll_sequence = next(_POLL_RENDER_SEQUENCE) if polling else None
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
                show_tool_definitions={'1': True, '0': False}.get(params.get('show_tool_definitions') or ''),
                matched_only={'1': True, '0': False}.get(params.get('matched_only') or ''),
                quick_view=params.get('quick_view') if params.get('quick_view') in QUICK_VIEWS else None,
            )
            view = await explorer.view()
            if view.state == 'loaded' and view.quick_view == 'conv_longest' and view.rows:
                await explorer.start_message_counting(retry=params.get('quick_view') == 'conv_longest')
        # The section's own timer swaps nothing itself (hx-swap="none"), so its render arrives out of band.
        body = await _explorer_html(req, oob=polling, poll=polling, poll_sequence=poll_sequence)
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
                    **await _current_fragment_kwargs(req, current),
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
                render_fragment(
                    req,
                    current,
                    settings,
                    error=str(exc),
                    **await _current_fragment_kwargs(req, current),
                ),
                status_code=422,
            )
        return _html(
            render_fragment(req, snapshot, settings, **await _current_fragment_kwargs(req, snapshot))
            + (await _explorer_html(req, oob=True) if not is_search(req) else '')
        )

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
                req,
                snapshot,
                settings,
                api_available=store is not None,
                **await _current_fragment_kwargs(req, snapshot),
            )
            + (await _explorer_html(req, oob=True) if not is_search(req) else '')
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
            await store.explorer.restore_narrowed()
            await store.explorer.clear_ai_view_state()
        return _html(
            render_fragment(
                req,
                snapshot,
                settings,
                api_available=store is not None,
                **await _current_fragment_kwargs(req, snapshot),
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
        from evaluatorq.dashboard.orq_workspace import cli_slug_render_scope
        from evaluatorq.dashboard.trace_finder.views import span_tree

        with cli_slug_render_scope():
            html = await asyncio.to_thread(span_tree, trace_id, spans, first_error_message=error_message)
        return _html(html)

    @app.get('/find/trace/{trace_id:path}')
    async def find_trace(trace_id: str, req: Request) -> Response:
        store = await store_for(req)
        if store is None:
            return _html('<p class="finder-empty">Trace finding is unavailable.</p>', status_code=404)
        raw_msg = req.query_params.get('msg', '')
        try:
            msg = int(raw_msg) if raw_msg.isdigit() else None
        except ValueError:
            # Python limits decimal string conversion length; an invalid message
            # selection should open the drawer at its default message.
            msg = None
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
        from evaluatorq.dashboard.orq_workspace import cli_slug_render_scope

        with cli_slug_render_scope():
            html = await asyncio.to_thread(
                drawer, detail, msg=msg, row=row, traces_layout=req.query_params.get('surface') == 'traces'
            )
        return _html(html)

    @app.get('/find/export.json')
    async def find_export(req: Request) -> Response:
        store = await store_for(req)
        if store is None:
            return Response('Not found', status_code=404, media_type='text/plain')
        snapshot = await store.snapshot()
        if snapshot.state != 'completed' or snapshot.request is None or snapshot.dimensions is None:
            return Response('Not found', status_code=404, media_type='text/plain')
        export_name = export_filename(snapshot)
        requested_export = req.query_params.get('export')
        if requested_export is not None and requested_export != export_name:
            return Response(
                'This Finder result changed. Refresh the page to download the current result.',
                status_code=409,
                media_type='text/plain',
            )
        payload = export_json(snapshot)
        export_dir = get_store_dir('finder-exports')
        try:
            await asyncio.to_thread(_save_finder_export, export_dir, export_name, payload)
        except OSError as exc:
            logger.warning('Could not save Finder export for Insights: {}', exc)
            return Response('Could not save Finder export for Insights.', status_code=500, media_type='text/plain')
        return Response(
            payload,
            media_type='application/json',
            headers={'Content-Disposition': f'attachment; filename="{export_name}"'},
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
                row_scoped=params.get('counts') == 'loaded',
                window_days=parsed.window_days,
            )
        )

    @app.get('/find/dismiss')
    def find_dismiss() -> Response:
        return Response('', media_type='text/html')
