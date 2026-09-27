"""Route handlers for launching and reviewing Insights runs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from loguru import logger
from pydantic import ValidationError
from starlette.requests import Request  # noqa: TC002 — FastHTML inspects this annotation at runtime
from starlette.responses import RedirectResponse, Response

from evaluatorq.common.orq_client import DEFAULT_ORQ_BASE_URL, close_orq_client, resolve_orq_client
from evaluatorq.common.run_manifest import list_manifests
from evaluatorq.dashboard import library
from evaluatorq.dashboard.insights_launch import InsightsLaunchSpec, launch_insights, reconcile_stale_worker
from evaluatorq.dashboard.insights_views import (
    TABS,
    facet_options,
    full_page,
    map_payload,
    new_run_page,
    overview_page,
    running_page,
    tab_content,
    unreadable_page,
)
from evaluatorq.dashboard.security import request_rejected
from evaluatorq.dashboard.trace_finder.routes import selected_orq_profile
from evaluatorq.insights.models import InsightsRun
from evaluatorq.insights.store import get_insights_runs_dir, list_run_paths
from evaluatorq.trace_finder.facets import load_facet_catalogue
from evaluatorq.trace_finder.models import FACET_NAMES, FacetSelection

if TYPE_CHECKING:
    from pathlib import Path

    from evaluatorq.contracts import RunManifest
    from evaluatorq.trace_finder.models import FacetCatalogue


def _entries(
    directory: Path,
) -> tuple[list[tuple[str, str, str]], dict[str, tuple[Path, InsightsRun | str]], dict[str, RunManifest]]:
    manifests = list_manifests(directory)
    try:
        for manifest in manifests:
            # This route owns only Insights worker leases. Other dashboard
            # surfaces share the manifest directory and must keep their state.
            if manifest.surface.value == 'insights':
                reconcile_stale_worker(directory, manifest.run_id)
    except (OSError, ValueError) as exc:
        logger.warning('Could not reconcile stale Insights workers in {}: {}', directory, exc)
    entries: list[tuple[str, str, str]] = []
    loaded: dict[str, tuple[Path, InsightsRun | str]] = {}
    aliases: list[tuple[str, tuple[Path, InsightsRun | str]]] = []
    seen_run_ids: set[str] = set()
    for path in list_run_paths(directory):
        try:
            run: InsightsRun | str = library.load_model_cached(path, InsightsRun.model_validate)
        except (OSError, ValueError, TypeError) as exc:
            run = f'{type(exc).__name__}: {exc}'
            logger.warning('Unreadable Insights run {}: {}', path, run)
        if isinstance(run, InsightsRun):
            key = run.run_id
            if key not in seen_run_ids:
                entries.append((key, run.run_name, run.status))
                loaded[key] = (path, run)
                seen_run_ids.add(key)
            aliases.append((path.stem, (path, run)))
        else:
            key = path.stem
            entries.append((key, path.stem, 'unreadable'))
            loaded.setdefault(key, (path, run))
    # Path stems are compatibility aliases. Install them only after every
    # persisted run ID is known so an alias can never shadow a canonical ID.
    for alias, target in aliases:
        loaded.setdefault(alias, target)
    manifests = list_manifests(directory)
    manifest_by_id = {manifest.run_id: manifest for manifest in manifests if manifest.surface.value == 'insights'}
    known = {item[0] for item in entries}
    manifest_entries = []
    for manifest in manifests:
        if manifest.surface.value != 'insights' or manifest.run_id in known:
            continue
        status = manifest.status.value
        manifest_entries.append((manifest.run_id, manifest.run_name, status))
        loaded[manifest.run_id] = (directory / f'{manifest.run_id}.json', status)
    return manifest_entries + entries, loaded, manifest_by_id


def _html(content: str, status_code: int = 200, media_type: str = 'text/html') -> Response:
    return Response(content, status_code=status_code, media_type=media_type)


def _resolve(run_id: str, loaded: dict[str, tuple[Path, InsightsRun | str]]) -> tuple[Path, InsightsRun | str] | None:
    return loaded.get(run_id)


async def _catalogue(app: Any, window_days: int) -> FacetCatalogue | None:
    """Cache Orq facet values for the selected window; failures remain visible in the wizard."""
    now = datetime.now(timezone.utc)
    try:
        profile = selected_orq_profile(app)
    except ValueError as exc:
        logger.warning('Insights facet values are unavailable: {}', exc)
        return None
    server = (
        (profile.server or DEFAULT_ORQ_BASE_URL)
        if profile
        else (os.environ.get('ORQ_BASE_URL') or DEFAULT_ORQ_BASE_URL)
    )
    api_key = profile.api_key if profile else os.environ.get('ORQ_API_KEY', '')
    credential_fingerprint = hashlib.sha256(f'{server}\0{api_key}'.encode()).hexdigest()
    cache_key = (profile.name if profile else None, credential_fingerprint, window_days)
    generation = getattr(app.state, 'finder_generation', 0)
    cache: dict[tuple[str | None, str, int], tuple[datetime, FacetCatalogue | None]] | None = getattr(
        app.state, 'insights_facet_catalogues', None
    )
    if cache is None:
        cache = {}
        app.state.insights_facet_catalogues = cache
    cached = cache.get(cache_key)
    if cached is not None and cached[0] > now:
        return cached[1]

    in_flight: dict[tuple[int, tuple[str | None, str, int]], asyncio.Task[FacetCatalogue | None]] = getattr(
        app.state, 'insights_facet_catalogue_tasks', {}
    )
    app.state.insights_facet_catalogue_tasks = in_flight
    task_key = (generation, cache_key)
    task = in_flight.get(task_key)
    if task is None:
        task = asyncio.create_task(
            _load_catalogue(app, cache, generation, cache_key, window_days, now, profile, server)
        )
        in_flight[task_key] = task

        def discard_finished(done: asyncio.Task[FacetCatalogue | None]) -> None:
            if in_flight.get(task_key) is done:
                del in_flight[task_key]

        task.add_done_callback(discard_finished)
    catalogue = await asyncio.shield(task)
    if generation != getattr(app.state, 'finder_generation', 0):
        return None
    return catalogue


async def _load_catalogue(
    app: Any,
    cache: dict[tuple[str | None, str, int], tuple[datetime, FacetCatalogue | None]],
    generation: int,
    cache_key: tuple[str | None, str, int],
    window_days: int,
    now: datetime,
    profile: Any,
    server: str,
) -> FacetCatalogue | None:
    """Fetch one window catalogue and cache it only while its settings are current."""
    orq = None
    try:
        orq = resolve_orq_client(profile.api_key if profile else None, base_url=server)
        catalogue = await load_facet_catalogue(orq, start=now - timedelta(days=window_days), end=now, limit=50)
    except Exception as exc:  # noqa: BLE001 — provider errors render a visible unavailable state
        logger.warning('Insights facet values are unavailable for the {}-day window: {}', window_days, exc)
        catalogue = None
    finally:
        if orq is not None:
            try:
                await close_orq_client(orq)
            except Exception as exc:  # noqa: BLE001 — cleanup must not hide the catalogue response
                logger.warning('Could not close the Insights facet catalogue client: {}', exc)
    if (
        generation == getattr(app.state, 'finder_generation', 0)
        and getattr(app.state, 'insights_facet_catalogues', None) is cache
    ):
        cache[cache_key] = (now + timedelta(minutes=5 if catalogue is not None else 1), catalogue)
    return catalogue


def register_insights_routes(app: Any) -> None:  # noqa: C901
    """Attach Insights page, fragment, and export routes to *app*."""

    @app.get('/insights')
    def insights_home() -> Response:
        directory = get_insights_runs_dir()
        entries, loaded, manifests = _entries(directory)
        runs = {run_id: item[1] for run_id, item in loaded.items() if isinstance(item[1], InsightsRun)}
        return _html(overview_page(entries, runs, manifests))

    @app.get('/insights/new')
    def insights_new() -> Response:
        return _html(new_run_page())

    @app.get('/insights/facets')
    async def insights_facets(req: Request) -> Response:
        try:
            window_days = int(req.query_params.get('window_days', '7'))
        except ValueError:
            return _html('<p class="insights-error">Enter a valid window to load facets.</p>', 422)
        if not 1 <= window_days <= 90:
            return _html('<p class="insights-error">Choose a window from 1 to 90 days.</p>', 422)
        selection = FacetSelection.model_validate({
            name: req.query_params.getlist(f'facet_{name}') for name in FACET_NAMES
        })
        return _html(facet_options(await _catalogue(req.app, window_days), selection))

    @app.post('/insights/runs')
    async def insights_start(req: Request) -> Response:
        form = await req.form()
        rejected = request_rejected(req, form)
        directory = get_insights_runs_dir()
        if rejected:
            return _html(new_run_page(error=rejected), 403)
        try:
            spec = await asyncio.to_thread(
                InsightsLaunchSpec.model_validate,
                {
                    'name': form.get('name', ''),
                    'source': form.get('source', 'recent'),
                    'query': form.get('query', ''),
                    'finder_export': form.get('finder_export', ''),
                    'window_days': form.get('window_days', 7),
                    'limit': form.get('limit', 100),
                    'facets': {name: form.getlist(f'facet_{name}') for name in FACET_NAMES},
                    'parallelism': form.get('parallelism', 20),
                    'labels': form.getlist('labels'),
                    'dimensions': form.getlist('dimensions'),
                },
            )
        except ValidationError as exc:
            message = '; '.join(error['msg'] for error in exc.errors())
            return _html(new_run_page(error=message), 422)
        try:
            profile = selected_orq_profile(req.app)
        except ValueError as exc:
            return _html(new_run_page(error=str(exc)), 422)
        run_id = await asyncio.to_thread(launch_insights, spec, directory, profile=profile)
        return RedirectResponse(f'/insights/{quote(run_id, safe="")}', status_code=303)

    @app.get('/insights/{run_id}')
    def insights_run(run_id: str) -> Response:
        directory = get_insights_runs_dir()
        _, loaded, manifests = _entries(directory)
        resolved = _resolve(run_id, loaded)
        if resolved is None:
            return _html('<h1>Insights run not found</h1>', 404)
        _, run = resolved
        if isinstance(run, InsightsRun):
            return _html(full_page(run, manifest=manifests.get(run.run_id)))
        if run in ('running', 'error', 'cancelled', 'completed') and run_id in manifests:
            return _html(running_page(run_id, manifests[run_id].run_name, manifests[run_id]))
        return _html(unreadable_page(str(run)))

    @app.get('/insights/{run_id}/tab/{tab}')
    def insights_tab(req: Request, run_id: str, tab: str) -> Response:
        if tab not in TABS:
            return _html('<p class="insights-empty">Unknown Insights tab.</p>', 404)
        directory = get_insights_runs_dir()
        _, loaded, manifests = _entries(directory)
        resolved = _resolve(run_id, loaded)
        if resolved is None or not isinstance(resolved[1], InsightsRun):
            return _html('<p class="insights-empty">Insights run not found.</p>', 404)
        query = {
            key: req.query_params[key]
            for key in ('dimension', 'cluster', 'label', 'value', 'row', 'row_value', 'column', 'column_value')
            if req.query_params.get(key)
        }
        if req.headers.get('HX-Request', '').casefold() == 'true':
            return _html(tab_content(resolved[1], tab, query=query))
        return _html(
            full_page(
                resolved[1],
                active_tab=tab,
                query=query,
                manifest=manifests.get(resolved[1].run_id),
            )
        )

    @app.get('/insights/{run_id}/cluster/{cluster_id}')
    def insights_cluster(run_id: str, cluster_id: str) -> Response:
        from evaluatorq.dashboard.insights_views import cluster_detail

        _, loaded, _ = _entries(get_insights_runs_dir())
        resolved = _resolve(run_id, loaded)
        if resolved is None or not isinstance(resolved[1], InsightsRun):
            return _html('<p class="insights-empty">Insights run not found.</p>', 404)
        return _html(cluster_detail(resolved[1], cluster_id))

    @app.get('/insights/{run_id}/traces')
    def insights_traces(req: Request, run_id: str) -> Response:
        from evaluatorq.dashboard.insights_views import traces

        _, loaded, _ = _entries(get_insights_runs_dir())
        resolved = _resolve(run_id, loaded)
        if resolved is None or not isinstance(resolved[1], InsightsRun):
            return _html('<p class="insights-empty">Insights run not found.</p>', 404)
        query = req.query_params
        return _html(
            traces(
                resolved[1],
                dimension=query.get('dimension'),
                cluster=query.get('cluster'),
                label=query.get('label'),
                value=query.get('value'),
            )
        )

    @app.get('/insights/{run_id}/map.json')
    def insights_map_json(req: Request, run_id: str) -> Response:
        _, loaded, _ = _entries(get_insights_runs_dir())
        resolved = _resolve(run_id, loaded)
        if resolved is None or not isinstance(resolved[1], InsightsRun):
            return Response(
                '{"points": [], "legend": [], "color_scale": null}', status_code=404, media_type='application/json'
            )
        payload = map_payload(
            resolved[1],
            req.query_params.get('dimension', next(iter(resolved[1].dimensions), '')),
            req.query_params.get('color_by', 'cluster'),
        )
        return Response(json.dumps(payload), media_type='application/json')

    @app.get('/insights/{run_id}/export.json')
    def insights_export(run_id: str) -> Response:
        _, loaded, _ = _entries(get_insights_runs_dir())
        resolved = _resolve(run_id, loaded)
        if resolved is None or not isinstance(resolved[1], InsightsRun):
            return Response('{"error": "Insights run not found"}', status_code=404, media_type='application/json')
        return Response(
            resolved[1].model_dump_json(indent=2),
            media_type='application/json',
            headers={'Content-Disposition': f'attachment; filename="{quote(run_id, safe="")}.json"'},
        )
