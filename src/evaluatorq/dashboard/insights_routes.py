"""Route handlers for launching and reviewing Insights runs."""

from __future__ import annotations

import json
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
from evaluatorq.dashboard.insights_launch import InsightsLaunchSpec, launch_insights
from evaluatorq.dashboard.insights_views import (
    TABS,
    facet_options,
    full_page,
    landing,
    map_payload,
    new_run_page,
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
    entries: list[tuple[str, str, str]] = []
    loaded: dict[str, tuple[Path, InsightsRun | str]] = {}
    for path in list_run_paths(directory):
        try:
            run: InsightsRun | str = library.load_model_cached(path, InsightsRun.model_validate)
        except (OSError, ValueError, TypeError) as exc:
            run = f'{type(exc).__name__}: {exc}'
            logger.warning('Unreadable Insights run {}: {}', path, run)
        if isinstance(run, InsightsRun):
            key = run.run_id
            entries.append((key, run.run_name, run.status))
            loaded[key] = (path, run)
        else:
            key = path.stem
            entries.append((key, path.stem, 'unreadable'))
            loaded[key] = (path, run)
            loaded[path.stem] = (path, run)
        loaded[path.stem] = (path, run)
    manifests = list_manifests(directory)
    manifest_by_id = {manifest.run_id: manifest for manifest in manifests if str(manifest.surface) == 'insights'}
    known = {item[0] for item in entries}
    manifest_entries = []
    for manifest in manifests:
        if str(manifest.surface) != 'insights' or manifest.run_id in known:
            continue
        status = str(manifest.status)
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
    cache_key = (profile.name if profile else None, window_days)
    cache: dict[tuple[str | None, int], tuple[datetime, FacetCatalogue | None]] = getattr(
        app.state, 'insights_facet_catalogues', {}
    )
    cached = cache.get(cache_key)
    if cached is not None and cached[0] > now:
        return cached[1]
    orq = None
    try:
        orq = resolve_orq_client(
            profile.api_key if profile else None,
            base_url=(profile.server or DEFAULT_ORQ_BASE_URL) if profile else None,
        )
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
    cache[cache_key] = (now + timedelta(minutes=5 if catalogue is not None else 1), catalogue)
    app.state.insights_facet_catalogues = cache
    return catalogue


def register_insights_routes(app: Any) -> None:  # noqa: C901
    """Attach Insights page, fragment, and export routes to *app*."""

    @app.get('/insights')
    def insights_home() -> Response:
        directory = get_insights_runs_dir()
        entries, loaded, manifests = _entries(directory)
        active = next((entry for entry in entries if entry[2] == 'running'), None)
        if active is not None:
            manifest = manifests.get(active[0])
            if manifest is not None:
                return _html(running_page(active[0], active[1], entries, manifest, manifests))
        latest = next(
            (
                loaded.get(entry[0])
                for entry in entries
                if isinstance(loaded.get(entry[0], (None, None))[1], InsightsRun)
            ),
            None,
        )
        if latest is not None and isinstance(latest[1], InsightsRun):
            return _html(full_page(latest[1], entries, manifest=manifests.get(latest[1].run_id), manifests=manifests))
        return _html(landing(entries, manifests))

    @app.get('/insights/new')
    def insights_new() -> Response:
        entries, _, manifests = _entries(get_insights_runs_dir())
        return _html(new_run_page(entries, manifests))

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
            entries, _, manifests = _entries(directory)
            return _html(new_run_page(entries, manifests, error=rejected), 403)
        try:
            spec = InsightsLaunchSpec.model_validate({
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
            })
        except ValidationError as exc:
            entries, _, manifests = _entries(directory)
            message = '; '.join(error['msg'] for error in exc.errors())
            return _html(new_run_page(entries, manifests, error=message), 422)
        try:
            profile = selected_orq_profile(req.app)
        except ValueError as exc:
            entries, _, manifests = _entries(directory)
            return _html(new_run_page(entries, manifests, error=str(exc)), 422)
        run_id = launch_insights(spec, directory, profile=profile)
        return RedirectResponse(f'/insights/{quote(run_id, safe="")}', status_code=303)

    @app.get('/insights/{run_id}')
    def insights_run(run_id: str) -> Response:
        directory = get_insights_runs_dir()
        entries, loaded, manifests = _entries(directory)
        resolved = _resolve(run_id, loaded)
        if resolved is None:
            return _html('<h1>Insights run not found</h1>', 404)
        _, run = resolved
        if isinstance(run, InsightsRun):
            return _html(full_page(run, entries, manifest=manifests.get(run.run_id), manifests=manifests))
        if run in ('running', 'error', 'cancelled', 'completed') and run_id in manifests:
            label = next((entry[1] for entry in entries if entry[0] == run_id), run_id)
            return _html(running_page(run_id, label, entries, manifests[run_id], manifests))
        return _html(unreadable_page(run_id, str(run), entries, manifests))

    @app.get('/insights/{run_id}/tab/{tab}')
    def insights_tab(req: Request, run_id: str, tab: str) -> Response:
        if tab not in TABS:
            return _html('<p class="insights-empty">Unknown Insights tab.</p>', 404)
        directory = get_insights_runs_dir()
        _, loaded, _ = _entries(directory)
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
        entries, _, manifests = _entries(directory)
        return _html(
            full_page(
                resolved[1],
                entries,
                active_tab=tab,
                query=query,
                manifest=manifests.get(resolved[1].run_id),
                manifests=manifests,
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
