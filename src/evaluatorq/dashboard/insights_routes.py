"""Read-only route handlers for persisted Insights runs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from loguru import logger
from starlette.requests import Request  # noqa: TC002 — FastHTML inspects this annotation at runtime
from starlette.responses import Response

from evaluatorq.common.run_manifest import list_manifests
from evaluatorq.dashboard import library
from evaluatorq.dashboard.insights_views import (
    TABS,
    full_page,
    landing,
    map_payload,
    running_page,
    tab_content,
    unreadable_page,
)
from evaluatorq.insights.models import InsightsRun
from evaluatorq.insights.store import get_insights_runs_dir, list_run_paths

if TYPE_CHECKING:
    from pathlib import Path


def _entries(directory: Path) -> tuple[list[tuple[str, str, str]], dict[str, tuple[Path, InsightsRun | str]]]:
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
    known = {item[0] for item in entries}
    for manifest in manifests:
        if str(manifest.surface) != 'insights' or manifest.run_id in known:
            continue
        status = str(manifest.status)
        if status == 'running':
            entries.insert(0, (manifest.run_id, manifest.run_name, 'running'))
            loaded[manifest.run_id] = (directory / f'{manifest.run_id}.json', 'running')
    return entries, loaded


def _html(content: str, status_code: int = 200, media_type: str = 'text/html') -> Response:
    return Response(content, status_code=status_code, media_type=media_type)


def _resolve(run_id: str, loaded: dict[str, tuple[Path, InsightsRun | str]]) -> tuple[Path, InsightsRun | str] | None:
    return loaded.get(run_id)


def register_insights_routes(app: Any) -> None:  # noqa: C901
    """Attach Insights page, fragment, and export routes to *app*."""

    @app.get('/insights')
    def insights_home() -> Response:
        directory = get_insights_runs_dir()
        entries, loaded = _entries(directory)
        latest = next(
            (
                loaded.get(entry[0])
                for entry in entries
                if isinstance(loaded.get(entry[0], (None, None))[1], InsightsRun)
            ),
            None,
        )
        if latest is not None and isinstance(latest[1], InsightsRun):
            return _html(full_page(latest[1], entries))
        return _html(landing(entries))

    @app.get('/insights/{run_id}')
    def insights_run(run_id: str) -> Response:
        directory = get_insights_runs_dir()
        entries, loaded = _entries(directory)
        resolved = _resolve(run_id, loaded)
        if resolved is None:
            return _html('<h1>Insights run not found</h1>', 404)
        _, run = resolved
        if isinstance(run, InsightsRun):
            return _html(full_page(run, entries))
        if run == 'running':
            label = next((entry[1] for entry in entries if entry[0] == run_id), run_id)
            return _html(running_page(run_id, label, entries))
        return _html(unreadable_page(run_id, str(run), entries))

    @app.get('/insights/{run_id}/tab/{tab}')
    def insights_tab(req: Request, run_id: str, tab: str) -> Response:
        if tab not in TABS:
            return _html('<p class="insights-empty">Unknown Insights tab.</p>', 404)
        directory = get_insights_runs_dir()
        _, loaded = _entries(directory)
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
        entries, _ = _entries(directory)
        return _html(full_page(resolved[1], entries, active_tab=tab, query=query))

    @app.get('/insights/{run_id}/cluster/{cluster_id}')
    def insights_cluster(run_id: str, cluster_id: str) -> Response:
        from evaluatorq.dashboard.insights_views import cluster_detail

        _, loaded = _entries(get_insights_runs_dir())
        resolved = _resolve(run_id, loaded)
        if resolved is None or not isinstance(resolved[1], InsightsRun):
            return _html('<p class="insights-empty">Insights run not found.</p>', 404)
        return _html(cluster_detail(resolved[1], cluster_id))

    @app.get('/insights/{run_id}/traces')
    def insights_traces(req: Request, run_id: str) -> Response:
        from evaluatorq.dashboard.insights_views import traces

        _, loaded = _entries(get_insights_runs_dir())
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
        _, loaded = _entries(get_insights_runs_dir())
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
        _, loaded = _entries(get_insights_runs_dir())
        resolved = _resolve(run_id, loaded)
        if resolved is None or not isinstance(resolved[1], InsightsRun):
            return Response('{"error": "Insights run not found"}', status_code=404, media_type='application/json')
        return Response(
            resolved[1].model_dump_json(indent=2),
            media_type='application/json',
            headers={'Content-Disposition': f'attachment; filename="{quote(run_id, safe="")}.json"'},
        )
