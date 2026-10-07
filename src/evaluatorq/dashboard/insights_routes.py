"""Route handlers for launching and reviewing Insights runs."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote, urlencode

from loguru import logger
from pydantic import ValidationError
from starlette.requests import Request  # noqa: TC002 — FastHTML inspects this annotation at runtime
from starlette.responses import RedirectResponse, Response

from evaluatorq.common.orq_client import close_orq_client, resolve_orq_client
from evaluatorq.common.reports import esc
from evaluatorq.common.run_manifest import list_manifests
from evaluatorq.dashboard import library
from evaluatorq.dashboard.auth import auth_identity
from evaluatorq.dashboard.insights_estimate_views import (
    render_compact_estimate,
    render_estimate,
    render_estimate_unavailable,
)
from evaluatorq.dashboard.insights_launch import InsightsLaunchSpec, launch_insights, reconcile_stale_worker
from evaluatorq.dashboard.insights_review_data import build_review_payload, build_signal_detail_payload
from evaluatorq.dashboard.insights_review_views import review_page
from evaluatorq.dashboard.insights_run_form import (
    INSIGHTS_MODEL_FIELDS,
    RunFormValues,
    facet_options,
    render_plan,
    render_run_form,
    render_run_page,
)
from evaluatorq.dashboard.insights_uploads import (
    UploadRequestTooLargeError,
    UploadTooLargeError,
    cleanup_expired_uploads,
    limit_request_body,
    receive_upload,
    store_upload,
)
from evaluatorq.dashboard.insights_views import (
    TABS,
    map_payload,
    overview_page,
    projection_notice,
    running_page,
    tab_content,
    tabs,
    trace_detail_page,
    unreadable_page,
)
from evaluatorq.dashboard.model_choices import catalogue_entry, model_groups
from evaluatorq.dashboard.security import csrf_token, request_rejected
from evaluatorq.dashboard.trace_finder.routes import selected_dashboard_auth
from evaluatorq.dashboard.view import model_control
from evaluatorq.insights.estimate import StageModels, estimate_run, stage_seconds, trace_bound
from evaluatorq.insights.models import InsightsRun, label_key
from evaluatorq.insights.population import PopulationError, preview_snapshot
from evaluatorq.insights.store import get_insights_runs_dir, list_run_paths
from evaluatorq.trace_finder.export import RunExport
from evaluatorq.trace_finder.facets import load_facet_catalogue
from evaluatorq.trace_finder.models import FACET_NAMES, FacetCatalogue, FacetSelection

if TYPE_CHECKING:
    from pathlib import Path

    from evaluatorq.common.model_catalogue import ModelInfo
    from evaluatorq.contracts import RunManifest
    from evaluatorq.dashboard.auth import DashboardAuth
    from evaluatorq.insights.estimate import RunEstimate


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


def _validation_message(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return '; '.join(error['msg'].removeprefix('Value error, ') for error in exc.errors())
    return str(exc)


async def _estimate(app: Any, spec: InsightsLaunchSpec) -> RunEstimate:
    """The traces, cost and time a run with *spec* would use, from the catalogue, prices and earlier runs."""
    finder_count: int | None = None
    snapshot_count: int | None = None
    catalogue = FacetCatalogue()
    if spec.source == 'finder':
        raw = spec.validated_finder_export_snapshot()
        finder_count = None if raw is None else len(RunExport.model_validate_json(raw).matched_trace_ids)
    elif spec.source == 'snapshot':
        try:
            snapshot_count = (await asyncio.to_thread(preview_snapshot, spec.snapshot_path))['n_traces']
        except PopulationError as exc:
            logger.warning('Could not count the traces in the local file for the estimate: {}', exc)
    else:
        loaded, _, _ = await _catalogue(app, spec.window_days)
        catalogue = loaded or FacetCatalogue()
    bound = trace_bound(
        spec.source,
        catalogue=catalogue,
        facets=spec.facets,
        limit=spec.limit,
        finder_count=finder_count,
        snapshot_count=snapshot_count,
    )
    models = StageModels(summary=spec.summary_model, classifier=spec.classifier_model, embedding=spec.embedding_model)
    prices: dict[str, ModelInfo | None] = {}
    try:
        auth = selected_dashboard_auth(app)
    except (RuntimeError, ValueError) as exc:
        logger.warning('No Orq catalogue for the Insights estimate prices: {}', exc)
    else:
        chosen = sorted({models.summary, models.classifier, models.embedding})
        entries = await asyncio.gather(*(catalogue_entry(auth, model) for model in chosen))
        prices = dict(zip(chosen, entries, strict=True))
    seconds = await asyncio.to_thread(stage_seconds, get_insights_runs_dir())
    return estimate_run(
        bound=bound,
        dimensions=spec.dimension_names(),
        question_count=len(spec.labels) + len(spec.custom_labels),
        coding=spec.coding_enabled,
        models=models,
        prices=prices,
        seconds=seconds,
        parallelism=spec.parallelism,
        query=spec.source == 'query',
        sentiment_selected='sentiment' in spec.labels,
    )


async def _model_rejection(auth: DashboardAuth, spec: InsightsLaunchSpec) -> str | None:
    """Why the catalogue says a chosen model cannot do its job, or `None` when it can or cannot be checked."""
    unchecked: list[str] = []
    for field, wrong_for in (
        ('classifier_model', lambda info: not info.supports_classify),
        ('embedding_model', lambda info: info.model_type != 'embedding'),
    ):
        model = getattr(spec, field)
        if not model:
            continue
        info = await catalogue_entry(auth, model)
        if info is None:
            unchecked.append(field)
        elif wrong_for(info):
            what = 'serve /classify' if field == 'classifier_model' else 'produce embeddings'
            return f'{INSIGHTS_MODEL_FIELDS[field][1]} {model} cannot {what}. Choose another from the list.'
    if unchecked:
        logger.warning(
            'Orq model catalogue has no entry for {}; accepting the typed values unchecked',
            ' and '.join(f'{field} {getattr(spec, field)!r}' for field in unchecked),
        )
    return None


def _run_form_response(values: RunFormValues, error: str | None, status_code: int) -> Response:
    """The run form as a page or a dialog fragment to match `values.mount`."""
    token = csrf_token()
    if values.mount == 'dialog':
        return _html(render_run_form(values, csrf=token, error=error), status_code)
    return _html(render_run_page(values, csrf=token, error=error), status_code)


def _resolve(run_id: str, loaded: dict[str, tuple[Path, InsightsRun | str]]) -> tuple[Path, InsightsRun | str] | None:
    return loaded.get(run_id)


def _review_hash(run: InsightsRun, tab: str, query: Any) -> str:
    """Translate a legacy tab and its compatible filters into the review URL state."""
    view = {'map': 'map', 'crosstab': 'compare'}.get(tab, 'themes')
    dimensions = run.dimensions
    dimension = query.get('dimension')
    if dimension not in dimensions:
        dimension = next(iter(dimensions), '')
    state: dict[str, str] = {}
    if dimension:
        state['dim'] = dimension
    state['view'] = view
    filters: list[str] = []
    cluster_id = query.get('cluster')
    if cluster_id:
        cluster = (
            next((item for item in dimensions[dimension].clusters if item.id == cluster_id), None)
            if dimension in dimensions
            else None
        )
        if cluster is not None:
            filters.append(f'c:{quote(cluster.id, safe="-_~")}')
    label_name, label_value = query.get('label'), query.get('value')
    label = next((item for item in run.config.labels if item.name == label_name), None)
    if label is not None and label_value:
        valid_value = any(
            trace.labels[label_name].error is None
            and trace.labels[label_name].value is not None
            and label_key(label, trace.labels[label_name].value) == label_value
            for trace in run.traces
            if label_name in trace.labels
        )
        if valid_value:
            filters.append(f'l:{quote(label_name, safe="-_~")}:{quote(label_value, safe="-_~")}')
    if filters:
        state['f'] = '|'.join(filters)
    return '#' + urlencode(state, quote_via=quote, safe='-_.!~*()')


_CatalogueResult = tuple[FacetCatalogue | None, str | None, bool]
_CatalogueKey = tuple[str | None, str, int]


async def _catalogue(app: Any, window_days: int, *, retry: bool = False) -> _CatalogueResult:
    """Cache Orq facet values for the selected window; failures remain visible in the wizard.

    Returns ``(catalogue, source_name, credential_rejected)``. ``retry`` skips a cached answer
    so a user who fixed their credentials is not held to the one-minute failure cache.
    """
    now = datetime.now(timezone.utc)
    try:
        auth = selected_dashboard_auth(app)
    except ValueError as exc:
        logger.warning('Insights facet values are unavailable: {}', exc)
        return None, None, False
    source_name = auth.profile.name if auth.profile else auth.label
    credential_fingerprint = hashlib.sha256(
        f'{auth.method}\0{auth.base_url}\0{auth.api_key or ""}'.encode()
    ).hexdigest()
    cache_key = (source_name, credential_fingerprint, window_days)
    generation = getattr(app.state, 'finder_generation', 0)
    cache: dict[_CatalogueKey, tuple[datetime, FacetCatalogue | None, bool]] | None = getattr(
        app.state, 'insights_facet_catalogues', None
    )
    if cache is None:
        cache = {}
        app.state.insights_facet_catalogues = cache
    cached = cache.get(cache_key)
    if not retry and cached is not None and cached[0] > now:
        return cached[1], source_name, cached[2]

    in_flight: dict[tuple[int, _CatalogueKey], asyncio.Task[tuple[FacetCatalogue | None, bool]]] = getattr(
        app.state, 'insights_facet_catalogue_tasks', {}
    )
    app.state.insights_facet_catalogue_tasks = in_flight
    task_key = (generation, cache_key)
    task = in_flight.get(task_key)
    if task is None:
        task = asyncio.create_task(_load_catalogue(app, cache, generation, cache_key, window_days, now, auth))
        in_flight[task_key] = task

        def discard_finished(done: asyncio.Task[tuple[FacetCatalogue | None, bool]]) -> None:
            if in_flight.get(task_key) is done:
                del in_flight[task_key]

        task.add_done_callback(discard_finished)
    catalogue, credential_rejected = await asyncio.shield(task)
    if generation != getattr(app.state, 'finder_generation', 0):
        return None, source_name, False
    return catalogue, source_name, credential_rejected


async def _load_catalogue(
    app: Any,
    cache: dict[_CatalogueKey, tuple[datetime, FacetCatalogue | None, bool]],
    generation: int,
    cache_key: _CatalogueKey,
    window_days: int,
    now: datetime,
    auth: DashboardAuth,
) -> tuple[FacetCatalogue | None, bool]:
    """Fetch one window catalogue and cache it only while its settings are current."""
    orq = None
    credential_rejected = False
    try:
        if auth.method == 'cli_oauth':
            from evaluatorq.common.cli_oauth import build_cli_oauth_clients

            settings = app.state.finder_settings
            orq, _ = build_cli_oauth_clients(
                server_url=auth.base_url, workspace=settings.orq_workspace, project=settings.orq_project_id
            )
        else:
            orq = resolve_orq_client(auth.api_key, base_url=auth.base_url)
        catalogue = await load_facet_catalogue(orq, start=now - timedelta(days=window_days), end=now, limit=50)
    except Exception as exc:  # noqa: BLE001 — provider errors render a visible unavailable state
        credential_rejected = _is_unauthorized(exc)
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
        cache[cache_key] = (
            now + timedelta(minutes=5 if catalogue is not None else 1),
            catalogue,
            credential_rejected,
        )
    return catalogue, credential_rejected


def _is_unauthorized(error: Exception) -> bool:
    """Recognize an Orq 401 without exposing provider error text in the UI."""
    status = getattr(error, 'status_code', None)
    if status is None:
        response = getattr(error, 'response', None)
        status = getattr(response, 'status_code', None)
    return status == 401


def register_insights_routes(app: Any) -> None:  # noqa: C901
    """Attach Insights page, fragment, and export routes to *app*."""

    @app.get('/insights')
    def insights_home() -> Response:
        directory = get_insights_runs_dir()
        cleanup_expired_uploads(directory)
        entries, loaded, manifests = _entries(directory)
        runs = {run_id: item[1] for run_id, item in loaded.items() if isinstance(item[1], InsightsRun)}
        return _html(overview_page(entries, runs, manifests))

    @app.get('/insights/new')
    def insights_new(req: Request) -> Response:
        directory = get_insights_runs_dir()
        rerun_id = req.query_params.get('rerun')
        if rerun_id:
            _, loaded, _ = _entries(directory)
            resolved = _resolve(rerun_id, loaded)
            if resolved is None or not isinstance(resolved[1], InsightsRun):
                return _html('<p class="insights-empty">Insights run not found.</p>', 404)
            values = RunFormValues.from_run(resolved[1], directory)
        else:
            values = RunFormValues.defaults()
        mount: Literal['page', 'dialog'] = 'dialog' if req.query_params.get('mount') == 'dialog' else 'page'
        return _run_form_response(replace(values, mount=mount), None, 200)

    @app.get('/insights/new/plan')
    async def insights_new_plan(req: Request) -> Response:
        compact = req.query_params.get('compact') == '1'

        def validated() -> tuple[InsightsLaunchSpec, list[tuple[str, str]]]:
            spec = InsightsLaunchSpec.model_validate(RunFormValues.from_form(req.query_params).launch_fields())
            return spec, [] if compact else spec.stages()

        try:
            spec, stages = await asyncio.to_thread(validated)
        except (ValidationError, ValueError, TypeError) as exc:
            message = _validation_message(exc)
            if compact:
                return _html(render_estimate_unavailable(message))
            return _html(f'<p class="insights-error" role="alert">{esc(message)}</p>', 422)
        estimate = await _estimate(req.app, spec)
        if compact:
            return _html(render_compact_estimate(estimate))
        return _html(
            f'<div data-part="estimate">{render_estimate(estimate, stages)}</div>'
            f'<div data-part="compact">{render_compact_estimate(estimate)}</div>'
            f'<div data-part="plan">{render_plan(spec, stages)}</div>'
        )

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
        catalogue, profile_name, credential_rejected = await _catalogue(
            req.app, window_days, retry=req.query_params.get('retry') == '1'
        )
        return _html(
            facet_options(
                catalogue,
                selection,
                profile_name=profile_name,
                credential_rejected=credential_rejected,
                window_days=window_days,
            )
        )

    @app.post('/insights/uploads')
    async def insights_upload(req: Request) -> Response:
        cleanup_expired_uploads(get_insights_runs_dir())
        limit_request_body(req)
        try:
            form = await req.form()
        except UploadRequestTooLargeError as exc:
            return Response(json.dumps({'error': str(exc)}), status_code=413, media_type='application/json')
        except Exception as exc:  # noqa: BLE001 — malformed multipart input is a client error
            logger.warning('Could not parse Insights source upload: {}', exc)
            return Response(
                json.dumps({'error': 'Could not read the uploaded file.'}),
                status_code=422,
                media_type='application/json',
            )
        try:
            rejected = request_rejected(req, form)
            if rejected:
                return Response(json.dumps({'error': rejected}), status_code=403, media_type='application/json')
            upload = form.get('file')
            if upload is None or not hasattr(upload, 'read') or not getattr(upload, 'filename', ''):
                return Response(
                    json.dumps({'error': 'Choose a JSON file to upload.'}),
                    status_code=422,
                    media_type='application/json',
                )
            try:
                kind, contents = await receive_upload(upload)
                path = await asyncio.to_thread(store_upload, get_insights_runs_dir(), contents, kind)
            except UploadTooLargeError as exc:
                return Response(json.dumps({'error': str(exc)}), status_code=413, media_type='application/json')
            except (OSError, ValueError) as exc:
                return Response(json.dumps({'error': str(exc)}), status_code=422, media_type='application/json')
            return Response(
                json.dumps({'kind': kind, 'path': str(path)}), status_code=201, media_type='application/json'
            )
        finally:
            await form.close()

    @app.post('/insights/snapshot-preview')
    async def insights_snapshot_preview(req: Request) -> Response:
        form = await req.form()
        rejected = request_rejected(req, form)
        if rejected:
            return _html(f'<p class="insights-error" role="alert">{rejected}</p>', 403)
        raw_path = str(form.get('snapshot_path', '')).strip()
        if not raw_path or len(raw_path) > 4096:
            return _html('<p class="insights-error" role="alert">Enter a trace snapshot JSON path.</p>', 422)
        try:
            coverage = await asyncio.to_thread(preview_snapshot, raw_path)
        except PopulationError:
            return _html('<p class="insights-error" role="alert">Could not read a valid local trace snapshot.</p>', 422)
        if coverage['n_traces'] == 0:
            return _html('<p class="insights-error" role="alert">This trace snapshot contains no traces.</p>', 422)
        return _html(projection_notice(coverage))

    @app.get('/insights/models')
    async def insights_models(req: Request) -> Response:
        field = req.query_params.get('field', '')
        if field not in INSIGHTS_MODEL_FIELDS:
            return _html('')
        kind, label = INSIGHTS_MODEL_FIELDS[field]
        try:
            groups = await model_groups(selected_dashboard_auth(req.app), kind)
        except (RuntimeError, ValueError) as exc:
            logger.warning('No model catalogue for the Insights run form ({}): {}', field, exc)
            groups = {}
        return _html(model_control(field, req.query_params.get(field, ''), groups, label=label))

    @app.post('/insights/runs')
    async def insights_start(req: Request) -> Response:
        form = await req.form()
        rejected = request_rejected(req, form)
        directory = get_insights_runs_dir()
        try:
            values = RunFormValues.from_form(form)
            unreadable = None
        except ValueError as exc:
            values = replace(RunFormValues.defaults(), mount='dialog' if form.get('mount') == 'dialog' else 'page')
            unreadable = _validation_message(exc)
        if rejected:
            return _run_form_response(values, rejected, 403)
        if unreadable:
            return _run_form_response(values, unreadable, 422)
        try:
            spec = await asyncio.to_thread(InsightsLaunchSpec.model_validate, values.launch_fields())
        except (ValidationError, ValueError, TypeError) as exc:
            return _run_form_response(values, _validation_message(exc), 422)
        try:
            auth = selected_dashboard_auth(req.app)
            identity = await asyncio.to_thread(auth_identity, auth, req.app.state.finder_settings)
        except (RuntimeError, ValueError) as exc:
            return _run_form_response(values, str(exc), 422)
        model_error = await _model_rejection(auth, spec)
        if model_error:
            return _run_form_response(values, model_error, 422)
        run_id = await asyncio.to_thread(
            launch_insights, spec, directory, auth_method=auth.method, auth_identity=identity
        )
        return RedirectResponse(f'/insights/{quote(run_id, safe="")}', status_code=303)

    @app.get('/insights/{run_id}')
    def insights_run(req: Request, run_id: str) -> Response:
        directory = get_insights_runs_dir()
        _, loaded, manifests = _entries(directory)
        resolved = _resolve(run_id, loaded)
        if resolved is None:
            return _html('<h1>Insights run not found</h1>', 404)
        _, run = resolved
        if isinstance(run, InsightsRun):
            return _html(review_page(run, manifest=manifests.get(run.run_id)))
        if run in ('running', 'error', 'cancelled', 'completed') and run_id in manifests:
            return _html(running_page(run_id, manifests[run_id].run_name, manifests[run_id]))
        return _html(unreadable_page(str(run)))

    @app.get('/insights/{run_id}/review-data.json')
    def insights_review_data(run_id: str) -> Response:
        _, loaded, _ = _entries(get_insights_runs_dir())
        resolved = _resolve(run_id, loaded)
        if resolved is None:
            return Response(
                json.dumps({'error': 'Insights run not found'}),
                status_code=404,
                media_type='application/json',
                headers={'Cache-Control': 'no-store'},
            )
        _, run = resolved
        if not isinstance(run, InsightsRun):
            return Response(
                json.dumps({'error': f'Insights run is unreadable: {run}'}),
                status_code=422,
                media_type='application/json',
                headers={'Cache-Control': 'no-store'},
            )
        return Response(
            json.dumps(build_review_payload(run), ensure_ascii=False, separators=(',', ':')),
            media_type='application/json',
            headers={'Cache-Control': 'no-store'},
        )

    @app.get('/insights/{run_id}/trace-signals.json')
    def insights_trace_signals(req: Request, run_id: str) -> Response:
        """Read the saved signal details for one trace; this route never loads source data or calls a model."""
        _, loaded, _ = _entries(get_insights_runs_dir())
        resolved = _resolve(run_id, loaded)
        if resolved is None:
            payload: dict[str, object] = {'error': 'Insights run not found'}
            status_code = 404
        elif not isinstance(resolved[1], InsightsRun):
            payload = {'error': f'Insights run is unreadable: {resolved[1]}'}
            status_code = 422
        else:
            trace_id = req.query_params.get('trace_id')
            span_id = req.query_params.get('span_id') if 'span_id' in req.query_params else None
            if not trace_id or span_id is None:
                payload = {'error': 'trace_id and span_id are required'}
                status_code = 400
            else:
                span_id = span_id or None
                detail = build_signal_detail_payload(resolved[1], trace_id, span_id)
                if detail is None:
                    payload = {'error': 'Trace not found in this Insights run'}
                    status_code = 404
                else:
                    payload = detail
                    status_code = 200
        return Response(
            json.dumps(payload, ensure_ascii=False, separators=(',', ':')),
            status_code=status_code,
            media_type='application/json',
            headers={'Cache-Control': 'no-store'},
        )

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
            return _html(tab_content(resolved[1], tab, query=query) + tabs(resolved[1], tab, oob=True))
        fragment = _review_hash(resolved[1], tab, req.query_params)
        return RedirectResponse(f'/insights/{quote(resolved[1].run_id, safe="")}{fragment}', status_code=302)

    @app.get('/insights/{run_id}/rerun')
    def insights_rerun(run_id: str) -> Response:
        _, loaded, _ = _entries(get_insights_runs_dir())
        resolved = _resolve(run_id, loaded)
        if resolved is None or not isinstance(resolved[1], InsightsRun):
            return _html('<p class="insights-empty">Insights run not found.</p>', 404)
        return RedirectResponse(f'/insights/{quote(resolved[1].run_id, safe="")}?rerun=1', status_code=302)

    @app.get('/insights/{run_id}/cluster/{cluster_id}')
    def insights_cluster(run_id: str, cluster_id: str) -> Response:
        from evaluatorq.dashboard.insights_views import cluster_detail

        _, loaded, _ = _entries(get_insights_runs_dir())
        resolved = _resolve(run_id, loaded)
        if resolved is None or not isinstance(resolved[1], InsightsRun):
            return _html('<p class="insights-empty">Insights run not found.</p>', 404)
        return _html(cluster_detail(resolved[1], cluster_id))

    @app.get('/insights/{run_id}/trace')
    def insights_trace(req: Request, run_id: str) -> Response:
        _, loaded, _ = _entries(get_insights_runs_dir())
        resolved = _resolve(run_id, loaded)
        if resolved is None or not isinstance(resolved[1], InsightsRun):
            return _html('<p class="insights-empty">Insights run not found.</p>', 404)
        trace_id = req.query_params.get('trace_id')
        span_id = req.query_params.get('span_id') if 'span_id' in req.query_params else None
        if span_id == '':
            span_id = None
        trace = next(
            (item for item in resolved[1].traces if item.trace_id == trace_id and item.span_id == span_id),
            None,
        )
        if trace is None:
            return _html('<p class="insights-empty">Trace not found in this Insights run.</p>', 404)
        return _html(trace_detail_page(resolved[1], trace))

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
                row=query.get('row'),
                row_value=query.get('row_value'),
                column=query.get('column'),
                column_value=query.get('column_value'),
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
