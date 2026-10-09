"""FastHTML dashboard application factory.

``build_app(roots)`` returns a configured FastHTML app with routes:

- ``GET /``                   → combined Dashboard landing (no ``surface``), or a
                               per-kind run list when ``?surface=redteam|sim|pairwise``
- ``GET /settings``           → editable settings plus read-only runtime details
- ``GET /settings/oauth-sessions?value=`` → CLI OAuth session field with checked token status (HTMX)
- ``GET /r/{rid}``            → embedded report view in the dashboard shell
- ``GET /r/{rid}/export``     → standalone HTML export (alias: export.html)
- ``GET /r/{rid}/export.html``→ standalone HTML export (full document)
- ``GET /r/{rid}/export.md``  → Markdown export (redteam and sim)
- ``GET /r/{rid}/export.csv`` → CSV of (filtered) result rows
- ``GET /r/{rid}/export.json``→ JSON of (filtered) result rows
- ``GET /r/{rid}/sim/transcript?idx=`` → sim transcript fragment (HTMX)
- ``GET /r/{rid}/sim/agent-card`` → asynchronously enriched sim agent card

The ``roots`` parameter overrides the default scan directories so the app can
be tested against a temporary fixture directory without touching the real run
stores.

FastHTML 0.12.x passes ``on_startup`` / ``on_shutdown`` positional args to
``Starlette.__init__``, which Starlette 1.3.x removed.  A targeted shim is
applied at import time so callers never see the error.  This is documented as
a concern in the task-3 report.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
from fasthtml.core import FastHTML, NotStr
from loguru import logger
from openai import APIError
from pydantic import ValidationError
from starlette.requests import Request  # noqa: TC002 — FastHTML inspects this annotation at runtime
from starlette.responses import JSONResponse, RedirectResponse, Response

# Starlette 1.3.x / FastHTML 0.12.x compat shim. Must be imported, not deferred to
# serve(): the patch has to be live when build_app() constructs the FastHTML app, and
# dashboard tests use build_app()+TestClient without ever calling serve(). See
# evaluatorq/dashboard/_compat.py.
import evaluatorq.dashboard._compat  # noqa: F401 — side-effect import
from evaluatorq.common.cli_oauth import list_oauth_sessions
from evaluatorq.common.llm_client import MissingLLMCredentialsError
from evaluatorq.common.orq_client import DEFAULT_ORQ_BASE_URL, OrqProfile, close_orq_client, list_orq_profiles
from evaluatorq.dashboard import library, metrics, model_choices, report_tabs
from evaluatorq.dashboard.apply_ui import register_apply_routes
from evaluatorq.dashboard.auth import build_auth_clients, resolve_dashboard_auth
from evaluatorq.dashboard.filter_request import parse_selections
from evaluatorq.dashboard.filters import FILTERS, apply_or_all
from evaluatorq.dashboard.insights_routes import register_insights_routes
from evaluatorq.dashboard.redteam_views import register_redteam_view_routes
from evaluatorq.dashboard.security import request_rejected
from evaluatorq.dashboard.shell import dashboard_css, dashboard_css_version, page
from evaluatorq.dashboard.sim_compare import register_sim_compare_routes
from evaluatorq.dashboard.sim_views import register_sim_view_routes
from evaluatorq.dashboard.surfaces import ADAPTERS
from evaluatorq.dashboard.trace_finder.routes import (
    initialize_finder_settings,
    register_finder_routes,
)
from evaluatorq.dashboard.trace_finder.sessions import TraceSessionMiddleware
from evaluatorq.dashboard.view import (
    MODEL_FIELDS,
    RUN_PAGE_SIZES,
    SURFACE_LABELS,
    filter_fragment,
    landing_body,
    model_field_source,
    oauth_session_field,
    redteam_overview_body,
    render_filter_form,
    report_actions,
    report_back_link,
    report_broken,
    report_not_found,
    report_view_with_filters,
    runs_screen_body,
    search_results,
    settings_body,
    settings_model_control,
    sim_overview_body,
)
from evaluatorq.trace_finder.settings import (
    DashboardSettings,
    credential_fingerprint,
    effective_settings,
    load_settings,
    save_settings,
    store_api_key,
)

_STATIC_DIR = Path(__file__).parent / 'static'


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _paging(req: Request) -> tuple[int, int]:
    """Parse ``?page=`` / ``?per_page=`` into a (page, per_page) pair. Bad input
    falls back to page 1 and the default page size; ``per_page`` is allow-listed."""
    try:
        page = max(1, int(req.query_params.get('page', 1)))
    except (TypeError, ValueError):
        page = 1
    try:
        per_page = int(req.query_params.get('per_page', RUN_PAGE_SIZES[0]))
    except (TypeError, ValueError):
        per_page = RUN_PAGE_SIZES[0]
    if per_page not in RUN_PAGE_SIZES:
        per_page = RUN_PAGE_SIZES[0]
    return page, per_page


def _mask_key(value: str) -> str:
    """Show only a key's last 4 characters so a user can validate *which* key is
    loaded without exposing the secret. Hidden chars render as ``*`` (capped so a
    long key doesn't blow out the row). Short keys reveal nothing but length."""
    if len(value) <= 4:
        return '*' * len(value)
    stars = '*' * min(len(value) - 4, 16)
    return f'{stars}{value[-4:]}'


_CSV_FORMULA_LEAD = ('=', '+', '-', '@', '\t', '\r')


def _csv_safe(value: object) -> object:
    """Neutralise spreadsheet formula injection in a CSV cell.

    Excel and Sheets evaluate a cell whose text begins with ``= + - @`` (or a
    tab / carriage return), so free text reaching an export can execute on the
    machine that opens the file. The pairwise export is the first here to carry
    free text — questions and judge model names are model-generated, and this
    tool renders adversarial red-team output — so a leading quote makes such a
    cell literal text.

    Values that only look like formulas are left alone: a number keeps its sign,
    and a bare ``-`` is the n/a placeholder the red-team export already writes.
    """
    if not isinstance(value, str) or not value.startswith(_CSV_FORMULA_LEAD):
        return value
    if value == '-':
        return value
    try:
        _ = float(value)
    except ValueError:
        return f"'{value}"
    return value


def _settings_config(
    roots: list[Path] | None,
    profile: OrqProfile | None = None,
    *,
    settings: DashboardSettings | None = None,
) -> list[tuple[str, str | list[str]]]:
    """Build the read-only runtime config shown on the Settings page: the run
    stores being scanned and API-key presence with a
    masked suffix (never the full value)."""
    from evaluatorq.dashboard.library import default_roots
    from evaluatorq.dashboard.orq_workspace import classify_host, resolve_base_url, resolve_slug

    scan_roots = roots if roots is not None else default_roots()
    store_paths = [str(p) for p in scan_roots] or ['—']
    settings = settings or effective_settings()
    config: list[tuple[str, str | list[str]]] = [
        ('Run stores', store_paths),
        ('Authentication method', settings.orq_auth_method.replace('_', ' ')),
    ]
    if settings.orq_auth_method == 'cli_profile' and settings.orq_profile is not None and profile is None:
        config.append(('Orq profile status', 'unavailable — choose Environment or another profile'))
    if profile is not None:
        config.extend((
            ('Selected profile API key', _mask_key(profile.api_key)),
            ('Selected profile host', profile.server or DEFAULT_ORQ_BASE_URL),
        ))
    for label, var in (('ORQ API key', 'ORQ_API_KEY'), ('OpenAI API key', 'OPENAI_API_KEY')):
        value = os.environ.get(var)
        config.append((label, _mask_key(value) if value else 'not set'))
    # Orq host/workspace are read-only here: deep-links derive them per-run from
    # each run's experiment_url. These env values are only the fallback for runs
    # with no experiment. See dashboard.orq_workspace.
    host = resolve_base_url()
    config.extend([
        ('Orq host', f'{host} ({classify_host(host)})'),
        ('Orq workspace', resolve_slug() or 'from experiment URL'),
        ('Orq project', settings.orq_project_name or 'all accessible projects'),
    ])
    return config


def _roots(req: Request) -> list[Path] | None:
    """Run-store roots stashed on the app by ``build_app``.

    Starlette's ``app.state`` is an untyped attribute bag, so reading
    ``req.app.state.roots`` inline degrades the value to ``Any`` at every handler.
    Every route reads the roots through here instead, so the declared type survives
    and there is one place to change if the stash moves.
    """
    roots: list[Path] | None = req.app.state.roots
    return roots


def _index(req: Request) -> NotStr:
    roots = _roots(req)
    surface = req.query_params.get('surface') or None
    if surface is None:
        # Combined Dashboard landing — aggregates across all run stores.
        body = landing_body(metrics.landing(roots))
        return NotStr(page('Dashboard', body, active_nav='dashboard'))
    # Agent Sim is the design's rich item-level overview; Red Team (and any
    # unknown surface) still render the run list.  Unknown surfaces fall
    # through to an empty run-list screen rather than 500.
    label = SURFACE_LABELS.get(surface, 'Reports')
    pg, per_pg = _paging(req)
    if surface == 'sim':
        # Compare picker options: sim runs only, excluding error-flagged runs
        # (they route to a 404), newest-first, capped so the dropdown stays
        # usable on large stores. library.scan reuses the mtime-keyed JSON
        # cache metrics.sim_overview already warmed, so this is not a full re-parse.
        sim_choices = [(c.id, c.name) for c in library.scan(roots) if c.surface == 'sim' and not c.error][:100]
        body = sim_overview_body(metrics.sim_overview(roots, page=pg, per_page=per_pg), compare_choices=sim_choices)
    elif surface == 'redteam':
        body = redteam_overview_body(metrics.redteam_overview(roots, page=pg, per_page=per_pg))
    else:
        rows = [r for r in metrics.run_rows(roots) if r.surface == surface]
        body = runs_screen_body(rows, surface)
    return NotStr(page(label, body, active_surface=surface))


def _dashboard_css(v: str = '') -> Response:
    # Only the current content hash (shell.dashboard_css_href) is safe to cache forever: a stale or missing
    # `v` would otherwise pin today's sheet under a URL that a later version must be able to change.
    cache = 'public, max-age=31536000, immutable' if v == dashboard_css_version() else 'no-cache'
    return Response(dashboard_css(), media_type='text/css', headers={'Cache-Control': cache})


async def _auth_status(req: Request) -> JSONResponse:
    """Verify the selected dashboard credential for the startup toast."""

    settings = effective_settings()
    try:
        auth = await asyncio.to_thread(resolve_dashboard_auth, settings)
    except (OSError, RuntimeError, ValueError) as exc:
        return JSONResponse({'status': 'action', 'method': settings.orq_auth_method, 'message': str(exc)})
    if auth.method == 'environment' and not auth.api_key:
        return JSONResponse({
            'status': 'action',
            'method': auth.method,
            'message': 'ORQ_API_KEY is not set. Choose an authentication method in Settings.',
        })
    orq = None
    llm = None
    try:
        orq, llm = build_auth_clients(auth, workspace=settings.orq_workspace, project=settings.orq_project_id)
        now = datetime.now(timezone.utc)
        await asyncio.wait_for(
            orq.traces.list_facet_values_async(field='agent_name', from_=now - timedelta(days=1), to=now, limit=1),
            timeout=12,
        )
    except Exception as exc:
        status_code = getattr(exc, 'status_code', None)
        response = getattr(exc, 'response', None)
        status_code = status_code or getattr(response, 'status_code', None)
        message = str(exc).lower()
        if status_code in (401, 403) or any(
            marker in message
            for marker in (
                'unauthorized',
                'not authenticated',
                'login required',
                'http 401',
                'http 403',
                'session expired',
                'cli is not installed',
                'sign-in needs attention',
                'orq auth login',
            )
        ):
            return JSONResponse({
                'status': 'action',
                'method': auth.method,
                'message': f'{auth.label} was rejected. Open Settings to choose or renew a credential.',
            })
        expected_provider_failure = status_code is not None or isinstance(exc, (httpx.HTTPError, APIError))
        if not expected_provider_failure:
            logger.opt(exception=True).error(
                'Unexpected dashboard auth status probe failure for {}: {}', auth.label, exc
            )
            return JSONResponse({
                'status': 'error',
                'method': auth.method,
                'message': 'The authentication check failed unexpectedly. Check the dashboard logs.',
            })
        logger.opt(exception=True).warning('Dashboard auth status probe failed for {}: {}', auth.label, exc)
        return JSONResponse({
            'status': 'unavailable',
            'method': auth.method,
            'message': f'Could not check {auth.label} right now. Try again shortly.',
        })
    finally:
        if orq is not None:
            try:
                await close_orq_client(orq)
            except Exception as exc:
                logger.opt(exception=True).warning('Could not close dashboard auth Orq client: {}', exc)
        if llm is not None:
            close = getattr(llm, 'close', None)
            if close is not None:
                try:
                    await close()
                except Exception as exc:
                    logger.opt(exception=True).warning('Could not close dashboard auth LLM client: {}', exc)
    return JSONResponse({'status': 'valid', 'method': auth.method})


async def _settings(req: Request) -> NotStr:
    roots = _roots(req)
    settings = effective_settings()
    profiles = await asyncio.to_thread(list_orq_profiles)
    if 'profile' in req.query_params:
        requested = req.query_params['profile']
        if not requested or any(profile.name == requested and '*' not in profile.api_key for profile in profiles):
            settings = settings.model_copy(
                update={
                    'orq_auth_method': 'cli_profile' if requested else 'environment',
                    'orq_profile': requested or None,
                    'orq_profile_host': None,
                    'orq_workspace': None,
                    'orq_project_id': None,
                    'orq_project_name': None,
                }
            )
    body = settings_body(
        _settings_config(
            roots,
            next(
                (p for p in profiles if settings.orq_auth_method == 'cli_profile' and p.name == settings.orq_profile),
                None,
            ),
            settings=settings,
        ),
        settings,
        saved=req.query_params.get('saved') == '1',
        preview='profile' in req.query_params,
        profiles=profiles,
    )
    return NotStr(page('Settings', body, active_nav='settings'))


async def _settings_oauth_sessions(req: Request) -> NotStr:
    """The CLI OAuth server field, loaded after the page because checking an expired login calls Orq."""
    value = req.query_params.get('value') or DEFAULT_ORQ_BASE_URL
    sessions = await asyncio.to_thread(list_oauth_sessions)
    return NotStr(oauth_session_field(value, sessions))


_MODEL_FIELD_KINDS: dict[str, Literal['chat', 'classify', 'embedding']] = {
    'fast_model': 'chat',
    'smart_model': 'chat',
    'classifier_model': 'classify',
    'embedding_model': 'embedding',
}


def _join_notes(*notes: str) -> str:
    return ' '.join(note for note in notes if note)


async def _settings_models(req: Request) -> NotStr:
    """One settings model field as its workspace menu, or as the text box when there is no catalogue."""
    field = req.query_params.get('field', '')
    if field not in MODEL_FIELDS:
        return NotStr('')
    value = req.query_params.get(field, '')
    params = req.query_params
    name = params.get('orq_profile') or params.get('profile') or ''
    settings = effective_settings()
    pinned, pinned_note = model_field_source(field)
    env_note = pinned_note if pinned and value == pinned else ''
    kind = _MODEL_FIELD_KINDS[field]
    requested_method = params.get('orq_auth_method') or params.get('auth_method')
    update: dict[str, Any] = {}
    if requested_method in ('environment', 'cli_profile', 'stored_api_key', 'cli_oauth'):
        update = {'orq_auth_method': requested_method, 'orq_profile': name or None}
    elif name:
        update = {'orq_auth_method': 'cli_profile', 'orq_profile': name}
    # The server must be one of the CLI's own logins: this GET carries no CSRF token, so an arbitrary
    # URL would let any caller point the Orq CLI at a host of their choosing. A saved value is checked
    # too whenever OAuth will use it, since a settings file can hold any URL.
    server = params.get('orq_oauth_server') or settings.orq_oauth_server
    method = update.get('orq_auth_method', settings.orq_auth_method)
    if server and (method == 'cli_oauth' or server != settings.orq_oauth_server):
        sessions = await asyncio.to_thread(list_oauth_sessions)
        if server not in {session.server for session in sessions}:
            server = None
    update['orq_oauth_server'] = server
    settings = settings.model_copy(update=update)
    if settings.orq_auth_method == 'cli_profile' and not settings.orq_profile:
        return NotStr(
            settings_model_control(
                field, value, {}, note=_join_notes(env_note, 'Choose a CLI profile to load its models.')
            )
        )
    profiles = await asyncio.to_thread(list_orq_profiles) if settings.orq_auth_method == 'cli_profile' else []
    try:
        auth = resolve_dashboard_auth(settings, profiles=profiles)
        async with model_choices.catalogue_client(auth) as client:
            groups = await model_choices.models_by_provider(client, kind=kind)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        groups = {}
        logger.warning('Could not load the model catalogue for {}: {}', field, exc)
        if isinstance(exc, MissingLLMCredentialsError):
            note = (
                "No Orq API key for the chosen authentication method, so the model list can't load. "
                'Set one under Authentication, or type a model id.'
            )
        else:
            note = "Couldn't load the model list from Orq. Type a model id."
    else:
        note = '' if groups else 'Orq returned no models for this field. Type a model id.'
    return NotStr(settings_model_control(field, value, groups, note=_join_notes(env_note, note)))


async def _save_settings(req: Request) -> Response | NotStr:  # noqa: C901
    """Validate and persist the settings form, or render field errors."""
    form_data = await req.form()
    rejected = request_rejected(req, form_data)
    if rejected:
        roots = _roots(req)
        config = await asyncio.to_thread(_settings_config, roots)
        body = settings_body(config, effective_settings(), errors={'form': rejected})
        return Response(page('Settings', body, active_nav='settings'), status_code=403, media_type='text/html')
    roots = _roots(req)
    # Use saved values as the baseline so unchanged environment overrides are not persisted.
    current = load_settings()
    values = _submitted_settings_values(form_data, current)
    profiles = await asyncio.to_thread(list_orq_profiles)
    errors: dict[str, str] = {}
    settings: DashboardSettings | None = None
    try:
        settings = DashboardSettings.model_validate(values)
    except ValidationError as exc:
        for detail in exc.errors():
            location = detail.get('loc', ())
            field = str(location[0]) if location else 'form'
            errors[field] = str(detail.get('msg', 'Invalid value'))
    if settings is not None and settings.orq_auth_method == 'stored_api_key':
        entered_key = str(form_data.get('orq_api_key_entry', '')).strip()
        if entered_key:
            try:
                settings = await asyncio.to_thread(store_api_key, settings, entered_key)
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning('Could not save dashboard API key securely: {}', type(exc).__name__)
                errors['orq_api_key_entry'] = 'Could not encrypt this key. Check the local Keychain and try again.'
        elif not settings.orq_api_key_ciphertext:
            errors['orq_api_key_entry'] = 'Enter an API key to use this method.'
    if (
        settings is not None
        and settings.orq_auth_method == 'cli_profile'
        and ('orq_auth_method' in form_data or 'orq_profile' in form_data)
        and settings.orq_profile is not None
        and settings.orq_profile not in {p.name for p in profiles}
    ):
        errors['orq_profile'] = 'The orq CLI does not know this profile'
    selected_profile = next(
        (
            p
            for p in profiles
            if settings is not None and settings.orq_auth_method == 'cli_profile' and p.name == settings.orq_profile
        ),
        None,
    )
    if (
        settings is not None
        and settings.orq_auth_method == 'cli_profile'
        and selected_profile is None
        and ('orq_auth_method' in form_data or 'orq_profile' in form_data)
    ):
        errors.setdefault('orq_profile', 'Choose an Orq CLI API-key profile.')
    if selected_profile is not None and '*' in selected_profile.api_key:
        errors['orq_profile'] = 'The installed orq CLI masks this profile key; use Environment credentials.'
    if settings is not None:
        settings = settings.model_copy(update={'orq_workspace': None, 'orq_project_id': None, 'orq_project_name': None})
        try:
            auth = resolve_dashboard_auth(settings, profiles=profiles)
        except ValueError:
            auth = None
        settings = settings.model_copy(
            update={
                'orq_profile_host': (selected_profile.server or DEFAULT_ORQ_BASE_URL)
                if selected_profile
                else settings.orq_profile_host
                if settings.orq_auth_method == 'stored_api_key'
                else None,
                'orq_credential_fingerprint': credential_fingerprint(
                    auth.api_key if auth else None,
                    auth.base_url if auth else None,
                ),
            }
        )
    if settings is None or errors:
        body = settings_body(_settings_config(roots), values, errors=errors, profiles=profiles)
        return Response(page('Settings', body, active_nav='settings'), status_code=422, media_type='text/html')

    await asyncio.to_thread(save_settings, settings)
    async with req.app.state.finder_store_lock:
        req.app.state.finder_profile = selected_profile
        if settings.orq_auth_method == 'cli_profile' and req.app.state.finder_profile is None:
            logger.warning(
                'Saved Orq profile {} is unavailable; select another profile or Environment', settings.orq_profile
            )
        old_search_store = getattr(req.app.state, 'finder_search_store', None)
        old_session_stores = await req.app.state.trace_sessions.detach_all()
        old_legacy_store = getattr(req.app.state, 'finder_store', None)
        old_warmup = getattr(req.app.state, 'finder_catalogue_warmup', None)
        req.app.state.finder_settings = effective_settings()
        req.app.state.finder_generation += 1
        for state_name in (
            'finder_store',
            'finder_search_store',
            'finder_catalogue_cache',
            'finder_catalogue_warmup',
            'insights_facet_catalogues',
        ):
            if hasattr(req.app.state, state_name):
                delattr(req.app.state, state_name)
    await _cancel_background_task(old_warmup[2] if old_warmup is not None else None)
    # Retire stores after removing them from app state so new requests cannot acquire them.
    await _close_finder_stores(*old_session_stores, old_legacy_store, old_search_store)
    req.app.state.finder_unavailable_reason = None
    return RedirectResponse('/settings?saved=1', status_code=303)


async def _cancel_background_task(task: asyncio.Task[Any] | None) -> None:
    if task is not None and not task.done():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def _close_finder_stores(*stores: Any) -> None:
    active_stores = [store for store in stores if store is not None]
    closing = asyncio.gather(*(store.close() for store in active_stores), return_exceptions=True)
    try:
        results = await asyncio.shield(closing)
    except asyncio.CancelledError:
        results = await closing
        _log_finder_store_close_failures(active_stores, results)
        raise
    _log_finder_store_close_failures(active_stores, results)
    cancellation = next((result for result in results if isinstance(result, asyncio.CancelledError)), None)
    if cancellation is not None:
        raise cancellation
    fatal = next(
        (result for result in results if isinstance(result, BaseException) and not isinstance(result, Exception)), None
    )
    if fatal is not None:
        raise fatal


def _log_finder_store_close_failures(stores: list[Any], results: list[Any]) -> None:
    for store, result in zip(stores, results, strict=True):
        if isinstance(result, Exception):
            logger.opt(exception=(type(result), result, result.__traceback__)).warning(
                'Could not close retired finder store {}: {}', type(store).__name__, result
            )


def _submitted_settings_values(form_data: Any, current: DashboardSettings) -> dict[str, object]:
    """Keep unchanged environment overrides out of the saved file."""
    values: dict[str, object] = {'model_overrides': current.model_overrides}
    for name in MODEL_FIELDS:
        saved = getattr(current, name)
        if name not in form_data:
            values[name] = saved
            continue
        submitted = str(form_data[name]).strip() or None
        # A value that only came from the environment or a flag is not the user's choice, so the saved one stays.
        values[name] = saved if submitted and submitted == model_field_source(name)[0] else submitted
    values['orq_profile'] = form_data.get('orq_profile', current.orq_profile)
    method = form_data.get('orq_auth_method')
    if method is None and 'orq_profile' in form_data:
        method = 'cli_profile' if form_data.get('orq_profile') else 'environment'
    values['orq_auth_method'] = method or current.orq_auth_method
    if values['orq_auth_method'] != 'cli_profile':
        values['orq_profile'] = None
    values['orq_profile_host'] = form_data.get('orq_stored_key_host', current.orq_profile_host)
    values['orq_api_key_ciphertext'] = current.orq_api_key_ciphertext
    values['orq_oauth_server'] = form_data.get('orq_oauth_server', current.orq_oauth_server)
    values['orq_credential_fingerprint'] = current.orq_credential_fingerprint
    values['orq_workspace'] = None
    values['orq_project_id'] = None
    values['orq_project_name'] = None
    values['ask_ai_mode'] = form_data.get('ask_ai_mode', current.ask_ai_mode)
    raw_input_cap = form_data.get('trace_input_chars')
    if raw_input_cap is None:
        values['trace_input_chars'] = current.trace_input_chars
    else:
        try:
            values['trace_input_chars'] = int(str(raw_input_cap).strip())
        except ValueError:
            values['trace_input_chars'] = raw_input_cap
    # Finder limits are per-run controls, so the settings form keeps their saved defaults.
    values.update(
        window_days=current.window_days,
        limit=current.limit,
        parallelism=current.parallelism,
        explorer_columns=current.explorer_columns,
    )
    return values


def _search(req: Request) -> NotStr:
    roots = _roots(req)
    q = req.query_params.get('q') or ''
    return NotStr(search_results(library.scan(roots), q))


def _report_view(rid: str, req: Request) -> NotStr | Response:
    roots = _roots(req)
    path = library.resolve(rid, roots)
    if path is None:
        # No report on disk — this may be an in-flight (running/error/
        # cancelled) run tracked only by a manifest. Render its status/stage.
        manifest = library.resolve_manifest(rid, roots)
        if manifest is not None:
            from evaluatorq.dashboard.view import report_in_flight

            in_flight_html = page(
                manifest.run_name,
                report_in_flight(manifest),
                active_surface=manifest.surface.value,
            )
            return NotStr(in_flight_html)
        not_found_html = page('Not found', report_not_found(rid))
        return Response(not_found_html, status_code=404, media_type='text/html')

    surface, _raw = library.load_surface(path)
    adapter = ADAPTERS.get(surface or '')
    if adapter is None:
        not_found_html = page('Not found', report_not_found(rid))
        return Response(not_found_html, status_code=404, media_type='text/html')

    try:
        report_obj = adapter.load(path)
    except Exception as exc:
        logger.warning('Failed to load report {}: {}', path.name, exc)
        broken_html = page(
            f'Error — {path.name}',
            report_broken(rid, path.name, str(exc)),
            active_surface=surface,
        )
        return Response(broken_html, status_code=200, media_type='text/html')

    name = adapter.name(report_obj)
    back_link = report_back_link(surface or '')

    # Render filter form alongside the body.  Both known surfaces
    # (redteam, sim) are registered in FILTERS; fall back to 404 for unknown surfaces.
    filter_def = FILTERS.get(surface or '')
    if filter_def is None:
        not_found_html = page('Not found', report_not_found(rid))
        return Response(not_found_html, status_code=404, media_type='text/html')

    # Tabbed body for the known surfaces; the interactive panels live inside
    # their tabs, so they are no longer appended separately.
    from evaluatorq.dashboard.orq_workspace import cli_slug_render_scope

    with cli_slug_render_scope():
        if surface == 'sim':
            from evaluatorq.dashboard.view import sim_run_compare_control

            # Same choice list as the overview picker (sim only, no error runs,
            # capped); the control itself drops the current run from the options.
            choices = [(c.id, c.name) for c in library.scan(roots) if c.surface == 'sim' and not c.error][:100]
            body_html = report_tabs.sim_report_tabs(rid, report_obj, compare_html=sim_run_compare_control(rid, choices))
        elif surface == 'redteam':
            body_html = report_tabs.redteam_report_tabs(rid, report_obj)
        else:
            body_html = adapter.body(report_obj)

    opts = filter_def.options(report_obj)
    total_results = len(filter_def.results(report_obj))
    form_html = render_filter_form(rid, surface or '', opts, {}, shown=total_results, total=total_results)
    body_with_filters = report_view_with_filters(rid, surface or '', body_html, form_html)

    html = page(
        name,
        body_with_filters,
        active_surface=surface,
        actions_html=report_actions(rid),
        back_html=back_link,
    )
    return NotStr(html)


async def _sim_agent_card(rid: str, req: Request) -> NotStr | Response:
    roots = _roots(req)
    path = library.resolve(rid, roots)
    if path is None:
        return Response('404 Not Found', status_code=404, media_type='text/plain')
    surface, _raw = library.load_surface(path)
    if surface != 'sim':
        return Response('404 Not Found', status_code=404, media_type='text/plain')
    adapter = ADAPTERS.get(surface)
    if adapter is None:
        return Response('404 Not Found', status_code=404, media_type='text/plain')
    try:
        run = adapter.load(path)
    except Exception as exc:
        logger.warning('Failed to load sim report for agent card {}: {}', path.name, exc)
        return Response('Error loading report', status_code=422, media_type='text/plain')
    return NotStr(await report_tabs.sim_agent_card_fragment(run))


async def _report_filter(rid: str, req: Request) -> NotStr | Response:
    roots = _roots(req)
    path = library.resolve(rid, roots)
    if path is None:
        return Response('404 Not Found', status_code=404, media_type='text/plain')

    surface, _raw = library.load_surface(path)
    adapter = ADAPTERS.get(surface or '')
    filter_def = FILTERS.get(surface or '')
    if adapter is None or filter_def is None:
        return Response('404 Not Found', status_code=404, media_type='text/plain')

    try:
        report_obj = adapter.load(path)
    except Exception as exc:
        logger.warning('Failed to load report for filter {}: {}', path.name, exc)
        return Response(
            f'Error loading report: {exc}',
            status_code=422,
            media_type='text/plain',
        )

    # Parse form data — build selections dict[str, list[str]]
    form_data = await req.form()
    selections: dict[str, list[str]] = {}
    for key, value in form_data.multi_items():
        selections.setdefault(key, []).append(str(value))

    # Apply filters to get the shown rows, but keep the option lists fixed
    # to the FULL dataset.  Recomputing options from the filtered rows made
    # a just-deselected value disappear from its own multi-select.
    filtered = filter_def.apply(report_obj, selections)
    new_opts = filter_def.options(report_obj)

    # Render the tabbed body from the filtered results so the static tab
    # content (tables, charts) tracks the filter, not just the HTMX panels.
    from evaluatorq.dashboard.orq_workspace import cli_slug_render_scope

    with cli_slug_render_scope():
        if surface == 'sim':
            body_html = await asyncio.to_thread(report_tabs.sim_report_tabs, rid, report_obj, filtered)
        elif surface == 'redteam':
            from evaluatorq.redteam.reports.converters import rebuild_filtered_report

            filtered_report = rebuild_filtered_report(report_obj, filtered)
            body_html = await asyncio.to_thread(report_tabs.redteam_report_tabs, rid, filtered_report)
        else:
            body_html = adapter.body_from_results(report_obj, filtered)

    form_html = render_filter_form(
        rid, surface or '', new_opts, selections, shown=len(filtered), total=len(filter_def.results(report_obj))
    )
    fragment_html = filter_fragment(rid, surface or '', body_html, form_html)

    # Signal interactive panels to refetch with the new filter.  Panels
    # that carry hx-trigger="load, orq:filter-changed from:body" and
    # hx-include="#filter-form" will catch this event, re-issue their
    # hx-get requests with the current form values, and re-render from the
    # filtered result set.
    return Response(
        fragment_html,
        media_type='text/html',
        headers={'HX-Trigger': 'orq:filter-changed'},
    )


def _do_html_export(rid: str, roots: list[Path] | None) -> Response:
    path = library.resolve(rid, roots)
    if path is None:
        return Response('404 Not Found', status_code=404, media_type='text/plain')

    surface, _raw = library.load_surface(path)
    adapter = ADAPTERS.get(surface or '')
    if adapter is None:
        return Response('404 Not Found', status_code=404, media_type='text/plain')

    try:
        report_obj = adapter.load(path)
    except Exception as exc:
        logger.warning('Failed to load report for export {}: {}', path.name, exc)
        return Response(
            f'Error loading report {path.name}: {exc}',
            status_code=422,
            media_type='text/plain',
        )
    return Response(
        adapter.export(report_obj),
        media_type='text/html',
        headers={'Content-Disposition': f'attachment; filename="{rid}.html"'},
    )


def _report_export(rid: str, req: Request) -> Response:
    roots = _roots(req)
    return _do_html_export(rid, roots)


def _report_export_html(rid: str, req: Request) -> Response:
    roots = _roots(req)
    return _do_html_export(rid, roots)


def _report_export_md(rid: str, req: Request) -> Response:
    roots = _roots(req)
    path = library.resolve(rid, roots)
    if path is None:
        return Response('404 Not Found', status_code=404, media_type='text/plain')

    surface, _raw = library.load_surface(path)
    adapter = ADAPTERS.get(surface or '')
    if adapter is None:
        return Response('404 Not Found', status_code=404, media_type='text/plain')

    if adapter.export_markdown is None:
        return Response(
            'no markdown export for this surface',
            status_code=404,
            media_type='text/plain',
        )

    try:
        report_obj = adapter.load(path)
    except Exception as exc:
        logger.warning('Failed to load report for md export {}: {}', path.name, exc)
        return Response(
            f'Error loading report {path.name}: {exc}',
            status_code=422,
            media_type='text/plain',
        )

    md_text = adapter.export_markdown(report_obj)
    return Response(
        md_text,
        media_type='text/markdown',
        headers={'Content-Disposition': f'attachment; filename="{rid}.md"'},
    )


def _report_export_csv(rid: str, req: Request) -> Response:
    roots = _roots(req)
    path = library.resolve(rid, roots)
    if path is None:
        return Response('404 Not Found', status_code=404, media_type='text/plain')

    surface, _raw = library.load_surface(path)
    adapter = ADAPTERS.get(surface or '')
    if adapter is None:
        return Response('404 Not Found', status_code=404, media_type='text/plain')

    # sim never had a CSV export — honest parity.
    if surface == 'sim':
        return Response(
            'no CSV export for simulation runs',
            status_code=404,
            media_type='text/plain',
        )

    if adapter.rows is None:
        return Response(
            'CSV export not supported for this surface',
            status_code=404,
            media_type='text/plain',
        )

    try:
        report_obj = adapter.load(path)
    except Exception as exc:
        logger.warning('Failed to load report for csv {}: {}', path.name, exc)
        return Response(
            f'Error loading report {path.name}: {exc}',
            status_code=422,
            media_type='text/plain',
        )

    # Apply filters from the query-string (same logic as POST /filter).
    selections = parse_selections(req, surface or '')
    filtered = apply_or_all(report_obj, surface or '', selections)

    row_dicts = adapter.rows(report_obj, filtered)

    if not row_dicts:
        # Return an empty CSV with just the header row (parity: empty filter)
        return Response(
            '',
            media_type='text/csv',
            headers={'Content-Disposition': f'attachment; filename="{rid}.csv"'},
        )

    # Headers as well as cells: a judge's model name becomes a column name.
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=[str(_csv_safe(k)) for k in row_dicts[0]])
    writer.writeheader()
    writer.writerows({str(_csv_safe(k)): _csv_safe(v) for k, v in row.items()} for row in row_dicts)

    return Response(
        buf.getvalue(),
        media_type='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{rid}.csv"'},
    )


def _report_export_json(rid: str, req: Request) -> Response:
    roots = _roots(req)
    path = library.resolve(rid, roots)
    if path is None:
        return Response('404 Not Found', status_code=404, media_type='text/plain')

    surface, _raw = library.load_surface(path)
    adapter = ADAPTERS.get(surface or '')
    if adapter is None:
        return Response('404 Not Found', status_code=404, media_type='text/plain')

    if adapter.rows is None:
        return Response(
            'JSON export not supported for this surface',
            status_code=404,
            media_type='text/plain',
        )

    try:
        report_obj = adapter.load(path)
    except Exception as exc:
        logger.warning('Failed to load report for json {}: {}', path.name, exc)
        return Response(
            f'Error loading report {path.name}: {exc}',
            status_code=422,
            media_type='text/plain',
        )

    # Apply filters from the query-string.
    selections = parse_selections(req, surface or '')
    filtered = apply_or_all(report_obj, surface or '', selections)

    row_dicts = adapter.rows(report_obj, filtered)

    json_str = json.dumps(row_dicts, indent=2, default=str)
    return Response(
        json_str,
        media_type='application/json',
        headers={'Content-Disposition': f'attachment; filename="{rid}.json"'},
    )


def register_report_routes(app: FastHTML) -> None:
    """Register the report routes on *app*."""
    app.get('/')(_index)
    app.get('/settings')(_settings)
    app.post('/settings')(_save_settings)
    app.get('/settings/models')(_settings_models)
    app.get('/settings/oauth-sessions')(_settings_oauth_sessions)
    app.get('/search')(_search)
    app.get('/r/{rid}')(_report_view)
    app.get('/r/{rid}/sim/agent-card')(_sim_agent_card)
    app.post('/r/{rid}/filter')(_report_filter)
    app.get('/r/{rid}/export')(_report_export)
    app.get('/r/{rid}/export.html')(_report_export_html)
    app.get('/r/{rid}/export.md')(_report_export_md)
    app.get('/r/{rid}/export.csv')(_report_export_csv)
    app.get('/r/{rid}/export.json')(_report_export_json)


def build_app(roots: list[Path] | None = None) -> FastHTML:
    """Create and return the configured FastHTML dashboard application.

    Args:
        roots: Override the default run-store directories.  When ``None`` the
            production defaults from ``evaluatorq.dashboard.library`` are used.
            Pass an explicit list to point the app at test fixture directories.

    Returns:
        A ``FastHTML`` ASGI application ready to be served or tested via
        ``starlette.testclient.TestClient``.
    """

    @asynccontextmanager
    async def lifespan(runtime: FastHTML) -> Any:
        try:
            yield
        finally:
            warmup = getattr(runtime.state, 'finder_catalogue_warmup', None)
            await _cancel_background_task(warmup[2] if warmup is not None else None)
            await runtime.state.trace_sessions.close_all()
            store = getattr(runtime.state, 'finder_store', None)
            if store is not None:
                await store.close()
                del runtime.state.finder_store
            search_store = getattr(runtime.state, 'finder_search_store', None)
            if search_store is not None:
                await search_store.close()
                del runtime.state.finder_search_store

    app = FastHTML(
        surreal=False,
        htmx=False,
        default_hdrs=False,
        pico=False,
        lifespan=lifespan,
    )
    app.state.roots = roots
    initialize_finder_settings(app)
    app.add_middleware(TraceSessionMiddleware, app_state=app.state)
    app.get('/auth/status')(_auth_status)
    app.get('/static/dashboard.css')(_dashboard_css)
    # NOTE: static_route_exts is registered AFTER all custom routes so that
    # its catch-all /{fname:path}.{ext:static} does not steal requests for
    # /r/{rid}/export.html, export.md, export.csv, export.json etc.
    # We call it at the end of build_app() instead.

    # ------------------------------------------------------------------
    # Routes: report pages and exports
    # ------------------------------------------------------------------
    register_report_routes(app)
    # ------------------------------------------------------------------
    # Routes: GET /r/{rid}/view/*  — redteam interactive fragment views
    # ------------------------------------------------------------------
    register_redteam_view_routes(app, roots)

    # ------------------------------------------------------------------
    # Routes: POST /r/{rid}/redteam/apply/*  — apply recommendations (RES-1143)
    # ------------------------------------------------------------------
    register_apply_routes(app, roots)

    # ------------------------------------------------------------------
    # Routes: GET /r/{rid}/sim/*  — sim interactive fragment views
    # ------------------------------------------------------------------
    register_sim_view_routes(app, roots)

    # ------------------------------------------------------------------
    # Routes: GET /compare/sim*  — side-by-side sim run comparison
    # ------------------------------------------------------------------
    register_sim_compare_routes(app, roots)

    # ------------------------------------------------------------------
    # Routes: /find — classifier trace finder
    # ------------------------------------------------------------------
    register_finder_routes(app)

    # ------------------------------------------------------------------
    # Routes: /insights — read-only trace intelligence review
    # ------------------------------------------------------------------
    register_insights_routes(app)

    # Register static file handler LAST so its catch-all /{fname}.{ext} does not
    # intercept the download routes above. Serve under /static/ to match the page
    # head's Script(src="/static/…") references (view.py head_assets); without the
    # explicit prefix every /static/*.js request 404s and htmx/vega never load.
    app.static_route_exts(prefix='/static', static_path=str(_STATIC_DIR))

    return app
