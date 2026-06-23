"""FastHTML dashboard application factory.

``build_app(roots)`` returns a configured FastHTML app with routes:

- ``GET /``                   → index page listing all discovered reports
- ``GET /r/{rid}``            → embedded report view in the dashboard shell
- ``GET /r/{rid}/export``     → standalone HTML export (alias: export.html)
- ``GET /r/{rid}/export.html``→ standalone HTML export (full document)
- ``GET /r/{rid}/export.md``  → Markdown export (redteam and sim)
- ``GET /r/{rid}/export.csv`` → CSV of (filtered) result rows
- ``GET /r/{rid}/export.json``→ JSON of (filtered) result rows
- ``GET /r/{rid}/sim/transcript?idx=`` → sim transcript fragment (HTMX)

The ``roots`` parameter overrides the default scan directories so the app can
be tested against a temporary fixture directory without touching the real run
stores.

FastHTML 0.12.x passes ``on_startup`` / ``on_shutdown`` positional args to
``Starlette.__init__``, which Starlette 1.3.x removed.  A targeted shim is
applied before the FastHTML import so callers never see the error.  This is
documented as a concern in the task-3 report.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Starlette 1.3.x compatibility shim for FastHTML 0.12.x
# FastHTML passes on_startup/on_shutdown kwargs that Starlette 1.3 dropped (it
# now wants a single ``lifespan``). The shim is a process-global class patch, so
# it must be SEMANTICS-PRESERVING: rather than dropping on_startup/on_shutdown
# (which would silently lobotomize any OTHER Starlette app constructed in the
# same interpreter), it translates them into a lifespan that still runs them.
# ---------------------------------------------------------------------------
import contextlib

import starlette.applications as _starlette_app

_orig_starlette_init = _starlette_app.Starlette.__init__


def _lifespan_from_handlers(on_startup, on_shutdown):
    """Fold Starlette's removed on_startup/on_shutdown lists into a lifespan."""

    @contextlib.asynccontextmanager
    async def _lifespan(_app):
        for handler in on_startup or ():
            result = handler()
            if result is not None:
                await result
        try:
            yield
        finally:
            for handler in on_shutdown or ():
                result = handler()
                if result is not None:
                    await result

    return _lifespan


def _starlette_compat_init(  # type: ignore[override]
    self: _starlette_app.Starlette,
    debug: object = False,  # noqa: FBT002
    routes: object = None,
    middleware: object = None,
    exception_handlers: object = None,
    lifespan: object = None,
    on_startup: object = None,
    on_shutdown: object = None,
    **kw: object,
) -> None:
    # Preserve startup/shutdown handlers by folding them into a lifespan,
    # instead of discarding them. Only synthesize when no explicit lifespan
    # was given (Starlette forbids passing both).
    if lifespan is None and (on_startup or on_shutdown):
        lifespan = _lifespan_from_handlers(on_startup, on_shutdown)  # type: ignore[arg-type]
    _orig_starlette_init(
        self,
        debug=debug,  # type: ignore[arg-type]
        routes=routes,  # type: ignore[arg-type]
        middleware=middleware,  # type: ignore[arg-type]
        exception_handlers=exception_handlers,  # type: ignore[arg-type]
        lifespan=lifespan,  # type: ignore[arg-type]
    )


_starlette_app.Starlette.__init__ = _starlette_compat_init  # type: ignore[method-assign]

# ---------------------------------------------------------------------------
# Normal imports (after shim)
# ---------------------------------------------------------------------------
import csv
import io
import json
from pathlib import Path

from fasthtml.core import FastHTML, NotStr
from loguru import logger
from starlette.requests import Request  # noqa: TC002 — FastHTML inspects this annotation at runtime
from starlette.responses import Response

from evaluatorq.dashboard import library
from evaluatorq.dashboard.filter_request import parse_selections
from evaluatorq.dashboard.filters import FILTERS, apply_or_all
from evaluatorq.dashboard.redteam_views import register_redteam_view_routes
from evaluatorq.dashboard.shell import page
from evaluatorq.dashboard.sim_views import register_sim_view_routes
from evaluatorq.dashboard.surfaces import ADAPTERS
from evaluatorq.dashboard.view import (
    download_sidebar,
    filter_fragment,
    index_body,
    redteam_interactive_panels,
    render_filter_form,
    report_broken,
    report_not_found,
    report_view_with_filters,
    sim_interactive_panels,
)

_STATIC_DIR = Path(__file__).parent / 'static'


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------



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
    app = FastHTML(
        surreal=False,
        htmx=False,
        default_hdrs=False,
        pico=False,
    )
    # NOTE: static_route_exts is registered AFTER all custom routes so that
    # its catch-all /{fname:path}.{ext:static} does not steal requests for
    # /r/{rid}/export.html, export.md, export.csv, export.json etc.
    # We call it at the end of build_app() instead.

    # ------------------------------------------------------------------
    # Route: GET /  — report index
    # ------------------------------------------------------------------
    @app.get('/')
    def index(req: Request) -> NotStr:
        active_surface = req.query_params.get('surface') or None
        cards = library.scan(roots)
        body = index_body(cards, active_surface=active_surface)
        html = page('Reports', body, active_surface=active_surface)
        return NotStr(html)

    # ------------------------------------------------------------------
    # Route: GET /r/{rid}  — embedded report view
    # ------------------------------------------------------------------
    @app.get('/r/{rid}')
    def report_view(rid: str) -> NotStr | Response:
        path = library.resolve(rid, roots)
        if path is None:
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

        body_html = adapter.body(report_obj)
        name = adapter.name(report_obj)

        # Render filter form alongside the body.  Both known surfaces
        # (redteam, sim) are registered in FILTERS; fall back to 404 for unknown surfaces.
        filter_def = FILTERS.get(surface or '')
        if filter_def is None:
            not_found_html = page('Not found', report_not_found(rid))
            return Response(not_found_html, status_code=404, media_type='text/html')
        opts = filter_def.options(report_obj)
        form_html = render_filter_form(rid, surface or '', opts, {})
        body_with_filters = report_view_with_filters(rid, surface or '', body_html, form_html)

        # Append surface-specific interactive panels.
        if surface == 'redteam':
            body_with_filters = body_with_filters + redteam_interactive_panels(rid)
        elif surface == 'sim':
            # Build typed entries for the conversation list panel.
            from evaluatorq.simulation.reports.sections import individual_entries
            from evaluatorq.simulation.types import SimulationEntry

            entries: list[SimulationEntry] = individual_entries(report_obj.results)
            body_with_filters = body_with_filters + sim_interactive_panels(rid, entries)

        # Download sidebar — available exports per surface.
        dl_sidebar = download_sidebar(
            rid,
            surface or '',
            has_markdown=(adapter.export_markdown is not None),
            has_csv=(surface == 'redteam'),
            has_json=True,
        )
        body_with_filters = body_with_filters + dl_sidebar

        html = page(name, body_with_filters, active_surface=surface)
        return NotStr(html)

    # ------------------------------------------------------------------
    # Route: POST /r/{rid}/filter  — HTMX filter round-trip
    # ------------------------------------------------------------------
    @app.post('/r/{rid}/filter')
    async def report_filter(rid: str, req: Request) -> NotStr | Response:
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

        # Apply filters once; pass the already-filtered list to recompute_options
        # so apply() runs exactly once per POST (Fix 4).
        filtered = filter_def.apply(report_obj, selections)
        new_opts = filter_def.recompute_options(filtered)

        # Render body based on surface
        if surface == 'redteam':
            from evaluatorq.redteam.reports.converters import rebuild_filtered_report
            from evaluatorq.redteam.reports.export_html import render_report_body

            rebuilt = rebuild_filtered_report(report_obj, filtered)
            body_html = render_report_body(rebuilt)
        else:
            from evaluatorq.simulation.reports.export_html import (
                render_report_body as sim_render_report_body,
            )

            body_html = sim_render_report_body(
                filtered,
                target=report_obj.target_kind,
                run_date=report_obj.created_at,
            )

        form_html = render_filter_form(rid, surface or '', new_opts, selections)
        fragment_html = filter_fragment(rid, surface or '', body_html, form_html)

        # OOB swap: re-render the download sidebar with the active filter
        # querystring so that CSV/JSON links point at the filtered export.
        # HTMX processes elements with hx-swap-oob="true" outside the primary
        # swap target, updating #download-sidebar in-place.
        oob_sidebar = download_sidebar(
            rid,
            surface or '',
            selections=selections,
            has_markdown=(adapter.export_markdown is not None),
            has_csv=(surface == 'redteam'),
            has_json=True,
            oob=True,
        )
        # Signal interactive panels to refetch with the new filter.  Panels
        # that carry hx-trigger="load, orq:filter-changed from:body" and
        # hx-include="#filter-form" will catch this event, re-issue their
        # hx-get requests with the current form values, and re-render from the
        # filtered result set.
        return Response(
            fragment_html + oob_sidebar,
            media_type='text/html',
            headers={'HX-Trigger': 'orq:filter-changed'},
        )

    # ------------------------------------------------------------------
    # Route: GET /r/{rid}/export  (legacy alias) + export.html
    # ------------------------------------------------------------------
    def _do_html_export(rid: str) -> Response:
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

    @app.get('/r/{rid}/export')
    def report_export(rid: str) -> Response:
        return _do_html_export(rid)

    @app.get('/r/{rid}/export.html')
    def report_export_html(rid: str) -> Response:
        return _do_html_export(rid)

    # ------------------------------------------------------------------
    # Route: GET /r/{rid}/export.md  — Markdown (redteam and sim)
    # ------------------------------------------------------------------
    @app.get('/r/{rid}/export.md')
    def report_export_md(rid: str) -> Response:
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

    # ------------------------------------------------------------------
    # Route: GET /r/{rid}/export.csv  — CSV of (filtered) result rows
    # ------------------------------------------------------------------
    @app.get('/r/{rid}/export.csv')
    def report_export_csv(rid: str, req: Request) -> Response:
        path = library.resolve(rid, roots)
        if path is None:
            return Response('404 Not Found', status_code=404, media_type='text/plain')

        surface, _raw = library.load_surface(path)
        adapter = ADAPTERS.get(surface or '')
        filter_def = FILTERS.get(surface or '')
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

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(row_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(row_dicts)

        return Response(
            buf.getvalue(),
            media_type='text/csv',
            headers={'Content-Disposition': f'attachment; filename="{rid}.csv"'},
        )

    # ------------------------------------------------------------------
    # Route: GET /r/{rid}/export.json  — JSON of (filtered) result rows
    # ------------------------------------------------------------------
    @app.get('/r/{rid}/export.json')
    def report_export_json(rid: str, req: Request) -> Response:
        path = library.resolve(rid, roots)
        if path is None:
            return Response('404 Not Found', status_code=404, media_type='text/plain')

        surface, _raw = library.load_surface(path)
        adapter = ADAPTERS.get(surface or '')
        filter_def = FILTERS.get(surface or '')
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

    # ------------------------------------------------------------------
    # Routes: GET /r/{rid}/view/*  — redteam interactive fragment views
    # ------------------------------------------------------------------
    register_redteam_view_routes(app, roots)

    # ------------------------------------------------------------------
    # Routes: GET /r/{rid}/sim/*  — sim interactive fragment views
    # ------------------------------------------------------------------
    register_sim_view_routes(app, roots)

    # Register static file handler LAST so its catch-all
    # /{fname:path}.{ext:static} does not intercept the download routes above.
    app.static_route_exts(static_path=str(_STATIC_DIR))

    return app
