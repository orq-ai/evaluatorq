"""FastHTML dashboard application factory.

``build_app(roots)`` returns a configured FastHTML app with three routes:

- ``GET /``             → index page listing all discovered reports
- ``GET /r/{rid}``      → embedded report view in the dashboard shell
- ``GET /r/{rid}/export`` → standalone HTML export (full document)

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
# FastHTML passes on_startup/on_shutdown kwargs that Starlette 1.3 dropped.
# ---------------------------------------------------------------------------
import starlette.applications as _starlette_app

_orig_starlette_init = _starlette_app.Starlette.__init__


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
from pathlib import Path

from fasthtml.core import FastHTML, NotStr
from loguru import logger
from starlette.responses import Response

from evaluatorq.dashboard import library
from evaluatorq.dashboard.shell import page
from evaluatorq.dashboard.surfaces import ADAPTERS
from evaluatorq.dashboard.view import index_body, report_broken, report_not_found

_STATIC_DIR = Path(__file__).parent / "static"


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
    app.static_route_exts(static_path=str(_STATIC_DIR))

    # ------------------------------------------------------------------
    # Route: GET /  — report index
    # ------------------------------------------------------------------
    @app.get("/")
    def index() -> NotStr:
        cards = library.scan(roots)
        body = index_body(cards)
        html = page("Reports", body)
        return NotStr(html)

    # ------------------------------------------------------------------
    # Route: GET /r/{rid}  — embedded report view
    # ------------------------------------------------------------------
    @app.get("/r/{rid}")
    def report_view(rid: str) -> NotStr | Response:
        path = library.resolve(rid, roots)
        if path is None:
            not_found_html = page("Not found", report_not_found(rid))
            return Response(not_found_html, status_code=404, media_type="text/html")

        surface, _raw = library.load_surface(path)
        adapter = ADAPTERS.get(surface or "")
        if adapter is None:
            not_found_html = page("Not found", report_not_found(rid))
            return Response(not_found_html, status_code=404, media_type="text/html")

        try:
            report_obj = adapter.load(path)
        except Exception as exc:
            logger.warning("Failed to load report {}: {}", path.name, exc)
            broken_html = page(
                f"Error — {path.name}",
                report_broken(rid, path.name, str(exc)),
                active_surface=surface,
            )
            return Response(broken_html, status_code=200, media_type="text/html")
        body_html = adapter.body(report_obj)
        name = adapter.name(report_obj)
        html = page(name, f'<section class="report-view">{body_html}</section>', active_surface=surface)
        return NotStr(html)

    # ------------------------------------------------------------------
    # Route: GET /r/{rid}/export  — standalone export
    # ------------------------------------------------------------------
    @app.get("/r/{rid}/export")
    def report_export(rid: str) -> Response:
        path = library.resolve(rid, roots)
        if path is None:
            return Response("404 Not Found", status_code=404, media_type="text/plain")

        surface, _raw = library.load_surface(path)
        adapter = ADAPTERS.get(surface or "")
        if adapter is None:
            return Response("404 Not Found", status_code=404, media_type="text/plain")

        try:
            report_obj = adapter.load(path)
        except Exception as exc:
            logger.warning("Failed to load report for export {}: {}", path.name, exc)
            return Response(
                f"Error loading report {path.name}: {exc}",
                status_code=422,
                media_type="text/plain",
            )
        return Response(adapter.export(report_obj), media_type="text/html")

    return app
