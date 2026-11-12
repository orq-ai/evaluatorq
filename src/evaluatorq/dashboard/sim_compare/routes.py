"""Run loading, error pages and the compare routes."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, NamedTuple

from loguru import logger
from starlette.requests import Request  # noqa: TC002 — FastHTML inspects at runtime
from starlette.responses import Response

from evaluatorq.common.reports import esc
from evaluatorq.dashboard import library
from evaluatorq.dashboard.orq_workspace import cli_slug_render_scope
from evaluatorq.dashboard.shell import page
from evaluatorq.dashboard.sim_compare.drawer import render_compare_transcript
from evaluatorq.dashboard.sim_compare.model import (
    Comparison,
    RunStamp,
    _side,
    cached_comparison,
)
from evaluatorq.dashboard.sim_compare.panels import _filter_strip, _headlines, _swap, render_compare_page
from evaluatorq.dashboard.sim_compare.state import canonical_url, parse_compare_params
from evaluatorq.dashboard.sim_compare.table import render_conversations, resolve_view
from evaluatorq.dashboard.ui_components import stylesheet_asset
from evaluatorq.dashboard.view import report_back_link

if TYPE_CHECKING:
    from pathlib import Path

    from evaluatorq.simulation.contracts import SimulationRun

# ---------------------------------------------------------------------------
# Loading (distinguishes not-found from load-failure, unlike sim_views._load_run)
# ---------------------------------------------------------------------------


def resolve_run(rid: str, roots: list[Path] | None, surface: str) -> tuple[str, Any | None]:
    """Load a run of the given *surface* ('sim' or 'pairwise') by report id.

    Returns ``('ok', run)``, ``('missing', None)`` when the id resolves to no
    file or a valid-but-non-sim report, or ``('error', None)`` when a matching
    sim file fails to load/parse — including a *syntactically* corrupt file, so it
    is reported as corrupt (422) rather than "not found" (404). The error case is
    logged with a stack so a corrupt report is debuggable.
    """
    status, run, _stamp = _resolve_stamped(rid, roots, surface)
    return status, run


def _resolve_stamped(rid: str, roots: list[Path] | None, surface: str) -> tuple[str, Any | None, RunStamp | None]:
    """``resolve_run`` plus the file's stamp, read before the load so a rewrite in between can only cause a miss."""
    # Local: surfaces -> pairwise_tabs -> pairwise_views imports this module.
    from evaluatorq.dashboard.surfaces import ADAPTERS

    path = library.resolve(rid, roots)
    if path is None:
        return 'missing', None, None
    try:
        stamp = (str(path.resolve()), path.stat().st_mtime_ns)
        # Strict read: a corrupt/unreadable file must surface as an error, not be masked
        # as an unknown surface (which load_surface does for lenient directory scans).
        found, _raw = library.load_surface_strict(path)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        logger.opt(exception=True).warning('compare: corrupt/unreadable {} report {}', surface, rid)
        return 'error', None, None
    if found != surface:
        return 'missing', None, None
    adapter = ADAPTERS.get(surface)
    if adapter is None:
        return 'missing', None, None
    try:
        return 'ok', adapter.load(path), stamp
    except (ValueError, OSError):  # JSON decode and pydantic validation errors are ValueErrors
        logger.opt(exception=True).warning('compare: failed to load {} report {}', surface, rid)
        return 'error', None, None


def error_page(message: str, status: int, surface: str) -> Response:
    """Full-page compare error for *surface* ('sim' or 'pairwise')."""
    body = (
        '<header class="report-hero"><h2 class="report-hero-title">Run comparison</h2></header>'
        f'<p class="sim-empty">{esc(message)}</p>'
    )
    return Response(
        page('Compare', body, active_surface=surface, back_html=report_back_link(surface)),
        status_code=status,
        media_type='text/html',
    )


def _fragment_error(message: str, status: int) -> Response:
    return Response(f'<p class="sim-empty">{esc(message)}</p>', status_code=status, media_type='text/html')


class _LoadedPair(NamedTuple):
    run_a: SimulationRun
    run_b: SimulationRun
    stamps: tuple[RunStamp, RunStamp]


def _load_pair(rid_a: str, rid_b: str, roots: list[Path] | None) -> _LoadedPair | tuple[str, int]:
    """Both runs, or ``(message, status)`` naming which one failed and why."""
    if not rid_a or not rid_b:
        return 'Pick two runs to compare.', 400
    if rid_a == rid_b:
        return 'Pick two different runs to compare.', 400
    status_a, run_a, stamp_a = _resolve_stamped(rid_a, roots, 'sim')
    status_b, run_b, stamp_b = _resolve_stamped(rid_b, roots, 'sim')
    if 'error' in (status_a, status_b):
        which = 'A' if status_a == 'error' else 'B'
        return f'Run {which} could not be loaded (the report may be corrupt).', 422
    if run_a is None or run_b is None:
        which = 'A' if run_a is None else 'B'
        return f'Run {which} was not found or is not a simulation run.', 404
    return _LoadedPair(run_a, run_b, (stamp_a, stamp_b))


def _comparison(rid_a: str, rid_b: str, loaded: _LoadedPair, selections: dict[str, list[str]]) -> Comparison:
    return cached_comparison(loaded.stamps, rid_a, rid_b, loaded.run_a, loaded.run_b, selections)


def _push(response: Response, url: str) -> Response:
    """Tell HTMX to put *url* in the address bar for this swap."""
    response.headers['HX-Push-Url'] = url
    return response


def _filter_response(roots: list[Path] | None, rid_a: str, rid_b: str, items: list[tuple[str, str]]) -> Response:
    """Blocking half of the filter route (disk loads and rendering), run off the event loop."""
    loaded = _load_pair(rid_a, rid_b, roots)
    if not isinstance(loaded, _LoadedPair):
        message, status = loaded
        return _fragment_error(str(message), int(status))
    selections, state = parse_compare_params(items)
    cmp = _comparison(rid_a, rid_b, loaded, selections)
    return _push(Response(_swap(cmp, selections, state), media_type='text/html'), canonical_url(cmp, selections, state))


# ---------------------------------------------------------------------------
# Route factory
# ---------------------------------------------------------------------------


def register_sim_compare_routes(app: Any, roots: list[Path] | None = None) -> None:
    """Register the compare routes on *app* (called from ``build_app``)."""

    def _ids(req: Request) -> tuple[str, str]:
        return req.query_params.get('a') or '', req.query_params.get('b') or ''

    def _items(req: Request) -> list[tuple[str, str]]:
        return list(req.query_params.multi_items())

    @app.get('/compare/sim')
    def compare_sim(req: Request) -> Response:
        rid_a, rid_b = _ids(req)
        loaded = _load_pair(rid_a, rid_b, roots)
        if not isinstance(loaded, _LoadedPair):
            message, status = loaded
            return error_page(str(message), int(status), 'sim')
        selections, state = parse_compare_params(_items(req))
        body = render_compare_page(_comparison(rid_a, rid_b, loaded, selections), selections, state)
        return Response(
            page(
                'Compare',
                body,
                active_surface='sim',
                back_html=report_back_link('sim'),
                head_html=stylesheet_asset(filename='review-kit.css'),
            ),
            media_type='text/html',
        )

    @app.post('/compare/sim/filter')
    async def compare_sim_filter(req: Request) -> Response:
        rid_a, rid_b = _ids(req)
        form = await req.form()
        items = [(key, str(value)) for key, value in form.multi_items()]
        return await asyncio.to_thread(_filter_response, roots, rid_a, rid_b, items)

    @app.get('/compare/sim/filter')
    async def compare_sim_filter_link(req: Request) -> Response:
        """The filter-bar chips: the same swap as the rail, with the selection in the query string."""
        rid_a, rid_b = _ids(req)
        return await asyncio.to_thread(_filter_response, roots, rid_a, rid_b, _items(req))

    @app.get('/compare/sim/conversations')
    def compare_sim_conversations(req: Request) -> Response:
        rid_a, rid_b = _ids(req)
        loaded = _load_pair(rid_a, rid_b, roots)
        if not isinstance(loaded, _LoadedPair):
            message, status = loaded
            return _fragment_error(str(message), int(status))
        selections, state = parse_compare_params(_items(req))
        cmp = _comparison(rid_a, rid_b, loaded, selections)
        oob = _headlines(cmp, resolve_view(cmp, state), state, oob=True) + _filter_strip(cmp, state, oob=True)
        return _push(
            Response(render_conversations(cmp, state) + oob, media_type='text/html'),
            canonical_url(cmp, selections, state),
        )

    @app.get('/compare/sim/transcript')
    def compare_sim_transcript(req: Request) -> Response:
        rid_a, rid_b = _ids(req)
        try:
            ia = int(req.query_params.get('ia', ''))
            ib = int(req.query_params.get('ib', ''))
        except (TypeError, ValueError):
            return _fragment_error('Invalid conversation index.', 400)
        loaded = _load_pair(rid_a, rid_b, roots)
        if not isinstance(loaded, _LoadedPair):
            message, status = loaded
            return _fragment_error(str(message), int(status))
        # Row links carry original indices, so resolve against the full, unfiltered matching.
        full = _comparison(rid_a, rid_b, loaded, {}).full
        pair = next((p for p in full.matched if p.a.index == ia and p.b.index == ib), None)
        if pair is None:
            return _fragment_error('Not a matched pair.', 404)
        side_a = _side(rid_a, loaded.run_a, [pair.a])
        side_b = _side(rid_b, loaded.run_b, [pair.b])
        with cli_slug_render_scope():
            body = render_compare_transcript(pair.a, pair.b, side_a, side_b)
        return Response(body, media_type='text/html')
