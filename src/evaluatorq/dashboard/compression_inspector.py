"""Local-only Settings inspector for readable traces and their Jev state."""

from __future__ import annotations

import asyncio
import html
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from starlette.requests import Request  # noqa: TC002 — FastHTML inspects this annotation at runtime
from starlette.responses import Response
from starlette.routing import Route

from evaluatorq.dashboard.compression_inspector_data import (
    CompressionPreview,
    InspectorSourceError,
    fetch_orq_document,
    preview_document,
)
from evaluatorq.dashboard.security import csrf_field, request_rejected
from evaluatorq.dashboard.shell import page
from evaluatorq.local_sessions import SessionQuery, SessionRef, load_session_document, search_sessions
from evaluatorq.local_sessions.roots import session_roots
from evaluatorq.trace_finder.settings import effective_settings, load_settings

if TYPE_CHECKING:
    from evaluatorq.local_sessions.models import SessionSearchResult, SessionSource


_BASE = '/compression-inspector'
_LOCAL_SOURCES: tuple[SessionSource, ...] = ('claude-code', 'claude-desktop', 'codex')
_SOURCE_LABELS: dict[SessionSource, str] = {
    'claude-code': 'Claude Code',
    'claude-desktop': 'Claude desktop',
    'codex': 'Codex',
}
_SESSION_SEARCH_ERROR = 'The local-session search could not be completed.'
_ORQ_ERROR = 'The Orq trace could not be loaded. Check the ID, selected dates, and dashboard access.'
_SESSION_ERROR = 'That local session could not be loaded from the configured session folders.'


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _date_bounds(form: Any) -> tuple[datetime | None, datetime | None]:
    start_text = str(form.get('start_date') or '').strip()
    end_text = str(form.get('end_date') or '').strip()
    if not start_text and not end_text:
        return None, None
    if not start_text or not end_text:
        raise ValueError
    start_day = date.fromisoformat(start_text)
    end_day = date.fromisoformat(end_text)
    if start_day > end_day:
        raise ValueError
    return (
        datetime.combine(start_day, time.min, tzinfo=timezone.utc),
        datetime.combine(end_day + timedelta(days=1), time.min, tzinfo=timezone.utc),
    )


def _preview_html(preview: CompressionPreview) -> str:
    state = json.dumps(preview.compressed, ensure_ascii=False, indent=2)
    original_html = (
        f'<pre>{_escape(preview.original)}</pre>'
        if preview.original
        else '<p class="compression-count-note" role="status">No readable input or output is available for this source.</p>'
    )
    return (
        '<section class="compression-preview" aria-labelledby="compression-preview-title">'
        '<div class="compression-preview-heading"><div>'
        '<h2 id="compression-preview-title">Preview</h2>'
        '<p>Original readable input/output with known sensitive values scrubbed; the canonical ATIF source is unchanged.</p>'
        '</div><p class="compression-question-note">No question included.</p></div>'
        '<dl class="compression-metrics">'
        f'<div><dt>Readable source characters</dt><dd>{preview.original_chars:,}</dd></div>'
        f'<div><dt>Source messages</dt><dd>{preview.message_count:,}</dd></div>'
        f'<div><dt>Serialized Jev state characters</dt><dd>{preview.compressed_chars:,}</dd></div>'
        f'<div><dt>Messages omitted from Jev state</dt><dd>{preview.omitted_messages:,}</dd></div>'
        f'<div><dt>Global character cap</dt><dd>{preview.global_cap:,}</dd></div>'
        '</dl>'
        '<div class="compression-panels">'
        '<section class="compression-panel" aria-labelledby="compression-original-title">'
        '<h3 id="compression-original-title">Original readable trace</h3>'
        f'{original_html}'
        '</section>'
        '<section class="compression-panel" aria-labelledby="compression-jev-title">'
        '<h3 id="compression-jev-title">Jev state</h3>'
        f'<pre>{_escape(state)}</pre>'
        '<p class="compression-count-note">The state character count uses compact serialization; this readable display adds formatting whitespace.</p>'
        '</section>'
        '</div></section>'
    )


def _session_results_html(result: SessionSearchResult) -> str:
    if not result.sessions:
        if not result.complete:
            empty = (
                'The search stopped before finding a local session. Its empty result may be incomplete. '
                'Try again or narrow the configured session roots.'
            )
        else:
            roots_exist = any(session_roots(family) for family in ('claude', 'codex'))
            empty = (
                'No Claude Code, Claude desktop, or Codex session folders were found on this computer.'
                if not roots_exist
                else 'No local sessions matched the available source folders.'
            )
        return f'<p class="compression-empty" role="status">{empty}</p>'

    items: list[str] = []
    for session in result.sessions:
        title = session.title or session.first_prompt or 'Untitled session'
        ref = f'{session.source}:{session.path}'
        updated = session.updated_at.astimezone(timezone.utc)
        updated_text = updated.strftime('%Y-%m-%d %H:%M UTC')
        items.append(
            '<li class="compression-session">'
            '<div class="compression-session-description">'
            f'<strong>{_escape(_SOURCE_LABELS[session.source])}</strong>'
            f'<span>{_escape(title)}</span>'
            f'<time datetime="{_escape(updated.isoformat())}">{_escape(updated_text)}</time>'
            '</div>'
            f'<form method="post" action="{_BASE}/sessions/load" autocomplete="off">'
            f'{csrf_field()}<input type="hidden" name="session" value="{_escape(ref)}">'
            '<button type="submit">Preview session</button></form>'
            '</li>'
        )
    note = (
        '<p class="compression-search-note">Search stopped before completion; these are the matches found so far.</p>'
        if not result.complete
        else '<p class="compression-search-note">Showing at most 50 matches. Session files remain on this computer.</p>'
    )
    return (
        f'<p class="compression-result-count" role="status">{len(result.sessions)} local sessions</p>'
        f'{note}<ul class="compression-session-list">'
        f'{"".join(items)}</ul>'
    )


def _body_html(
    *,
    notice: str | None = None,
    preview: CompressionPreview | None = None,
    sessions: SessionSearchResult | None = None,
) -> str:
    notice_html = f'<p class="compression-notice" role="alert">{_escape(notice)}</p>' if notice else ''
    sessions_html = _session_results_html(sessions) if sessions is not None else ''
    preview_html = _preview_html(preview) if preview is not None else ''
    return (
        '<div class="dash-wrap compression-inspector">'
        '<header class="compression-intro">'
        '<h2>Inspect trace compression</h2>'
        '<p>Compare the readable conversation with the structured Jev state '
        'without sending a question or calling a model.</p>'
        '</header>'
        f'{notice_html}'
        '<div class="compression-sources">'
        '<section class="compression-source" aria-labelledby="compression-orq-title">'
        '<h3 id="compression-orq-title">Orq trace ID</h3>'
        '<p>Search the most recent 7 days by default. Set both dates to find an older trace; '
        'the end date includes that full day.</p>'
        f'<form method="post" action="{_BASE}/source/orq" autocomplete="off">'
        f'{csrf_field()}'
        '<label for="compression-trace-id">Trace ID</label>'
        '<input id="compression-trace-id" name="trace_id" type="text" '
        'maxlength="512" autocomplete="off" required>'
        '<div class="compression-date-fields">'
        '<div><label for="compression-start-date">Start date (optional)</label>'
        '<input id="compression-start-date" name="start_date" type="date"></div>'
        '<div><label for="compression-end-date">End date (optional)</label>'
        '<input id="compression-end-date" name="end_date" type="date"></div>'
        '</div>'
        '<button type="submit">Load Orq trace</button>'
        '</form></section>'
        '<section class="compression-source" aria-labelledby="compression-local-title">'
        '<h3 id="compression-local-title">Local sessions</h3>'
        '<p>Search up to 50 sessions from Claude Code, Claude desktop, and Codex. '
        'The selected session is validated under its configured local session roots before reading.</p>'
        f'<form method="post" action="{_BASE}/sessions/search" autocomplete="off">'
        f'{csrf_field()}<button type="submit">Search local sessions</button></form>'
        f'{sessions_html}</section>'
        '</div>'
        '<section class="compression-limits" aria-labelledby="compression-limits-title">'
        '<h3 id="compression-limits-title">Character limits</h3>'
        f'<p>Global cap: <strong>{effective_settings().trace_input_chars:,}</strong> '
        'characters (500,000 by default). '
        'Jev state is capped at 112,000 characters; question-aware ceilings are '
        '128,000 or 256,000 characters. '
        'This preview includes no question, so its state count is not fitted to a question payload.</p>'
        '</section>'
        f'{preview_html}'
        '</div>'
    )


def _page_response(
    *,
    notice: str | None = None,
    preview: CompressionPreview | None = None,
    sessions: SessionSearchResult | None = None,
    status_code: int = 200,
) -> Response:
    body = page(
        title='Compression inspector',
        body_html=_body_html(notice=notice, preview=preview, sessions=sessions),
        active_nav='compression-inspector',
    )
    return Response(
        content=body,
        status_code=status_code,
        media_type='text/html',
        headers={'Cache-Control': 'no-store'},
    )


def _disabled() -> Response:
    return Response(content='Not found', status_code=404, media_type='text/plain')


def _enabled() -> bool:
    return load_settings().compression_inspector_enabled


def compression_inspector_page(_req: Request) -> Response:
    if not _enabled():
        return _disabled()
    return _page_response()


async def compression_inspector_orq(req: Request) -> Response:
    if not _enabled():
        return _disabled()
    form = await req.form()
    rejected = request_rejected(req, form)
    if rejected:
        return _page_response(notice=rejected, status_code=403)
    trace_id = str(form.get('trace_id') or '').strip()
    if not trace_id or len(trace_id) > 512:
        return _page_response(notice='Enter a valid Orq trace ID.', status_code=422)
    try:
        start, end = _date_bounds(form)
    except (OverflowError, TypeError, ValueError):
        return _page_response(
            notice='Enter both valid dates, with the start date on or before the end date.',
            status_code=422,
        )
    try:
        document = await fetch_orq_document(req.app, trace_id, start=start, end=end)
        if document is None:
            return _page_response(notice='No Orq trace matched that ID in the selected date range.')
        cap = effective_settings().trace_input_chars
        preview = preview_document(document, global_char_cap=cap)
    except InspectorSourceError:
        return _page_response(notice=_ORQ_ERROR)
    except Exception:  # noqa: BLE001 — never expose provider or trace data in source failures
        return _page_response(notice=_ORQ_ERROR)
    return _page_response(preview=preview)


async def compression_inspector_sessions_search(req: Request) -> Response:
    if not _enabled():
        return _disabled()
    form = await req.form()
    rejected = request_rejected(req, form)
    if rejected:
        return _page_response(notice=rejected, status_code=403)
    query = SessionQuery(sources=_LOCAL_SOURCES, limit=50)
    try:
        result = await asyncio.to_thread(search_sessions, query=query)
    except Exception:  # noqa: BLE001 — session-reader errors can expose local paths
        return _page_response(notice=_SESSION_SEARCH_ERROR)
    return _page_response(sessions=result)


async def compression_inspector_sessions_load(req: Request) -> Response:
    if not _enabled():
        return _disabled()
    form = await req.form()
    rejected = request_rejected(req, form)
    if rejected:
        return _page_response(notice=rejected, status_code=403)
    raw_ref = str(form.get('session') or '')
    if len(raw_ref) > 4096:
        return _page_response(notice=_SESSION_ERROR, status_code=422)
    source, separator, raw_path = raw_ref.partition(':')
    if not separator or source not in _LOCAL_SOURCES or not raw_path:
        return _page_response(notice=_SESSION_ERROR, status_code=422)
    try:
        ref = SessionRef(source=source, path=Path(raw_path))
        document = await asyncio.to_thread(load_session_document, ref=ref)
        cap = effective_settings().trace_input_chars
        preview = preview_document(document, global_char_cap=cap)
    except Exception:  # noqa: BLE001 — loader errors can contain local paths or transcript data
        return _page_response(notice=_SESSION_ERROR)
    return _page_response(preview=preview)


def register_compression_inspector_routes(app: Any) -> None:
    """Register the private, persisted-setting-gated compression inspector routes."""
    app.add_route(Route(_BASE, endpoint=compression_inspector_page, methods=['GET']))
    app.add_route(
        Route(
            f'{_BASE}/source/orq',
            endpoint=compression_inspector_orq,
            methods=['POST'],
        )
    )
    app.add_route(
        Route(
            f'{_BASE}/sessions/search',
            endpoint=compression_inspector_sessions_search,
            methods=['POST'],
        )
    )
    app.add_route(
        Route(
            f'{_BASE}/sessions/load',
            endpoint=compression_inspector_sessions_load,
            methods=['POST'],
        )
    )
