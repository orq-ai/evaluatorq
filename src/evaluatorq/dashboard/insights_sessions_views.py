"""The results partial of the Insights "Local sessions" picker."""

from __future__ import annotations

from datetime import timezone
from typing import TYPE_CHECKING

from evaluatorq.common.reports import esc
from evaluatorq.insights.population import format_size
from evaluatorq.local_sessions import SEARCH_DEADLINE_SECONDS

if TYPE_CHECKING:
    from evaluatorq.local_sessions import SessionSearchResult, SessionSummary

_NO_MATCH = 'No local sessions match these filters.'
_NO_ROOTS = 'No Claude Code, Claude desktop, Codex or omp session folders were found on this computer.'


def encode_ref(summary: SessionSummary) -> str:
    """The form value that names one session: `{source}:{path}`, split on the first colon."""
    return f'{summary.source}:{summary.path}'


def _row(summary: SessionSummary, selected: frozenset[str]) -> str:
    ref = encode_ref(summary)
    utc = summary.updated_at.astimezone(timezone.utc)
    stamp = utc.strftime('%Y-%m-%dT%H:%M:%S')
    title = summary.title or summary.first_prompt or '(untitled)'
    return (
        f'<tr><td><input type="checkbox" name="session" value="{esc(ref)}"'
        f'{" checked" if ref in selected else ""} aria-label="Select session {esc(title)}"></td>'
        f'<td><time data-utc="{stamp}">{utc.strftime("%Y-%m-%d %H:%M")} UTC</time></td>'
        f'<td>{esc(summary.source)}</td>'
        f'<td class="irf-sessions-project">{esc(summary.project_dir)}</td>'
        f'<td class="irf-sessions-title">{esc(title)}</td>'
        f'<td>{esc(format_size(summary.size_bytes))}</td></tr>'
    )


def render_session_results(result: SessionSearchResult, *, selected: frozenset[str], roots_found: bool) -> str:
    """The search results as a selectable table, with an empty state and a note when the scan was cut short."""
    note = (
        ''
        if result.complete
        else (
            f'<p class="irf-hint" role="status">Searched {result.scanned_files} of {result.candidate_files} '
            f'session files before the {SEARCH_DEADLINE_SECONDS:g}-second limit. Narrow the dates or text to see the rest.</p>'
        )
    )
    if not result.sessions:
        message = _NO_MATCH if roots_found else _NO_ROOTS
        return f'<p class="irf-hint" role="status">{esc(message)}</p>{note}'
    refs = {encode_ref(summary) for summary in result.sessions}
    chosen = len(refs & selected)
    rows = ''.join(_row(summary, selected) for summary in result.sessions)
    count = len(result.sessions)
    return (
        f'<p class="irf-hint" role="status"><span data-sessions-count>{count} session{"" if count == 1 else "s"}</span> · '
        f'<span data-sessions-selected>{chosen} selected</span></p>{note}'
        '<div class="irf-sessions-scroll"><table class="irf-sessions-table"><thead><tr>'
        '<th><input type="checkbox" data-sessions-all aria-label="Select all sessions"></th>'
        '<th>Updated</th><th>Source</th><th>Project</th><th>Title</th><th>Size</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )
