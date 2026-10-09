"""`eq agent-sessions`: search local coding-agent sessions and export them as an Insights snapshot."""

from __future__ import annotations

from datetime import date
from pathlib import Path  # noqa: TC003 — Typer resolves option annotations at runtime
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from evaluatorq.common.cli_errors import emit_error
from evaluatorq.common.cli_json import echo_json
from evaluatorq.common.private_files import write_private_atomic
from evaluatorq.local_sessions.models import (
    MAX_SELECTED_SESSIONS,
    MAX_SESSION_TEXT_CHARS,
    SEARCH_DEADLINE_SECONDS,
    SESSION_SOURCES,
    SessionLoadError,
    SessionQuery,
    SessionSearchResult,
    SessionSource,
    SessionSummary,
    day_window,
)
from evaluatorq.local_sessions.search import build_session_snapshot, search_sessions

_TITLE_CHARS = 60
_BAD_USAGE = 2


def _fail(message: str) -> typer.Exit:
    emit_error(message)
    return typer.Exit(code=_BAD_USAGE)


def _day(value: str, *, flag: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise _fail(f'{flag} must be a date like 2026-10-01, got {value!r}') from None


def _sources(values: list[str] | None) -> tuple[SessionSource, ...]:
    if not values:
        return SESSION_SOURCES
    for value in values:
        if value not in SESSION_SOURCES:
            raise _fail(f'unknown --source {value!r}; choose from {", ".join(SESSION_SOURCES)}')
    return tuple(source for source in SESSION_SOURCES if source in values)


def _query(
    sources: tuple[SessionSource, ...],
    *,
    start: str | None,
    end: str | None,
    project_dir: Path | None,
    text: str | None,
    limit: int,
) -> SessionQuery:
    """Build the shared local-session query, including date and text validation."""
    start_day = _day(start, flag='--from') if start else None
    end_day = _day(end, flag='--to') if end else None
    try:
        start_at, end_at = day_window(start_day, end_day, None)
    except ValueError as exc:
        raise _fail(str(exc)) from None
    if text is not None and len(text) > MAX_SESSION_TEXT_CHARS:
        raise _fail(f'--text is limited to {MAX_SESSION_TEXT_CHARS} characters, got {len(text)}')
    return SessionQuery(
        sources=sources,
        start=start_at,
        end=end_at,
        project_dir=str(project_dir) if project_dir is not None else None,
        text=text or None,
        limit=limit,
    )


def _search_notices(result: SessionSearchResult, query: SessionQuery) -> None:
    if not result.complete:
        typer.echo(
            f'Searched {result.scanned_files} of {result.candidate_files} session files before the '
            f'{SEARCH_DEADLINE_SECONDS:g}s limit; narrow --from/--to or --text to see the rest.',
            err=True,
        )
    if len(result.sessions) == query.limit:
        more = (
            f'raise --limit (up to {MAX_SELECTED_SESSIONS}) to see more.'
            if query.limit < MAX_SELECTED_SESSIONS
            else 'narrow --from/--to or --text to see more.'
        )
        typer.echo(f'Listed the newest {query.limit} sessions; {more}', err=True)


def search_session_options(
    sources: list[str] | None,
    *,
    start: str | None,
    end: str | None,
    project_dir: Path | None,
    text: str | None,
    limit: int,
) -> SessionSearchResult:
    """Search sources using the same option parsing and notices as `eq agent-sessions`."""
    query = _query(_sources(sources), start=start, end=end, project_dir=project_dir, text=text, limit=limit)
    result = search_sessions(query)
    _search_notices(result, query)
    return result


def _print_table(sessions: tuple[SessionSummary, ...]) -> None:
    console = Console()
    if not sessions:
        console.print('No local sessions match.')
        return
    table = Table(title='Local sessions')
    for column in ('Updated', 'Source', 'Project', 'Title'):
        table.add_column(column)
    table.add_column('Path', overflow='fold')
    for session in sessions:
        title = (session.title or session.first_prompt).replace('\n', ' ')[:_TITLE_CHARS]
        table.add_row(
            session.updated_at.astimezone().strftime('%Y-%m-%d %H:%M'),
            session.source,
            session.project_dir,
            title,
            str(session.path),
        )
    console.print(table)


def _export(sessions: tuple[SessionSummary, ...], target: Path) -> None:
    if not sessions:
        raise _fail('no sessions to export')
    try:
        frozen = build_session_snapshot([session.ref() for session in sessions])
    except SessionLoadError as exc:
        emit_error(exc)
        raise typer.Exit(code=1) from None
    try:
        write_private_atomic(path=target, contents=frozen.data, private_directory=False)
    except OSError as exc:
        emit_error(f'could not write {target}: {exc.strerror or type(exc).__name__}')
        raise typer.Exit(code=_BAD_USAGE) from None
    count = frozen.n_sessions
    typer.echo(f'Wrote {count} session{"" if count == 1 else "s"} to {target}', err=True)
    for _ref, reason in frozen.failed:
        typer.echo(f'Skipped {reason}', err=True)


def sessions_cmd(
    source: Annotated[
        list[str] | None,
        typer.Option('--source', help='claude-code, claude-desktop, codex or omp; repeatable. Defaults to all.'),
    ] = None,
    start: Annotated[str | None, typer.Option('--from', help='First day to include, YYYY-MM-DD, local time.')] = None,
    end: Annotated[str | None, typer.Option('--to', help='Last day to include, YYYY-MM-DD, local time.')] = None,
    project_dir: Annotated[
        Path | None,
        typer.Option(
            '--project-dir', help='Only sessions run in this directory, below it, or in a git worktree of it.'
        ),
    ] = None,
    text: Annotated[
        str | None,
        typer.Option('--text', help='Only sessions whose messages contain this text (case-insensitive).'),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            '--limit',
            min=1,
            max=MAX_SELECTED_SESSIONS,
            help=f'Maximum sessions to list. Default 50, or {MAX_SELECTED_SESSIONS} with --export.',
        ),
    ] = None,
    json_output: Annotated[  # noqa: FBT002
        bool, typer.Option('--json', help='Emit sessions as a JSON array on stdout (machine-readable).')
    ] = False,
    export: Annotated[
        Path | None,
        typer.Option(
            '--export',
            help='Write the listed sessions as an Insights snapshot JSON for `eq insights --from-snapshot`.',
        ),
    ] = None,
) -> None:
    """`eq agent-sessions`: search local session files. Reads on this machine only."""
    result = search_session_options(
        source,
        start=start,
        end=end,
        project_dir=project_dir,
        text=text,
        limit=limit if limit is not None else (MAX_SELECTED_SESSIONS if export is not None else 50),
    )

    if json_output:
        echo_json([session.model_dump(mode='json') for session in result.sessions])
    else:
        _print_table(result.sessions)
    if export is not None and not result.complete:
        typer.echo('The export is partial: it holds only the sessions found before the search stopped.', err=True)
    if export is not None:
        _export(result.sessions, export)
