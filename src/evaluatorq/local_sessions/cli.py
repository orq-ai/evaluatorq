"""`eq sessions`: search local coding-agent sessions and export them as an Insights snapshot."""

from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta
from pathlib import Path  # noqa: TC003 - typer resolves annotations at runtime
from typing import Annotated, get_args

import typer
from rich.console import Console
from rich.table import Table

from evaluatorq.common.cli_errors import emit_error
from evaluatorq.common.cli_json import echo_json
from evaluatorq.local_sessions.models import (
    MAX_SELECTED_SESSIONS,
    SEARCH_DEADLINE_SECONDS,
    SESSION_SOURCES,
    SessionLoadError,
    SessionQuery,
    SessionSource,
    SessionSummary,
)
from evaluatorq.local_sessions.search import build_session_snapshot, search_sessions

_TITLE_CHARS = 60
_BAD_USAGE = 2


def _fail(message: str) -> typer.Exit:
    emit_error(message)
    return typer.Exit(code=_BAD_USAGE)


def _day(value: str, *, flag: str, offset_days: int) -> datetime:
    try:
        day = date.fromisoformat(value)
    except ValueError:
        raise _fail(f'{flag} must be a date like 2026-10-01, got {value!r}') from None
    return datetime.combine(day + timedelta(days=offset_days), time.min).astimezone()


def _sources(values: list[str] | None) -> tuple[SessionSource, ...]:
    if not values:
        return SESSION_SOURCES
    known = get_args(SessionSource)
    for value in values:
        if value not in known:
            raise _fail(f'unknown --source {value!r}; choose from {", ".join(known)}')
    return tuple(source for source in SESSION_SOURCES if source in values)


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
        snapshot, failed = build_session_snapshot([session.ref() for session in sessions])
    except SessionLoadError as exc:
        emit_error(exc)
        raise typer.Exit(code=1) from None
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        handle.write(snapshot.model_dump_json())
    target.chmod(0o600)
    typer.echo(f'Wrote {len(snapshot.documents)} sessions to {target}', err=True)
    for _ref, reason in failed:
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
        int, typer.Option('--limit', min=1, max=MAX_SELECTED_SESSIONS, help='Maximum sessions to list.')
    ] = 50,
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
    """Search Claude Code, Claude desktop, Codex and omp sessions on this machine. Reads local files only."""
    sources = _sources(source)
    start_at = _day(start, flag='--from', offset_days=0) if start else None
    end_at = _day(end, flag='--to', offset_days=1) if end else None
    if start_at is not None and end_at is not None and start_at >= end_at:
        raise _fail('--from must not be after --to')
    query = SessionQuery(
        sources=sources,
        start=start_at,
        end=end_at,
        project_dir=str(project_dir) if project_dir is not None else None,
        text=text or None,
        limit=limit,
    )
    result = search_sessions(query)

    if json_output:
        echo_json([session.model_dump(mode='json') for session in result.sessions])
    else:
        _print_table(result.sessions)
    if not result.complete:
        typer.echo(
            f'Searched {result.scanned_files} of {result.candidate_files} session files before the '
            f'{SEARCH_DEADLINE_SECONDS:g}s limit; narrow --from/--to or --text to see the rest.',
            err=True,
        )
        if export is not None:
            typer.echo('The export is partial: it holds only the sessions found before the search stopped.', err=True)
    if len(result.sessions) == query.limit:
        more = (
            f'raise --limit (up to {MAX_SELECTED_SESSIONS}) to see more.'
            if query.limit < MAX_SELECTED_SESSIONS
            else 'narrow --from/--to or --text to see more.'
        )
        typer.echo(f'Listed the newest {query.limit} sessions; {more}', err=True)
    if export is not None:
        _export(result.sessions, export)
