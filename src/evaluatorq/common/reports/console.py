"""Shared console helpers for evaluatorq CLIs.

Provides two reusable primitives consumed by both the simulation and red-team
CLI hooks, avoiding duplicated inline logic in each CLI module.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from rich.console import Console


def render_plan_table(
    console: Console,
    *,
    title: str,
    rows: list[tuple[str, str]],
    parameter_width: int = 18,
) -> None:
    """Render a ROUNDED Parameter/Value table.

    Args:
        console: Rich Console to print to.
        title: Table title shown in the header.
        rows: Sequence of (parameter, value) pairs to add as table rows.
        parameter_width: Minimum width of the Parameter column.
    """
    import rich.box as box
    from rich.table import Table

    table = Table(title=title, show_header=True, header_style='bold', box=box.ROUNDED)
    table.add_column('Parameter', style='white', min_width=parameter_width)
    table.add_column('Value', style='cyan')
    for name, value in rows:
        table.add_row(name, value)
    console.print(table)


def _read_terminal_line(fd: int) -> str | None:
    """Read one line from ``fd`` with ``os.read``; ``None`` on end of input before any text.

    One byte per read, so a second line already waiting (type-ahead, a paste) stays unread.
    """
    data = b''
    while not data.endswith(b'\n'):
        chunk = os.read(fd, 1)
        if not chunk:
            break
        data += chunk
    if not data:
        return None
    return data.decode(sys.stdin.encoding or 'utf-8', errors='replace')


def _write_terminal(text: str) -> None:
    """Write ``text`` to stdout's file descriptor, bypassing the lock ``sys.stdout`` holds while writing."""
    data = text.encode(sys.stdout.encoding or 'utf-8', errors='replace')
    fd = sys.stdout.fileno()
    while data:
        data = data[os.write(fd, data) :]


def _terminal_prompt(prompt: str) -> str:
    return f'{prompt} [Y/n]: '


def _confirm_from_terminal(prompt: str) -> bool:
    """Same answers and errors as ``typer.confirm(prompt, default=True)``, read with ``os.read``.

    The caller shows the first prompt; this repeats it after an invalid answer.

    ``input()`` is not safe on the background reader thread: with ``readline`` loaded (libedit on
    macOS) the library swaps its own process-wide SIGINT handler in and out around the read, and a
    Ctrl-C landing in that window never reaches Python, so the prompt ignores it. Without
    ``readline``, the stdio fallback holds the stdin ``FILE`` lock and blocks interpreter shutdown
    after Ctrl-C. A raw ``os.read`` holds no lock and installs no handler; the terminal's own line
    discipline still provides backspace editing.

    Output goes through `_write_terminal` for the same reason: if Ctrl-C lands while this thread
    is inside a ``sys.stdout`` write, it still holds that stream's lock when the interpreter shuts
    down, and CPython aborts on it.
    """
    import click

    fd = sys.stdin.fileno()
    while True:
        line = _read_terminal_line(fd)
        if line is None:
            # Ctrl-D, as click.confirm treats EOFError; click's handler prints the newline.
            raise click.Abort
        value = line.strip().lower()
        if value in ('y', 'yes', ''):
            return True
        if value in ('n', 'no'):
            return False
        _write_terminal(f'Error: invalid input\n{_terminal_prompt(prompt)}')


async def ask_confirm(prompt: str) -> bool:
    """Ask ``typer.confirm(prompt, default=True)`` without pinning the event loop.

    The read runs on a daemon thread rather than ``asyncio.to_thread``: on Ctrl-C ``asyncio.run``
    cancels the awaiting task and then waits for its default executor, which would block on the
    unfinished stdin read until the user pressed Enter.

    On a POSIX terminal the thread reads with ``os.read`` instead of ``input()`` (see
    `_confirm_from_terminal`), so Ctrl-C always reaches the event loop and shutdown never waits on
    the reader. Call this only from a CLI entry point that exits after the run, not from a
    long-lived process. If the awaiting task is cancelled without interrupting the read, the daemon
    thread remains blocked until a line arrives, and discards it.
    """
    import typer

    read_terminal = os.name == 'posix' and sys.stdin.isatty()

    loop = asyncio.get_running_loop()
    answer: asyncio.Future[bool] = loop.create_future()

    def settle(outcome: Callable[[], None]) -> None:
        # The loop is already closed after an interrupt; nobody awaits the answer then.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(lambda: None if answer.done() else outcome())

    def ask() -> None:
        try:
            result = _confirm_from_terminal(prompt) if read_terminal else typer.confirm(prompt, default=True)
        except BaseException as exc:  # noqa: BLE001 (handed to the awaiting task, e.g. click.Abort on Ctrl-D)
            error = exc
            settle(lambda: answer.set_exception(error))
        else:
            settle(lambda: answer.set_result(result))

    if read_terminal:
        # The reader writes to the raw descriptor; flush first so earlier output stays in order.
        sys.stdout.flush()
    threading.Thread(target=ask, name='evaluatorq-confirm', daemon=True).start()
    if read_terminal:
        # Shown only once start() has returned: on Python 3.10 a Ctrl-C raised inside
        # Thread.start() corrupts its lock ("release unlocked lock") instead of interrupting.
        _write_terminal(_terminal_prompt(prompt))
    # A terminal SIGINT can be delivered to the stdin worker instead of the
    # event-loop thread. Periodic wakeups let the loop process that signal.
    try:
        while not answer.done():
            await asyncio.wait(fs=(answer,), timeout=0.1)
        return answer.result()
    finally:
        answer.cancel()


async def confirm_run_plan(
    console: Console,
    *,
    title: str,
    rows: list[tuple[str, str]],
    prompt: str,
    skip_confirm: bool,
) -> bool:
    """Render a ROUNDED Parameter/Value table then optionally prompt for confirmation.

    Args:
        console: Rich Console to print to.
        title: Table title shown in the header.
        rows: Sequence of (parameter, value) pairs to add as table rows.
        prompt: Confirmation prompt text passed to ``typer.confirm``.
        skip_confirm: When True, return immediately after rendering the table
            (useful for ``--yes`` / ``--no-confirm`` flags).

    Returns:
        ``True`` if ``skip_confirm`` is set, otherwise the answer from `ask_confirm`.
    """
    render_plan_table(console, title=title, rows=rows)
    if skip_confirm:
        return True
    return await ask_confirm(prompt)


def write_text_report(
    directory: Path,
    *,
    stem: str,
    fmt: str,
    content: str,
) -> Path:
    """Create ``directory`` and write ``{stem}.{fmt}`` (UTF-8); echo path to stderr.

    Args:
        directory: Target directory; created (with parents) if absent.
        stem: Filename stem, e.g. ``"sim-report-2025-01-01"``.
        fmt: File extension without dot, e.g. ``"md"`` or ``"html"``.
        content: Text content to write.

    Returns:
        Absolute `Path` of the written file.

    Raises:
        `typer.Exit`: With code 1 if the directory cannot be created.
    """
    import typer

    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        typer.echo(f'Error: cannot create report directory {directory}: {exc}', err=True)
        raise typer.Exit(code=1) from exc

    path = directory / f'{stem}.{fmt}'
    path.write_text(content, encoding='utf-8')
    typer.echo(f'Report written to {path}', err=True)
    return path
