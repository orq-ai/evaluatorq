"""Shared console helpers for evaluatorq CLIs.

Provides two reusable primitives consumed by both the simulation and red-team
CLI hooks, avoiding duplicated inline logic in each CLI module.
"""

from __future__ import annotations

import asyncio
import contextlib
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


async def ask_confirm(prompt: str) -> bool:
    """Ask ``typer.confirm(prompt, default=True)`` without pinning the event loop.

    The read runs on a daemon thread rather than ``asyncio.to_thread``: on Ctrl-C ``asyncio.run``
    cancels the awaiting task and then waits for its default executor, which would block on the
    unfinished stdin read until the user pressed Enter.

    A blocking stdin read cannot be interrupted, so after cancellation the thread stays blocked
    until a line arrives or the process exits, and that line is discarded. Call it only from
    a CLI entry point that exits after the run, not from a long-lived process.
    """
    import typer

    loop = asyncio.get_running_loop()
    answer: asyncio.Future[bool] = loop.create_future()

    def settle(outcome: Callable[[], None]) -> None:
        # The loop is already closed after an interrupt; nobody awaits the answer then.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(lambda: None if answer.done() else outcome())

    def ask() -> None:
        try:
            result = typer.confirm(prompt, default=True)
        except BaseException as exc:  # noqa: BLE001 (handed to the awaiting task, e.g. click.Abort on Ctrl-D)
            error = exc
            settle(lambda: answer.set_exception(error))
        else:
            settle(lambda: answer.set_result(result))

    threading.Thread(target=ask, name='evaluatorq-confirm', daemon=True).start()
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
