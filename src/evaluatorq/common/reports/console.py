"""Shared console helpers for evaluatorq CLIs.

Provides two reusable primitives consumed by both the simulation and red-team
CLI hooks, avoiding duplicated inline logic in each CLI module.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from rich.console import Console


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
        ``True`` if ``skip_confirm`` is set, otherwise the result of
        ``typer.confirm(prompt, default=True)``.
    """
    import rich.box as box
    from rich.table import Table

    table = Table(title=title, show_header=True, header_style="bold", box=box.ROUNDED)
    table.add_column("Parameter", style="white", min_width=18)
    table.add_column("Value", style="cyan")
    for name, value in rows:
        table.add_row(name, value)
    console.print(table)

    if skip_confirm:
        return True

    import typer

    # Blocking stdin read; offload so event loop is not pinned.
    return await asyncio.to_thread(typer.confirm, prompt, default=True)


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
        Absolute :class:`~pathlib.Path` of the written file.

    Raises:
        :class:`typer.Exit`: With code 1 if the directory cannot be created.
    """
    import typer

    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        typer.echo(f"Error: cannot create report directory {directory}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    path = directory / f"{stem}.{fmt}"
    path.write_text(content, encoding="utf-8")
    typer.echo(f"Report written to {path}", err=True)
    return path
