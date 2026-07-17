"""Shared ``--help`` epilog builder for the evaluatorq CLIs."""

from __future__ import annotations


def examples(*lines: str) -> str:
    """Build a command ``--help`` epilog from example lines.

    Under ``rich_markup_mode='rich'`` the epilog is flowed like HTML — single
    newlines collapse to spaces — so each visual line must be its own paragraph
    (blank line between) to render one-per-row. Lines starting with ``#`` are
    dimmed as comments; command lines render verbatim.
    """
    from rich.markup import escape

    def render(line: str) -> str:
        return f'[dim]{escape(line)}[/]' if line.lstrip().startswith('#') else escape(line)

    return '\n\n'.join(['[bold]Examples[/]', *(render(line) for line in lines)])
