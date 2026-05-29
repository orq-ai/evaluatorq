"""Markdown formatting helpers shared across report renderers.

Used by ``redteam.reports.export_md`` and ``simulation.reports.export_md``
so both report flavors produce consistent tables, progress bars, and
collapsible blocks.
"""

from __future__ import annotations

import textwrap


def pct(rate: float) -> str:
    """Format a float rate as a percentage string, e.g. ``0.75`` -> ``'75%'``."""
    return f"{rate:.0%}"


def bar(rate: float, width: int = 10) -> str:
    """Render a Unicode block-character progress bar with a numeric percentage.

    Uses U+2588 (full block) for filled segments and U+2591 (light shade) for
    empty segments. Always ``width`` characters wide, followed by the numeric
    percentage. Example: ``'████░░░░░░ 40%'``.
    """
    filled = round(rate * width)
    return "█" * filled + "░" * (width - filled) + f" {rate:.0%}"


def bold_bar(rate: float, threshold: float = 0.5) -> str:
    """Return a Unicode bar, bolded when rate exceeds ``threshold``."""
    cell = bar(rate)
    return f"**{cell}**" if rate > threshold else cell


def md_table(
    headers: list[str],
    rows: list[list[str]],
    right_align: set[int] | None = None,
) -> str:
    """Render a Markdown table from headers and string rows.

    Args:
        headers: Column header labels.
        rows: Table data rows; each element is a list of cell values.
        right_align: Optional set of zero-based column indices that should be
            right-aligned (rendered with ``---:`` separator).
    """
    right_align = right_align or set()
    lines: list[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    separators = ["---:" if i in right_align else "---" for i in range(len(headers))]
    lines.append("| " + " | ".join(separators) + " |")
    for row in rows:
        sanitized = [str(cell).replace("|", "\\|").replace("\n", " ") for cell in row]
        lines.append("| " + " | ".join(sanitized) + " |")
    return "\n".join(lines)


def center_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown table with all columns center-aligned."""
    sep = " | ".join(":---:" for _ in headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + sep + " |",
    ]
    for row in rows:
        sanitized = [str(c).replace("|", "\\|").replace("\n", " ") for c in row]
        lines.append("| " + " | ".join(sanitized) + " |")
    return "\n".join(lines)


def truncate(text: str, max_chars: int = 800) -> str:
    """Truncate long text with an ellipsis indicator."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n*[truncated — full text in report JSON]*"


def details_block(summary: str, body: str) -> str:
    """Wrap content in a collapsible ``<details>`` block."""
    inner = textwrap.indent(body.strip(), "  ")
    return f"<details>\n<summary>{summary}</summary>\n\n{inner}\n\n</details>"
