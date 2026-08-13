"""Markdown formatting helpers shared across report renderers.

Used by ``redteam.reports.export_md`` and ``simulation.reports.export_md``
so both report flavors produce consistent tables, progress bars, and
collapsible blocks.
"""

from __future__ import annotations

import html
import textwrap

from evaluatorq.common.reports.html_helpers import pct as _html_pct
from evaluatorq.contracts import resolve_cost_source

# Deliberately the same object as the HTML helper, not a parallel implementation:
# both are re-exported from evaluatorq.common.reports under the name ``pct``, so a
# second definition here silently shadows one of them depending on import order —
# which is how a float-only pct once defeated the None-safe one package-wide.
pct = _html_pct


def fmt_cost(cost: float | None) -> str:
    """Format a USD cost value, e.g. ``0.0032`` -> ``'$0.0032'``.

    The single cost formatter for every surface — reports and dashboard alike —
    so the same value never renders as ``$0.0032`` in one place and ``$0.00`` in
    another. Fixed at 4dp because per-call costs are routinely sub-cent, and
    rounding those to cents rounds them to nothing.

    ``None`` means the provider did not report a cost — distinct from a real
    ``0.0`` — and renders as an em dash. Callers that would rather omit the row
    entirely should check for ``None`` themselves rather than render the dash.
    """
    if cost is None:
        return '—'
    return f'${cost:,.4f}'


def cost_coverage(priced_calls: int, calls: int, *, estimated_calls: int = 0) -> str:
    """Label a cost total with how many calls contributed and, if any, whether
    the total is billed or client-side estimated.

    Returns ``''`` when every call was priced by the provider (nothing to qualify)
    or when ``priced_calls`` is 0 — reports written before coverage was tracked have
    no coverage data, and claiming "0 of N" for them would be a lie in the other
    direction.

    ``estimated_calls`` counts how many of ``priced_calls`` were priced client-side
    from ``common.model_catalogue`` rather than billed by the provider — the same
    provenance ``Usage.cost_source`` derives from. Combined with call coverage, the
    returned string is one of:

    - ``''`` — every call priced, all by the provider (nothing to qualify).
    - ``' (3 of 10 calls)'`` — only some calls priced, all by the provider.
    - ``' (estimated)'`` — every call priced, all client-side estimates.
    - ``' (partly estimated)'`` — every call priced, a mix of provider and estimate.
    - ``' (3 of 10 calls, estimated)'`` — only some calls priced, all client-side estimates.
    - ``' (3 of 10 calls, partly estimated)'`` — only some calls priced, a mix of provider and estimate.
    """
    if priced_calls <= 0:
        return ''
    coverage = f'{priced_calls:,} of {calls:,} calls' if priced_calls < calls else ''
    source = resolve_cost_source(priced_calls, estimated_calls)
    if source in (None, 'provider'):
        provenance = ''
    elif source == 'catalogue':
        provenance = 'estimated'
    else:
        provenance = 'partly estimated'
    parts = [part for part in (coverage, provenance) if part]
    if not parts:
        return ''
    return f' ({", ".join(parts)})'


def bar(rate: float | None, width: int = 10) -> str:
    """Render a Unicode block-character progress bar with a numeric percentage.

    Uses U+2588 (full block) for filled segments and U+2591 (light shade) for
    empty segments. Always ``width`` characters wide, followed by the numeric
    percentage. Example: ``'████░░░░░░ 40%'``. An unknown rate renders as an empty
    bar labelled ``n/a`` — there is no bar length that honestly means "not measured".
    """
    if rate is None:
        return '░' * width + f' {pct(rate)}'
    filled = round(rate * width)
    return '█' * filled + '░' * (width - filled) + f' {pct(rate)}'


def bold_bar(rate: float | None, threshold: float = 0.5) -> str:
    """Return a Unicode bar, bolded when rate exceeds ``threshold``."""
    cell = bar(rate)
    return f'**{cell}**' if rate is not None and rate > threshold else cell


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
    lines.append('| ' + ' | '.join(headers) + ' |')
    separators = ['---:' if i in right_align else '---' for i in range(len(headers))]
    lines.append('| ' + ' | '.join(separators) + ' |')
    for row in rows:
        sanitized = [str(cell).replace('|', '\\|').replace('\n', ' ') for cell in row]
        lines.append('| ' + ' | '.join(sanitized) + ' |')
    return '\n'.join(lines)


def center_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown table with all columns center-aligned."""
    sep = ' | '.join(':---:' for _ in headers)
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + sep + ' |',
    ]
    for row in rows:
        sanitized = [str(c).replace('|', '\\|').replace('\n', ' ') for c in row]
        lines.append('| ' + ' | '.join(sanitized) + ' |')
    return '\n'.join(lines)


def truncate(text: str, max_chars: int = 800) -> str:
    """Truncate long text with an ellipsis indicator."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + '\n\n*[truncated — full text in report JSON]*'


def details_block(summary: str, body: str) -> str:
    """Wrap content in a collapsible ``<details>`` block.

    The ``summary`` argument is HTML-escaped so titles containing ``&``,
    ``<``, ``>`` render correctly in GitHub-Flavored Markdown.
    """
    inner = textwrap.indent(body.strip(), '  ')
    escaped_summary = html.escape(summary)
    return f'<details>\n<summary>{escaped_summary}</summary>\n\n{inner}\n\n</details>'
