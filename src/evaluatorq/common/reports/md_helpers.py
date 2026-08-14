"""Markdown formatting helpers shared across report renderers.

Used by ``redteam.reports.export_md`` and ``simulation.reports.export_md``
so both report flavors produce consistent tables, progress bars, and
collapsible blocks.
"""

from __future__ import annotations

import html
import textwrap
from types import MappingProxyType

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


# Cost provenance -> the clause it contributes to a coverage label. The single
# vocabulary both `coverage_parts` and the dashboard's `view._coverage` draw
# from — a `resolve_cost_source` value with no entry here (a fourth literal
# added later) renders as an empty clause instead of silently falling into
# whichever branch happened to be `else`, which is how 'mixed' used to get
# mislabelled by a plain if/elif/else before this table existed.
_PROVENANCE_LABELS: MappingProxyType[str, str] = MappingProxyType({
    'provider': '',
    'catalogue': 'estimated',
    'mixed': 'partly estimated',
})


def coverage_parts(priced_calls: int, calls: int, *, estimated_calls: int = 0) -> list[str]:
    """List the qualifier clauses for a priced-call coverage figure.

    The single vocabulary behind `cost_coverage` (markdown/HTML reports) and
    the dashboard's ``view._coverage`` — both used to re-derive these strings
    independently and drifted in clause order; this is the one place they may
    be spelled out. Returns ``[]`` when ``priced_calls <= 0``: reports written
    before coverage was tracked have no coverage data, and claiming "0 of N"
    for them would be a lie in the other direction.

    ``estimated_calls`` defaults to 0, which reads as "every priced call was
    billed by the provider" — the same optimistic default `cost_coverage` and
    `view._coverage` carry, and callers that actually know the provenance must
    pass the real count rather than rely on it.

    The returned list is at most two clauses, in order: the "N of M calls"
    call-coverage clause (only when ``priced_calls < calls``), then the
    provenance clause ("estimated" / "partly estimated", omitted for
    provider-billed). See `cost_coverage` for how they compose into a string.
    """
    if priced_calls <= 0:
        return []
    parts: list[str] = []
    if priced_calls < calls:
        parts.append(f'{priced_calls:,} of {calls:,} calls')
    source = resolve_cost_source(priced_calls, estimated_calls)
    label = _PROVENANCE_LABELS.get(source, '') if source else ''
    if label:
        parts.append(label)
    return parts


def cost_coverage(priced_calls: int, calls: int, *, estimated_calls: int = 0) -> str:
    """Label a cost total with how many calls contributed and, if any, whether
    the total is billed or client-side estimated.

    A thin join over `coverage_parts` — see it for the clause vocabulary and
    the ``estimated_calls`` default's meaning. Returns ``''`` when every call
    was priced by the provider (nothing to qualify) or when ``priced_calls``
    is 0. Combined with call coverage, the returned string is one of:

    - ``''`` — every call priced, all by the provider (nothing to qualify).
    - ``' (3 of 10 calls)'`` — only some calls priced, all by the provider.
    - ``' (estimated)'`` — every call priced, all client-side estimates.
    - ``' (partly estimated)'`` — every call priced, a mix of provider and estimate.
    - ``' (3 of 10 calls, estimated)'`` — only some calls priced, all client-side estimates.
    - ``' (3 of 10 calls, partly estimated)'`` — only some calls priced, a mix of provider and estimate.
    """
    parts = coverage_parts(priced_calls, calls, estimated_calls=estimated_calls)
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
