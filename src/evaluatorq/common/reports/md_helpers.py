"""Markdown formatting helpers shared across report renderers.

Used by ``redteam.reports.export_md`` and ``simulation.reports.export_md``
so both report flavors produce consistent tables, progress bars, and
collapsible blocks.
"""

from __future__ import annotations

import html
import textwrap

from evaluatorq.common.reports.html_helpers import pct as _html_pct

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


def cost_coverage(priced_calls: int, calls: int, *, noun: str = 'calls') -> str:
    """Label a cost total that only some calls contributed to, e.g. ``' (3 of 10 calls)'``.

    Returns ``''`` when every call was priced, or when ``priced_calls`` is 0 —
    reports written before this was tracked have no coverage data, and claiming
    "0 of N" for them would be a lie in the other direction. *noun* names what
    was counted when the items are not calls, e.g. ``'conversations'``.
    """
    if 0 < priced_calls < calls:
        return f' ({priced_calls:,} of {calls:,} {noun})'
    return ''


def coverage_qualifier(*, priced_calls: int, calls: int, reported: int, items: int, noun: str) -> str:
    """Both coverage gaps of a summed cost in one label, e.g. ``'(1 of 2 calls, 3 of 4 attacks)'``.

    *priced_calls* of the *calls* had a cost; *reported* of the *items* recorded
    any usage at all. Either gap makes the sum a lower bound. ``''`` when there is
    no gap, under the same rules as `cost_coverage`. Every report and dashboard
    surface words a target cost through this, so they cannot drift apart.
    """
    labels = [
        f'{shown:,} of {total:,} {what}'
        for shown, total, what in ((priced_calls, calls, 'calls'), (reported, items, noun))
        if cost_coverage(shown, total)
    ]
    return f'({", ".join(labels)})' if labels else ''


def fmt_cost_coverage(
    cost: float | None, *, priced_calls: int, calls: int, reported: int, items: int, noun: str
) -> str:
    """A summed cost with its coverage label, e.g. ``'$0.0040 (1 of 2 calls, 3 of 4 attacks)'``.

    ``'unknown'`` when *cost* is ``None``. See `coverage_qualifier` for the label.
    """
    if cost is None:
        return 'unknown'
    qualifier = coverage_qualifier(priced_calls=priced_calls, calls=calls, reported=reported, items=items, noun=noun)
    return f'{fmt_cost(cost)} {qualifier}' if qualifier else fmt_cost(cost)


CACHED_TOKENS_LABEL = 'Cached Input Tokens'
"""Row/card label for `fmt_cached_tokens`, shared so the two surfaces cannot drift apart."""


def fmt_cached_tokens(cached_tokens: float, input_tokens: float) -> str:
    """Format cached input tokens with their share of input, e.g. ``'12,345 (68% of input tokens)'``.

    The share is the number that answers "did caching work" — a raw count says
    nothing without the denominator it was cached against.

    Says **of input tokens**, never bare "of input", because this row renders
    two rows above `Total Cost`: a cached token bills at roughly a tenth of a
    fresh one, so 68% of input cached is nowhere near 68% of spend, and the
    provider does not break out a cached-vs-uncached cost split for us to show
    instead. Omits the share entirely when ``input_tokens`` is 0 — the
    denominator is missing, not zero, and dividing would raise.
    """
    if input_tokens <= 0:
        return f'{cached_tokens:,.0f}'
    return f'{cached_tokens:,.0f} ({pct(cached_tokens / input_tokens)} of input tokens)'


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
