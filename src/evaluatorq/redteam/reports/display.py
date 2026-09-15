"""Rich terminal display for red team report summaries."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from evaluatorq.common.reports import rate_style
from evaluatorq.common.reports.html_helpers import pct
from evaluatorq.redteam.contracts import OWASP_CATEGORY_NAMES, RedTeamReport, ReportSummary
from evaluatorq.redteam.vulnerability_registry import VULNERABILITY_DEFS, Vulnerability

if TYPE_CHECKING:
    from collections.abc import Iterator


def _format_vulnerability_label(vuln_str: str) -> str:
    """Format 'goal_hijacking' -> 'Goal Hijacking (ASI01)'."""
    try:
        vuln = Vulnerability(vuln_str)
    except ValueError:
        return vuln_str
    vdef = VULNERABILITY_DEFS.get(vuln)
    if vdef:
        cats = ', '.join(code for codes in vdef.framework_mappings.values() for code in codes)
        return f'{vdef.name} ({cats})' if cats else vdef.name
    return vuln_str


def _format_category_label(category: str) -> str:
    """Format ``'ASI01'`` → ``'ASI01 - Goal Hijacking'``."""
    name = OWASP_CATEGORY_NAMES.get(category)
    if name:
        return f'{category} - {name}'
    return category


def _print_section(console: Console, title: str, rows: Table | None) -> None:
    """Print a populated report section."""
    if rows is None:
        return
    console.print(f'[bold white]{title}:[/bold white]')
    console.print(rows)
    console.print()


def _build_stats_table(report: RedTeamReport, summary: ReportSummary) -> Table:
    """Build the high-level summary stats table."""
    stats = Table(show_header=True, header_style='bold', box=box.ROUNDED)
    stats.add_column('Metric', style='white', width=22)
    stats.add_column('Value', width=15)

    stats.add_row('Total Attacks', Text(str(summary.total_attacks), style='cyan'))
    stats.add_row('Evaluated', Text(str(summary.evaluated_attacks), style='cyan'))
    stats.add_row(
        'Vulnerabilities',
        Text(str(summary.vulnerabilities_found), style='red' if summary.vulnerabilities_found else 'green'),
    )
    asr_style = (
        'dim' if summary.vulnerability_rate is None else rate_style(summary.vulnerability_rate, higher_is_better=False)
    )
    stats.add_row('ASR', Text(pct(summary.vulnerability_rate), style=asr_style))
    stats.add_row(
        'Eval Coverage',
        Text(f'{summary.evaluation_coverage:.0%}', style=rate_style(summary.evaluation_coverage)),
    )
    if report.duration_seconds is not None:
        mins, secs = divmod(int(report.duration_seconds), 60)
        stats.add_row('Duration', Text(f'{mins}m {secs}s', style='cyan'))
    if summary.total_errors:
        stats.add_row('Errors', Text(str(summary.total_errors), style='red'))

    # Panel-of-judges reliability (RES-739) — only present for multi-judge runs.
    rel = summary.jury_reliability
    if rel is not None:
        if rel.krippendorff_alpha is None:
            value = Text(f'n/a ({rel.samples} samples)', style='dim')
        else:
            value = Text(
                f'alpha={rel.krippendorff_alpha:.2f} ({rel.samples} samples)',
                style=rate_style(rel.krippendorff_alpha),
            )
        stats.add_row('Jury Agreement', value)

    # Datapoint breakdown (hybrid runs)
    breakdown = summary.datapoint_breakdown
    if breakdown:
        parts = []
        if breakdown.get('static', 0):
            parts.append(f'{breakdown["static"]} static')
        if breakdown.get('template_dynamic', 0):
            parts.append(f'{breakdown["template_dynamic"]} template')
        if breakdown.get('generated_dynamic', 0):
            parts.append(f'{breakdown["generated_dynamic"]} generated')
        if parts:
            stats.add_row('Breakdown', Text(', '.join(parts), style='cyan'))

    return stats


def _build_vulnerability_table(summary: ReportSummary) -> Table | None:
    """Build the per-vulnerability breakdown table when data is available."""
    if not summary.by_vulnerability:
        return None

    # Show worst-first (lowest resistance first, most attacks as tiebreaker).
    # None (unevaluated) sorts last — it's neither best nor worst.
    sorted_vulns = sorted(
        summary.by_vulnerability.values(),
        key=lambda v: (v.resistance_rate is None, v.resistance_rate or 0.0, -v.total_attacks),
    )

    vuln_table = Table(show_header=True, header_style='bold', box=box.ROUNDED)
    vuln_table.add_column('Vulnerability', style='white', min_width=35)
    vuln_table.add_column('Domain', style='white', min_width=18)
    vuln_table.add_column('Tested', justify='right', width=12)
    vuln_table.add_column('Passed', justify='right', width=8)
    vuln_table.add_column('ASR', justify='right', width=11)

    for vuln_summary in sorted_vulns:
        # Passed is counted over *evaluated* attacks, never over tested ones: an
        # attack with no verdict did not resist, and "total - found" would print
        # a fully unevaluated vulnerability as having passed everything, in green.
        evaluated = vuln_summary.evaluated_attacks
        passed_count = evaluated - vuln_summary.vulnerabilities_found
        asr = None if vuln_summary.resistance_rate is None else 1 - vuln_summary.resistance_rate
        tested_label = (
            str(vuln_summary.total_attacks)
            if evaluated == vuln_summary.total_attacks
            else f'{evaluated} of {vuln_summary.total_attacks}'
        )
        vuln_table.add_row(
            _format_vulnerability_label(vuln_summary.vulnerability),
            Text(vuln_summary.domain.replace('_', ' ').title(), style='white'),
            Text(tested_label, style='cyan' if evaluated == vuln_summary.total_attacks else 'yellow'),
            Text(
                str(passed_count),
                style='green' if evaluated and passed_count == vuln_summary.total_attacks else 'yellow',
            ),
            Text(
                pct(asr),
                style='dim' if asr is None else rate_style(asr, higher_is_better=False),
            ),
        )

    return vuln_table


def _build_category_table(summary: ReportSummary) -> Table | None:
    """Build the per-category breakdown table when data is available."""
    if not summary.by_category:
        return None

    # Show worst-first (lowest resistance first, most attacks as tiebreaker).
    # None (unevaluated) sorts last — it's neither best nor worst.
    sorted_cats = sorted(
        summary.by_category.values(),
        key=lambda c: (c.resistance_rate is None, c.resistance_rate or 0.0, -c.total_attacks),
    )

    cat_table = Table(show_header=True, header_style='bold', box=box.ROUNDED)
    cat_table.add_column('Category', style='white', min_width=30)
    cat_table.add_column('Attacks', justify='right', width=9)
    cat_table.add_column('Vulnerable', justify='right', width=11)
    cat_table.add_column('ASR', justify='right', width=11)

    for cat_summary in sorted_cats:
        asr = None if cat_summary.resistance_rate is None else 1 - cat_summary.resistance_rate
        cat_table.add_row(
            _format_category_label(cat_summary.category),
            Text(str(cat_summary.total_attacks), style='cyan'),
            Text(
                str(cat_summary.vulnerabilities_found),
                style='red' if cat_summary.vulnerabilities_found else 'green',
            ),
            Text(
                pct(asr),
                style='dim' if asr is None else rate_style(asr, higher_is_better=False),
            ),
        )

    return cat_table


def _build_technique_table(summary: ReportSummary) -> Table | None:
    """Build the top vulnerable techniques table when data is available."""
    if not summary.by_technique:
        return None

    top_techniques = sorted(
        summary.by_technique.items(),
        key=lambda t: t[1].vulnerabilities_found,
        reverse=True,
    )[:5]
    tech_table = Table(show_header=True, header_style='bold', box=box.ROUNDED)
    tech_table.add_column('Technique', style='white', min_width=25)
    tech_table.add_column('Vulnerabilities', justify='right', width=16)

    for technique, tech_summary in top_techniques:
        tech_table.add_row(technique, Text(str(tech_summary.vulnerabilities_found), style='red'))

    return tech_table


def _build_error_table(report: RedTeamReport, summary: ReportSummary) -> Table | None:
    """Build the top error causes table when data is available."""
    if not summary.errors_by_type:
        return None

    top_errors = sorted(summary.errors_by_type.items(), key=operator.itemgetter(1), reverse=True)[:5]

    # Collect a sample error message and stage per error code
    error_samples: dict[str, str] = {}
    error_stages: dict[str, str] = {}
    for r in report.results:
        if r.error:
            etype = r.error_code or r.error_type or 'unknown'
            if etype not in error_samples:
                msg = r.error[:100] + '...' if len(r.error) > 100 else r.error
                error_samples[etype] = msg
                error_stages[etype] = r.error_stage or ''

    err_table = Table(show_header=True, header_style='bold', box=box.ROUNDED)
    err_table.add_column('Error', style='white', min_width=16)
    err_table.add_column('Stage', style='cyan', min_width=10)
    err_table.add_column('Count', justify='right', width=6)
    err_table.add_column('Example', style='dim', max_width=70)

    for error_code, count in top_errors:
        err_table.add_row(
            error_code,
            error_stages.get(error_code, ''),
            Text(str(count), style='red'),
            error_samples.get(error_code, ''),
        )

    return err_table


def _iter_sections(report: RedTeamReport, summary: ReportSummary) -> Iterator[tuple[str, Table | None]]:
    """Build report sections one at a time in display order.

    Yields:
        Each section's title and optional table.
    """
    yield 'Per-Vulnerability Breakdown', _build_vulnerability_table(summary)
    yield 'Per-Category Breakdown', _build_category_table(summary)
    yield 'Top Vulnerable Techniques', _build_technique_table(summary)
    yield 'Top Error Causes', _build_error_table(report, summary)


def print_report_summary(report: RedTeamReport, *, console: Console | None = None) -> None:
    """Print a Rich summary of a `RedTeamReport` to the terminal.

    Displays:

    * High-level stats (total attacks, vulnerabilities, resistance rate, …)
    * Per-category breakdown sorted by vulnerability rate (worst first)
    * Top vulnerable techniques (if any)
    * Top error causes (if any)
    """
    console = console or Console()
    summary = report.summary

    # ── Title ──────────────────────────────────────────────────────────
    console.print()
    console.print('[bold underline white]RED TEAM REPORT SUMMARY[/bold underline white]')
    console.print()

    if report.executive_summary:
        from rich.panel import Panel

        console.print(Panel(report.executive_summary, title='Executive Summary', border_style='cyan'))
        console.print()

    # ── Summary stats table ────────────────────────────────────────────
    stats = _build_stats_table(report, summary)

    console.print(stats)
    console.print()

    for title, rows in _iter_sections(report, summary):
        _print_section(console, title, rows)
