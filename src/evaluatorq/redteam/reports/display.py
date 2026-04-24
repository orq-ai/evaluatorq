"""Rich terminal display for red team report summaries."""

from __future__ import annotations

import operator

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from evaluatorq.redteam.contracts import OWASP_CATEGORY_NAMES, RedTeamReport
from evaluatorq.redteam.vulnerability_registry import VULNERABILITY_DEFS, Vulnerability


def _format_vulnerability_label(vuln_str: str) -> str:
    """Format 'goal_hijacking' -> 'Goal Hijacking (ASI01)'."""
    try:
        vuln = Vulnerability(vuln_str)
    except ValueError:
        return vuln_str
    vdef = VULNERABILITY_DEFS.get(vuln)
    if vdef:
        cats = ', '.join(
            code for codes in vdef.framework_mappings.values() for code in codes
        )
        return f"{vdef.name} ({cats})" if cats else vdef.name
    return vuln_str


def _format_category_label(category: str) -> str:
    """Format ``'ASI01'`` → ``'ASI01 - Goal Hijacking'``."""
    name = OWASP_CATEGORY_NAMES.get(category)
    if name:
        return f"{category} - {name}"
    return category


def _rate_style(rate: float) -> str:
    """Return a Rich color style for a resistance/coverage rate."""
    if rate >= 0.8:
        return "green"
    if rate >= 0.5:
        return "yellow"
    return "red"


def _asr_style(asr: float) -> str:
    """Return a Rich color style for an ASR value (lower is better)."""
    if asr <= 0.2:
        return "green"
    if asr <= 0.5:
        return "yellow"
    return "red"


def print_report_summary(report: RedTeamReport, *, console: Console | None = None) -> None:
    """Print a Rich summary of a :class:`RedTeamReport` to the terminal.

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
    console.print("[bold underline white]RED TEAM REPORT SUMMARY[/bold underline white]")
    console.print()

    # ── Summary stats table ────────────────────────────────────────────
    stats = Table(show_header=True, header_style="bold", box=box.ROUNDED)
    stats.add_column("Metric", style="white", width=22)
    stats.add_column("Value", width=15)

    stats.add_row("Total Attacks", Text(str(summary.total_attacks), style="cyan"))
    stats.add_row("Evaluated", Text(str(summary.evaluated_attacks), style="cyan"))
    stats.add_row(
        "Vulnerabilities",
        Text(str(summary.vulnerabilities_found), style="red" if summary.vulnerabilities_found else "green"),
    )
    stats.add_row(
        "ASR",
        Text(f"{summary.vulnerability_rate:.0%}", style=_asr_style(summary.vulnerability_rate)),
    )
    stats.add_row(
        "Eval Coverage",
        Text(f"{summary.evaluation_coverage:.0%}", style=_rate_style(summary.evaluation_coverage)),
    )
    if report.duration_seconds is not None:
        mins, secs = divmod(int(report.duration_seconds), 60)
        stats.add_row("Duration", Text(f"{mins}m {secs}s", style="cyan"))
    if summary.total_errors:
        stats.add_row("Errors", Text(str(summary.total_errors), style="red"))

    # Datapoint breakdown (hybrid runs)
    breakdown = summary.datapoint_breakdown
    if breakdown:
        parts = []
        if breakdown.get("static", 0):
            parts.append(f"{breakdown['static']} static")
        if breakdown.get("template_dynamic", 0):
            parts.append(f"{breakdown['template_dynamic']} template")
        if breakdown.get("generated_dynamic", 0):
            parts.append(f"{breakdown['generated_dynamic']} generated")
        if parts:
            stats.add_row("Breakdown", Text(", ".join(parts), style="cyan"))

    console.print(stats)
    console.print()

    # ── Per-vulnerability breakdown (primary) ─────────────────────────
    if summary.by_vulnerability:
        # Show worst-first (lowest resistance first, most attacks as tiebreaker)
        sorted_vulns = sorted(
            summary.by_vulnerability.values(),
            key=lambda v: (v.resistance_rate, -v.total_attacks),
        )

        vuln_table = Table(show_header=True, header_style="bold", box=box.ROUNDED)
        vuln_table.add_column("Vulnerability", style="white", min_width=35)
        vuln_table.add_column("Domain", style="white", min_width=18)
        vuln_table.add_column("Tested", justify="right", width=8)
        vuln_table.add_column("Passed", justify="right", width=8)
        vuln_table.add_column("ASR", justify="right", width=11)

        for vuln_summary in sorted_vulns:
            passed_count = vuln_summary.total_attacks - vuln_summary.vulnerabilities_found
            asr = 1 - vuln_summary.resistance_rate
            vuln_table.add_row(
                _format_vulnerability_label(vuln_summary.vulnerability),
                Text(vuln_summary.domain.replace("_", " ").title(), style="white"),
                Text(str(vuln_summary.total_attacks), style="cyan"),
                Text(
                    str(passed_count),
                    style="green" if passed_count == vuln_summary.total_attacks else "yellow",
                ),
                Text(
                    f"{asr:.0%}",
                    style=_asr_style(asr),
                ),
            )

        console.print("[bold white]Per-Vulnerability Breakdown:[/bold white]")
        console.print(vuln_table)
        console.print()

    # ── Per-category breakdown (secondary) ────────────────────────────
    if summary.by_category:
        # Show worst-first (lowest resistance first, most attacks as tiebreaker)
        sorted_cats = sorted(
            summary.by_category.values(),
            key=lambda c: (c.resistance_rate, -c.total_attacks),
        )

        cat_table = Table(show_header=True, header_style="bold", box=box.ROUNDED)
        cat_table.add_column("Category", style="white", min_width=30)
        cat_table.add_column("Attacks", justify="right", width=9)
        cat_table.add_column("Vulnerable", justify="right", width=11)
        cat_table.add_column("ASR", justify="right", width=11)

        for cat_summary in sorted_cats:
            asr = 1 - cat_summary.resistance_rate
            cat_table.add_row(
                _format_category_label(cat_summary.category),
                Text(str(cat_summary.total_attacks), style="cyan"),
                Text(
                    str(cat_summary.vulnerabilities_found),
                    style="red" if cat_summary.vulnerabilities_found else "green",
                ),
                Text(
                    f"{asr:.0%}",
                    style=_asr_style(asr),
                ),
            )

        console.print("[bold white]Per-Category Breakdown:[/bold white]")
        console.print(cat_table)
        console.print()

    # ── Top vulnerable techniques ──────────────────────────────────────
    if summary.by_technique:
        top_techniques = sorted(
            summary.by_technique.items(),
            key=lambda t: t[1].vulnerabilities_found,
            reverse=True,
        )[:5]
        tech_table = Table(show_header=True, header_style="bold", box=box.ROUNDED)
        tech_table.add_column("Technique", style="white", min_width=25)
        tech_table.add_column("Vulnerabilities", justify="right", width=16)

        for technique, tech_summary in top_techniques:
            tech_table.add_row(technique, Text(str(tech_summary.vulnerabilities_found), style="red"))

        console.print("[bold white]Top Vulnerable Techniques:[/bold white]")
        console.print(tech_table)
        console.print()

    # ── Top error causes ───────────────────────────────────────────────
    if summary.errors_by_type:
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

        err_table = Table(show_header=True, header_style="bold", box=box.ROUNDED)
        err_table.add_column("Error", style="white", min_width=16)
        err_table.add_column("Stage", style="cyan", min_width=10)
        err_table.add_column("Count", justify="right", width=6)
        err_table.add_column("Example", style="dim", max_width=70)

        for error_code, count in top_errors:
            err_table.add_row(
                error_code,
                error_stages.get(error_code, ''),
                Text(str(count), style="red"),
                error_samples.get(error_code, ''),
            )

        console.print("[bold white]Top Error Causes:[/bold white]")
        console.print(err_table)
        console.print()
