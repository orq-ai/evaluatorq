"""Rich terminal renderer for agent-simulation results.

Renders the same ``build_report_sections`` data layer that drives
Markdown/HTML exports — no re-aggregation happens here.  Output goes to the
``Console`` passed by the caller; the CLI passes a stderr Console so progress
lines land in ``2>log`` and the summary stays on screen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from evaluatorq.common.reports import rate_style
from evaluatorq.simulation.reports.sections import build_report_sections

if TYPE_CHECKING:
    from rich.console import Console

    from evaluatorq.contracts import ReportSection
    from evaluatorq.simulation.types import SimulationResult


def print_simulation_summary(
    results: list[SimulationResult],
    *,
    console: Console | None = None,
) -> None:
    """Print a Rich multi-section summary of simulation results to *console*.

    Parameters
    ----------
    results:
        The list of :class:`~evaluatorq.simulation.types.SimulationResult`
        objects produced by a simulation run.
    console:
        Rich :class:`~rich.console.Console` to write to.  Defaults to a
        stderr console so the summary does not interfere with piped stdout.
    """
    from rich import box
    from rich.console import Console as RichConsole
    from rich.table import Table
    from rich.text import Text

    console = console or RichConsole(stderr=True)
    total = len(results)

    console.print()
    console.print(f"[bold underline]SIMULATION SUMMARY[/bold underline] ({total} simulations)")

    if total == 0:
        console.print("[dim]No results (run aborted or produced nothing).[/dim]")
        return

    sections = {s.kind: s for s in build_report_sections(results)}

    # ── Summary stats ──────────────────────────────────────────────────
    summ = sections["summary"].data
    rate = summ["success_rate"]

    stats = Table(show_header=True, header_style="bold", box=box.ROUNDED)
    stats.add_column("Metric", style="white", width=28)
    stats.add_column("Value", width=16)

    stats.add_row(
        "Conversations",
        Text(str(summ["total_conversations"]), style="cyan"),
    )
    stats.add_row(
        "Goals Achieved",
        Text(
            f"{summ['goals_achieved']}/{summ['total_conversations']}",
            style="green" if rate >= 0.8 else ("yellow" if rate >= 0.5 else "red"),
        ),
    )
    stats.add_row(
        "Success Rate",
        Text(f"{rate:.0%}", style=rate_style(rate)),
    )
    stats.add_row(
        "Avg Completion Score",
        Text(f"{summ['avg_goal_completion_score']:.2f}", style=rate_style(summ["avg_goal_completion_score"])),
    )
    stats.add_row(
        "Avg Turn Count",
        Text(f"{summ['avg_turn_count']:.1f}", style="cyan"),
    )
    stats.add_row(
        "Total Tokens",
        Text(f"{summ['total_tokens']:,}", style="cyan"),
    )
    if summ.get("errors"):
        stats.add_row("Errors", Text(str(summ["errors"]), style="red"))

    console.print(stats)
    console.print()

    # ── Per-persona breakdown ──────────────────────────────────────────
    _breakdown(console, sections["persona_breakdown"], key="persona", title="Per-Persona Breakdown")
    console.print()

    # ── Per-scenario breakdown ─────────────────────────────────────────
    _breakdown(console, sections["scenario_breakdown"], key="scenario", title="Per-Scenario Breakdown")
    console.print()

    # ── Judge verdicts ─────────────────────────────────────────────────
    jv = sections["judge_verdicts"].data
    if jv.get("terminated_by"):
        jt = Table(title="Judge Verdicts", show_header=True, header_style="bold", box=box.ROUNDED)
        jt.add_column("Terminated By")
        jt.add_column("Count", justify="right")
        for reason, count in jv["terminated_by"].items():
            jt.add_row(reason, str(count))
        console.print(jt)
        console.print()

    # ── Errors (optional) ─────────────────────────────────────────────
    err_section = sections.get("errors")
    if err_section is not None:
        err = err_section.data
        et = Table(title="Errors", show_header=True, header_style="bold", box=box.ROUNDED)
        et.add_column("Message")
        et.add_column("Count", justify="right", style="red")
        for msg, count in err["by_message"].items():
            display_msg = (msg[:80] + "…") if len(msg) > 80 else msg
            et.add_row(display_msg, str(count))
        console.print(et)
        console.print()


def _breakdown(console: Console, section: ReportSection, *, key: str, title: str) -> None:
    """Render a per-persona or per-scenario breakdown table."""
    from rich import box
    from rich.table import Table
    from rich.text import Text

    rows = section.data["rows"]
    if not rows:
        return

    t = Table(title=title, show_header=True, header_style="bold", box=box.ROUNDED)
    t.add_column(key.title())
    t.add_column("Runs", justify="right")
    t.add_column("Achieved", justify="right")
    t.add_column("Rate", justify="right")

    for row in rows:  # sections.py already sorts worst-first
        r = row["success_rate"]
        t.add_row(
            str(row[key]),
            str(row["conversations"]),
            str(row["goals_achieved"]),
            Text(f"{r:.0%}", style=rate_style(r)),
        )
    console.print(t)
