"""Render two red_team reports side-by-side with rich."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    results = report.get("results") or report.get("datapoints") or []
    out: list[dict[str, Any]] = []
    for r in results:
        vuln = r.get("attack", {}).get("vulnerability") or r.get("vulnerability") or "unknown"
        messages = r.get("messages", [])
        last_user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        preview = (last_user or "")[:60]
        ev = r.get("evaluation") or {}
        vulnerable = r.get("vulnerable")
        verdict = (
            "VULNERABLE"
            if vulnerable is True
            else "RESISTANT"
            if vulnerable is False
            else (ev.get("explanation", "?")[:20] if ev else "?")
        )
        out.append({"vulnerability": vuln, "preview": preview, "verdict": verdict})
    return out


def render_side_by_side(hal_path: str, jarvis_path: str) -> None:
    console = Console()
    hal_report = _load(hal_path)
    jarvis_report = _load(jarvis_path)
    hal_rows = _rows(hal_report)
    jarvis_rows = _rows(jarvis_report)

    hal_summary = hal_report.get("summary", {})
    jarvis_summary = jarvis_report.get("summary", {})

    console.print(
        f"\n[bold]HAL[/bold]:    "
        f"{hal_summary.get('vulnerabilities_found', '?')}/{hal_summary.get('total_attacks', '?')} attacks succeeded · "
        f"resistance {hal_summary.get('resistance_rate', 0.0):.0%}"
    )
    console.print(
        f"[bold]JARVIS[/bold]: "
        f"{jarvis_summary.get('vulnerabilities_found', '?')}/{jarvis_summary.get('total_attacks', '?')} attacks succeeded · "
        f"resistance {jarvis_summary.get('resistance_rate', 0.0):.0%}\n"
    )

    table = Table(title="Attack-by-attack comparison", show_lines=True)
    table.add_column("Vulnerability", style="bold")
    table.add_column("Attack preview")
    table.add_column("HAL", justify="center")
    table.add_column("JARVIS", justify="center")

    if len(hal_rows) != len(jarvis_rows):
        console.print(
            f"[yellow]WARN: row count mismatch — HAL {len(hal_rows)}, JARVIS {len(jarvis_rows)}. "
            "Extra rows truncated.[/yellow]"
        )
    for hal_row, jarvis_row in zip(hal_rows, jarvis_rows):
        hal_verdict = (
            f"[red]{hal_row['verdict']}[/red]"
            if "VULN" in hal_row["verdict"].upper()
            else f"[green]{hal_row['verdict']}[/green]"
        )
        jarvis_verdict = (
            f"[red]{jarvis_row['verdict']}[/red]"
            if "VULN" in jarvis_row["verdict"].upper()
            else f"[green]{jarvis_row['verdict']}[/green]"
        )
        table.add_row(hal_row["vulnerability"], hal_row["preview"], hal_verdict, jarvis_verdict)

    console.print(table)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <hal.json> <jarvis.json>", file=sys.stderr)
        sys.exit(2)
    render_side_by_side(sys.argv[1], sys.argv[2])
