"""Render two red_team reports side-by-side with rich."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    os.environ.setdefault("COLUMNS", str(os.get_terminal_size().columns))
except OSError:
    os.environ.setdefault("COLUMNS", "220")

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

    def _summary(rows: list[dict[str, Any]]) -> tuple[int, int, float]:
        total = len(rows)
        vuln = sum(1 for r in rows if r["verdict"] == "VULNERABLE")
        rate = (total - vuln) / total if total else 0.0
        return vuln, total, rate

    jarvis_vuln, jarvis_total, jarvis_rate = _summary(jarvis_rows)
    hal_vuln, hal_total, hal_rate = _summary(hal_rows)

    console.print(
        f"\n[bold]JARVIS[/bold]: "
        f"{jarvis_vuln}/{jarvis_total} attacks succeeded · "
        f"resistance {jarvis_rate:.0%}"
    )
    console.print(
        f"[bold]HAL[/bold]:    "
        f"{hal_vuln}/{hal_total} attacks succeeded · "
        f"resistance {hal_rate:.0%}\n"
    )

    table = Table(title="Attack-by-attack comparison", show_lines=True)
    table.add_column("Vulnerability", style="bold")
    table.add_column("Attack preview")
    table.add_column("JARVIS", justify="center")
    table.add_column("HAL", justify="center")

    if len(jarvis_rows) != len(hal_rows):
        console.print(
            f"[yellow]WARN: row count mismatch — JARVIS {len(jarvis_rows)}, HAL {len(hal_rows)}. "
            "Extra rows truncated.[/yellow]"
        )
    for jarvis_row, hal_row in zip(jarvis_rows, hal_rows):
        if jarvis_row["vulnerability"] != hal_row["vulnerability"]:
            console.print(
                f"[yellow]WARN: row order mismatch — "
                f"JARVIS '{jarvis_row['vulnerability']}' vs HAL '{hal_row['vulnerability']}'. "
                "Comparison may be unreliable.[/yellow]"
            )
        jarvis_verdict = (
            f"[red]{jarvis_row['verdict']}[/red]"
            if "VULN" in jarvis_row["verdict"].upper()
            else f"[green]{jarvis_row['verdict']}[/green]"
        )
        hal_verdict = (
            f"[red]{hal_row['verdict']}[/red]"
            if "VULN" in hal_row["verdict"].upper()
            else f"[green]{hal_row['verdict']}[/green]"
        )
        table.add_row(jarvis_row["vulnerability"], jarvis_row["preview"], jarvis_verdict, hal_verdict)

    console.print(table)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <hal.json> <jarvis.json>", file=sys.stderr)
        sys.exit(2)
    render_side_by_side(sys.argv[1], sys.argv[2])
