"""Smoke tests for redteam.reports rendering (RES-846).

The redteam report renderers were refactored to consume shared helpers
from evaluatorq.common.reports. These smoke tests catch regressions
during that refactor and any future common/reports changes that drop a
helper the redteam renderers depend on.
"""

from __future__ import annotations

from datetime import datetime, timezone

from evaluatorq.redteam.contracts import Pipeline, RedTeamReport, ReportSummary
from evaluatorq.redteam.reports import export_html, export_markdown


def _make_empty_report() -> RedTeamReport:
    """Smallest valid RedTeamReport — no attacks run yet."""
    return RedTeamReport(
        created_at=datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc),
        description="smoke",
        pipeline=Pipeline.DYNAMIC,
        framework=None,
        categories_tested=["ASI01"],
        tested_agents=["agent:test"],
        total_results=0,
        results=[],
        summary=ReportSummary(),
    )


def test_redteam_export_markdown_smoke():
    """Smoke: redteam export_markdown produces a usable Markdown document."""
    assert export_markdown is not None, "export_markdown should be importable"
    md = export_markdown(_make_empty_report())
    assert "# Red Team Security Report" in md
    assert "Executive Summary" in md
    # Must not crash on an empty result set.
    assert "**Target:**" in md


def test_redteam_export_html_smoke():
    """Smoke: redteam export_html produces a self-contained HTML document."""
    assert export_html is not None, "export_html should be importable"
    html = export_html(_make_empty_report())
    assert html.startswith("<!DOCTYPE html>")
    assert "<title>Red Team Security Report</title>" in html
    # Inlined CSS - check a class that should be defined.
    assert "<style>" in html
    # The %% bug regression: rendered CSS must not contain '%%'.
    assert "%%" not in html


def test_write_report_helpers_exit_on_oserror(tmp_path, monkeypatch):
    """The CLI write_*_report helpers must convert OSError into a friendly
    typer.Exit instead of leaking a raw traceback.
    """
    import typer

    from evaluatorq.redteam.cli import write_html_report, write_markdown_report

    report = _make_empty_report()
    out_dir = tmp_path / "reports"

    def fail_write(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("pathlib.Path.write_text", fail_write)

    import pytest

    with pytest.raises(typer.Exit) as exc:
        write_markdown_report(report, out_dir, target="t")
    assert exc.value.exit_code == 1

    with pytest.raises(typer.Exit) as exc:
        write_html_report(report, out_dir, target="t")
    assert exc.value.exit_code == 1
