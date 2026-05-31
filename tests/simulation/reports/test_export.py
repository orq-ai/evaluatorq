"""Tests for simulation/reports/export_md.py and export_html.py (RES-846)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evaluatorq.contracts import Message, TokenUsage
from evaluatorq.simulation.reports import export_html, export_markdown
from evaluatorq.simulation.types import SimulationResult, TerminatedBy, TurnMetrics


def _make_result(
    *,
    goal_achieved=True,
    score=1.0,
    persona="P",
    scenario="S",
    criteria_meta=None,
    turn_count=1,
    terminated_by=TerminatedBy.judge,
    target_model=None,
    turn_metrics=None,
) -> SimulationResult:
    meta = {"persona": persona, "scenario": scenario}
    if criteria_meta is not None:
        meta["criteria_meta"] = criteria_meta
    if target_model is not None:
        meta["target_model"] = target_model
    if turn_metrics is None:
        turn_metrics = [
            TurnMetrics(
                turn_number=1,
                token_usage=TokenUsage(),
                judge_reason="ok",
                response_quality=0.8,
            )
        ]
    return SimulationResult(
        messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="response"),
        ],
        terminated_by=terminated_by,
        reason="judge decided",
        goal_achieved=goal_achieved,
        goal_completion_score=score,
        rules_broken=[] if goal_achieved else ["criteria_0"],
        turn_count=turn_count,
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        turn_metrics=turn_metrics,
        metadata=meta,
    )


@pytest.fixture
def make_result():
    return _make_result


@pytest.fixture
def sample_results():
    return [
        _make_result(
            goal_achieved=True,
            score=0.95,
            persona="Alice",
            scenario="Billing",
            target_model="gpt-4o-mini",
            criteria_meta=[
                {"id": "criteria_0", "description": "explain charge",
                 "type": "must_happen", "passed": True},
            ],
        ),
        _make_result(
            goal_achieved=False,
            score=0.1,
            persona="Bob",
            scenario="Refund",
            criteria_meta=[
                {"id": "criteria_0", "description": "explain charge",
                 "type": "must_happen", "passed": False},
                {"id": "criteria_1", "description": "no rudeness",
                 "type": "must_not_happen", "passed": False},
            ],
        ),
    ]


def test_html_has_hero_kpis_and_cards(sample_results):
    html = export_html(sample_results, target="Acme support agent")
    assert 'class="hero"' in html
    assert 'class="kpi-band"' in html
    assert 'class="report-card"' in html
    assert "Acme support agent" in html


def test_html_no_raw_criteria_id_leak(sample_results):
    html = export_html(sample_results, target="t")
    assert "criteria_0" not in html
    assert "criteria_1" not in html


def test_html_renders_without_plotly(sample_results, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_plotly(name, *a, **k):
        if name.startswith("plotly") or name == "kaleido":
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_plotly)
    html = export_html(sample_results, target="t")
    assert "<svg" in html  # charts still render (hand-SVG)


def test_html_renders_new_charts(sample_results):
    html = export_html(sample_results, target="t")
    assert "heatmap-table" in html  # criteria + persona/scenario heatmaps
    assert "Goal Score Distribution" in html
    assert "Turn Quality Timeline" in html
    assert "Failures" in html
    # achieved/failed rendered as semantic badges, not plain text
    assert "status-badge--fail" in html


def test_html_persona_rows_have_sparklines(sample_results):
    html = export_html(sample_results, target="t")
    assert "sparkline" in html


def test_html_heatmap_pass_is_green_fail_is_red(sample_results):
    from evaluatorq.simulation.reports.export_html import (
        _render_criteria_heatmap_html,
    )
    from evaluatorq.simulation.reports.sections import build_report_sections

    sections = build_report_sections(sample_results)
    heat = next(s for s in sections if s.kind == "criteria_heatmap")
    html = _render_criteria_heatmap_html(heat).lower()
    assert "pass" in html and "fail" in html
    # PASS (value 1.0) must be the success green, FAIL (value 0.0) the error red.
    assert "background:#2ebd85" in html  # pass -> green
    assert "background:#d92d20" in html  # fail -> red
    # sample_results' first conversation has no criteria_1, so that cell is
    # absent and must render neutral grey, NOT red.
    assert "background:#e4e2df" in html  # absent -> neutral, not a failure
    assert "—" in html


def test_html_omits_model_row_when_unknown(make_result):
    r = make_result(goal_achieved=True)  # no target_model in metadata
    html = export_html([r], target="t")
    assert "Model:</strong> unknown" not in html
    assert ">unknown<" not in html  # no stray "unknown" model cell


def test_html_full_transcript_no_json_note(sample_results):
    html = export_html(sample_results, target="t")
    assert "full text in report JSON" not in html


def _result(persona: str, *, achieved: bool, score: float, turns: int = 1) -> SimulationResult:
    return SimulationResult(
        messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="response"),
        ],
        terminated_by=TerminatedBy.judge,
        reason="judge decided",
        goal_achieved=achieved,
        goal_completion_score=score,
        rules_broken=[] if achieved else ["off-topic"],
        turn_count=turns,
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        turn_metrics=[
            TurnMetrics(turn_number=1, token_usage=TokenUsage(), judge_reason="ok")
        ],
        metadata={"persona": persona, "scenario": "Smoke", "model": "gpt-4o-mini"},
    )


def test_export_markdown_includes_header_and_summary():
    results = [
        _result("Alice", achieved=True, score=1.0),
        _result("Bob", achieved=False, score=0.2),
    ]
    md = export_markdown(
        results,
        target="demo-agent",
        run_date=datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc),
    )
    assert "# Agent Simulation Report" in md
    assert "**Target:** demo-agent" in md
    assert "**Date:** 2026-05-29 12:00 UTC" in md
    assert "Executive Summary" in md
    assert "Per-Persona Breakdown" in md
    assert "Alice" in md
    assert "Bob" in md
    # Success rate 1/2 = 50%
    assert "50%" in md


def test_export_markdown_handles_empty_results():
    md = export_markdown([], target="nothing")
    assert "Agent Simulation Report" in md
    # Should not crash; should reflect zero conversations.
    assert "**Conversations:** 0" in md


def test_export_html_produces_self_contained_document():
    results = [_result("Alice", achieved=True, score=1.0)]
    html = export_html(
        results,
        target="demo-agent",
        run_date=datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc),
    )
    assert html.startswith("<!DOCTYPE html>")
    assert "<title>Agent Simulation Report</title>" in html
    assert "<style>" in html  # CSS is inlined
    assert "Alice" in html
    assert "demo-agent" in html


def test_export_html_handles_empty_results():
    html = export_html([], target="nothing")
    assert "<!DOCTYPE html>" in html


def test_export_html_css_has_no_double_percent_artifacts():
    """The CSS must not contain ``%%`` (a leftover from the old %-format era).

    Regression: when load_css used Python's % string formatting, CSS rules
    that needed a literal % (``width: 100%``) had to be escaped as ``%%``.
    The switch to string.Template removes the need to escape, so ``%%`` in
    the rendered CSS now produces invalid rules that browsers silently drop.
    """
    html = export_html(
        [_result("Alice", achieved=True, score=1.0)],
        target="t",
    )
    assert "%%" not in html, "double-percent artifacts in CSS will break browser rendering"
    # Sanity: width: 100% should be present and well-formed.
    assert "width: 100%" in html


def test_export_html_uses_only_defined_css_classes():
    """Sim HTML only references CSS classes that exist in report.css."""
    from evaluatorq.common.reports import load_css
    css = load_css()
    html = export_html(
        [_result("Alice", achieved=True, score=1.0)],
        target="t",
    )
    # Inspect status classes the sim renderer applies.
    for cls in ("status-success", "status-failure", "transcript-message"):
        if cls in html:
            assert f".{cls}" in css, f"sim HTML uses .{cls} but it's not defined in report.css"


def test_export_html_truncates_long_transcript_messages():
    """Very long messages must be truncated so HTML files stay manageable."""
    long_content = "x" * 5000
    result = SimulationResult(
        messages=[
            Message(role="user", content=long_content),
            Message(role="assistant", content="ok"),
        ],
        terminated_by=TerminatedBy.judge,
        reason="done",
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        turn_metrics=[],
        metadata={"persona": "p", "scenario": "s", "model": "m"},
    )
    html = export_html([result], target="t")
    # The raw 5000-char run must not appear verbatim.
    assert long_content not in html
    # The truncation marker should be present.
    assert "truncated" in html


def test_export_markdown_collapses_token_usage_and_individual_results():
    results = [_result("Alice", achieved=True, score=1.0)]
    md = export_markdown(results, target="t")
    # Collapsed sections wrap content in <details> blocks.
    assert "<details>" in md
    assert "<summary>Token Usage</summary>" in md
    assert "<summary>Individual Conversations</summary>" in md
