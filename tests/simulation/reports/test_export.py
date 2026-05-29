"""Tests for simulation/reports/export_md.py and export_html.py (RES-846)."""

from __future__ import annotations

from datetime import datetime, timezone

from evaluatorq.contracts import Message, TokenUsage
from evaluatorq.simulation.reports import export_html, export_markdown
from evaluatorq.simulation.types import SimulationResult, TerminatedBy, TurnMetrics


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
