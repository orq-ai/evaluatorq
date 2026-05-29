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


def test_export_html_escapes_user_controlled_metadata():
    """Persona/scenario/transcript content must be HTML-escaped.

    Regression: a future refactor dropping an _esc() call in the renderers
    would silently introduce HTML injection. Lock it in with a test
    asserting that script tags from user-controlled fields don't appear
    raw in the output.
    """
    nasty = "<script>alert('xss')</script>"
    result = SimulationResult(
        messages=[
            Message(role="user", content=nasty),
            Message(role="assistant", content="hi"),
        ],
        terminated_by=TerminatedBy.judge,
        reason="ok " + nasty,
        goal_achieved=False,
        goal_completion_score=0.5,
        rules_broken=[nasty],
        turn_count=1,
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        turn_metrics=[],
        metadata={
            "persona": nasty,
            "scenario": "S&T",
            "model": "m & m",
            "evaluator_scores": {nasty: 1.0},
            "error": nasty,
        },
    )
    html = export_html([result], target=nasty)
    # The raw script tag must NEVER appear in HTML output.
    assert "<script>alert" not in html
    assert nasty not in html
    # And the escaped form should be there at least once (in the persona field).
    assert "&lt;script&gt;" in html
    # Ampersands also escaped (e.g., persona "S&T", model "m & m").
    assert "S&amp;T" in html or "m &amp; m" in html


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


def test_export_markdown_collapses_token_usage():
    results = [_result("Alice", achieved=True, score=1.0)]
    md = export_markdown(results, target="t")
    # token_usage is wrapped in a collapsible block by the renderer.
    assert "<details>" in md
    assert "<summary>Token Usage</summary>" in md


def test_individual_results_section_not_doubly_collapsed():
    """Each conversation in individual_results is already in its own
    <details>; the section as a whole must NOT be wrapped in an outer
    <details> or readers see doubly-collapsed transcripts.
    """
    results = [_result("Alice", achieved=True, score=1.0)]
    md = export_markdown(results, target="t")
    # The section title appears as a Markdown heading, not as a <summary>.
    assert "## Individual Conversations" in md
    assert "<summary>Individual Conversations</summary>" not in md
