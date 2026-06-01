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
    persona='P',
    scenario='S',
    criteria_meta=None,
    turn_count=1,
    terminated_by=TerminatedBy.judge,
    target_model=None,
    turn_metrics=None,
) -> SimulationResult:
    meta = {'persona': persona, 'scenario': scenario}
    if criteria_meta is not None:
        meta['criteria_meta'] = criteria_meta
    if target_model is not None:
        meta['target_model'] = target_model
    if turn_metrics is None:
        turn_metrics = [
            TurnMetrics(
                turn_number=1,
                token_usage=TokenUsage(),
                judge_reason='ok',
                response_quality=0.8,
            )
        ]
    return SimulationResult(
        messages=[
            Message(role='user', content='hi'),
            Message(role='assistant', content='response'),
        ],
        terminated_by=terminated_by,
        reason='judge decided',
        goal_achieved=goal_achieved,
        goal_completion_score=score,
        rules_broken=[] if goal_achieved else ['criteria_0'],
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
            persona='Alice',
            scenario='Billing',
            target_model='gpt-4o-mini',
            criteria_meta=[
                {'id': 'criteria_0', 'description': 'explain charge', 'type': 'must_happen', 'passed': True},
            ],
        ),
        _make_result(
            goal_achieved=False,
            score=0.1,
            persona='Bob',
            scenario='Refund',
            criteria_meta=[
                {'id': 'criteria_0', 'description': 'explain charge', 'type': 'must_happen', 'passed': False},
                {'id': 'criteria_1', 'description': 'no rudeness', 'type': 'must_not_happen', 'passed': False},
            ],
        ),
    ]


def test_html_has_hero_kpis_and_cards(sample_results):
    html = export_html(sample_results, target='Acme support agent')
    assert 'class="hero"' in html
    assert 'class="kpi-band"' in html
    assert 'class="report-card"' in html
    assert 'Acme support agent' in html


def test_html_no_raw_criteria_id_leak(sample_results):
    html = export_html(sample_results, target='t')
    assert 'criteria_0' not in html
    assert 'criteria_1' not in html


def test_html_renders_without_plotly(sample_results, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_plotly(name, *a, **k):
        if name.startswith('plotly') or name == 'kaleido':
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, '__import__', no_plotly)
    html = export_html(sample_results, target='t')
    assert '<svg' in html  # charts still render (hand-SVG)


def test_html_renders_new_charts(sample_results):
    html = export_html(sample_results, target='t')
    assert 'heatmap-table' in html  # criteria + persona/scenario heatmaps
    assert 'Goal Score Distribution' in html
    assert 'Turn Quality Timeline' in html
    assert 'Failures' in html
    # achieved/failed rendered as semantic badges, not plain text
    assert 'status-badge--fail' in html


def test_html_persona_rows_show_success_rate_not_sparkline(sample_results):
    # Sparklines were dropped: a [rate, 1-rate] minibar in one colour misled
    # (100% could look weaker than 33%) and duplicated the Success % column.
    html = export_html(sample_results, target='t')
    assert 'class="sparkline"' not in html
    assert 'Success' in html  # the success-rate column header remains


def test_html_criterion_description_single_escaped(make_result):
    """Criterion descriptions are escaped exactly once (status_badge escapes
    internally), so special chars must not be double-encoded."""
    r = make_result(
        goal_achieved=True,
        criteria_meta=[
            {'id': 'criteria_0', 'description': 'fees & "charges"', 'type': 'must_happen', 'passed': True},
        ],
    )
    html = export_html([r], target='t')
    # Correct single escaping.
    assert '&amp;' in html
    assert '&quot;' in html
    # Not double-escaped.
    assert '&amp;amp;' not in html
    assert '&amp;quot;' not in html


def test_html_summary_empty_state_message():
    """With no conversations the summary section still surfaces a no-data note
    so the report never looks truncated (the headline numbers otherwise live in
    the hero KPI band, so the summary renders no standalone card)."""
    html = export_html([], target='t')
    assert 'No conversations to summarize.' in html
    # The pass-rate donut was dropped — its data duplicates the KPI band.
    assert 'Goal Outcomes' not in html


def test_html_omits_model_row_when_unknown(make_result):
    r = make_result(goal_achieved=True)  # no target_model in metadata
    html = export_html([r], target='t')
    assert 'Model:</strong> unknown' not in html
    assert '>unknown<' not in html  # no stray "unknown" model cell


def test_html_full_transcript_no_json_note(sample_results):
    html = export_html(sample_results, target='t')
    assert 'full text in report JSON' not in html


def _result(persona: str, *, achieved: bool, score: float, turns: int = 1) -> SimulationResult:
    return SimulationResult(
        messages=[
            Message(role='user', content='hi'),
            Message(role='assistant', content='response'),
        ],
        terminated_by=TerminatedBy.judge,
        reason='judge decided',
        goal_achieved=achieved,
        goal_completion_score=score,
        rules_broken=[] if achieved else ['off-topic'],
        turn_count=turns,
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        turn_metrics=[TurnMetrics(turn_number=1, token_usage=TokenUsage(), judge_reason='ok')],
        metadata={'persona': persona, 'scenario': 'Smoke', 'model': 'gpt-4o-mini'},
    )


def test_export_markdown_includes_header_and_summary():
    results = [
        _result('Alice', achieved=True, score=1.0),
        _result('Bob', achieved=False, score=0.2),
    ]
    md = export_markdown(
        results,
        target='demo-agent',
        run_date=datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc),
    )
    assert '# Agent Simulation Report' in md
    assert '**Target:** demo-agent' in md
    assert '**Date:** 2026-05-29 12:00 UTC' in md
    assert 'Executive Summary' in md
    assert 'Per-Persona Breakdown' in md
    assert 'Alice' in md
    assert 'Bob' in md
    # Success rate 1/2 = 50%
    assert '50%' in md


def test_export_markdown_handles_empty_results():
    md = export_markdown([], target='nothing')
    assert 'Agent Simulation Report' in md
    # Should not crash; should reflect zero conversations.
    assert '**Conversations:** 0' in md


def test_export_html_produces_self_contained_document():
    results = [_result('Alice', achieved=True, score=1.0)]
    html = export_html(
        results,
        target='demo-agent',
        run_date=datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc),
    )
    assert html.startswith('<!DOCTYPE html>')
    assert '<title>Agent Simulation Report</title>' in html
    assert '<style>' in html  # CSS is inlined
    assert 'Alice' in html
    assert 'demo-agent' in html


def test_export_html_handles_empty_results():
    html = export_html([], target='nothing')
    assert '<!DOCTYPE html>' in html


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
        [_result('Alice', achieved=True, score=1.0)],
        target='t',
    )
    assert '%%' not in html, 'double-percent artifacts in CSS will break browser rendering'
    # Sanity: width: 100% should be present and well-formed.
    assert 'width: 100%' in html


def test_export_html_uses_only_defined_css_classes():
    """Sim HTML only references CSS classes that exist in report.css."""
    from evaluatorq.common.reports import load_css

    css = load_css()
    html = export_html(
        [_result('Alice', achieved=True, score=1.0)],
        target='t',
    )
    # Inspect status classes the sim renderer applies.
    for cls in ('status-success', 'status-failure', 'transcript-message'):
        if cls in html:
            assert f'.{cls}' in css, f"sim HTML uses .{cls} but it's not defined in report.css"


def test_export_html_truncates_long_transcript_messages():
    """Very long messages must be truncated so HTML files stay manageable."""
    long_content = 'x' * 5000
    result = SimulationResult(
        messages=[
            Message(role='user', content=long_content),
            Message(role='assistant', content='ok'),
        ],
        terminated_by=TerminatedBy.judge,
        reason='done',
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        turn_metrics=[],
        metadata={'persona': 'p', 'scenario': 's', 'model': 'm'},
    )
    html = export_html([result], target='t')
    # The raw 5000-char run must not appear verbatim.
    assert long_content not in html
    # The truncation marker should be present.
    assert 'truncated' in html


def test_export_markdown_collapses_token_usage_and_individual_results():
    results = [_result('Alice', achieved=True, score=1.0)]
    md = export_markdown(results, target='t')
    # Collapsed sections wrap content in <details> blocks.
    assert '<details>' in md
    assert '<summary>Token Usage</summary>' in md
    assert '<summary>Individual Conversations</summary>' in md


def test_md_has_new_sections_and_no_raw_ids(sample_results):
    md = export_markdown(sample_results, target='t')
    assert 'Failures' in md
    assert 'Persona x Scenario' in md or 'Persona × Scenario' in md
    assert 'Goal Score Distribution' in md
    assert 'Turn Quality Timeline' in md
    assert 'criteria_0' not in md and 'criteria_1' not in md


def test_md_omits_model_when_unknown(make_result):
    md = export_markdown([make_result(goal_achieved=True)], target='t')
    assert 'Model:** unknown' not in md


def test_overview_html_and_md_show_traits_and_goal():
    from evaluatorq.simulation.reports import export_html, export_markdown
    from evaluatorq.simulation.types import SimulationResult, TerminatedBy, TokenUsage

    r = SimulationResult(
        messages=[], terminated_by=TerminatedBy.judge, reason='r',
        goal_achieved=True, goal_completion_score=1.0, rules_broken=[],
        turn_count=1, turn_metrics=[],
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        metadata={
            'persona': 'Frustrated Customer', 'scenario': 'Billing',
            'persona_traits': {'patience': 0.2, 'assertiveness': 0.8, 'politeness': 0.4,
                               'technical_level': 0.3, 'communication_style': 'casual',
                               'background': 'Annoyed.'},
            'scenario_goal': 'Explain the invoice', 'scenario_context': 'Unexpected charge.',
            'criteria_meta': [{'id': 'criteria_0', 'description': 'explains charge',
                               'type': 'must_happen', 'passed': True}],
        },
    )
    html = export_html([r], target='Agent')
    md = export_markdown([r], target='Agent')
    assert 'Explain the invoice' in html and 'Explain the invoice' in md
    assert 'Annoyed.' in html and 'Annoyed.' in md


def test_overview_renders_zero_trait_and_fallback_without_metadata():
    from evaluatorq.simulation.reports import export_html, export_markdown
    from evaluatorq.simulation.types import SimulationResult, TerminatedBy, TokenUsage

    def _result(metadata):
        return SimulationResult(
            messages=[], terminated_by=TerminatedBy.judge, reason='r',
            goal_achieved=True, goal_completion_score=1.0, rules_broken=[],
            turn_count=1, turn_metrics=[],
            token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            metadata=metadata,
        )

    # patience 0.0 must NOT be dropped by a truthiness guard
    with_zero = _result({
        'persona': 'P', 'scenario': 'S',
        'persona_traits': {'patience': 0.0, 'assertiveness': 0.8, 'politeness': 0.4,
                           'technical_level': 0.3, 'communication_style': 'casual',
                           'background': 'bg'},
        'scenario_goal': 'g',
    })
    html = export_html([with_zero], target='Agent')
    md = export_markdown([with_zero], target='Agent')
    assert 'patience 0.0' in html and 'patience 0.0' in md

    # older result: no traits/goal -> name-only, no empty "Goal:" label, no stray separators
    legacy = _result({'persona': 'Old', 'scenario': 'Leg'})
    html2 = export_html([legacy], target='Agent')
    md2 = export_markdown([legacy], target='Agent')
    assert 'Old' in html2 and 'Old' in md2
    assert 'Goal:' not in html2 and '**Goal:**' not in md2
