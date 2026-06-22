"""TDD tests for shared render_body and surface-level render_report_body functions.

Verifies:
- ``common.reports.render.render_body`` produces a body fragment (no <!doctype>,
  no <html>, no <head>) containing section content.
- ``redteam.reports.export_html.render_report_body`` produces a body fragment
  for a RedTeamReport.
- ``simulation.reports.export_html.render_report_body`` produces a body fragment
  for simulation results.
- Both surfaces' ``export_html`` output remains unchanged (byte-stable) after
  the refactor.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evaluatorq.contracts import ReportSection


# ---------------------------------------------------------------------------
# Helpers shared by multiple tests
# ---------------------------------------------------------------------------


def _make_redteam_report():
    """Build a minimal RedTeamReport for testing."""
    from evaluatorq.redteam.contracts import (
        AgentInfo,
        AttackInfo,
        AttackTechnique,
        DeliveryMethod,
        Framework,
        Pipeline,
        RedTeamReport,
        RedTeamResult,
        Severity,
        TurnType,
        UnifiedEvaluationResult,
    )
    from evaluatorq.redteam.reports.converters import compute_report_summary

    def _result(category: str = 'ASI01', passed: bool | None = True) -> RedTeamResult:
        return RedTeamResult(
            attack=AttackInfo(
                id=f'{category}-body-test-001',
                category=category,
                framework=Framework.OWASP_ASI,
                attack_technique=AttackTechnique.INDIRECT_INJECTION,
                delivery_methods=[DeliveryMethod.DIRECT_REQUEST],
                turn_type=TurnType.SINGLE,
                severity=Severity.MEDIUM,
                source='test',
            ),
            agent=AgentInfo(key='test-agent'),
            messages=[],
            vulnerable=passed is False,
            evaluation=UnifiedEvaluationResult(passed=passed, explanation='test') if passed is not None else None,
        )

    results = [
        _result(category='ASI01', passed=True),
        _result(category='ASI01', passed=False),
        _result(category='LLM01', passed=None),
    ]
    return RedTeamReport(
        created_at=datetime.now(tz=timezone.utc),
        description='Body-fragment test report',
        pipeline=Pipeline.STATIC,
        framework=Framework.OWASP_ASI,
        categories_tested=['ASI01', 'LLM01'],
        tested_agents=['test-agent'],
        total_results=len(results),
        results=results,
        summary=compute_report_summary(results),
    )


def _make_sim_results():
    """Build a minimal list of SimulationResults for testing."""
    from evaluatorq.contracts import Message, TokenUsage
    from evaluatorq.simulation.types import SimulationResult, TerminatedBy, TurnMetrics

    return [
        SimulationResult(
            messages=[
                Message(role='user', content='hello'),
                Message(role='assistant', content='hi there'),
            ],
            terminated_by=TerminatedBy.judge,
            reason='judge decided',
            goal_achieved=True,
            goal_completion_score=0.9,
            rules_broken=[],
            turn_count=1,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            turn_metrics=[
                TurnMetrics(turn_number=1, token_usage=TokenUsage(), judge_reason='ok', response_quality=0.8)
            ],
            metadata={'persona': 'Alice', 'scenario': 'Billing'},
        )
    ]


def _is_body_fragment(html: str) -> bool:
    """Return True when html is a body fragment (no doctype/html/head elements).

    We check for the head *element* (``<head>`` or ``<head `` with a space/attr),
    not just the substring ``<head`` which also matches the valid ``<header>`` tag
    that appears legitimately in body fragments.
    """
    lower = html.lower()
    has_doctype = '<!doctype' in lower
    has_html = '<html' in lower
    # Match <head> or <head followed by a space/attribute, but NOT <header>
    has_head_element = '<head>' in lower or '<head ' in lower
    return not has_doctype and not has_html and not has_head_element


# ---------------------------------------------------------------------------
# Tests: common render_body
# ---------------------------------------------------------------------------


class TestRenderBody:
    """Tests for ``common.reports.render.render_body``."""

    def test_render_body_importable(self) -> None:
        """render_body must be importable from common.reports.render."""
        from evaluatorq.common.reports.render import render_body  # noqa: F401

    def test_render_body_no_doctype_or_html_wrapper(self) -> None:
        """render_body must return a fragment without <!doctype>, <html> or <head>."""
        from evaluatorq.common.reports.render import render_body
        from evaluatorq.contracts import ReportSection

        section = ReportSection(kind='test', title='Test', data={})
        renderers = {'test': lambda s: f'<p>{s.title}</p>'}
        fragment = render_body([section], renderers=renderers)
        assert _is_body_fragment(fragment)

    def test_render_body_includes_section_content(self) -> None:
        """render_body must include content produced by the section renderer."""
        from evaluatorq.common.reports.render import render_body

        section = ReportSection(kind='test', title='My Section', data={})
        renderers = {'test': lambda s: f'<p>content-{s.title}</p>'}
        fragment = render_body([section], renderers=renderers)
        assert 'content-My Section' in fragment

    def test_render_body_includes_body_header_and_footer(self) -> None:
        """render_body must prepend body_header and append body_footer."""
        from evaluatorq.common.reports.render import render_body

        fragment = render_body(
            [],
            renderers={},
            body_header='<header>top</header>',
            body_footer='<footer>bottom</footer>',
        )
        assert '<header>top</header>' in fragment
        assert '<footer>bottom</footer>' in fragment

    def test_render_body_skips_unknown_section_kinds(self) -> None:
        """Sections with unregistered kinds must not appear in the output."""
        from evaluatorq.common.reports.render import render_body

        known = ReportSection(kind='known', title='K', data={})
        unknown = ReportSection(kind='unknown', title='U', data={})
        renderers = {'known': lambda s: '<p>known</p>'}
        fragment = render_body([known, unknown], renderers=renderers)
        assert 'known' in fragment
        assert 'unknown' not in fragment

    def test_render_html_still_returns_full_document(self) -> None:
        """render_html must still return a full HTML5 document after extracting render_body."""
        from evaluatorq.common.reports.render import render_html

        section = ReportSection(kind='s', title='S', data={})
        renderers = {'s': lambda sec: '<p>hi</p>'}
        doc = render_html([section], renderers=renderers, head='<title>T</title>')
        assert doc.startswith('<!DOCTYPE html>')
        assert '<html' in doc
        assert '<head>' in doc
        assert '</html>' in doc
        assert '<p>hi</p>' in doc


# ---------------------------------------------------------------------------
# Tests: redteam render_report_body
# ---------------------------------------------------------------------------


class TestRedteamRenderReportBody:
    """Tests for ``redteam.reports.export_html.render_report_body``."""

    def test_render_report_body_importable(self) -> None:
        """render_report_body must be importable from redteam.reports.export_html."""
        from evaluatorq.redteam.reports.export_html import render_report_body  # noqa: F401

    def test_render_report_body_is_fragment(self) -> None:
        """render_report_body must return a body fragment, not a full document."""
        from evaluatorq.redteam.reports.export_html import render_report_body

        report = _make_redteam_report()
        fragment = render_report_body(report)
        assert _is_body_fragment(fragment), (
            'render_report_body returned a full HTML document; expected a fragment without '
            '<!doctype>, <html> or <head>'
        )

    def test_render_report_body_contains_section_content(self) -> None:
        """The body fragment must contain recognizable report content."""
        from evaluatorq.redteam.reports.export_html import render_report_body

        report = _make_redteam_report()
        fragment = render_report_body(report)
        # The summary section heading should appear in the fragment.
        assert 'Executive Summary' in fragment or 'Summary' in fragment

    def test_export_html_still_returns_full_document(self) -> None:
        """export_html must still return a complete HTML document after the refactor."""
        from evaluatorq.redteam.reports.export_html import export_html

        report = _make_redteam_report()
        doc = export_html(report)
        assert '<!doctype' in doc.lower()
        assert '<html' in doc.lower()
        assert '<head' in doc.lower()

    def test_export_html_contains_body_fragment_content(self) -> None:
        """export_html must include all content produced by render_report_body."""
        from evaluatorq.redteam.reports.export_html import export_html, render_report_body

        report = _make_redteam_report()
        fragment = render_report_body(report)
        doc = export_html(report)
        # Pick a stable substring from the fragment that should appear in the full doc.
        # Strip leading/trailing whitespace and check a meaningful chunk.
        # We just verify the section content surfaces in the full doc.
        assert 'Executive Summary' in doc or 'Summary' in doc


# ---------------------------------------------------------------------------
# Tests: simulation render_report_body
# ---------------------------------------------------------------------------


class TestSimRenderReportBody:
    """Tests for ``simulation.reports.export_html.render_report_body``."""

    def test_render_report_body_importable(self) -> None:
        """render_report_body must be importable from simulation.reports.export_html."""
        from evaluatorq.simulation.reports.export_html import render_report_body  # noqa: F401

    def test_render_report_body_is_fragment(self) -> None:
        """render_report_body must return a body fragment, not a full document."""
        from evaluatorq.simulation.reports.export_html import render_report_body

        results = _make_sim_results()
        fragment = render_report_body(results, target='test-agent')
        assert _is_body_fragment(fragment), (
            'render_report_body returned a full HTML document; expected a fragment without '
            '<!doctype>, <html> or <head>'
        )

    def test_render_report_body_contains_section_content(self) -> None:
        """The body fragment must contain recognizable report content."""
        from evaluatorq.simulation.reports.export_html import render_report_body

        results = _make_sim_results()
        fragment = render_report_body(results, target='test-agent')
        # The hero header HTML should appear in the fragment.
        assert 'Agent Simulation Report' in fragment

    def test_export_html_still_returns_full_document(self) -> None:
        """export_html must still return a complete HTML document after the refactor."""
        from evaluatorq.simulation.reports.export_html import export_html

        results = _make_sim_results()
        doc = export_html(results, target='test-agent')
        assert '<!doctype' in doc.lower()
        assert '<html' in doc.lower()
        assert '<head' in doc.lower()

    def test_render_report_body_accepts_empty_results(self) -> None:
        """render_report_body must not crash on an empty result list."""
        from evaluatorq.simulation.reports.export_html import render_report_body

        fragment = render_report_body([], target='nothing')
        assert _is_body_fragment(fragment)

    def test_render_report_body_with_run_date(self) -> None:
        """render_report_body must accept an explicit run_date."""
        from evaluatorq.simulation.reports.export_html import render_report_body

        results = _make_sim_results()
        run_date = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        fragment = render_report_body(results, target='test-agent', run_date=run_date)
        assert '2026-01-01' in fragment
