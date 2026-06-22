"""Tests that redteam HTML export uses Vega-Lite, not plotly/kaleido.

A4 acceptance tests:
1. No plotly or kaleido imports in the redteam export module.
2. try_render_svg is gone from common.reports.
3. Smoke test: a full RedTeamReport renders to a valid HTML doc with SVG charts.
"""

from __future__ import annotations

import importlib
from datetime import datetime, timezone
from pathlib import Path

from evaluatorq.redteam.contracts import (
    AgentInfo,
    AttackInfo,
    AttackTechnique,
    DeliveryMethod,
    Framework,
    Pipeline,
    RedTeamReport,
    RedTeamResult,
    ReportSummary,
    Severity,
    TurnType,
    UnifiedEvaluationResult,
)
from evaluatorq.redteam.reports.converters import compute_report_summary
from evaluatorq.redteam.reports.export_html import export_html


def test_no_plotly_or_kaleido_in_redteam_export() -> None:
    """The redteam export module must contain no plotly or try_render_svg references."""
    eh_module = importlib.import_module('evaluatorq.redteam.reports.export_html')
    src = Path(eh_module.__file__).read_text()
    assert 'plotly' not in src and 'try_render_svg' not in src


def test_try_render_svg_gone_from_common() -> None:
    """try_render_svg must be removed from evaluatorq.common.reports."""
    from evaluatorq.common import reports

    assert not hasattr(reports, 'try_render_svg')


def _make_result(
    category: str = 'ASI01',
    passed: bool | None = True,
    agent_key: str = 'test-agent',
) -> RedTeamResult:
    return RedTeamResult(
        attack=AttackInfo(
            id=f'{category}-a4-smoke-001',
            category=category,
            framework=Framework.OWASP_ASI,
            attack_technique=AttackTechnique.INDIRECT_INJECTION,
            delivery_methods=[DeliveryMethod.DIRECT_REQUEST],
            turn_type=TurnType.SINGLE,
            severity=Severity.MEDIUM,
            source='test',
        ),
        agent=AgentInfo(key=agent_key),
        messages=[],
        vulnerable=passed is False,
        evaluation=UnifiedEvaluationResult(passed=passed, explanation='test') if passed is not None else None,
    )


def test_redteam_export_html_smoke() -> None:
    """export_html produces a valid HTML doc with SVG charts (Vega-Lite path)."""
    results = [
        _make_result(category='ASI01', passed=True),
        _make_result(category='ASI01', passed=False),
        _make_result(category='LLM01', passed=None),
    ]
    report = RedTeamReport(
        created_at=datetime.now(tz=timezone.utc),
        description='A4 smoke test report',
        pipeline=Pipeline.STATIC,
        framework=Framework.OWASP_ASI,
        categories_tested=['ASI01', 'LLM01'],
        tested_agents=['test-agent'],
        total_results=len(results),
        results=results,
        summary=compute_report_summary(results),
    )
    html = export_html(report)
    assert '<!DOCTYPE html>' in html
    assert '<html' in html
    assert '</html>' in html
    assert '<table' in html
    assert '<svg' in html
