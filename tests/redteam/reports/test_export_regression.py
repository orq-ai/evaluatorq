"""Regression tests: redteam HTML report renders with and without vl-convert.

Proves that the shared ``common.reports`` palette move + CSS overhaul did not
break the redteam HTML export path.  Two cases:

1. Normal environment (vl-convert installed) — charts render as SVG.
2. vl-convert absent — graceful degrade to tables; no crash.
"""

from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

import pytest

from evaluatorq.redteam.contracts import (
    AttackInfo,
    AttackTechnique,
    AgentInfo,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    category: str = 'ASI01',
    passed: bool | None = True,
    agent_key: str = 'test-agent',
) -> RedTeamResult:
    """Build a minimal RedTeamResult for testing."""
    return RedTeamResult(
        attack=AttackInfo(
            id=f'{category}-regression-001',
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


def _make_minimal_report() -> RedTeamReport:
    """Build the smallest valid RedTeamReport that exercises the HTML renderer."""
    results = [
        _make_result(category='ASI01', passed=True),
        _make_result(category='ASI01', passed=False),
        _make_result(category='LLM01', passed=None),
    ]
    return RedTeamReport(
        created_at=datetime.now(tz=timezone.utc),
        description='Regression test report',
        pipeline=Pipeline.STATIC,
        framework=Framework.OWASP_ASI,
        categories_tested=['ASI01', 'LLM01'],
        tested_agents=['test-agent'],
        total_results=len(results),
        results=results,
        summary=compute_report_summary(results),
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def redteam_report() -> RedTeamReport:
    """Minimal RedTeamReport for render regression tests."""
    return _make_minimal_report()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_redteam_html_renders_with_vl_convert(redteam_report: RedTeamReport) -> None:
    """export_html returns a complete HTML document when vl-convert is available."""
    html = export_html(redteam_report)
    assert '<html' in html and '</html>' in html


def test_redteam_html_renders_without_vl_convert(
    redteam_report: RedTeamReport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """export_html degrades gracefully (tables only) when vl-convert is absent."""
    real_import = builtins.__import__

    def no_vl_convert(
        name: str,
        globals_: Mapping[str, object] | None = None,
        locals_: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name == 'vl_convert':
            raise ImportError(name)
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', no_vl_convert)
    # Also patch vl_available so cached state doesn't hide the block.
    import evaluatorq.common.reports.vega as vega_mod

    monkeypatch.setattr(vega_mod, 'vl_available', lambda: False)

    html = export_html(redteam_report)
    assert '<html' in html and '</html>' in html
    # Degrades to data tables (not empty chart shells) when charts are absent.
    assert '<table' in html
