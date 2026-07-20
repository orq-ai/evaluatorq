"""Dashboard trace-link buttons — the three remaining insertions
(``_sim_hero``, ``_redteam_hero``, ``redteam_transcripts.render_attack_fragment``).

Confirms the deep-link anchor appears in real rendered HTML when
``ORQ_WORKSPACE_SLUG`` is configured, and is absent when it isn't (so
older reports without ids, or unconfigured deployments, render fine).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from evaluatorq.dashboard.redteam_transcripts import render_attack_fragment
from evaluatorq.dashboard.report_tabs import _redteam_hero, _sim_hero
from evaluatorq.dashboard.trace_links import run_trace_url, thread_trace_url, trace_link_button
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
from evaluatorq.simulation.types import SimulationRun


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ('ORQ_UI_BASE_URL', 'ORQ_BASE_URL', 'ORQ_WORKSPACE_SLUG', 'ORQ_WORKSPACE'):
        monkeypatch.delenv(var, raising=False)


def _make_run(run_id: str | None) -> SimulationRun:
    return SimulationRun(
        run_name='test-run',
        created_at=datetime.now(tz=timezone.utc),
        mode='run',
        target_kind='openai_model',
        evaluator_names=[],
        total_results=0,
        scorer_averages={},
        results=[],
        run_id=run_id,
    )


def _make_result(thread_id: str | None) -> RedTeamResult:
    return RedTeamResult(
        attack=AttackInfo(
            id='ASI01-test-001',
            category='ASI01',
            framework=Framework.OWASP_ASI,
            attack_technique=AttackTechnique.INDIRECT_INJECTION,
            delivery_methods=[DeliveryMethod.DIRECT_REQUEST],
            turn_type=TurnType.SINGLE,
            severity=Severity.MEDIUM,
            source='test',
        ),
        agent=AgentInfo(key='agent-a'),
        messages=[],
        vulnerable=False,
        evaluation=UnifiedEvaluationResult(passed=True, explanation='test'),
        thread_id=thread_id,
    )


def _make_report(run_id: str | None) -> RedTeamReport:
    results = [_make_result('run1:0')]
    return RedTeamReport(
        created_at=datetime.now(tz=timezone.utc),
        pipeline=Pipeline.DYNAMIC,
        framework=Framework.OWASP_ASI,
        categories_tested=['ASI01'],
        tested_agents=['agent-a'],
        total_results=len(results),
        results=results,
        summary=compute_report_summary(results),
        run_id=run_id,
    )


class TestSimHeroTraceButton:
    def test_button_absent_without_workspace_slug(self) -> None:
        html = _sim_hero(_make_run('run1'))
        assert 'trace-link' not in html
        assert 'View all run traces' not in html

    def test_button_absent_without_run_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ORQ_WORKSPACE_SLUG', 'orq-research')
        html = _sim_hero(_make_run(None))
        assert 'trace-link' not in html

    def test_button_present_when_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ORQ_WORKSPACE_SLUG', 'orq-research')
        html = _sim_hero(_make_run('run1'))
        assert 'class="btn-secondary trace-link"' in html
        assert 'View all run traces' in html
        assert f'href="{run_trace_url("run1")}"' in html


class TestRedteamHeroTraceButton:
    def test_button_absent_without_workspace_slug(self) -> None:
        html = _redteam_hero(None, _make_report('run1'))
        assert 'trace-link' not in html

    def test_button_present_when_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ORQ_WORKSPACE_SLUG', 'orq-research')
        html = _redteam_hero(None, _make_report('run1'))
        assert 'class="btn-secondary trace-link"' in html
        assert 'View all run traces' in html
        assert f'href="{run_trace_url("run1")}"' in html


class TestRedteamConversationTraceButton:
    def test_button_absent_without_workspace_slug(self) -> None:
        html = render_attack_fragment(_make_result('run1:0'))
        assert 'trace-link' not in html

    def test_button_absent_without_thread_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ORQ_WORKSPACE_SLUG', 'orq-research')
        html = render_attack_fragment(_make_result(None))
        assert 'trace-link' not in html

    def test_button_present_when_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ORQ_WORKSPACE_SLUG', 'orq-research')
        html = render_attack_fragment(_make_result('run1:0'))
        assert 'class="btn-secondary trace-link"' in html
        assert 'View Traces' in html
        assert f'href="{thread_trace_url("run1:0")}"' in html


def _make_entry(thread_id: str | None) -> Any:
    from evaluatorq.simulation.types import SimulationEntry

    return SimulationEntry(
        index=0,
        persona='p',
        scenario='s',
        model='m',
        target_model=None,
        terminated_by='judge',
        goal_achieved=False,
        goal_completion_score=0.0,
        rules_broken=[],
        criteria=[],
        turn_count=2,
        total_tokens=0,
        judge_reason='',
        error=None,
        evaluator_scores={},
        transcript=[],
        thread_id=thread_id,
    )


class TestSimRowTraceButton:
    """The conversation trace button lives on the summary row (next to the
    outcome badge), not in the expanded transcript fragment."""

    def test_button_absent_without_workspace_slug(self) -> None:
        from evaluatorq.dashboard.sim_views import render_sim_row_list

        html = render_sim_row_list('rid', [_make_entry('run1:0')])
        assert 'trace-link' not in html

    def test_button_present_on_row_when_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from evaluatorq.dashboard.sim_views import render_sim_row_list

        monkeypatch.setenv('ORQ_WORKSPACE_SLUG', 'orq-research')
        html = render_sim_row_list('rid', [_make_entry('run1:0')])
        assert 'class="btn-secondary trace-link"' in html
        assert 'View Traces' in html
        assert f'href="{thread_trace_url("run1:0")}"' in html
        trace_start = html.index('<a class="btn-secondary trace-link"')
        trace_end = html.index('</a>', trace_start) + len('</a>')
        # Aligned columns put the trace anchor in a cell inside the row; its
        # ``data-no-drawer`` marker is what stops a click from firing the drawer.
        assert 'data-no-drawer' in html[trace_start:trace_end]
        assert '<tr class="sim-conv-row ' in html


def test_button_helper_smoke() -> None:
    # Sanity: trace_link_button itself is exercised transitively above; this
    # locks in the '' fallback contract the callers depend on.
    assert trace_link_button(None, 'x') == ''


if __name__ == '__main__':
    import subprocess
    import sys

    raise SystemExit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-q']))
