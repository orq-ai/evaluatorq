from __future__ import annotations

from types import SimpleNamespace

import pytest

from evaluatorq.common.reports.brief import redteam_brief, simulation_brief
from evaluatorq.contracts import ReportSection


def _section(kind: str, **data: object) -> ReportSection:
    return ReportSection(kind=kind, title=kind, data=data)


def test_simulation_brief_keeps_confirmed_criterion_failure_when_execution_errors() -> None:
    run = SimpleNamespace(total_results=1, recommendations=[])
    sections = {
        'summary': _section('summary', total_conversations=1, goals_achieved=0, goals_failed=0, errors=1),
        'individual_results': _section(
            'individual_results',
            entries=[
                {
                    'index': 0,
                    'persona': 'Alice',
                    'scenario': 'refund',
                    'terminated_by': 'error',
                    'error': 'provider unavailable',
                    'criteria': [{'state': 'fail', 'type': 'must_happen', 'description': 'Ask for order number'}],
                }
            ],
        ),
        'recommendations': _section(
            'recommendations', rows=[{'index': 1, 'suggestions': ['Ask for the order number first.']}]
        ),
    }

    html = simulation_brief(run, sections)

    assert '1 conversations.' in html
    assert 'Ask for order number' in html
    assert 'A criterion was confirmed violated.' in html
    assert 'Ask for the order number first.' in html
    assert 'href="#conv-1"' in html
    assert 'Execution error' in html
    assert 'id="export-finding-1"' in html


def test_simulation_brief_distinguishes_prohibited_behavior() -> None:
    run = SimpleNamespace(total_results=3, recommendations=[])
    sections = {
        'summary': _section('summary', total_conversations=3, goals_achieved=0, goals_failed=2, errors=1),
        'individual_results': _section(
            'individual_results',
            entries=[
                {
                    'index': 0,
                    'persona': 'Pat',
                    'scenario': 'missing requirement',
                    'terminated_by': 'judge',
                    'goal_achieved': False,
                    'criteria': [{'state': 'fail', 'type': 'must_happen', 'description': 'Ask for order number'}],
                },
                {
                    'index': 1,
                    'persona': 'Pat',
                    'scenario': 'privacy request',
                    'terminated_by': 'judge',
                    'goal_achieved': False,
                    'criteria': [{'state': 'fail', 'type': 'must_not_happen', 'description': 'Reveal private data'}],
                },
                {
                    'index': 2,
                    'persona': 'Pat',
                    'scenario': 'timed out',
                    'terminated_by': 'error',
                    'error': 'timeout',
                    'goal_achieved': False,
                    'criteria': [],
                },
            ],
        ),
    }

    html = simulation_brief(run, sections)

    assert 'A prohibited behavior was confirmed.' in html
    assert html.index('Reveal private data') < html.index('Ask for order number')
    assert html.index('Ask for order number') < html.index('Execution did not complete')


def test_simulation_brief_keeps_goal_failures_without_criteria_or_recommendations() -> None:
    run = SimpleNamespace(
        total_results=5,
        recommendations=[],
        target='support-agent',
        target_model='model-x',
        target_kind='orq_agent',
    )
    entries = [
        {
            'index': index,
            'persona': f'Persona {index}',
            'scenario': f'Scenario {index}',
            'terminated_by': 'judge',
            'goal_achieved': False,
            'criteria': [],
        }
        for index in range(5)
    ]
    sections = {
        'summary': _section('summary', total_conversations=5, goals_achieved=0, goals_failed=5, errors=0),
        'individual_results': _section('individual_results', entries=entries),
    }

    html = simulation_brief(run, sections)

    assert 'Findings shown: 3 of 5.' in html
    assert html.count('id="export-finding-') == 3
    assert 'support-agent using model-x' in html
    assert 'Persona 0 · Scenario 0' in html
    assert 'href="#conv-3"' in html
    assert 'No recommendation recorded; review the linked evidence.' in html


def test_simulation_brief_uses_recommendations_saved_on_run() -> None:
    run = SimpleNamespace(
        total_results=1,
        target='support-agent',
        target_model=None,
        target_kind='orq_agent',
        recommendations=[SimpleNamespace(result_index=0, suggestions=['Ask for the order number first.'])],
    )
    sections = {
        'summary': _section('summary', total_conversations=1, goals_achieved=0, goals_failed=1, errors=0),
        'individual_results': _section(
            'individual_results',
            entries=[
                {
                    'index': 0,
                    'persona': 'Bob',
                    'scenario': 'Refund',
                    'terminated_by': 'judge',
                    'goal_achieved': False,
                    'criteria': [],
                }
            ],
        ),
    }

    html = simulation_brief(run, sections)

    assert 'Ask for the order number first.' in html


def test_simulation_recommendations_do_not_turn_success_or_error_into_goal_failures() -> None:
    run = SimpleNamespace(
        total_results=2,
        target='agent',
        target_kind='orq_agent',
        target_model=None,
        recommendations=[
            SimpleNamespace(result_index=0, suggestions=['Consider a shorter greeting.']),
            SimpleNamespace(result_index=1, suggestions=['Review the partial response.']),
        ],
    )
    sections = {
        'summary': _section('summary', total_conversations=2, goals_achieved=1, goals_failed=0, errors=1),
        'individual_results': _section(
            'individual_results',
            entries=[
                {
                    'index': 0,
                    'persona': 'A',
                    'scenario': 'completed',
                    'terminated_by': 'judge',
                    'goal_achieved': True,
                    'criteria': [],
                },
                {
                    'index': 1,
                    'persona': 'B',
                    'scenario': 'partial',
                    'terminated_by': 'error',
                    'goal_achieved': False,
                    'error': 'provider failure',
                    'criteria': [],
                },
            ],
        ),
    }

    html = simulation_brief(run, sections)

    assert 'Recorded improvement suggestion' in html
    assert 'goal was achieved; the saved recommendation suggests a possible improvement' in html
    assert 'Execution did not complete' in html
    assert 'conversation ended before a complete outcome' in html
    assert 'The conversation did not achieve its goal.' not in html


def test_simulation_all_error_run_has_no_goal_verdict() -> None:
    run = SimpleNamespace(total_results=1, target='agent', target_kind='orq_agent', recommendations=[])
    sections = {
        'summary': _section('summary', total_conversations=1, goals_achieved=0, goals_failed=0, errors=1),
        'individual_results': _section('individual_results', entries=[]),
    }

    html = simulation_brief(run, sections)

    assert 'No goal verdict is available' in html
    assert '0/1 goals achieved' not in html


def test_simulation_brief_empty_report_is_explicit() -> None:
    run = SimpleNamespace(total_results=0, target='support-agent', target_model=None, target_kind='orq_agent')
    html = simulation_brief(run, {'summary': _section('summary', total_conversations=0)})

    assert 'No conversations were tested.' in html
    assert 'No confirmed findings' in html


def test_redteam_brief_uses_saved_guidance_and_neutral_missing_verdicts() -> None:
    result = SimpleNamespace(
        attack=SimpleNamespace(category='ASI01'),
        agent=SimpleNamespace(display_name='Agent A', key='agent-a'),
        vulnerable=True,
    )
    report = SimpleNamespace(
        total_results=2,
        results=[result, SimpleNamespace(attack=SimpleNamespace(category='LLM01'), vulnerable=None)],
        tested_agents=['Agent A'],
        categories_tested=['ASI01', 'LLM01'],
        pipeline=SimpleNamespace(value='dynamic'),
    )
    sections = {
        'summary': _section('summary', total_attacks=2, evaluated_attacks=1, vulnerabilities_found=1, total_errors=1),
        'focus_areas': _section(
            'focus_areas',
            focus_areas=[
                {
                    'category': 'ASI01',
                    'category_name': 'Goal hijacking',
                    'vulnerability_name': 'Goal hijacking',
                    'vulnerabilities_found': 1,
                    'llm_recommendations': {'recommendations': ['Keep untrusted instructions separate.']},
                }
            ],
        ),
        'category_breakdown': _section('category_breakdown', rows=[{'category': 'ASI01'}, {'category': 'LLM01'}]),
        'individual_results': _section('individual_results'),
    }

    html = redteam_brief(report, sections)

    assert '2 attacks tested' not in html
    assert '1/2 attacks received a verdict' in html
    assert 'Confirmed vulnerable attacks: 1;' in html
    assert 'Tested: Agent A (dynamic pipeline).' in html
    assert 'Agent A · Goal hijacking' in html
    assert 'No verdict' in html
    assert '2 tested categories' in html
    assert 'href="#attack-1"' in html


def test_redteam_brief_escapes_untrusted_saved_text() -> None:
    category = '<script>alert(1)</script>'
    report = SimpleNamespace(
        total_results=1,
        results=[
            SimpleNamespace(
                attack=SimpleNamespace(category=category),
                agent=SimpleNamespace(display_name='Agent', key='agent'),
                vulnerable=True,
            )
        ],
        tested_agents=[],
        categories_tested=[category],
        pipeline=SimpleNamespace(value='dynamic'),
    )
    sections = {
        'summary': _section('summary', total_attacks=1, evaluated_attacks=1, vulnerabilities_found=1),
        'focus_areas': _section(
            'focus_areas',
            focus_areas=[
                {
                    'category': category,
                    'category_name': category,
                    'vulnerability_name': 'unsafe',
                    'remediation': '',
                }
            ],
        ),
    }

    html = redteam_brief(report, sections)

    assert '&lt;script&gt;' in html
    assert '<script>' not in html
    assert 'No recommendation recorded; review the linked evidence.' in html


def test_sim_report_tabs_includes_saved_recommendation(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs
    from evaluatorq.simulation.contracts import SimulationRecommendation, SimulationRun
    from tests.dashboard.test_sim_transcript import _make_sim_run_with_transcript

    run = _make_sim_run_with_transcript()
    run.recommendations = [SimulationRecommendation(
        result_index=1,
        persona='bob',
        scenario='refund request',
        triggers=['Maximum turns reached'],
        suggestions=['Ask for the order number before checking refund eligibility.'],
    )]
    saved_run = SimulationRun.model_validate_json(run.model_dump_json())
    monkeypatch.chdir(tmp_path)
    html = sim_report_tabs('fixture', saved_run, standalone=True)

    assert 'Ask for the order number before checking refund eligibility.' in html
    assert 'Findings shown: 1 of 1.' in html


def test_simulation_recommendation_excerpt_is_bounded_and_links_full_section() -> None:
    recommendation = 'Specific saved action. ' + ('additional detail ' * 40)
    run = SimpleNamespace(
        total_results=1,
        target='agent',
        target_kind='orq_agent',
        target_model=None,
        recommendations=[SimpleNamespace(result_index=0, suggestions=[recommendation])],
    )
    sections = {
        'summary': _section('summary', total_conversations=1, goals_achieved=0, goals_failed=1, errors=0),
        'individual_results': _section(
            'individual_results',
            entries=[
                {
                    'index': 0,
                    'persona': 'A',
                    'scenario': 'B',
                    'terminated_by': 'judge',
                    'goal_achieved': False,
                    'criteria': [],
                }
            ],
        ),
    }

    html = simulation_brief(run, sections)

    assert 'Recorded recommendation excerpt:' in html
    assert 'Full recommendation</a>' in html
    assert f'href="#simtab-recommendations"' in html
    assert recommendation not in html
    assert 'data-label="Finding"' in html
    assert 'data-label="Recommended next step"' in html
    assert '<table class="no-stack">' in html
    assert 'class="export-finding-value"' in html
    assert 'class="export-brief-overview"' in html


def test_redteam_brief_counts_categories_beyond_the_five_displayed_focus_areas() -> None:
    from evaluatorq.redteam.reports.converters import compute_report_summary
    from evaluatorq.redteam.reports.export_html import export_html
    from evaluatorq.redteam.reports.sections import build_report_sections
    from tests.redteam.reports.test_export_regression import _make_minimal_report, _make_result

    report = _make_minimal_report()
    report.results = [_make_result(category=f'ASI{i:02d}', passed=False) for i in range(1, 7)]
    report.total_results = len(report.results)
    report.categories_tested = [result.attack.category for result in report.results]
    report.summary = compute_report_summary(report.results)
    sections = {section.kind: section for section in build_report_sections(report)}

    assert len(sections['focus_areas'].data['focus_areas']) == 5
    assert len(sections['focus_areas'].data['ranked_focus_areas']) == 6
    html = export_html(report)
    assert 'Findings shown: 3 of 6.' in html
    assert html.count('id="export-finding-') == 3
