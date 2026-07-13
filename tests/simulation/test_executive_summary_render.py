from __future__ import annotations

from evaluatorq.simulation.reports.export_html import export_html
from evaluatorq.simulation.reports.export_md import export_markdown
from tests.simulation.test_executive_summary_facts import _result

_NARRATIVE = 'Across 3 simulations the agent achieved 33% of goals but leaked PII twice.'


def _results():
    return [
        _result(goal=True, rules=[]),
        _result(goal=False, rules=['must_not_reveal_pii']),
        _result(goal=False, rules=['must_not_reveal_pii']),
    ]


def test_sim_html_includes_narrative_and_pill():
    html = export_html(_results(), executive_summary=_NARRATIVE)
    assert _NARRATIVE in html
    assert 'status-badge' in html
    assert 'CONFIDENCE' in html
    assert html.index(_NARRATIVE) < html.index('<div class="kpi-band">')


def test_sim_markdown_includes_narrative():
    md = export_markdown(_results(), executive_summary=_NARRATIVE)
    assert _NARRATIVE in md
