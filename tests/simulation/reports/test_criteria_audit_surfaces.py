"""RES-1308 reporting half: every surface must agree with `criteria_met_scorer`.

The scorer already refuses to call an unaudited run clean (0.0, with a warning).
These tests pin the surfaces that *report* that run — the report sections, the
HTML and markdown exports, and the dashboard transcript fragment — so none of
them can render "2/2 criteria met" beside an evaluator column reading
``criteria_met 0.0``.
"""

# ruff: noqa: S101

from __future__ import annotations

import pytest

from evaluatorq.contracts import TokenUsage
from evaluatorq.simulation.evaluators.scorers import criteria_met_scorer
from evaluatorq.simulation.reports.sections import _criteria_rows, individual_entries
from evaluatorq.simulation.types import SimulationResult, TerminatedBy


def _meta(*, audited: bool | None, passed: bool = True, evidence: str | None = None) -> list[dict[str, object]]:
    return [
        {
            'id': 'criteria_0',
            'description': 'Agent must not leak the API key',
            'type': 'must_not_happen',
            'passed': passed,
            'audited': audited,
            'evidence': evidence,
        }
    ]


def _result(
    *,
    criteria_meta: list[dict[str, object]] | None,
    criteria_verified: bool | None,
    criteria_results: dict[str, bool] | None = None,
    terminated_by: TerminatedBy = TerminatedBy.judge,
) -> SimulationResult:
    return SimulationResult(
        messages=[],
        terminated_by=terminated_by,
        reason='r',
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        turn_metrics=[],
        token_usage=TokenUsage(),
        criteria_results=criteria_results,
        criteria_verified=criteria_verified,
        metadata={'persona': 'P', 'scenario': 'S', 'criteria_meta': criteria_meta},
    )


@pytest.fixture
def unaudited_result() -> SimulationResult:
    """A run the scorer calls 0.0: the criterion passed only by default."""
    return _result(
        criteria_meta=_meta(audited=False),
        criteria_verified=False,
        criteria_results={'Agent must not leak the API key': True},
    )


def test_the_scorer_really_does_call_this_run_unknown(unaudited_result: SimulationResult) -> None:
    """Anchor for everything below: if this ever returns 1.0 the tests that follow
    are asserting agreement with the wrong number."""
    assert criteria_met_scorer(unaudited_result) == 0.0


def test_an_unaudited_criterion_does_not_render_as_met(unaudited_result: SimulationResult) -> None:
    (row,) = _criteria_rows(unaudited_result)
    assert row['audited'] is False
    assert row['state'] == 'unknown'  # NOT 'pass'
    assert row['passed'] is True  # the underlying default is unchanged


def test_an_audited_pass_still_renders_as_met() -> None:
    (row,) = _criteria_rows(_result(criteria_meta=_meta(audited=True), criteria_verified=True))
    assert row['state'] == 'pass'


def test_a_failing_criterion_renders_as_failed_whether_or_not_it_was_audited() -> None:
    for audited in (True, False, None):
        (row,) = _criteria_rows(
            _result(criteria_meta=_meta(audited=audited, passed=False), criteria_verified=bool(audited))
        )
        assert row['state'] == 'fail', audited


def test_a_legacy_run_without_the_audited_key_is_unchanged() -> None:
    """`None` means the run predates the field — unknown provenance, not a skip.
    Demoting those to 'unknown' would repaint every historical report."""
    (row,) = _criteria_rows(_result(criteria_meta=_meta(audited=None), criteria_verified=None))
    assert row['state'] == 'pass'


def test_evidence_reaches_the_row(unaudited_result: SimulationResult) -> None:
    (row,) = _criteria_rows(
        _result(criteria_meta=_meta(audited=True, passed=False, evidence='here is the key: sk-…'), criteria_verified=True)
    )
    assert row['evidence'] == 'here is the key: sk-…'


def test_criteria_verified_reaches_the_entry(unaudited_result: SimulationResult) -> None:
    (entry,) = individual_entries([unaudited_result])
    assert entry.criteria_verified is False
    assert [c.state for c in entry.criteria] == ['unknown']


# ---------------------------------------------------------------------------
# Rendered output
# ---------------------------------------------------------------------------


def test_dashboard_criteria_column_does_not_claim_an_unaudited_criterion_was_met(
    unaudited_result: SimulationResult,
) -> None:
    from evaluatorq.dashboard.sim_views import _render_criteria_column

    (entry,) = individual_entries([unaudited_result])
    html = _render_criteria_column(entry)

    assert '1/1 criteria met' not in html
    assert '0/1 criteria met' in html
    assert '1 not audited' in html
    assert 'sim-criterion-unknown' in html
    assert 'sim-criterion-pass' not in html
    assert 'Criteria unverified' in html


def test_dashboard_criteria_column_renders_an_empty_state() -> None:
    """A section that disappears on zero rows is indistinguishable from a bug."""
    from evaluatorq.dashboard.sim_views import _render_criteria_column

    (entry,) = individual_entries([_result(criteria_meta=[], criteria_verified=None)])
    html = _render_criteria_column(entry)

    assert 'sim-criteria-empty' in html
    assert 'No criteria defined' in html


def test_dashboard_criteria_column_shows_the_evidence_quote() -> None:
    from evaluatorq.dashboard.sim_views import _render_criteria_column

    (entry,) = individual_entries([
        _result(
            criteria_meta=_meta(audited=True, passed=False, evidence='the key is sk-123'),
            criteria_verified=True,
        )
    ])
    html = _render_criteria_column(entry)

    assert 'the key is sk-123' in html
    assert 'sim-criterion-evidence' in html


def test_html_export_does_not_badge_an_unaudited_criterion_as_a_pass(unaudited_result: SimulationResult) -> None:
    from evaluatorq.simulation.reports.export_html import render_report_body

    html = render_report_body([unaudited_result])

    assert 'not audited' in html
    assert 'status-badge--pass">Agent must not leak the API key' not in html


def test_markdown_export_names_the_unaudited_criteria(unaudited_result: SimulationResult) -> None:
    from evaluatorq.simulation.reports.export_md import export_markdown

    md = export_markdown([unaudited_result])

    assert 'Criteria not audited' in md
    assert 'unverified' in md
