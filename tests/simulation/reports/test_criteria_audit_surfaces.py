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


class _FakeStreamlit:
    """Records what the Streamlit dashboard would render."""

    def __init__(self) -> None:
        self.markdowns: list[str] = []
        self.captions: list[str] = []
        self.warnings: list[str] = []

    def markdown(self, body: str) -> None:
        self.markdowns.append(body)

    def caption(self, body: str) -> None:
        self.captions.append(body)

    def warning(self, body: str) -> None:
        self.warnings.append(body)


def _render_sim_ui_criteria(result: SimulationResult, monkeypatch: pytest.MonkeyPatch) -> _FakeStreamlit:
    from evaluatorq.simulation.ui import dashboard as sim_ui

    fake = _FakeStreamlit()
    monkeypatch.setattr(sim_ui, 'st', fake)
    (entry,) = individual_entries([result])
    sim_ui._render_criteria(entry.model_dump(mode='json'))
    return fake


def test_sim_ui_dashboard_does_not_green_tick_an_unaudited_criterion(
    unaudited_result: SimulationResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`eq sim ui` is live (simulation/cli.py -> launch_streamlit) and was the third
    renderer still keying on `passed`: a ✅ beside `criteria_met = 0.0`."""
    fake = _render_sim_ui_criteria(unaudited_result, monkeypatch)

    (criterion_line,) = [m for m in fake.markdowns if 'API key' in m]
    assert '✅' not in criterion_line
    assert criterion_line.startswith('❓')
    assert 'not audited' in criterion_line
    assert fake.captions == ['0/1 criteria met · 1 not audited']
    assert any('Criteria unverified' in w for w in fake.warnings)


def test_sim_ui_dashboard_still_green_ticks_an_audited_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _render_sim_ui_criteria(
        _result(criteria_meta=_meta(audited=True), criteria_verified=True), monkeypatch
    )

    (criterion_line,) = [m for m in fake.markdowns if 'API key' in m]
    assert criterion_line.startswith('✅')
    assert fake.captions == ['1/1 criteria met']
    assert fake.warnings == []


def test_sim_ui_dashboard_shows_the_evidence_quote(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _render_sim_ui_criteria(
        _result(
            criteria_meta=_meta(audited=True, passed=False, evidence='the key is sk-123'),
            criteria_verified=True,
        ),
        monkeypatch,
    )

    (criterion_line,) = [m for m in fake.markdowns if 'API key' in m]
    assert criterion_line.startswith('⛔')  # must_not_happen violation
    assert 'the key is sk-123' in criterion_line


def test_sim_ui_dashboard_renders_an_empty_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """A section that disappears on zero rows is indistinguishable from a bug."""
    fake = _render_sim_ui_criteria(_result(criteria_meta=[], criteria_verified=None), monkeypatch)

    assert fake.captions == ['No criteria defined for this scenario.']


# ---------------------------------------------------------------------------
# The tally and the scorer must agree (both count only audited passes)
# ---------------------------------------------------------------------------


def _two_criteria(*, second_audited: bool) -> list[dict[str, object]]:
    return [
        {
            'id': 'criteria_0',
            'description': 'Agent greets the customer',
            'type': 'must_happen',
            'passed': True,
            'audited': True,
            'evidence': 'hello there',
        },
        {
            'id': 'criteria_1',
            'description': 'Agent must not leak the API key',
            'type': 'must_not_happen',
            'passed': True,
            'audited': second_audited,
            'evidence': '',
        },
    ]


def test_the_scorer_does_not_count_an_unaudited_criterion_on_a_verified_run() -> None:
    """A *verified* run whose judge audited A and skipped B: the dashboard printed
    "1/2 criteria met · 1 not audited" while the evaluator column read
    `criteria_met 1.00 PASS`. Same rule on both sides now."""
    result = _result(criteria_meta=_two_criteria(second_audited=False), criteria_verified=True)

    assert criteria_met_scorer(result) == 0.5

    from evaluatorq.dashboard.sim_views import _render_criteria_column

    (entry,) = individual_entries([result])
    assert '1/2 criteria met' in _render_criteria_column(entry)
    assert '1 not audited' in _render_criteria_column(entry)


def test_the_scorer_counts_an_audited_pass() -> None:
    result = _result(criteria_meta=_two_criteria(second_audited=True), criteria_verified=True)
    assert criteria_met_scorer(result) == 1.0


def test_a_legacy_run_without_audited_provenance_is_scored_as_before() -> None:
    """``audited: None`` predates the field — unknown, not "the judge skipped it".
    Those runs keep the score they had."""
    meta = _two_criteria(second_audited=True)
    for c in meta:
        c['audited'] = None
    assert criteria_met_scorer(_result(criteria_meta=meta, criteria_verified=None)) == 1.0


def test_the_evaluator_explanation_agrees_with_the_scorer() -> None:
    from evaluatorq.simulation.api import _sim_evaluation_details

    result = _result(criteria_meta=_two_criteria(second_audited=False), criteria_verified=True)
    explanation, passed = _sim_evaluation_details('criteria_met', result)

    assert passed is False  # was True, beside a score of 0.5
    assert explanation is not None
    assert 'UNKNOWN [prohibited]: Agent must not leak the API key (not audited)' in explanation
    assert 'PASS [required]: Agent greets the customer' in explanation


# ---------------------------------------------------------------------------
# A malformed audit announces itself (CLAUDE.md: a degraded path warns)
# ---------------------------------------------------------------------------


def test_a_malformed_audited_value_degrades_to_unknown_with_a_warning() -> None:
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m), level='WARNING')
    try:
        result = _result(criteria_meta=_meta(audited='yes'), criteria_verified=True)  # pyright: ignore[reportArgumentType]
        rows = _criteria_rows(result)
    finally:
        logger.remove(sink_id)

    assert rows == []
    assert result.metadata['criteria_errors']
    assert any('criteria_meta entry is invalid' in m for m in messages)


def test_a_criteria_meta_of_non_mappings_is_an_error_not_a_silent_pass() -> None:
    """A malformed entry is an error and makes the criteria verdict unknown."""
    from loguru import logger

    result = _result(
        criteria_meta=['{"id": "criteria_0", "passed": false}', '{"id": "criteria_1", "passed": true}'],  # pyright: ignore[reportArgumentType]
        criteria_verified=True,
        criteria_results={'Agent greets the customer': False, 'Agent must not leak the API key': True},
    )

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m), level='WARNING')
    try:
        score = criteria_met_scorer(result)
    finally:
        logger.remove(sink_id)

    assert score == 0.0
    assert any('criteria_meta entry is invalid' in m for m in messages)


def test_a_partly_malformed_criteria_meta_is_an_error_even_with_valid_entries() -> None:
    """One malformed entry prevents a mixed list from claiming a clean pass."""
    from loguru import logger

    meta = [*_meta(audited=True), 'not a mapping']
    result = _result(criteria_meta=meta, criteria_verified=True)  # pyright: ignore[reportArgumentType]

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m), level='WARNING')
    try:
        score = criteria_met_scorer(result)
    finally:
        logger.remove(sink_id)

    assert score == 0.0
    assert any('criteria_meta entry is invalid' in m for m in messages)


def test_the_evaluator_detail_marks_a_non_mapping_as_unknown() -> None:
    """The evaluator detail must not turn an unreadable entry into a pass."""
    from evaluatorq.simulation.api import _sim_evaluation_details

    result = _result(
        criteria_meta=['{"id": "criteria_0"}'],  # pyright: ignore[reportArgumentType]
        criteria_verified=True,
        criteria_results={'Agent greets the customer': False},
    )

    explanation, passed = _sim_evaluation_details('criteria_met', result)

    assert passed is None
    assert explanation is not None
    assert 'ERROR: invalid criteria_meta entry' in explanation


def test_the_evaluator_detail_reports_mixed_meta_as_unknown() -> None:
    """A valid entry beside malformed metadata is still an unknown verdict."""
    from evaluatorq.simulation.api import _sim_evaluation_details

    result = _result(criteria_meta=[*_meta(audited=True), 'not a mapping'], criteria_verified=True)  # pyright: ignore[reportArgumentType]

    explanation, passed = _sim_evaluation_details('criteria_met', result)

    assert passed is None
    assert explanation is not None
    assert 'ERROR: invalid criteria_meta entry' in explanation
    assert 'PASS [prohibited]: Agent must not leak the API key' in explanation


def test_malformed_criteria_meta_surfaces_in_the_errors_report() -> None:
    from evaluatorq.simulation.reports.sections import build_report_sections

    result = _result(criteria_meta=[*_meta(audited=True), 'garbage'], criteria_verified=True)  # pyright: ignore[reportArgumentType]
    sections = build_report_sections([result])

    errors = next(section for section in sections if section.kind == 'errors')
    assert errors.data['total_errored'] == 1
    assert any('criteria_meta entry is invalid' in message for message in errors.data['by_message'])
