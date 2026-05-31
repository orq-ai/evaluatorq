"""Tests for simulation/reports/sections.py (RES-846)."""

from __future__ import annotations

import pytest

from evaluatorq.contracts import Message, TokenUsage
from evaluatorq.simulation.reports.sections import build_report_sections, _criteria_rows
from evaluatorq.simulation.types import SimulationResult, TerminatedBy, TurnMetrics


@pytest.fixture
def make_result():
	def _make(*, goal_achieved=True, score=1.0, persona='P', scenario='S',
			  criteria_meta=None, turn_count=1, terminated_by=TerminatedBy.judge):
		meta = {'persona': persona, 'scenario': scenario}
		if criteria_meta is not None:
			meta['criteria_meta'] = criteria_meta
		return SimulationResult(
			messages=[], terminated_by=terminated_by, reason='r',
			goal_achieved=goal_achieved, goal_completion_score=score,
			rules_broken=[], turn_count=turn_count, turn_metrics=[],
			token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
			metadata=meta,
		)
	return _make


def _make_result(
    *,
    persona: str = "Tester",
    scenario: str = "Smoke",
    goal_achieved: bool = True,
    goal_completion_score: float = 1.0,
    rules_broken: list[str] | None = None,
    turn_count: int = 1,
    tokens: tuple[int, int, int] = (10, 5, 15),
    evaluator_scores: dict[str, float] | None = None,
    error: str | None = None,
    terminated_by: TerminatedBy = TerminatedBy.judge,
) -> SimulationResult:
    metadata: dict[str, object] = {
        "persona": persona,
        "scenario": scenario,
        "model": "gpt-4o-mini",
    }
    if evaluator_scores is not None:
        metadata["evaluator_scores"] = evaluator_scores
    if error is not None:
        metadata["error"] = error
    return SimulationResult(
        messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ],
        terminated_by=terminated_by,
        reason="done",
        goal_achieved=goal_achieved,
        goal_completion_score=goal_completion_score,
        rules_broken=rules_broken or [],
        turn_count=turn_count,
        token_usage=TokenUsage(
            prompt_tokens=tokens[0],
            completion_tokens=tokens[1],
            total_tokens=tokens[2],
        ),
        turn_metrics=[
            TurnMetrics(turn_number=1, token_usage=TokenUsage(), judge_reason="ok")
        ],
        metadata=metadata,
    )


def test_build_report_sections_emits_core_sections():
    results = [_make_result(), _make_result(goal_achieved=False, goal_completion_score=0.2)]
    sections = build_report_sections(results)
    kinds = [s.kind for s in sections]
    # Mandatory kinds, in order.
    for kind in (
        "summary",
        "persona_breakdown",
        "scenario_breakdown",
        "judge_verdicts",
        "turn_metrics",
        "token_usage",
        "individual_results",
    ):
        assert kind in kinds, f"missing {kind}: {kinds}"
    # Optional kinds: not present without data.
    assert "evaluator_scores" not in kinds
    assert "errors" not in kinds


def test_summary_section_computes_success_rate_and_tokens():
    results = [
        _make_result(goal_achieved=True, tokens=(10, 5, 15)),
        _make_result(goal_achieved=False, tokens=(20, 10, 30)),
    ]
    sections = build_report_sections(results)
    summary = next(s for s in sections if s.kind == "summary")
    assert summary.data["total_conversations"] == 2
    assert summary.data["goals_achieved"] == 1
    assert summary.data["success_rate"] == 0.5
    assert summary.data["total_tokens"] == 45


def test_persona_breakdown_aggregates_per_persona():
    results = [
        _make_result(persona="A", goal_achieved=True),
        _make_result(persona="A", goal_achieved=False, goal_completion_score=0.0),
        _make_result(persona="B", goal_achieved=True),
    ]
    sections = build_report_sections(results)
    persona = next(s for s in sections if s.kind == "persona_breakdown")
    rows = {r["persona"]: r for r in persona.data["rows"]}
    assert rows["A"]["conversations"] == 2
    assert rows["A"]["goals_achieved"] == 1
    assert rows["A"]["success_rate"] == 0.5
    assert rows["B"]["success_rate"] == 1.0


def test_judge_verdicts_section_counts_terminated_by_and_rules():
    results = [
        _make_result(terminated_by=TerminatedBy.judge, rules_broken=["rude", "off-topic"]),
        _make_result(terminated_by=TerminatedBy.max_turns, rules_broken=["rude"]),
    ]
    sections = build_report_sections(results)
    verdict = next(s for s in sections if s.kind == "judge_verdicts")
    assert verdict.data["terminated_by"]["judge"] == 1
    assert verdict.data["terminated_by"]["max_turns"] == 1
    assert verdict.data["rules_broken"]["rude"] == 2
    assert verdict.data["rules_broken"]["off-topic"] == 1


def test_evaluator_scores_section_present_when_scores_attached():
    results = [
        _make_result(evaluator_scores={"goal_achieved": 1.0, "criteria_met": 0.8}),
        _make_result(evaluator_scores={"goal_achieved": 0.0, "criteria_met": 0.5}),
    ]
    sections = build_report_sections(results)
    evaluators = next(s for s in sections if s.kind == "evaluator_scores")
    rows = {r["evaluator"]: r for r in evaluators.data["rows"]}
    assert rows["goal_achieved"]["mean_score"] == 0.5
    assert rows["criteria_met"]["mean_score"] == 0.65
    assert rows["goal_achieved"]["min_score"] == 0.0
    assert rows["goal_achieved"]["max_score"] == 1.0


def test_errors_section_present_when_failures_present():
    results = [
        _make_result(error="rate limit"),
        _make_result(terminated_by=TerminatedBy.error, error="upstream timeout"),
        _make_result(),
    ]
    sections = build_report_sections(results)
    errors = next(s for s in sections if s.kind == "errors")
    assert errors.data["total_errored"] == 2
    assert errors.data["by_message"]["rate limit"] == 1


def test_persona_breakdown_excludes_errored_runs_from_achieved():
    """Per-persona success rate must not count errored runs as achieved.

    Regression: a result with both ``goal_achieved=True`` and an error
    metadata key would previously count in the persona breakdown's
    ``goals_achieved`` but not in the summary's. The per-persona rate
    could paradoxically exceed the overall rate.
    """
    results = [
        _make_result(persona="A", goal_achieved=True),                   # achieved
        _make_result(persona="A", goal_achieved=True, error="oh no"),    # errored, not achieved
    ]
    sections = build_report_sections(results)
    persona = next(s for s in sections if s.kind == "persona_breakdown")
    row_a = next(r for r in persona.data["rows"] if r["persona"] == "A")
    assert row_a["conversations"] == 2
    assert row_a["goals_achieved"] == 1
    assert row_a["success_rate"] == 0.5

    summary = next(s for s in sections if s.kind == "summary")
    # The persona rate must not exceed the overall rate for the same data.
    assert row_a["success_rate"] <= summary.data["success_rate"] + 1e-9 or \
           summary.data["success_rate"] == row_a["success_rate"]
    assert summary.data["goals_achieved"] == row_a["goals_achieved"]


def test_scenario_breakdown_excludes_errored_runs_from_achieved():
    """Same partition discipline as persona breakdown."""
    results = [
        _make_result(scenario="X", goal_achieved=True),
        _make_result(scenario="X", goal_achieved=True, error="boom"),
    ]
    sections = build_report_sections(results)
    scenario = next(s for s in sections if s.kind == "scenario_breakdown")
    row_x = next(r for r in scenario.data["rows"] if r["scenario"] == "X")
    assert row_x["goals_achieved"] == 1
    assert row_x["success_rate"] == 0.5


def test_summary_partitions_achieved_failed_errored_disjointly():
    """An errored run never counts as achieved, and goals_failed is never negative.

    Regression: a result with both ``goal_achieved=True`` and a metadata
    ``error`` would previously be double-counted (in achieved AND errored),
    producing ``goals_failed = total - achieved - errored < 0``.
    """
    results = [
        _make_result(goal_achieved=True),                  # clean win
        _make_result(goal_achieved=False),                 # clean loss
        _make_result(goal_achieved=True, error="weird"),   # achieved + error -> errored
        _make_result(terminated_by=TerminatedBy.error),    # terminated_by=error
    ]
    sections = build_report_sections(results)
    summary = next(s for s in sections if s.kind == "summary")
    assert summary.data["goals_achieved"] == 1
    assert summary.data["errors"] == 2
    assert summary.data["goals_failed"] == 1
    assert (
        summary.data["goals_achieved"]
        + summary.data["errors"]
        + summary.data["goals_failed"]
        == summary.data["total_conversations"]
    )


def test_errors_section_count_matches_summary_section_count():
    """Both sections must agree on what 'errored' means.

    Regression: the summary counted only metadata['error'] results, while the
    errors section also counted ``terminated_by == error``.
    """
    results = [
        _make_result(error="rate limit"),
        _make_result(terminated_by=TerminatedBy.error),  # no metadata error
        _make_result(),
    ]
    sections = build_report_sections(results)
    summary = next(s for s in sections if s.kind == "summary")
    errors = next(s for s in sections if s.kind == "errors")
    assert summary.data["errors"] == errors.data["total_errored"] == 2


def test_individual_results_section_carries_transcript_and_meta():
    results = [_make_result(persona="A", goal_achieved=True)]
    sections = build_report_sections(results)
    individual = next(s for s in sections if s.kind == "individual_results")
    entry = individual.data["entries"][0]
    assert entry["persona"] == "A"
    assert entry["goal_achieved"] is True
    assert entry["transcript"][0]["role"] == "user"
    assert entry["transcript"][1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# Task 4.1 — criteria_meta accessor + summary verdict
# ---------------------------------------------------------------------------


def test_summary_section_has_hero_kpis(make_result):
	results = [make_result(goal_achieved=True, score=1.0), make_result(goal_achieved=False, score=0.0)]
	summary = next(s for s in build_report_sections(results) if s.kind == 'summary')
	d = summary.data
	assert d['total_conversations'] == 2
	assert d['goals_achieved'] == 1
	assert d['success_rate'] == 0.5
	assert 'verdict' in d  # "pass" | "warn" | "fail"


def test_criteria_rows_uses_meta_ids_not_descriptions(make_result):
	r = make_result(
		goal_achieved=False, score=0.0,
		criteria_meta=[
			{'id': 'criteria_0', 'description': 'explain charge', 'type': 'must_happen', 'passed': False},
		],
	)
	rows = _criteria_rows(r)
	assert rows[0]['id'] == 'criteria_0'
	assert rows[0]['description'] == 'explain charge'
	assert rows[0]['passed'] is False
	assert rows[0]['safety'] is False  # must_happen miss is not a safety violation
