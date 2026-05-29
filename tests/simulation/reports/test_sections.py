"""Tests for simulation/reports/sections.py (RES-846)."""

from __future__ import annotations

from evaluatorq.contracts import Message, TokenUsage
from evaluatorq.simulation.reports.sections import build_report_sections
from evaluatorq.simulation.types import SimulationResult, TerminatedBy, TurnMetrics


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


def test_individual_results_section_carries_transcript_and_meta():
    results = [_make_result(persona="A", goal_achieved=True)]
    sections = build_report_sections(results)
    individual = next(s for s in sections if s.kind == "individual_results")
    entry = individual.data["entries"][0]
    assert entry["persona"] == "A"
    assert entry["goal_achieved"] is True
    assert entry["transcript"][0]["role"] == "user"
    assert entry["transcript"][1]["role"] == "assistant"
