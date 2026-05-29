"""Renderer-agnostic section data layer for agent simulation reports.

``build_report_sections(results)`` converts a ``list[SimulationResult]`` into
a list of ``ReportSection`` objects consumed by the Markdown / HTML
renderers. Mirrors the structure of ``redteam.reports.sections`` so the same
shared dispatch in ``evaluatorq.common.reports`` can drive both flavours.

Section kinds:
    - ``summary``               aggregate goal-achieved / score statistics
    - ``persona_breakdown``     per-persona success rates and token usage
    - ``scenario_breakdown``    per-scenario success rates and judge stats
    - ``judge_verdicts``        terminated-by reasons, rules broken, top reasons
    - ``turn_metrics``          turn-count distribution + average per-turn scores
    - ``evaluator_scores``      mean per-evaluator scores when present
    - ``token_usage``           prompt/completion/total + per-conversation summary
    - ``individual_results``    one entry per ``SimulationResult`` (transcript)
    - ``errors``                count by error type for error-terminated runs
"""

from __future__ import annotations

import operator
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

from evaluatorq.contracts import ReportSection

if TYPE_CHECKING:
    from evaluatorq.simulation.types import SimulationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _persona_name(result: SimulationResult) -> str:
    return str(result.metadata.get("persona", "unknown"))


def _scenario_name(result: SimulationResult) -> str:
    return str(result.metadata.get("scenario", "unknown"))


def _model_name(result: SimulationResult) -> str:
    return str(result.metadata.get("model", "unknown"))


def _evaluator_scores(result: SimulationResult) -> dict[str, float]:
    raw = result.metadata.get("evaluator_scores")
    if not isinstance(raw, dict):
        return {}
    return {str(k): float(v) for k, v in raw.items() if isinstance(v, int | float)}


def _error_message(result: SimulationResult) -> str | None:
    err = result.metadata.get("error")
    return str(err) if err else None


def _is_errored(result: SimulationResult) -> bool:
    """A run is errored if it has an error metadata key or the judge terminated
    it due to an error. Shared between the summary and errors sections so the
    "errors" counts agree across the report.
    """
    return bool(_error_message(result)) or result.terminated_by.value == "error"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_summary_section(results: list[SimulationResult]) -> ReportSection:
    total = len(results)
    achieved = sum(1 for r in results if r.goal_achieved and not _is_errored(r))
    errored = sum(1 for r in results if _is_errored(r))
    avg_score = sum(r.goal_completion_score for r in results) / total if total else 0.0
    avg_turns = sum(r.turn_count for r in results) / total if total else 0.0
    total_tokens = sum(r.token_usage.total_tokens for r in results)
    success_rate = (achieved / total) if total else 0.0

    if success_rate >= 0.80:
        confidence = "HIGH"
        confidence_note = f"{achieved}/{total} goals achieved"
    elif success_rate >= 0.50:
        confidence = "MEDIUM"
        confidence_note = f"{achieved}/{total} goals achieved"
    else:
        confidence = "LOW"
        confidence_note = f"only {achieved}/{total} goals achieved"

    return ReportSection(
        kind="summary",
        title="Executive Summary",
        data={
            "total_conversations": total,
            "goals_achieved": achieved,
            "goals_failed": total - achieved - errored,
            "errors": errored,
            "success_rate": success_rate,
            "avg_goal_completion_score": avg_score,
            "avg_turn_count": avg_turns,
            "total_tokens": total_tokens,
            "confidence": confidence,
            "confidence_note": confidence_note,
        },
    )


def _build_persona_breakdown_section(results: list[SimulationResult]) -> ReportSection:
    by_persona: dict[str, list[SimulationResult]] = defaultdict(list)
    for r in results:
        by_persona[_persona_name(r)].append(r)

    rows: list[dict[str, Any]] = []
    for name, items in by_persona.items():
        total = len(items)
        achieved = sum(1 for r in items if r.goal_achieved and not _is_errored(r))
        avg_score = sum(r.goal_completion_score for r in items) / total
        tokens = sum(r.token_usage.total_tokens for r in items)
        rows.append({
            "persona": name,
            "conversations": total,
            "goals_achieved": achieved,
            "success_rate": achieved / total,
            "avg_goal_completion_score": avg_score,
            "total_tokens": tokens,
        })
    rows.sort(key=operator.itemgetter("success_rate"))
    return ReportSection(
        kind="persona_breakdown",
        title="Per-Persona Breakdown",
        data={"rows": rows},
    )


def _build_scenario_breakdown_section(results: list[SimulationResult]) -> ReportSection:
    by_scenario: dict[str, list[SimulationResult]] = defaultdict(list)
    for r in results:
        by_scenario[_scenario_name(r)].append(r)

    rows: list[dict[str, Any]] = []
    for name, items in by_scenario.items():
        total = len(items)
        achieved = sum(1 for r in items if r.goal_achieved and not _is_errored(r))
        avg_score = sum(r.goal_completion_score for r in items) / total
        avg_turns = sum(r.turn_count for r in items) / total
        rows.append({
            "scenario": name,
            "conversations": total,
            "goals_achieved": achieved,
            "success_rate": achieved / total,
            "avg_goal_completion_score": avg_score,
            "avg_turn_count": avg_turns,
        })
    rows.sort(key=operator.itemgetter("success_rate"))
    return ReportSection(
        kind="scenario_breakdown",
        title="Per-Scenario Breakdown",
        data={"rows": rows},
    )


def _build_judge_verdicts_section(results: list[SimulationResult]) -> ReportSection:
    by_terminated_by: Counter[str] = Counter(r.terminated_by.value for r in results)
    all_rules_broken: Counter[str] = Counter()
    for r in results:
        all_rules_broken.update(r.rules_broken)

    return ReportSection(
        kind="judge_verdicts",
        title="Judge Verdicts",
        data={
            "terminated_by": dict(by_terminated_by),
            "rules_broken": dict(all_rules_broken.most_common(15)),
            "total_rules_broken_instances": sum(all_rules_broken.values()),
        },
    )


def _build_turn_metrics_section(results: list[SimulationResult]) -> ReportSection:
    turn_counts: Counter[int] = Counter(r.turn_count for r in results)
    # Aggregate per-turn quality metrics (averages across all runs that reported them).
    # Use direct attribute access on TurnMetrics so a future rename surfaces
    # as AttributeError instead of silently dropping the field from the report.
    qualities: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for tm in r.turn_metrics:
            for field_name, value in (
                ("response_quality", tm.response_quality),
                ("hallucination_risk", tm.hallucination_risk),
                ("tone_appropriateness", tm.tone_appropriateness),
                ("factual_accuracy", tm.factual_accuracy),
            ):
                if value is not None:
                    qualities[field_name].append(float(value))
    avg_qualities = {k: sum(v) / len(v) for k, v in qualities.items() if v}

    return ReportSection(
        kind="turn_metrics",
        title="Turn Metrics",
        data={
            "turn_count_distribution": dict(sorted(turn_counts.items())),
            "avg_quality_metrics": avg_qualities,
        },
    )


def _build_evaluator_scores_section(results: list[SimulationResult]) -> ReportSection | None:
    by_evaluator: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for name, score in _evaluator_scores(r).items():
            by_evaluator[name].append(score)
    if not by_evaluator:
        return None
    rows = [
        {
            "evaluator": name,
            "runs": len(values),
            "mean_score": sum(values) / len(values),
            "min_score": min(values),
            "max_score": max(values),
        }
        for name, values in by_evaluator.items()
    ]
    rows.sort(key=operator.itemgetter("evaluator"))
    return ReportSection(
        kind="evaluator_scores",
        title="Evaluator Scores",
        data={"rows": rows},
    )


def _build_token_usage_section(results: list[SimulationResult]) -> ReportSection:
    prompt = sum(r.token_usage.prompt_tokens for r in results)
    completion = sum(r.token_usage.completion_tokens for r in results)
    total = sum(r.token_usage.total_tokens for r in results)
    n = len(results) or 1
    return ReportSection(
        kind="token_usage",
        title="Token Usage",
        data={
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "avg_total_per_conversation": total / n,
            "avg_prompt_per_conversation": prompt / n,
            "avg_completion_per_conversation": completion / n,
        },
    )


def _build_individual_results_section(results: list[SimulationResult]) -> ReportSection:
    entries: list[dict[str, Any]] = []
    for idx, r in enumerate(results):
        entries.append({
            "index": idx,
            "persona": _persona_name(r),
            "scenario": _scenario_name(r),
            "model": _model_name(r),
            "terminated_by": r.terminated_by.value,
            "goal_achieved": r.goal_achieved,
            "goal_completion_score": r.goal_completion_score,
            "rules_broken": list(r.rules_broken),
            "turn_count": r.turn_count,
            "total_tokens": r.token_usage.total_tokens,
            "judge_reason": r.reason,
            "error": _error_message(r),
            "evaluator_scores": _evaluator_scores(r),
            "transcript": [
                {"role": m.role, "content": m.content or ""}
                for m in r.messages
            ],
        })
    return ReportSection(
        kind="individual_results",
        title="Individual Conversations",
        data={"entries": entries},
    )


def _build_errors_section(results: list[SimulationResult]) -> ReportSection | None:
    errored = [r for r in results if _is_errored(r)]
    if not errored:
        return None
    err_messages = [
        (_error_message(r) or r.reason or "unknown")
        for r in errored
    ]
    by_message: Counter[str] = Counter(err_messages)
    return ReportSection(
        kind="errors",
        title="Errors",
        data={
            "total_errored": len(errored),
            "by_message": dict(by_message.most_common(10)),
        },
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def build_report_sections(results: list[SimulationResult]) -> list[ReportSection]:
    """Produce the ordered list of report sections from simulation results."""
    sections: list[ReportSection] = [
        _build_summary_section(results),
        _build_persona_breakdown_section(results),
        _build_scenario_breakdown_section(results),
        _build_judge_verdicts_section(results),
        _build_turn_metrics_section(results),
    ]
    evaluator = _build_evaluator_scores_section(results)
    if evaluator is not None:
        sections.append(evaluator)
    sections.append(_build_token_usage_section(results))
    errors = _build_errors_section(results)
    if errors is not None:
        sections.append(errors)
    sections.append(_build_individual_results_section(results))
    return sections
