"""Built-in evaluators for agent simulation.

These evaluators assess simulation results using a scorer pattern
compatible with evaluatorq integration.
"""

from __future__ import annotations

from collections.abc import Callable

from loguru import logger

from evaluatorq.simulation.types import SimulationResult, TerminatedBy

# A run that ended this way never reached the judge's criteria audit, so its
# criteria outcome is unknown — not met. Shared with `api._sim_evaluation_details`
# so the reported pass/fail cannot disagree with the score computed here.
UNEVALUATED_TERMINATIONS = (TerminatedBy.error, TerminatedBy.timeout)

SimulationScorer = Callable[[SimulationResult], float]


# ---------------------------------------------------------------------------
# Individual scorers
# ---------------------------------------------------------------------------


def goal_achieved_scorer(result: SimulationResult) -> float:
    """Evaluate if the simulation goal was achieved. Returns 1 if achieved, 0 otherwise."""
    return 1.0 if result.goal_achieved else 0.0


def criteria_met_scorer(result: SimulationResult) -> float:
    """Fraction of the scenario's criteria that were met, 0..1.

    A run that errored or timed out is scored **0.0**: it terminated before the
    judge could audit anything, so its criteria outcome is unknown, and scoring an
    unchecked run 1.0 let a dead target inflate the run average (RES-1308).
    A run whose `criteria_verified` is False scores 0.0 for the same reason: the
    judge never returned an occurrence audit, so the verdicts came from the
    free-text `rules_broken` list, which cannot fail a `must_happen` criterion.
    A run with no criteria at all still scores 1.0 — nothing to fail.

    An individual criterion the judge never audited is **not met** either, even on
    an otherwise verified run: it is passing only by its not-observed default, and
    that is the same flattering silence at criterion granularity. This is read from
    ``metadata['criteria_meta']`` (the only place ``audited`` survives —
    ``criteria_results`` is keyed by description and carries no provenance), so the
    score matches the ``N/M criteria met`` tally the reports print, which counts
    `CriteriaRow.state == 'pass'`. A criterion the judge settled early *is* audited
    (`_CriteriaTracker` records a verdict before it can become settled), so
    `JudgeAgent.mark_settled` never costs a run a point.

    A ``criteria_meta`` list with no usable (mapping) entry is an unknown shape:
    it warns and falls through to ``criteria_results`` rather than scoring the
    empty tally 1.0.
    """
    if result.terminated_by in UNEVALUATED_TERMINATIONS:
        logger.warning(
            'criteria_met: run terminated by {} before any criteria audit; scoring 0.0 (unknown, not met).',
            result.terminated_by.value,
        )
        return 0.0

    # `None` is a run saved before the flag existed — leave those scored as they were.
    if result.criteria_verified is False:
        logger.warning(
            'criteria_met: judge returned no per-criterion occurrence audit, so these verdicts cannot '
            'fail a must_happen criterion; scoring 0.0 (unverified, not met).'
        )
        return 0.0

    meta = result.metadata.get('criteria_meta')
    if isinstance(meta, list) and meta:
        entries = [c for c in meta if isinstance(c, dict)]
        if len(entries) != len(meta):
            logger.warning(
                'criteria_met: {} of {} criteria_meta entries are not mappings (e.g. {}); ignoring them.',
                len(meta) - len(entries),
                len(meta),
                type(next(c for c in meta if not isinstance(c, dict))).__name__,
            )
        # No usable entry is an unknown shape, not a perfect score: returning 1.0
        # here scored a run whose criteria_meta round-tripped as JSON strings a
        # clean 1.0 while criteria_results held real failures.
        if not entries:
            logger.warning(
                'criteria_met: criteria_meta holds {} entries but none is a mapping; falling back to '
                'criteria_results, which carries no audit provenance.',
                len(meta),
            )
        else:
            unaudited = [c for c in entries if c.get('audited') is False and c.get('passed')]
            if unaudited:
                logger.warning(
                    'criteria_met: {} of {} criteria passed only by default (the judge never audited them); '
                    'counting them as not met.',
                    len(unaudited),
                    len(entries),
                )
            met = sum(1 for c in entries if c.get('passed') and c.get('audited') is not False)
            return met / len(entries)

    criteria_results = result.criteria_results or {}
    if not criteria_results:
        return 1.0

    met = sum(1 for v in criteria_results.values() if v)
    return met / len(criteria_results)


def turn_efficiency_scorer(result: SimulationResult) -> float:
    """Evaluate conversation efficiency (fewer turns = better).

    Returns a value between 0 and 1.
    """
    total_turns = result.turn_count
    goal_achieved = result.goal_achieved

    if not goal_achieved:
        return 0.0

    if total_turns <= 2:
        return 1.0
    if total_turns <= 4:
        return 0.9
    if total_turns <= 6:
        return 0.7

    return max(0.3, 1.0 - (total_turns - 6) * 0.1)


def conversation_quality_scorer(result: SimulationResult) -> float:
    """Evaluate overall conversation quality.

    Composite score based on:
    - Goal achievement (40%)
    - Criteria met (30%)
    - Turn efficiency (30%)
    """
    goal_score = goal_achieved_scorer(result)
    criteria_score = criteria_met_scorer(result)
    efficiency_score = turn_efficiency_scorer(result)

    score = goal_score * 0.4 + criteria_score * 0.3 + efficiency_score * 0.3
    return round(score * 100) / 100


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SIMULATION_EVALUATORS: dict[str, SimulationScorer] = {
    'goal_achieved': goal_achieved_scorer,
    'criteria_met': criteria_met_scorer,
    'turn_efficiency': turn_efficiency_scorer,
    'conversation_quality': conversation_quality_scorer,
}


def get_evaluator(name: str) -> SimulationScorer:
    """Get a built-in simulation evaluator by name.

    Raises:
            ValueError: If evaluator not found.
    """
    evaluator = SIMULATION_EVALUATORS.get(name)
    if not evaluator:
        available = ', '.join(SIMULATION_EVALUATORS.keys())
        raise ValueError(f'Unknown evaluator: {name}. Available: {available}')
    return evaluator


def get_all_evaluators() -> dict[str, SimulationScorer]:
    """Get all built-in simulation evaluators."""
    return dict(SIMULATION_EVALUATORS)
