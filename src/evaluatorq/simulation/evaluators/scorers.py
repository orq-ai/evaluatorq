"""Built-in evaluators for agent simulation.

These evaluators assess simulation results using a scorer pattern
compatible with evaluatorq integration.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evaluatorq.simulation.types import CriteriaMeta, SimulationResult, TerminatedBy, parse_criteria_meta

# A run that ended this way never reached the judge's criteria audit, so its
# criteria outcome is unknown — not met. Shared with `api._sim_evaluation_details`
# so the reported pass/fail cannot disagree with the score computed here.
UNEVALUATED_TERMINATIONS = (TerminatedBy.error, TerminatedBy.timeout)


def failure_reason(result: SimulationResult) -> str | None:
    """The reason a run ended before the judge could audit it, or ``None`` if it did.

    One derivation for every caller. `simulate()`'s job read `metadata['error']` first
    while `wrap_simulation_agent`'s job read `reason` alone, so the same dead run
    described itself two ways depending on the entry point. The fallback keeps the
    string non-empty: a job's `error` key is a failure signal, and an empty one reads
    downstream as a clean row.
    """
    if result.terminated_by not in UNEVALUATED_TERMINATIONS:
        return None
    metadata_error = result.metadata.get('error')
    return str(metadata_error or result.reason or '') or f'simulation terminated by {result.terminated_by.value}'


SimulationScorer = Callable[[SimulationResult], float]

_WEIGHT_SUM_TOLERANCE = 1e-9


class SimulationScoringConfig(BaseModel):
    """Tunables for the two built-in scorers that encode a *policy*, not a fact.

    ``goal_achieved`` and ``criteria_met`` read the judge's verdicts and have nothing to
    tune. The other two make judgement calls this class exposes, mirroring
    ``SimulationRecommendationConfig``: bounded fields, ``extra='forbid'``, and defaults
    that carry over the previously hardcoded cliffs verbatim. The one deliberate
    difference is the tail past the last cliff, which now decays from that cliff's score
    instead of restarting from ``1.0`` — the old curve scored 7 turns higher than 6, so a
    run past the last cliff scores lower than it used to. See ``CHANGELOG.md``. Pass an
    instance as ``simulate(scoring=...)`` /
    ``generate_and_simulate(scoring=...)``, or to ``get_evaluator`` /
    ``get_all_evaluators`` when driving the scorers directly.

    **What `turn_efficiency` measures.** It is a *cost* proxy, not a quality one: given
    that the goal was reached, how many conversational turns did it take? The assumption
    is that a user who got what they came for in two turns had a better experience — and
    cost less to serve — than one who needed twelve, because the extra turns are usually
    the agent asking for something it could have inferred, re-asking, or wandering. A run
    that did **not** reach the goal scores ``0.0`` outright: efficiency at failing is not
    a virtue worth crediting.

    **Where that assumption breaks.** A task that legitimately needs many turns — a long
    intake form, a multi-step troubleshooting tree, a negotiation — is penalised by this
    metric for doing its job properly. If your scenarios look like that, either move the
    cliffs out (``turn_efficiency_cliffs=((6, 1.0), (10, 0.9), (16, 0.7))``) so the curve
    matches a realistic conversation length, or leave ``turn_efficiency`` out of
    ``evaluator_names`` and ignore the score. It is not in
    ``DEFAULT_EVALUATOR_NAMES`` precisely because it is not universally meaningful.

    Worked example — a 4-turn conversation that reached its goal and met 1 of 2 criteria,
    scored with the defaults:

    ```python
    from evaluatorq.simulation.evaluators import SimulationScoringConfig, conversation_quality_scorer

    # goal_achieved   = 1.0    (the judge set goal_achieved)
    # criteria_met    = 0.5    (1 of 2 criteria met)
    # turn_efficiency = 0.9    (4 turns: past the <=2 cliff, inside the <=4 one)
    #
    # conversation_quality = 1.0 * 0.4 + 0.5 * 0.3 + 0.9 * 0.3
    #                      = 0.4 + 0.15 + 0.27 = 0.82
    conversation_quality_scorer(result)  # 0.82

    # Same conversation, scored as if only the goal mattered:
    goal_only = SimulationScoringConfig(
        goal_achieved_weight=1.0, criteria_met_weight=0.0, turn_efficiency_weight=0.0
    )
    conversation_quality_scorer(result, goal_only)  # 1.0
    ```
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    turn_efficiency_cliffs: tuple[tuple[int, float], ...] = ((2, 1.0), (4, 0.9), (6, 0.7))
    """``(max_turns, score)`` steps, first match wins: a conversation of ``turn_count``
    turns scores the first entry whose ``max_turns`` it does not exceed. Must be ordered
    by strictly increasing ``max_turns`` and non-increasing ``score`` — a curve that pays
    *more* for a longer conversation is a config bug, and is rejected at construction
    rather than producing a quietly nonsensical report."""

    turn_efficiency_decay_per_turn: float = Field(default=0.1, ge=0.0, le=1.0)
    """Beyond the last cliff, each additional turn costs this much, starting from the last
    cliff's score. With the defaults: 7 turns -> ``0.7 - 0.1 = 0.6``, 8 -> ``0.5``."""

    turn_efficiency_floor: float = Field(default=0.3, ge=0.0, le=1.0)
    """The decay stops here, so a very long conversation that *did* reach the goal keeps a
    non-zero score — it is inefficient, not a failure. Must not exceed the last cliff's
    score, or the floor would raise long conversations above shorter ones."""

    goal_achieved_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    """Weight in ``conversation_quality`` of ``goal_achieved``: 1.0 when the judge decided
    the persona's goal was reached, 0.0 otherwise."""

    criteria_met_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    """Weight in ``conversation_quality`` of ``criteria_met``: the fraction of the
    scenario's ``must_happen`` / ``must_not_happen`` criteria the judge audited and found
    satisfied (0.0 for an errored, timed-out or unverified run)."""

    turn_efficiency_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    """Weight in ``conversation_quality`` of ``turn_efficiency``, the turn-count cost proxy
    described above."""

    @field_validator('turn_efficiency_cliffs')
    @classmethod
    def _validate_cliffs(cls, value: tuple[tuple[int, float], ...]) -> tuple[tuple[int, float], ...]:
        if not value:
            raise ValueError('turn_efficiency_cliffs must contain at least one (max_turns, score) step')
        previous_turns = 0
        previous_score = None
        for max_turns, score in value:
            if max_turns < 1:
                raise ValueError(f'turn_efficiency_cliffs: max_turns must be >= 1, got {max_turns}')
            if max_turns <= previous_turns:
                raise ValueError(
                    f'turn_efficiency_cliffs: max_turns must strictly increase, got {max_turns} after {previous_turns}'
                )
            if not 0.0 <= score <= 1.0:
                raise ValueError(f'turn_efficiency_cliffs: score must be in [0.0, 1.0], got {score}')
            if previous_score is not None and score > previous_score:
                raise ValueError(
                    f'turn_efficiency_cliffs: score must not increase with turns, got {score} after {previous_score} '
                    '(that curve rewards a longer conversation)'
                )
            previous_turns, previous_score = max_turns, score
        return value

    @model_validator(mode='after')
    def _validate_shape(self) -> SimulationScoringConfig:
        last_score = self.turn_efficiency_cliffs[-1][1]
        if self.turn_efficiency_floor > last_score:
            raise ValueError(
                f'turn_efficiency_floor ({self.turn_efficiency_floor}) exceeds the last cliff score ({last_score}); '
                'that scores a long conversation above a shorter one'
            )
        total = self.goal_achieved_weight + self.criteria_met_weight + self.turn_efficiency_weight
        if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f'conversation_quality weights must sum to 1.0, got {total} '
                f'(goal_achieved={self.goal_achieved_weight}, criteria_met={self.criteria_met_weight}, '
                f'turn_efficiency={self.turn_efficiency_weight}); an unnormalised sum makes the composite '
                'incomparable with other runs'
            )
        return self


DEFAULT_SCORING_CONFIG = SimulationScoringConfig()
"""The shipped policy. Used whenever a scorer is called without a ``config``."""


def read_criteria_meta(result: SimulationResult) -> tuple[list[CriteriaMeta], list[object]]:
    """Read the persisted criteria contract and record malformed entries."""
    valid, invalid = parse_criteria_meta(result.metadata.get('criteria_meta'))
    if invalid:
        errors = result.metadata.setdefault('criteria_errors', [])
        if not isinstance(errors, list):
            errors = result.metadata['criteria_errors'] = []
        for entry in invalid:
            message = f'criteria_meta entry is invalid: {entry!r}'
            logger.warning(message)
            if message not in errors:
                errors.append(message)
    return valid, invalid


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

    A malformed ``criteria_meta`` entry is an error: it is logged, recorded on
    the result for downstream reports, and scores 0.0 rather than being dropped.
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

    raw_meta = result.metadata.get('criteria_meta')
    if raw_meta is not None:
        entries, invalid = read_criteria_meta(result)
        if invalid:
            logger.warning('criteria_met: invalid criteria_meta is an error; scoring 0.0 (unknown, not met).')
            return 0.0
        if entries:
            unaudited = [c for c in entries if c.audited is False and c.passed]
            if unaudited:
                logger.warning(
                    'criteria_met: {} of {} criteria passed only by default (the judge never audited them); '
                    'counting them as not met.',
                    len(unaudited),
                    len(entries),
                )
            met = sum(1 for c in entries if c.passed and c.audited is not False)
            return met / len(entries)
        logger.warning(
            'criteria_met: criteria_meta is present but empty; falling back to criteria_results, '
            'which carries no audit provenance.'
        )

    criteria_results = result.criteria_results or {}
    if not criteria_results:
        return 1.0

    met = sum(1 for v in criteria_results.values() if v)
    return met / len(criteria_results)


def turn_efficiency_scorer(result: SimulationResult, config: SimulationScoringConfig | None = None) -> float:
    """How cheaply the goal was reached, 0..1 — fewer turns scores higher.

    A run that did not reach its goal scores ``0.0``: this is a cost metric conditioned on
    success, and "failed quickly" is not efficiency. Otherwise the turn count is mapped
    through ``config.turn_efficiency_cliffs`` (first match wins), then decayed by
    ``turn_efficiency_decay_per_turn`` from the last cliff's score down to
    ``turn_efficiency_floor``.

    The assumption — extra turns mean re-asking, clarifying or wandering — does not hold
    for tasks that legitimately need many turns. See ``SimulationScoringConfig`` for how to
    move the cliffs, or drop this scorer for such scenarios.
    """
    config = config or DEFAULT_SCORING_CONFIG

    if not result.goal_achieved:
        return 0.0

    total_turns = result.turn_count
    for max_turns, score in config.turn_efficiency_cliffs:
        if total_turns <= max_turns:
            return score

    last_turns, last_score = config.turn_efficiency_cliffs[-1]
    decayed = last_score - (total_turns - last_turns) * config.turn_efficiency_decay_per_turn
    # Round: repeated float subtraction turns 0.6 into 0.6000000000000001, which then
    # renders in a report as a 16-digit score.
    return round(max(config.turn_efficiency_floor, decayed), 4)


def conversation_quality_scorer(result: SimulationResult, config: SimulationScoringConfig | None = None) -> float:
    """Weighted composite of the other three scorers, 0..1, rounded to 2 decimals.

    Defaults: ``goal_achieved`` 0.4, ``criteria_met`` 0.3, ``turn_efficiency`` 0.3. The
    weights are validated to sum to 1.0 at config construction, so the composite stays on
    the same 0..1 scale as its parts and is comparable across runs.
    """
    config = config or DEFAULT_SCORING_CONFIG
    goal_score = goal_achieved_scorer(result)
    criteria_score = criteria_met_scorer(result)
    efficiency_score = turn_efficiency_scorer(result, config)

    score = (
        goal_score * config.goal_achieved_weight
        + criteria_score * config.criteria_met_weight
        + efficiency_score * config.turn_efficiency_weight
    )
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


def get_evaluator(name: str, config: SimulationScoringConfig | None = None) -> SimulationScorer:
    """Get a built-in simulation evaluator by name, bound to ``config`` when given.

    ``config`` only affects ``turn_efficiency`` and ``conversation_quality``; the other two
    read the judge's verdicts and have nothing to tune, so they are returned unwrapped.

    Raises:
            ValueError: If evaluator not found.
    """
    evaluator = get_all_evaluators(config).get(name)
    if not evaluator:
        available = ', '.join(SIMULATION_EVALUATORS.keys())
        raise ValueError(f'Unknown evaluator: {name}. Available: {available}')
    return evaluator


def get_all_evaluators(config: SimulationScoringConfig | None = None) -> dict[str, SimulationScorer]:
    """Get all built-in simulation evaluators, bound to ``config`` when given."""
    evaluators = dict(SIMULATION_EVALUATORS)
    if config is not None:
        evaluators['turn_efficiency'] = partial(turn_efficiency_scorer, config=config)
        evaluators['conversation_quality'] = partial(conversation_quality_scorer, config=config)
    return evaluators
