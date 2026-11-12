"""Matching, aggregation and score rows for the sim run comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING

from evaluatorq.common.matching import pair_indices_by_key
from evaluatorq.dashboard.filters import FILTERS
from evaluatorq.dashboard.sim_views import (
    _entries_from_run,
    filtered_entries,
)
from evaluatorq.simulation.metrics import TURN_METRICS, is_errored_result
from evaluatorq.simulation.reports.sections import (
    _build_evaluator_scores_section,
    _build_summary_section,
    _build_turn_metrics_section,
)

if TYPE_CHECKING:
    from evaluatorq.simulation.contracts import SimulationEntry, SimulationResult, SimulationRun

_FLAT = 0.005


# ---------------------------------------------------------------------------
# Conversation matching (by persona + scenario)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchedPair:
    a: SimulationEntry
    b: SimulationEntry


@dataclass(frozen=True)
class Matching:
    matched: list[MatchedPair]
    a_only: list[SimulationEntry]
    b_only: list[SimulationEntry]

    @property
    def match_rate(self) -> float:
        """Share of all conversations that paired up (0..1). 1.0 = identical key sets."""
        total = 2 * len(self.matched) + len(self.a_only) + len(self.b_only)
        return (2 * len(self.matched) / total) if total else 0.0


def match_entries(entries_a: list[SimulationEntry], entries_b: list[SimulationEntry]) -> Matching:
    """Pair conversations across runs by ``(persona, scenario)``.

    Duplicate keys within a run are paired positionally (first A with first B,
    etc.) — an approximation surfaced by the low-overlap note on the page; any
    leftovers on either side fall into ``a_only`` / ``b_only``.
    """
    matched_indices, a_only_indices, b_only_indices = pair_indices_by_key(
        entries_a,
        entries_b,
        key=lambda entry: (entry.persona, entry.scenario),
    )
    return Matching(
        [MatchedPair(entries_a[i], entries_b[j]) for i, j in matched_indices],
        [entries_a[i] for i in a_only_indices],
        [entries_b[i] for i in b_only_indices],
    )


# ---------------------------------------------------------------------------
# Comparison model: both runs, filtered, plus the matching over them
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Side:
    """One run's view: the entries that survived the filter (pair-level, see
    ``build_comparison``) and which original indices errored (``is_errored_result``
    needs the raw result)."""

    rid: str
    run: SimulationRun
    entries: list[SimulationEntry]
    errored: frozenset[int]
    results: list[SimulationResult]
    """The raw results behind ``entries``, built once for the sections builders."""

    @property
    def name(self) -> str:
        return self.run.run_name

    def status(self, entry: SimulationEntry) -> str:
        """``'error'`` / ``'met'`` / ``'missed'`` — the sim row-list's outcome vocabulary."""
        if entry.index in self.errored:
            return 'error'
        return 'met' if entry.goal_achieved else 'missed'


@dataclass(frozen=True)
class Comparison:
    a: Side
    b: Side
    matching: Matching
    """Surviving rows: a pair stays when either side passes the filter, an unmatched entry when it passes."""
    full: Matching
    """The unfiltered matching, for the status line and the rail's shown/total counter."""
    narrowing: dict[str, list[str]]
    """The filter selections that drop at least one conversation, in rail order."""

    @property
    def total_rows(self) -> int:
        """Rows (pairs + unmatched) before filtering."""
        return len(self.full.matched) + len(self.full.a_only) + len(self.full.b_only)

    @property
    def shown_rows(self) -> int:
        return len(self.matching.matched) + len(self.matching.a_only) + len(self.matching.b_only)

    @property
    def outcome_filter(self) -> str:
        """The filter that picks conversations by the outcome being compared, as
        the page names it, or ``''``. While one is on, the goal rate and goal
        score can no longer be compared between runs."""
        return ' and '.join(name for dim, name in _OUTCOME_DIMS.items() if dim in self.narrowing)

    @cached_property
    def aggregates(self) -> tuple[Aggregates, Aggregates]:
        return aggregates(self.a), aggregates(self.b)

    @cached_property
    def scores(self) -> list[ScoreRow]:
        return score_rows(self)

    @cached_property
    def changes(self) -> list[tuple[MatchedPair, str]]:
        return [(p, pair_change(self.a, self.b, p)) for p in self.matching.matched]


# Filters that select on the goal outcome itself (directly or through the score).
_OUTCOME_DIMS = {'goal_outcome': 'goal outcome', 'max_goal_score': 'goal score'}


def narrowing_selections(
    run_a: SimulationRun, run_b: SimulationRun, selections: dict[str, list[str]]
) -> dict[str, list[str]]:
    """The selections that drop a conversation in either run, in rail order.

    The rail posts every control, defaults included, so a selection is only a
    real filter when applying it alone removes something.
    """
    sim = FILTERS['sim']
    return {
        dim: selections[dim]
        for dim in sim.dimensions
        if selections.get(dim)
        and any(len(sim.apply(run, {dim: selections[dim]})) < len(run.results) for run in (run_a, run_b))
    }


def _side(rid: str, run: SimulationRun, entries: list[SimulationEntry]) -> Side:
    errored = frozenset(i for i, r in enumerate(run.results) if is_errored_result(result=r))
    entries = sorted(entries, key=lambda e: e.index)
    return Side(rid, run, entries, errored, [run.results[e.index] for e in entries])


def build_comparison(
    rid_a: str, rid_b: str, run_a: SimulationRun, run_b: SimulationRun, selections: dict[str, list[str]]
) -> Comparison:
    full = match_entries(_entries_from_run(run_a), _entries_from_run(run_b))
    keep_a = {e.index for e in filtered_entries(run_a, selections)}
    keep_b = {e.index for e in filtered_entries(run_b, selections)}
    matching = Matching(
        [p for p in full.matched if p.a.index in keep_a or p.b.index in keep_b],
        [e for e in full.a_only if e.index in keep_a],
        [e for e in full.b_only if e.index in keep_b],
    )
    side_a = _side(rid_a, run_a, [*(p.a for p in matching.matched), *matching.a_only])
    side_b = _side(rid_b, run_b, [*(p.b for p in matching.matched), *matching.b_only])
    return Comparison(side_a, side_b, matching, full, narrowing_selections(run_a, run_b, selections))


# (resolved path, mtime_ns) of a report file: a rewrite changes the stamp, so a stale entry is never hit.
RunStamp = tuple[str, int]


@dataclass(frozen=True)
class _Runs:
    """The loaded runs, riding along an ``lru_cache`` call without being part of its key."""

    a: SimulationRun = field(compare=False)
    b: SimulationRun = field(compare=False)


# Two runs per entry, each holding every transcript: keep few. 16 covers a session's back-and-forth between slices.
@lru_cache(maxsize=16)
def _build_cached(
    stamp_a: RunStamp,
    stamp_b: RunStamp,
    rid_a: str,
    rid_b: str,
    frozen_selections: tuple[tuple[str, tuple[str, ...]], ...],
    runs: _Runs,
) -> Comparison:
    return build_comparison(rid_a, rid_b, runs.a, runs.b, {dim: list(values) for dim, values in frozen_selections})


def cached_comparison(
    stamps: tuple[RunStamp, RunStamp],
    rid_a: str,
    rid_b: str,
    run_a: SimulationRun,
    run_b: SimulationRun,
    selections: dict[str, list[str]],
) -> Comparison:
    """``build_comparison`` memoised on the two files' stamps and the selections.

    The result is shared between requests: treat it as read-only.
    """
    frozen = tuple((dim, tuple(values)) for dim, values in selections.items())
    return _build_cached(stamps[0], stamps[1], rid_a, rid_b, frozen, _Runs(run_a, run_b))


def pair_change(a: Side, b: Side, pair: MatchedPair) -> str:
    """``'regressed'`` (met -> missed), ``'improved'`` (missed -> met), ``'error'``
    (the statuses differ and either side errored), or ``'same'``."""
    status_a, status_b = a.status(pair.a), b.status(pair.b)
    if status_a == status_b:
        return 'same'
    if 'error' in (status_a, status_b):
        return 'error'
    return 'regressed' if status_a == 'met' else 'improved'


@dataclass(frozen=True)
class Aggregates:
    conversations: int
    goal_rate: float | None
    avg_score: float | None
    avg_turns: float | None
    errors: int


def aggregates(side: Side) -> Aggregates:
    """Headline numbers for one run's surviving conversations, taken from the
    single-run summary section so they match the run's own report by
    construction: errored conversations count in the denominator of the goal
    rate and the average score. An empty side has no rates (n/a)."""
    if not side.results:
        return Aggregates(0, None, None, None, 0)
    data = _build_summary_section(side.results).data
    return Aggregates(
        conversations=data['total_conversations'],
        goal_rate=data['success_rate'],
        avg_score=data['avg_goal_completion_score'],
        avg_turns=data['avg_turn_count'],
        errors=data['errors'],
    )


@dataclass(frozen=True)
class ScoreRow:
    """One scores-panel row. ``higher_better`` is False for risk metrics.
    ``dropped_*`` counts conversations whose scorer failed in that run, so a
    ``None`` mean with drops reads as failed rather than not measured."""

    label: str
    a: float | None
    b: float | None
    higher_better: bool = True
    dropped_a: int = 0
    dropped_b: int = 0

    @property
    def delta(self) -> float | None:
        return None if self.a is None or self.b is None else self.b - self.a

    @property
    def in_unit_range(self) -> bool:
        return all(v is None or 0.0 <= v <= 1.0 for v in (self.a, self.b))


def _scorer_stats(side: Side) -> dict[str, tuple[float | None, int]]:
    """Scorer name -> (mean score, dropped count) over the side's results."""
    section = _build_evaluator_scores_section(side.results)
    if section is None:
        return {}
    return {row['evaluator']: (row['mean_score'], row['dropped']) for row in section.data['rows']}


def _quality_means(side: Side) -> dict[str, float]:
    return dict(_build_turn_metrics_section(side.results).data['avg_quality_metrics'])


def score_rows(cmp: Comparison) -> list[ScoreRow]:
    """Per-scorer and per-turn-quality means over each run's surviving
    conversations. A metric one run never measured stays ``None`` there (n/a),
    never 0.0, so it cannot read as a regression; a scorer that failed on every
    conversation is ``None`` with a drop count."""
    sa, sb = _scorer_stats(cmp.a), _scorer_stats(cmp.b)
    rows = []
    for key in sorted(set(sa) | set(sb)):
        mean_a, dropped_a = sa.get(key, (None, 0))
        mean_b, dropped_b = sb.get(key, (None, 0))
        rows.append(ScoreRow(key, mean_a, mean_b, dropped_a=dropped_a, dropped_b=dropped_b))
    qa, qb = _quality_means(cmp.a), _quality_means(cmp.b)
    rows.extend(
        ScoreRow(m.label, qa.get(m.key), qb.get(m.key), higher_better=not m.high_is_risky)
        for m in TURN_METRICS
        if m.key in qa or m.key in qb
    )
    return rows
