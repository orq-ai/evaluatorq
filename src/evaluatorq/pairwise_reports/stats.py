"""Render-time statistics for the pairwise dashboard report.

Pure functions over a saved ``PairwiseRun``: nothing here is persisted, so an
old run file gets every number without a migration. No HTML lives here.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from evaluatorq.pairwise_reports.sections import POSITION_BIAS_WARN, is_split, observed_swap

if TYPE_CHECKING:
    from collections.abc import Sequence

    from evaluatorq.pairwise_run import PairwiseEntry, PairwiseRun

Tone = Literal['good', 'warn', '']

_CATEGORIES = ('A', 'B', 'tie')
# Share of comparisons with a winner below which the verdict rests on a thin slice.
DECIDED_GOOD = 0.7
# Share of votes discarded because a judge contradicted itself across orderings.
THROWN_OUT_WARN = 0.1


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials (95% by default)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def outcome_counts(run: PairwiseRun) -> dict[str, int]:
    counts = Counter(e.comparison.winner for e in run.entries)
    return {key: counts.get(key, 0) for key in ('A', 'B', 'tie', 'inconclusive')}


@dataclass(frozen=True)
class Verdict:
    state: Literal['clear', 'too_close', 'no_decided']
    leader: Literal['A', 'B'] | None
    leader_label: str | None
    wins_a: int
    wins_b: int
    ties: int
    inconclusive: int
    decided: int
    total: int
    win_rate: float | None
    low: float | None
    high: float | None


def verdict(run: PairwiseRun) -> Verdict:
    """Whether the run names a winner, with a 95% interval on the leader's win rate."""
    c = outcome_counts(run)
    a, b = c['A'], c['B']
    decided = a + b
    base = {
        'wins_a': a,
        'wins_b': b,
        'ties': c['tie'],
        'inconclusive': c['inconclusive'],
        'decided': decided,
        'total': len(run.entries),
    }
    if decided == 0:
        return Verdict(state='no_decided', leader=None, leader_label=None, win_rate=None, low=None, high=None, **base)
    leader: Literal['A', 'B'] | None = 'A' if a > b else 'B' if b > a else None
    k = max(a, b)
    low, high = wilson_interval(k, decided)
    state: Literal['clear', 'too_close'] = 'clear' if leader is not None and low > 0.5 else 'too_close'
    label = None if leader is None else (run.label_a if leader == 'A' else run.label_b)
    return Verdict(state=state, leader=leader, leader_label=label, win_rate=k / decided, low=low, high=high, **base)


def fleiss_kappa(run: PairwiseRun) -> tuple[float | None, int]:
    """Fleiss' kappa over the configured panel, on comparisons where every panel judge voted.

    Stand-in (replacement) judges are not part of the panel; a comparison where a
    panel judge failed is skipped. Returns ``(kappa, rows_used)``; ``(None, 0)``
    with fewer than two judges or no full-panel comparison.
    """
    judges = list(run.judges)
    n = len(judges)
    if n < 2:
        return (None, 0)
    rows: list[Counter[str]] = []
    for e in run.entries:
        votes = {v.model: v.vote for v in e.comparison.votes}
        if all(votes.get(j) is not None for j in judges):
            rows.append(Counter(str(votes[j]) for j in judges))
    if not rows:
        return (None, 0)
    p_bar = sum((sum(c[k] ** 2 for k in _CATEGORIES) - n) / (n * (n - 1)) for c in rows) / len(rows)
    p_e = sum((sum(c[k] for c in rows) / (len(rows) * n)) ** 2 for k in _CATEGORIES)
    if p_e >= 1:
        return (1.0, len(rows))
    return ((p_bar - p_e) / (1 - p_e), len(rows))


def kappa_label(k: float) -> str:
    """Landis and Koch reading of a kappa value."""
    if k < 0:
        return 'poor'
    for cut, name in ((0.2, 'slight'), (0.4, 'fair'), (0.6, 'moderate'), (0.8, 'substantial')):
        if k < cut:
            return name
    return 'almost perfect'


@dataclass(frozen=True)
class LengthLean:
    longer_wins: int
    compared: int


def length_lean(run: PairwiseRun) -> LengthLean | None:
    """How often the longer response (in characters) won a decided comparison."""
    compared = longer = 0
    for e in run.entries:
        w = e.comparison.winner
        la, lb = len(e.response_a), len(e.response_b)
        if w not in ('A', 'B') or la == lb:
            continue
        compared += 1
        longer += (w == 'A') == (la > lb)
    return LengthLean(longer_wins=longer, compared=compared) if compared else None


@dataclass(frozen=True)
class ComparisonFlags:
    split: bool
    thrown_out: bool
    judge_failed: bool


def comparison_flags(entry: PairwiseEntry, judges: Sequence[str]) -> ComparisonFlags:
    votes = entry.comparison.votes
    present = {v.model for v in votes}
    return ComparisonFlags(
        split=is_split(entry),
        thrown_out=any(v.flipped for v in votes),
        judge_failed=any(v.repetition_failures > 0 for v in votes) or any(j not in present for j in judges),
    )


@dataclass(frozen=True)
class UsageTotals:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float | None
    with_usage: int


def judge_usage(run: PairwiseRun) -> UsageTotals:
    """Sum recorded tokens; report cost only when every comparison was priced."""
    usages = [e.comparison.token_usage for e in run.entries if e.comparison.token_usage is not None]
    costs = [u.total_cost for u in usages if u.total_cost is not None]
    return UsageTotals(
        input_tokens=sum(u.input_tokens for u in usages),
        output_tokens=sum(u.output_tokens for u in usages),
        total_tokens=sum(u.total_tokens or (u.input_tokens + u.output_tokens) for u in usages),
        cost=math.fsum(costs) if costs and len(costs) == len(run.entries) else None,
        with_usage=len(usages),
    )


@dataclass(frozen=True)
class TrustSignal:
    label: str
    value: str
    detail: str
    tone: Tone


def trust_signals(run: PairwiseRun) -> list[TrustSignal]:
    """Run-confidence rows for the "How much to trust this" table, in display order."""
    total = len(run.entries)
    v = verdict(run)
    out: list[TrustSignal] = []
    share = v.decided / total if total else 0.0
    out.append(
        TrustSignal(
            'Decided',
            f'{v.decided} of {total} ({share:.0%})',
            'comparisons had a winner',
            'good' if share >= DECIDED_GOOD else 'warn',
        )
    )
    votes = [vote for e in run.entries for vote in e.comparison.votes]
    flipped = sum(vote.flipped for vote in votes)
    thrown = flipped / len(votes) if votes else 0.0
    out.append(
        TrustSignal(
            'Votes thrown out',
            f'{thrown:.0%} ({flipped}/{len(votes)})',
            'flipped when the sides were swapped',
            'warn' if thrown > THROWN_OUT_WARN else '',
        )
    )
    kappa, rows = fleiss_kappa(run)
    if kappa is None:
        out.append(
            TrustSignal(
                'Judge agreement', 'not measurable', 'needs two or more judges voting on the same comparison', ''
            )
        )
    else:
        out.append(
            TrustSignal(
                'Judge agreement',
                f'κ {kappa:.2f}',
                f'{kappa_label(kappa)}, 1 = every judge agreed, over {rows} full-panel comparisons',
                '',
            )
        )
    if observed_swap(run):
        report = run.rollup()
        measured = {v.model for e in run.entries for v in e.comparison.votes if v.completed}
        measured_stats = [s for s in report.per_judge if s.model in measured]
        flagged = [s.model for s in measured_stats if s.position_bias >= POSITION_BIAS_WARN]
        out.append(
            TrustSignal(
                'Position bias',
                f'{len(flagged)} of {len(measured_stats)} judges',
                f'flagged at a flip rate of {POSITION_BIAS_WARN:g} or more',
                'warn' if flagged else 'good',
            )
        )
    lean = length_lean(run)
    if lean is None:
        out.append(TrustSignal('Length', 'not measurable', 'no decided comparison had answers of different length', ''))
    else:
        longer = lean.longer_wins * 2 >= lean.compared
        wins = lean.longer_wins if longer else lean.compared - lean.longer_wins
        out.append(
            TrustSignal(
                'Length',
                f'{"longer" if longer else "shorter"} won {wins / lean.compared:.0%}',
                f'{wins} of {lean.compared} decided comparisons with answers of different length',
                '',
            )
        )
    failures = sum(vote.repetition_failures for vote in votes)
    out.append(
        TrustSignal(
            'Judge failures',
            str(failures),
            'judge passes failed or returned an unusable answer',
            'warn' if failures else 'good',
        )
    )
    return out
