"""Synthetic-recovery tests for the BT-sigma fit (arXiv:2602.16610).

These encode the paper's claims as regressions: under heterogeneous judge
noise the fit recovers the true ranking, and the fitted discriminators order
judges by their injected noise level.
"""

from __future__ import annotations

import math
import random

import pytest

from evaluatorq.ranking import JudgedComparison, binarize, comparisons_per_judge, cycle_rate, fit_bt

_ITEMS = ['m1', 'm2', 'm3', 'm4', 'm5']
_TRUE_SKILLS = {'m1': 2.0, 'm2': 1.0, 'm3': 0.0, 'm4': -1.0, 'm5': -2.0}


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _soft_comparisons(sigmas: dict[str, float]) -> list[JudgedComparison]:
    """Exact model-generated soft preferences for every ordered pair per judge."""
    out = []
    for judge, sigma in sigmas.items():
        for i in _ITEMS:
            for j in _ITEMS:
                if i >= j:
                    continue
                p = _logistic((_TRUE_SKILLS[i] - _TRUE_SKILLS[j]) / sigma)
                out.append(JudgedComparison(judge=judge, item_a=i, item_b=j, p_a=p))
    return out


def _sampled_hard_comparisons(sigmas: dict[str, float], rounds: int, seed: int = 0) -> list[JudgedComparison]:
    """Hard votes sampled from the model with per-judge noise, deterministic seed."""
    rng = random.Random(seed)
    out = []
    for judge, sigma in sigmas.items():
        for _ in range(rounds):
            for i in _ITEMS:
                for j in _ITEMS:
                    if i >= j:
                        continue
                    p = _logistic((_TRUE_SKILLS[i] - _TRUE_SKILLS[j]) / sigma)
                    vote = 1.0 if rng.random() < p else 0.0
                    out.append(JudgedComparison(judge=judge, item_a=i, item_b=j, p_a=vote))
    return out


def test_soft_bt_sigma_recovers_ranking_and_noise_ordering() -> None:
    sigmas = {'sharp': 0.5, 'mid': 1.0, 'noisy': 4.0}
    fit = fit_bt(_soft_comparisons(sigmas), judge_sigma=True)
    assert fit.ranking == ['m1', 'm2', 'm3', 'm4', 'm5']
    # Fitted discriminator ordering matches the injected noise ordering.
    assert fit.sigmas['sharp'] < fit.sigmas['mid'] < fit.sigmas['noisy']
    assert fit.converged


def test_hard_bt_sigma_recovers_ranking_under_sampled_noise() -> None:
    sigmas = {'sharp': 0.4, 'noisy': 5.0}
    fit = fit_bt(_sampled_hard_comparisons(sigmas, rounds=40), judge_sigma=True, hard=True)
    assert fit.ranking == ['m1', 'm2', 'm3', 'm4', 'm5']
    assert fit.sigmas['sharp'] < fit.sigmas['noisy']


def test_bt_sigma_downweights_an_adversarial_judge() -> None:
    # Two sharp judges and one voting the REVERSE ranking. With a single good
    # judge this is ill-posed (the paper's limitations section: a consistent
    # minority is indistinguishable from a correct one), so the majority
    # breaks the symmetry. BT-sigma should keep the true order and hand the
    # adversary the largest discriminator.
    good = _soft_comparisons({'good-1': 0.5, 'good-2': 0.7})
    adversary = [
        JudgedComparison(judge='bad', item_a=c.item_a, item_b=c.item_b, p_a=1.0 - c.p_a)
        for c in _soft_comparisons({'bad': 0.5})
    ]
    fit = fit_bt(good + adversary, judge_sigma=True)
    assert fit.ranking == ['m1', 'm2', 'm3', 'm4', 'm5']
    assert fit.sigmas['bad'] > fit.sigmas['good-1']
    assert fit.sigmas['bad'] > fit.sigmas['good-2']


def test_identifiability_constraints_hold() -> None:
    fit = fit_bt(_soft_comparisons({'a': 0.7, 'b': 2.0}), judge_sigma=True)
    assert sum(fit.skills.values()) == pytest.approx(0.0, abs=1e-6)
    # Sigma's absolute scale is only loosely anchored (soft prior); the
    # contract is that ratios within a fit are meaningful and finite.
    assert all(v > 0 and math.isfinite(v) for v in fit.sigmas.values())
    assert fit.sigmas['a'] < fit.sigmas['b']  # ratio reflects injected noise


def test_single_judge_falls_back_to_plain_bt() -> None:
    fit = fit_bt(_soft_comparisons({'only': 1.0}), judge_sigma=True)
    assert fit.sigmas == {}
    assert any('single judge' in w for w in fit.warnings)
    assert fit.ranking == ['m1', 'm2', 'm3', 'm4', 'm5']


def test_fit_is_deterministic() -> None:
    comparisons = _sampled_hard_comparisons({'a': 0.5, 'b': 2.0}, rounds=5)
    first = fit_bt(comparisons, judge_sigma=True, hard=True)
    second = fit_bt(comparisons, judge_sigma=True, hard=True)
    assert first.skills == second.skills
    assert first.sigmas == second.sigmas


def test_perfect_separation_stays_finite() -> None:
    # Every judge says m1 beats m2 every time: the unregularised MLE diverges.
    comparisons = [
        JudgedComparison(judge=j, item_a='m1', item_b='m2', p_a=1.0) for j in ('a', 'b') for _ in range(10)
    ]
    fit = fit_bt(comparisons, judge_sigma=True, hard=True)
    assert fit.ranking == ['m1', 'm2']
    assert all(abs(v) < 50 for v in fit.skills.values())


def test_disconnected_graph_stays_finite() -> None:
    comparisons = [
        JudgedComparison(judge='a', item_a='x1', item_b='x2', p_a=0.9),
        JudgedComparison(judge='b', item_a='y1', item_b='y2', p_a=0.2),
    ]
    fit = fit_bt(comparisons, judge_sigma=True)
    assert set(fit.ranking) == {'x1', 'x2', 'y1', 'y2'}
    assert all(math.isfinite(v) for v in fit.skills.values())
    assert any('disconnected' in warning for warning in fit.warnings)


def test_sparse_judge_coverage_is_reported() -> None:
    comparisons = [
        JudgedComparison(judge='sparse', item_a='a', item_b='b', p_a=1.0),
        JudgedComparison(judge='steady', item_a='a', item_b='b', p_a=1.0),
        JudgedComparison(judge='steady', item_a='a', item_b='b', p_a=1.0),
        JudgedComparison(judge='steady', item_a='a', item_b='b', p_a=1.0),
    ]
    fit = fit_bt(comparisons, judge_sigma=True, hard=True)
    assert any('fewer than 3 observations' in warning for warning in fit.warnings)


def test_empty_comparisons_raise() -> None:
    with pytest.raises(ValueError, match='at least one comparison'):
        fit_bt([])


def test_binarize_maps_votes_and_keeps_ties() -> None:
    soft = [
        JudgedComparison(judge='j', item_a='a', item_b='b', p_a=0.8),
        JudgedComparison(judge='j', item_a='a', item_b='b', p_a=0.2),
        JudgedComparison(judge='j', item_a='a', item_b='b', p_a=0.5),
    ]
    assert [c.p_a for c in binarize(soft)] == [1.0, 0.0, 0.5]


def test_cycle_rate_transitive_cyclic_and_undefined() -> None:
    def pref(a: str, b: str) -> JudgedComparison:
        return JudgedComparison(judge='j', item_a=a, item_b=b, p_a=1.0)

    transitive = [pref('a', 'b'), pref('b', 'c'), pref('a', 'c')]
    assert cycle_rate(transitive, 'j') == 0.0
    cyclic = [pref('a', 'b'), pref('b', 'c'), pref('c', 'a')]
    assert cycle_rate(cyclic, 'j') == 1.0
    # Two items: no triplet exists, the diagnostic is undefined (A/B setting).
    assert cycle_rate([pref('a', 'b')], 'j') is None


def test_cycle_rate_uses_the_majority_of_duplicate_edges() -> None:
    comparisons = [
        JudgedComparison(judge='j', item_a='a', item_b='b', p_a=1.0),
        JudgedComparison(judge='j', item_a='a', item_b='b', p_a=0.0),
        JudgedComparison(judge='j', item_a='a', item_b='b', p_a=0.0),
        JudgedComparison(judge='j', item_a='b', item_b='c', p_a=1.0),
        JudgedComparison(judge='j', item_a='a', item_b='c', p_a=1.0),
    ]
    assert cycle_rate(comparisons, 'j') == 0.0


def test_noisier_judge_has_higher_cycle_rate_and_higher_sigma() -> None:
    # The paper's validation signal: 1/sigma tracks cycle consistency.
    comparisons = _sampled_hard_comparisons({'sharp': 0.3, 'noisy': 6.0}, rounds=1, seed=3)
    fit = fit_bt(comparisons, judge_sigma=True, hard=True)
    sharp_cycles = cycle_rate(comparisons, 'sharp')
    noisy_cycles = cycle_rate(comparisons, 'noisy')
    assert sharp_cycles is not None
    assert noisy_cycles is not None
    assert sharp_cycles <= noisy_cycles
    assert fit.sigmas['sharp'] < fit.sigmas['noisy']


def test_comparisons_per_judge_counts() -> None:
    comparisons = _soft_comparisons({'a': 1.0, 'b': 1.0})
    counts = comparisons_per_judge(comparisons)
    assert counts == {'a': 10, 'b': 10}


def test_comparisons_per_judge_sums_weights() -> None:
    weighted = [JudgedComparison(judge='j', item_a='a', item_b='b', p_a=1.0, weight=7.0)]
    assert comparisons_per_judge(weighted) == {'j': 7.0}


def test_fit_matches_independent_derivation() -> None:
    # One judge, hard 3-1 votes for m1 over m2: with skills centred at +-g/2,
    # plain-BT stationarity is 4*(0.75 - logistic(g)) = _RIDGE * g / 2. Solve
    # it by bisection, independently of the optimiser, and pin the fit to it.
    # A scaling bug in the gradient that preserved monotonicity would fail here.
    votes = [JudgedComparison(judge='j', item_a='m1', item_b='m2', p_a=1.0) for _ in range(3)] + [
        JudgedComparison(judge='j', item_a='m1', item_b='m2', p_a=0.0)
    ]
    fit = fit_bt(votes, judge_sigma=False)
    gap = fit.skills['m1'] - fit.skills['m2']

    def stationarity(g: float) -> float:
        return 4.0 * (0.75 - _logistic(g)) - 1e-2 * g / 2.0

    lo, hi = 0.0, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if stationarity(mid) > 0:
            lo = mid
        else:
            hi = mid
    # abs=5e-3: the optimiser stops within ~_TOL-sized residue of the optimum;
    # the tolerance is far below any scaling error (a 2x gradient bug lands
    # the gap near 0.55 off) while not fighting the convergence criterion.
    assert gap == pytest.approx(lo, abs=5e-3)
    assert _logistic(gap) == pytest.approx(0.75, abs=2e-3)


def test_weighted_records_match_repeated_records() -> None:
    # The likelihood is additive over identical records, so a weight-5 record
    # must fit exactly like five copies - this is what lets the pairwise
    # aggregation collapse votes without changing the optimum.
    def records(collapse: bool) -> list[JudgedComparison]:
        spec = [('j1', 1.0, 5), ('j1', 0.0, 1), ('j2', 0.0, 2), ('j2', 1.0, 1)]
        if collapse:
            return [
                JudgedComparison(judge=j, item_a='m1', item_b='m2', p_a=p, weight=float(n)) for j, p, n in spec
            ]
        return [
            JudgedComparison(judge=j, item_a='m1', item_b='m2', p_a=p) for j, p, n in spec for _ in range(n)
        ]

    expanded = fit_bt(records(collapse=False), judge_sigma=True, hard=True)
    collapsed = fit_bt(records(collapse=True), judge_sigma=True, hard=True)
    assert collapsed.skills['m1'] == pytest.approx(expanded.skills['m1'], abs=1e-6)
    assert collapsed.sigmas['j1'] == pytest.approx(expanded.sigmas['j1'], rel=1e-6)
    assert collapsed.sigmas['j2'] == pytest.approx(expanded.sigmas['j2'], rel=1e-6)


def test_cycle_rate_aggregates_duplicate_judgements() -> None:
    # A judge that rated the same pair twice in opposite directions has no net
    # preference on it: the edge must cancel (mean p = 0.5), not resolve to
    # whichever judgement happened to come last.
    base = [
        JudgedComparison(judge='j', item_a='m1', item_b='m2', p_a=1.0),
        JudgedComparison(judge='j', item_a='m2', item_b='m3', p_a=1.0),
        JudgedComparison(judge='j', item_a='m3', item_b='m1', p_a=1.0),
    ]
    assert cycle_rate(base, 'j') == 1.0
    # The duplicate arrives in the REVERSED frame (m2, m1): canonicalisation
    # must fold it onto the same edge before averaging.
    dup = [*base, JudgedComparison(judge='j', item_a='m2', item_b='m1', p_a=1.0)]
    assert cycle_rate(dup, 'j') is None
