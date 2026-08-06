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
