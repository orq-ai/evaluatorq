"""Bradley-Terry ranking with per-judge reliability (BT-sigma).

Implements the judge-aware Bradley-Terry family from "Who can we trust?
LLM-as-a-jury for Comparative Assessment" (arXiv:2602.16610), whose model class
is independently derived with identifiability and consistency proofs in
arXiv:2601.21817. Given pairwise comparisons from K judges, it jointly fits
latent item skills ``s_i`` and per-judge discriminators ``sigma_k``::

    P_k(i beats j) = logistic((s_i - s_j) / sigma_k)

by unsupervised maximum likelihood on the run's own comparisons - no human
labels, no training data. A small ``sigma_k`` marks a sharp, internally
consistent judge; a large one marks a noisy judge whose votes get down-weighted
in the inferred ranking. Four variants, selected by two flags:

- ``fit_bt(c, judge_sigma=False, hard=False)`` - soft BT (paper Eq. 3)
- ``fit_bt(c, judge_sigma=False, hard=True)``  - hard BT (paper Eq. 2)
- ``fit_bt(c, judge_sigma=True,  hard=False)`` - BT-sigma (paper Eq. 12)
- ``fit_bt(c, judge_sigma=True,  hard=True)``  - hard BT-sigma

The paper's guidance: hard BT-sigma is the most robust when judges are highly
inconsistent; soft BT-sigma wins under moderate noise. The per-aspect variant
(BT-sigma-asp) is deliberately not implemented - the paper finds it marginal.

Everything is pure Python on purpose: the parameter space is tiny (N items +
K judges) and evaluatorq's runtime dependency set stays unchanged (no
scipy/numpy). The cost is O(iterations x records) with iterations growing with
the data, so collapse repeated identical judgements into one record with a
``weight`` (see :class:`JudgedComparison`) before fitting large runs — the
pairwise ``bt_sigma_aggregation`` does this and stays fast at any run size;
an uncollapsed fit over thousands of records can take seconds.

Optimiser choice, on the record (PR #113 review): plain Bradley-Terry has the
hyperparameter-free, monotone Zermelo/MM update, and if this file ever drops
the judge discriminator that is the solver to switch to. The per-judge
``sigma_k`` exponent breaks the algebraic form the MM update relies on (the
likelihood is no longer a ratio of sums of positive weights), so this uses a
generic first-order optimiser instead, and the constants below exist to manage
the failure modes that choice introduces.

Comparisons carry judge identity so the records are reusable by other judging
methods that need per-judge accounting (e.g. round-robin panel assignment).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# Fit hyperparameters. Adam is deliberately boring: the loss is smooth and
# low-dimensional, and a fixed recipe keeps runs reproducible (deterministic
# zero init, no random restarts - unlike the paper's random init, which only
# matters for its L-BFGS implementation).
_LR = 0.1
_LR_DECAY_STEPS = 200  # lr_t = _LR / (1 + step/_LR_DECAY_STEPS): kills Adam's oscillation floor
_MAX_ITER = 20000  # strongly separated data needs the decay tail to play out; cost is per-record, so collapse first
# Converged when the iterate stops moving (max-abs parameter change after the
# identifiability projections). A gradient criterion is wrong here: the ridge
# leaves a constant radial gradient along the scale direction that the
# projection undoes every step, so the raw gradient never reaches zero even at
# the constrained optimum. 1e-4 is far below anything that moves a ranking or
# a vote weight (skills live on a ~+-10 scale, sigmas are reported to 3
# decimals); tighter tolerances just fight Adam's decay tail, which unanimous
# panels ride to the iteration cap at ~3e-5 steps.
_TOL = 1e-4
# A single sub-_TOL step is not convergence: when Adam's momentum flips sign
# crossing the optimum, one step can transiently stall far from stationarity
# (caught by the review's pin-the-fit test: a 4-vote fit "converged" with a
# 3e-2 gradient). Genuine stationarity keeps steps small on consecutive
# iterations; a transient stall rebuilds momentum within a step or two.
_TOL_STREAK = 5
_ADAM_B1 = 0.9
_ADAM_B2 = 0.999
_ADAM_EPS = 1e-8
# Weak gaussian prior on skills: keeps disconnected comparison graphs and
# perfectly separated items (all wins one way -> unbounded skill MLE, which
# otherwise drifts to the iteration cap exactly like the sigma case below)
# finite and quick to converge. Mostly shrinks the MAGNITUDE of extreme gaps;
# because low-coverage items shrink harder than well-covered ones, it can in
# principle reorder two items whose unpenalised skills are very close under
# unbalanced coverage — a small effect at 1e-2, but not a guarantee.
_RIDGE = 1e-2
# Log-normal prior on sigma (ridge on tau), deliberately stronger than _RIDGE:
# a perfectly consistent judge's ideal sigma is 0, so its tau otherwise drifts
# for thousands of iterations (the fit never converges) while the judge's vote
# weight 1/sigma explodes without bound. The prior gives that drift a finite
# equilibrium: reliability separation stays large but bounded, and ambiguous
# magnitudes shrink toward sigma = 1.
_TAU_RIDGE = 1e-2
# Clamp for probabilities entering the log-likelihood.
_P_EPS = 1e-6
# Gradients below this are float dust, not signal. Without the floor, Adam's
# scale invariance turns a 1e-16 residue at a symmetric stationary point (e.g.
# two judges in perfect opposition) into a full-size step, and the fit rolls
# into an arbitrary asymmetric optimum: one judge crowned ultra-sharp, the
# other pure noise, decided by rounding error. Stopping at the stationary
# point keeps perfectly ambiguous data perfectly ambiguous.
_GRAD_FLOOR = 1e-10
_MIN_JUDGE_COMPARISONS = 3


class JudgedComparison(BaseModel):
    """One judge's preference on one item pair, in a fixed (a, b) frame.

    ``p_a`` is the judge's preference probability that ``item_a`` beats
    ``item_b``. Callers that only have categorical votes map them as
    A -> 1.0, B -> 0.0, tie -> 0.5. Position-bias symmetrisation (paper Eq. 4)
    is the caller's job: build ``p_a`` from both orderings, e.g. via the
    pairwise module's swap/reconcile machinery, before fitting.
    """

    judge: str = Field(description='Judge model ID')
    item_a: str = Field(description='First item ID in the canonical frame')
    item_b: str = Field(description='Second item ID in the canonical frame')
    p_a: float = Field(ge=0.0, le=1.0, description='P(item_a beats item_b) according to this judge')
    weight: float = Field(
        default=1.0,
        gt=0.0,
        description='Multiplicity: how many identical judgements this record stands for. The fit cost is '
        'O(iterations x records), so collapsing repeats keeps large runs fast without changing the optimum.',
    )


class BTFit(BaseModel):
    """Result of a Bradley-Terry family fit."""

    skills: dict[str, float] = Field(description='Latent skill per item, centred at mean 0')
    sigmas: dict[str, float] = Field(
        description='Per-judge discriminator; smaller = sharper/more reliable. Only RATIOS between judges '
        'in the same fit are meaningful (the log-normal prior anchors the scale loosely); do not compare '
        'absolute values across fits. Empty when fitted without judge_sigma.'
    )
    ranking: list[str] = Field(description='Item IDs, best first')
    converged: bool = Field(description='True when the iterate stopped moving before the iteration cap')
    iterations: int = Field(description='Optimizer iterations used')
    warnings: list[str] = Field(default_factory=list, description='Degradations applied during the fit')

    def reliability(self, judge: str) -> float:
        """``1 / sigma`` for a judge - the paper's unsupervised reliability signal."""
        return 1.0 / self.sigmas[judge]


def _logistic(x: float) -> float:
    # Guard against overflow on wild intermediate values.
    if x >= 0:
        z = math.exp(-min(x, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -60.0))
    return z / (1.0 + z)


def binarize(comparisons: Iterable[JudgedComparison]) -> list[JudgedComparison]:
    """Map soft preferences to hard ones (paper: ``y = 1[p >= 0.5]``), keeping exact ties at 0.5."""
    out: list[JudgedComparison] = []
    for c in comparisons:
        # 0.5 is the tie point of the preference scale.
        hard = 0.5 if c.p_a == 0.5 else (1.0 if c.p_a > 0.5 else 0.0)
        out.append(c.model_copy(update={'p_a': hard}))
    return out


def fit_bt(
    comparisons: Sequence[JudgedComparison],
    *,
    judge_sigma: bool = True,
    hard: bool = False,
) -> BTFit:
    """Fit the Bradley-Terry family by MLE on the given comparisons.

    ``judge_sigma`` adds the per-judge discriminator (BT-sigma). ``hard``
    binarizes preferences first (hard BT / hard BT-sigma) - the paper's most
    robust variant when judges are highly inconsistent.

    Degradations (recorded in ``warnings``, never raised):

    - a single judge with ``judge_sigma=True`` falls back to plain BT - a lone
      sigma is absorbed into the skill scale and carries no information
      (paper section 3.3);
    - a tiny ridge keeps disconnected graphs / perfect separation finite.
    """
    if not comparisons:
        msg = 'fit_bt needs at least one comparison'
        raise ValueError(msg)
    if hard:
        comparisons = binarize(comparisons)

    items = sorted({c.item_a for c in comparisons} | {c.item_b for c in comparisons})
    judges = sorted({c.judge for c in comparisons})
    warnings: list[str] = []

    judge_counts = comparisons_per_judge(comparisons)
    sparse = sorted(judge for judge, count in judge_counts.items() if count < _MIN_JUDGE_COMPARISONS)
    if sparse:
        warnings.append(
            f'judge(s) {", ".join(sparse)} have fewer than {_MIN_JUDGE_COMPARISONS} observations; '
            'their sigma estimates are weakly supported'
        )

    adjacency: dict[str, set[str]] = defaultdict(set)
    for c in comparisons:
        adjacency[c.item_a].add(c.item_b)
        adjacency[c.item_b].add(c.item_a)
    components = 0
    unseen = set(items)
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            item = stack.pop()
            for neighbor in adjacency[item]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    if components > 1:
        warnings.append(
            f'comparison graph is disconnected ({components} components); '
            'relative ranks across components are not identified'
        )

    use_sigma = judge_sigma
    if judge_sigma and len(judges) < 2:
        use_sigma = False
        warnings.append('single judge: sigma is unidentifiable, fitted plain BT instead')
        logger.warning('fit_bt: {}', warnings[-1])

    s = dict.fromkeys(items, 0.0)  # skills
    tau = dict.fromkeys(judges, 0.0)  # log sigmas
    # Adam state
    m_s = dict.fromkeys(items, 0.0)
    v_s = dict.fromkeys(items, 0.0)
    m_t = dict.fromkeys(judges, 0.0)
    v_t = dict.fromkeys(judges, 0.0)

    iterations = 0
    converged = False
    delta = math.inf
    still_streak = 0
    for step in range(1, _MAX_ITER + 1):
        iterations = step
        g_s = dict.fromkeys(items, 0.0)
        g_t = dict.fromkeys(judges, 0.0)
        for c in comparisons:
            p = min(max(c.p_a, _P_EPS), 1.0 - _P_EPS)
            sigma_k = math.exp(tau[c.judge]) if use_sigma else 1.0
            z = (s[c.item_a] - s[c.item_b]) / sigma_k
            p_hat = _logistic(z)
            err = (p - p_hat) * c.weight  # d(loglik)/dz, times record multiplicity
            g_s[c.item_a] += err / sigma_k
            g_s[c.item_b] -= err / sigma_k
            if use_sigma:
                g_t[c.judge] += -err * z  # chain rule through sigma_k = exp(tau)
        for i in items:
            g_s[i] -= _RIDGE * s[i]
        if use_sigma:
            for k in judges:
                g_t[k] -= _TAU_RIDGE * tau[k]

        max_g = max(abs(g) for g in g_s.values())
        if use_sigma:
            max_g = max(max_g, max(abs(g) for g in g_t.values()))
        if max_g < _GRAD_FLOOR:
            converged = True
            break

        # Adam ascent step with decaying lr, so the iterate settles instead of
        # oscillating around the optimum at a constant-lr noise floor.
        lr = _LR / (1 + step / _LR_DECAY_STEPS)
        prev_s = dict(s)
        prev_tau = dict(tau)
        for i in items:
            g = g_s[i]
            m_s[i] = _ADAM_B1 * m_s[i] + (1 - _ADAM_B1) * g
            v_s[i] = _ADAM_B2 * v_s[i] + (1 - _ADAM_B2) * g * g
            m_hat = m_s[i] / (1 - _ADAM_B1**step)
            v_hat = v_s[i] / (1 - _ADAM_B2**step)
            s[i] += lr * m_hat / (math.sqrt(v_hat) + _ADAM_EPS)
        if use_sigma:
            for k in judges:
                g = g_t[k]
                m_t[k] = _ADAM_B1 * m_t[k] + (1 - _ADAM_B1) * g
                v_t[k] = _ADAM_B2 * v_t[k] + (1 - _ADAM_B2) * g * g
                m_hat = m_t[k] / (1 - _ADAM_B1**step)
                v_hat = v_t[k] / (1 - _ADAM_B2**step)
                tau[k] += lr * m_hat / (math.sqrt(v_hat) + _ADAM_EPS)

        # Identifiability: skills are mean-centred (additive, Adam-neutral).
        # The sigma scale is anchored SOFTLY by the log-normal prior instead of
        # a hard geometric-mean projection: the projection's compensating skill
        # rescale is a multiplicative ratchet that bypasses Adam - on unanimous
        # panels every tau drifts down together each step, the rescale inflates
        # skills without bound, and Adam's normalised steps cap the ridge's
        # corrective pull at the learning rate. The prior has no such loophole.
        s_mean = sum(s.values()) / len(s)
        for i in items:
            s[i] -= s_mean

        delta = max(abs(s[i] - prev_s[i]) for i in items)
        if use_sigma:
            delta = max(delta, max(abs(tau[k] - prev_tau[k]) for k in judges))
        still_streak = still_streak + 1 if delta < _TOL else 0
        if still_streak >= _TOL_STREAK:
            converged = True
            break

    if not converged:
        warnings.append(f'fit stopped at the {_MAX_ITER}-iteration cap (max|step|={delta:.2e})')
        logger.warning('fit_bt: {}', warnings[-1])

    sigmas = {k: math.exp(tau[k]) for k in judges} if use_sigma else {}
    ranking = sorted(items, key=lambda i: -s[i])
    return BTFit(
        skills={i: s[i] for i in items},
        sigmas=sigmas,
        ranking=ranking,
        converged=converged,
        iterations=iterations,
        warnings=warnings,
    )


def cycle_rate(comparisons: Sequence[JudgedComparison], judge: str) -> float | None:
    """Directed 3-cycle rate of one judge's preferences (paper Eq. 9).

    The paper's independent validation signal: ``1/sigma`` correlates ~0.9 with
    ``1 - cycle_rate``. Needs at least 3 items with pairwise coverage; returns
    ``None`` when no complete triplet exists (e.g. the two-item A/B setting).
    Ties (``p == 0.5``) break no cycles and are treated as no-preference edges.

    One deliberate deviation from the paper's Eq. 9: the denominator here is
    the number of ASSESSABLE triplets (all three edges present and untied),
    not all C(n, 3) triplets. On fully covered tie-free data the two agree;
    on sparse or tie-heavy data the paper's form understates inconsistency by
    counting unassessable triplets as consistent.

    Repeated judgements of the same pair (e.g. from ``repetitions``) are
    aggregated by weighted mean preference before the sign is taken, so a
    judge that split evenly on a pair contributes a no-preference edge rather
    than whichever judgement happened to come last.
    """
    p_sum: dict[tuple[str, str], float] = defaultdict(float)
    w_sum: dict[tuple[str, str], float] = defaultdict(float)
    for c in comparisons:
        if c.judge != judge:
            continue
        # Canonicalize the pair frame so (a, b, p) and (b, a, 1-p) aggregate together.
        a, b, p = (c.item_a, c.item_b, c.p_a) if c.item_a <= c.item_b else (c.item_b, c.item_a, 1.0 - c.p_a)
        p_sum[a, b] += p * c.weight
        w_sum[a, b] += c.weight
    prefers: dict[tuple[str, str], bool] = {}
    for (a, b), total_w in w_sum.items():
        mean_p = p_sum[a, b] / total_w
        if mean_p > 0.5:
            prefers[a, b] = True
            prefers[b, a] = False
        elif mean_p < 0.5:
            prefers[a, b] = False
            prefers[b, a] = True

    items = sorted({i for pair in prefers for i in pair})
    n = len(items)
    if n < 3:
        return None

    def _wins(a: str, b: str) -> bool | None:
        return prefers.get((a, b))

    triplets = 0
    cycles = 0
    for x in range(n):
        for y in range(x + 1, n):
            for z in range(y + 1, n):
                a, b, c_ = items[x], items[y], items[z]
                ab, bc, ca = _wins(a, b), _wins(b, c_), _wins(c_, a)
                if ab is None or bc is None or ca is None:
                    continue  # incomplete or tied triplet: not assessable
                triplets += 1
                # a>b>c>a or a<b<c<a - either orientation is a cycle
                if (ab and bc and ca) or (not ab and not bc and not ca):
                    cycles += 1
    if triplets == 0:
        return None
    return cycles / triplets


def comparisons_per_judge(comparisons: Sequence[JudgedComparison]) -> dict[str, float]:
    """Comparison counts by judge (record weights included) - a cheap sanity
    check before trusting sigmas. ``bt_sigma_aggregation`` uses it to warn when
    a judge has too few decisive votes for its sigma to mean much."""
    counts: dict[str, float] = defaultdict(float)
    for c in comparisons:
        counts[c.judge] += c.weight
    return dict(counts)


__all__ = [
    'BTFit',
    'JudgedComparison',
    'binarize',
    'comparisons_per_judge',
    'cycle_rate',
    'fit_bt',
]
