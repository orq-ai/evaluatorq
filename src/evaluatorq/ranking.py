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

Everything is pure Python on purpose: the problem is tiny (N items + K judges
parameters), closed-form gradients converge in milliseconds, and evaluatorq's
runtime dependency set stays unchanged (no scipy/numpy).

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
_MAX_ITER = 2000
# Converged when the iterate stops moving (max-abs parameter change after the
# identifiability projections). A gradient criterion is wrong here: the ridge
# leaves a constant radial gradient along the scale direction that the
# projection undoes every step, so the raw gradient never reaches zero even at
# the constrained optimum. 1e-4 is far below anything that moves a ranking or
# a vote weight (skills live on a ~+-10 scale, sigmas are reported to 3
# decimals); tighter tolerances just fight Adam's decay tail, which unanimous
# panels ride to the iteration cap at ~3e-5 steps. Steps this small only occur
# near stationarity, and the exact-saddle case has its own gradient floor.
_TOL = 1e-4
_ADAM_B1 = 0.9
_ADAM_B2 = 0.999
_ADAM_EPS = 1e-8
# Weak gaussian prior on skills: keeps disconnected comparison graphs and
# perfectly separated items (all wins one way -> unbounded skill MLE, which
# otherwise drifts to the iteration cap exactly like the sigma case below)
# finite and quick to converge. Shrinks only the MAGNITUDE of extreme gaps;
# orderings, and therefore rankings and weighted winners, are untouched.
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


class BTFit(BaseModel):
    """Result of a Bradley-Terry family fit."""

    skills: dict[str, float] = Field(description='Latent skill per item, centred at mean 0')
    sigmas: dict[str, float] = Field(
        description='Per-judge discriminator; smaller = sharper/more reliable. Geometric mean fixed at 1. '
        'Empty when fitted without judge_sigma.'
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
    for step in range(1, _MAX_ITER + 1):
        iterations = step
        g_s = dict.fromkeys(items, 0.0)
        g_t = dict.fromkeys(judges, 0.0)
        for c in comparisons:
            p = min(max(c.p_a, _P_EPS), 1.0 - _P_EPS)
            sigma_k = math.exp(tau[c.judge]) if use_sigma else 1.0
            z = (s[c.item_a] - s[c.item_b]) / sigma_k
            p_hat = _logistic(z)
            err = p - p_hat  # d(loglik)/dz
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

        # Identifiability projections: skills mean-zero; geometric mean of
        # sigma fixed at 1 (only relative sigmas are meaningful, paper 3.3).
        s_mean = sum(s.values()) / len(s)
        for i in items:
            s[i] -= s_mean
        if use_sigma:
            t_mean = sum(tau.values()) / len(tau)
            if t_mean:
                for k in judges:
                    tau[k] -= t_mean
                # Rescaling sigma rescales the skill space; undo it so the
                # constraint does not fight the gradient direction.
                scale = math.exp(-t_mean)
                for i in items:
                    s[i] *= scale

        delta = max(abs(s[i] - prev_s[i]) for i in items)
        if use_sigma:
            delta = max(delta, max(abs(tau[k] - prev_tau[k]) for k in judges))
        if delta < _TOL:
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
    """
    prefers: dict[tuple[str, str], bool] = {}
    for c in comparisons:
        if c.judge != judge:
            continue
        if c.p_a > 0.5:
            prefers[c.item_a, c.item_b] = True
            prefers[c.item_b, c.item_a] = False
        elif c.p_a < 0.5:
            prefers[c.item_a, c.item_b] = False
            prefers[c.item_b, c.item_a] = True

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


def comparisons_per_judge(comparisons: Sequence[JudgedComparison]) -> dict[str, int]:
    """Comparison counts by judge - a cheap sanity check before trusting sigmas."""
    counts: dict[str, int] = defaultdict(int)
    for c in comparisons:
        counts[c.judge] += 1
    return dict(counts)


__all__ = [
    'BTFit',
    'JudgedComparison',
    'binarize',
    'comparisons_per_judge',
    'cycle_rate',
    'fit_bt',
]
