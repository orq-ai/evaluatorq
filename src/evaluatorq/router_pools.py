"""Smart Router pool presets (RES-1345), generated rather than chosen.

A router pool is the candidate set the autorouter chooses among per request. It
differs from a jury in the way that matters for selection: a jury calls every
member and wants their errors to be independent, while a router calls exactly
one and wants the best value available under the caller's constraints.

RES-1345 names the pool shapes — which constraints customers actually have —
but never says what belongs inside one. Membership is therefore not authored
in this file: each pool is derived from its constraint predicate by
`common.model_frontier.derive_trio`, whose four criteria are each traceable to what
Arian's RES-1285 benchmark measured (see its docstring). The same function
that seats a model reseats it when the live garden changes, which is the
auto-update story: pools are derived from `/v2/models` on first use, never from
a committed copy.

Constraints (region, vendor, open weights) are applied *before* the frontier
is taken, never after. A pool is the best available set given what the caller
has ruled out, which is why the EU pool contains models the unconstrained
frontier does not.

Candidates additionally pass `common.model_frontier.seatable`: an index floor that
removes the noise tier, and the NEVER_SEAT families the platform will not
recommend at any score (gpt-oss, on the PR #330 review verdict).

Not every named shape ships: `DROPPED_POOLS` records each retirement reason, keyed
by pool name, so the ticket's list is answered name by name.

On cost: a pool has a range, not a figure. What a request actually costs
depends on the routing policy that picks within the pool, and routing policy
presets are explicitly out of scope here (INN-2686, ENG-1551). Quoting a
single number would mean inventing a traffic mix. `cost_per_1k_range` is the
honest form, computed from the same 1,500 input / 500 output assumption the
jury presets use.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — pydantic resolves `_PoolSpec.predicate` at runtime
from functools import lru_cache

from loguru import logger
from pydantic import BaseModel, Field, model_validator

from evaluatorq.common.model_frontier import (
    ScoredModel,
    derive_trio,
    derived_successor,
    load_garden_sync,
    pareto_frontier,
    seatable,
)
from evaluatorq.jury_presets import (
    ESTIMATED_COMPLETION_TOKENS,
    ESTIMATED_PROMPT_TOKENS,
    judge_family,
    model_identity,
)


class RouterPool(BaseModel):
    """A named candidate set for the autorouter, plus the constraint that shaped it."""

    name: str
    models: tuple[str, ...]
    use_when: str
    reserve_models: tuple[str, ...] = ()
    reserve_gap_reason: str = ''
    constraint: str = Field(
        description='What was ruled out before the frontier was taken, in words.',
    )
    index_range: tuple[float, float] = Field(
        description='Lowest and highest index in the pool, each at the effort named in `default_efforts`.'
    )
    default_efforts: dict[str, str | None] = Field(
        default_factory=dict,
        description='The catalogue default effort each member runs and is ranked at. None means '
        'the card exposes no effort parameter.',
    )
    pareto_scores: dict[str, float] = Field(
        default_factory=dict,
        description='Default-effort intelligence index per blended USD per 1M tokens, per member.',
    )
    price_range: tuple[float, float] = Field(
        description='Cheapest and dearest blended USD per 1M tokens, 3:1 input/output.'
    )
    cost_per_1k_range: tuple[float, float] = Field(
        description='USD per 1,000 requests at 1,500 input / 500 output tokens, if every '
        'request went to the cheapest member, and if every request went to the dearest.'
    )
    tier_ratios: tuple[float, ...] = Field(
        default=(),
        description='Price ratio between adjacent members, dearest first. What the cost profile has to work with.',
    )
    relaxed_tier_rule: bool = Field(
        default=False,
        description='True when no trio under this constraint reaches 2.75x tiers and the pool '
        'took the widest spread available instead. A relaxed pool routes for less '
        'saving than an open one, and the flag is how a caller can say so.',
    )
    measured: bool = Field(
        default=False,
        description='True only for the RES-1285 benchmarked trio, whose membership is a measurement, not a derivation.',
    )

    @model_validator(mode='after')
    def _pool_is_well_formed(self) -> RouterPool:
        """A one-model pool is not a pool, and a repeat is not a choice."""
        if len(self.models) < 2:
            raise ValueError(f'{self.name}: a pool needs at least two models')
        if len(set(self.models)) != len(self.models):
            raise ValueError(f'{self.name}: duplicate model')
        if set(self.reserve_models) & set(self.models):
            raise ValueError(f'{self.name}: a reserve cannot also be seated')
        if not self.reserve_models and not self.reserve_gap_reason:
            raise ValueError(f'{self.name}: name a reserve or write down why none exists')
        return self


def _load_scored() -> dict[str, ScoredModel]:
    return load_garden_sync()


def _family(model: ScoredModel) -> str | None:
    try:
        return judge_family(model.router_id)
    except ValueError:
        return None


# Lineages that publish downloadable weights, which is what the Open-Weight /
# Portable constraint actually asks for. `minimax` was missing while the jury of
# the same name held `MiniMax-M2.7` as its reserve, so the preset's own reserve
# would have broken the preset's own pitch. M2.7 publishes 228B safetensors and
# is served by three hosts other than MiniMax, so it is portable in the sense
# the constraint means; the licence is MiniMax's own rather than MIT or Apache,
# which is a redistribution question and not a self-hosting one.
_OPEN_WEIGHT_FAMILIES = {'zhipu', 'moonshot', 'deepseek', 'alibaba', 'meta', 'minimax'}


def _is_open_weight(model: ScoredModel) -> bool:
    return _family(model) in _OPEN_WEIGHT_FAMILIES or 'gpt-oss' in model.router_id


class _PoolSpec(BaseModel):
    """One RES-1345 shape: its constraint in words and as a predicate."""

    model_config = {'arbitrary_types_allowed': True}

    name: str
    constraint: str
    predicate: Callable[[ScoredModel], bool]
    use_when: str


# EU-pinned endpoints are excluded from the general-purpose shapes for the same
# reason `model_frontier.region_compatible` exists: seating one would silently
# pin an unconstrained preset to a region the caller never asked for.
_SPECS: tuple[_PoolSpec, ...] = (
    _PoolSpec(
        name='Balanced Pareto Trio',
        constraint='none',
        predicate=lambda m: m.region != 'europe',
        use_when=(
            'The default when nothing else is constrained. The same shape RES-1285 '
            'measured (99.2% retention at 17.1% savings, on an earlier garden), '
            'refreshed to the current frontier. The figure belongs to the '
            'measurement, not to these seats: revalidate before quoting it for '
            'this exact lineup.'
        ),
    ),
    _PoolSpec(
        name='OpenAI Only',
        constraint='single vendor: OpenAI lineage',
        predicate=lambda m: _family(m) == 'openai' and m.region != 'europe',
        use_when=(
            'Procurement is already through OpenAI and adding a vendor means a new '
            'contract. Members come from the live constrained frontier, not '
            'a fixed generation label. The relaxed flag records a tier spread '
            'the single-vendor catalogue cannot satisfy at the normal threshold.'
        ),
    ),
    _PoolSpec(
        name='Anthropic Only',
        constraint='single vendor: Anthropic lineage',
        predicate=lambda m: _family(m) == 'anthropic' and m.region != 'europe',
        use_when=(
            'Same procurement argument, Anthropic side. The tier rule is relaxed '
            'here and the pool is weaker for it: the served lineup spans a fifth of '
            'the price range a mixed pool routes across. A single-vendor pool '
            'simply has less room, and this is where that shows. EU-pinned '
            'endpoints are excluded like everywhere else; residency buyers take '
            'EU Region Only.'
        ),
    ),
    _PoolSpec(
        name='EU Region Only',
        constraint='region: europe',
        predicate=lambda m: m.region == 'europe',
        use_when=(
            'Data residency. The residency filter is applied before the frontier is '
            'taken, never after, so these are the best available inside the EU '
            'rather than the best overall minus the ones that failed a check. The '
            'widest tier spread in the library, because EU hosting now includes '
            'cheap open-weight endpoints alongside frontier Claude.'
        ),
    ),
)

# RES-1345 names these shapes, but derivation can collapse one onto another, and
# two names for one pool is padding, not choice. Recorded here so the ticket's
# list is answered in full.
DROPPED_POOLS: dict[str, str] = {
    'Max Savings': (
        'Retired 2026-08-24 when the seatability floor left the sub-dollar '
        'frontier as two DeepSeek models, briefly reinstated on the 2026-08-25 '
        're-capture when luna and MiniMax M3 came in under a dollar, and retired '
        'again the same day: a live probe found wafer/MiniMax-M3 returns 404 on a '
        'completion call despite being listed, so it is now in UNROUTABLE and the '
        'cheap frontier is back to DeepSeek plus luna. The derivation will seat '
        'deepseek-v4-pro and deepseek-v4-flash together to fill the third chair, '
        'which is one vendor holding two of three seats and the correlated '
        'failure a pool exists to avoid. Cost-driven routing takes Balanced '
        'Pareto Trio. Reinstate when the sub-dollar band spans three lineages.'
    ),
    'Measured Default': (
        'Retired on PR #330 review (2026-08-24): it seated an EU-pinned endpoint '
        '(google/eu.claude-sonnet-5) in an unconstrained pool and a NEVER_SEAT '
        'model (groq/openai/gpt-oss-120b). A measured configuration cannot be '
        'edited without invalidating the measurement, so it retires whole. The '
        'RES-1285 run (99.2% retention at 17.1% savings over 524 reference-graded '
        'prompts) stays documented with that ticket; Balanced Pareto Trio is the '
        'same shape derived against the current frontier.'
    ),
    'Bedrock Only': (
        'Retired on PR #330 review follow-up (2026-08-25): the Bedrock EU '
        'catalog serves exactly three Anthropic models (opus-4-5, sonnet-4, '
        'haiku-4-5), so the pool was the constrained garden verbatim, with no '
        'derivation and no choice in any seat. Its middle seat, sonnet-4, '
        'scores 25.5 at three times the haiku price, which is close to dead '
        'weight. On 2026-11-10 the catalog had five cards but a two-model '
        'seatable frontier (haiku-5-5, opus-5-5): haiku-4-5 and sonnet-4 are '
        'dominated by haiku-5-5 and opus-4-5 is unrankable at its default. '
        'Reinstate when the seatable Bedrock frontier is wide enough for the '
        'derivation to have something to decide.'
    ),
    'Open-Weight / Portable': (
        'Derives to the same three models as Balanced Pareto Trio: ranked at '
        'their default efforts, the open-weight models are the whole '
        'unconstrained frontier. One pool, one name. Reinstate when a closed '
        'model re-enters the default trio.'
    ),
    'Value Stack': (
        'Derives to the same three models as Balanced Pareto Trio: the value '
        'lineages are the open-weight frontier, which is the default frontier. '
        'One pool, one name.'
    ),
}

# The spec remains available for derivation; this is an explicit data blocker,
# not a retirement or permission to seat a dominated/unmeasured third model.
REVIEWED_UNDERIVABLE_POOLS: dict[str, tuple[tuple[str, ...], str]] = {
    'Anthropic Only': (
        ('claude-haiku-4-5-20251001', 'claude-opus-5-5'),
        (
            'The 2026-10-02 catalogue supports only Haiku 4.5 (no effort '
            'parameter, 23.7, $2.00 blend) and Opus 5.5 (medium, '
            '51.2434931792768, $8.00 blend). Ten other scored non-EU cards '
            'lack their default score in this cache; Sonnet 5.5 is unscored. '
            'Do not fold the non-reasoning 4.6 high headlines into adaptive high. '
            'Current coherent AA v4.3.2 measures four relevant defaults: '
            'Sonnet 5 high 31.6623246664651, Sonnet 5.5 high 46.7533528066958, '
            'Opus 5 high 48.1219182595908 and Opus 5.5 medium 51.2434931792768. '
            'Dominance leaves Sonnet 5.5 and Opus 5.5; current Haiku 15.410949885381 '
            'is estimated, not measured. The cache is ingestion-blocked, not proof '
            'of upstream absence. Keep the spec; never splice versions to add a third.'
        ),
    ),
    'OpenAI Only': (
        ('gpt-6-luna', 'gpt-6-sol'),
        (
            'The 2026-11-10 non-EU OpenAI frontier is two models: GPT-6 Sol '
            '(medium 39.8, $2/$10) and GPT-6 Luna (medium 29.9, $0.10/$0.50). '
            'They dominate every other seatable card: GPT-5.6 Sol 39.2 at $4/$20, '
            'GPT-5.6 Terra 30.1 at $2/$12, GPT-5.6 Luna 25.0 at $0.20/$1.20, '
            'GPT-5.4 18.2 and GPT-5 22.9. GPT-6.1 Sol is EU-only and GPT-6.1 '
            'Terra is unscored. Keep the spec; it derives again once a third '
            'GPT-6-generation tier is scored.'
        ),
    ),
}


def request_cost_per_1k(model: ScoredModel) -> float:
    """USD per 1,000 requests to one model at the jury estimate, from its live card rates."""
    return (model.input_price * ESTIMATED_PROMPT_TOKENS + model.output_price * ESTIMATED_COMPLETION_TOKENS) / 1_000


def _build_pool(
    spec_name: str,
    constraint: str,
    use_when: str,
    members: list[ScoredModel],
    frontier: list[ScoredModel],
    *,
    relaxed: bool,
    measured: bool = False,
    reserve_gap_reason: str = '',
) -> RouterPool:
    members = sorted(members, key=lambda m: -m.price)
    seated_identities = {model_identity(m.router_id) for m in members}
    remaining = [
        m
        for m in frontier
        if m.router_id not in {s.router_id for s in members} and model_identity(m.router_id) not in seated_identities
    ]
    reserves = (max(remaining, key=lambda m: m.ranked_index).router_id,) if remaining else ()
    costs = [request_cost_per_1k(m) for m in members]
    prices = [m.price for m in members]
    indices = [m.ranked_index for m in members]
    return RouterPool(
        name=spec_name,
        models=tuple(m.router_id for m in members),
        use_when=use_when,
        reserve_models=reserves,
        reserve_gap_reason='' if reserves else reserve_gap_reason,
        constraint=constraint,
        index_range=(min(indices), max(indices)),
        default_efforts={m.router_id: m.default_effort for m in members},
        pareto_scores={m.router_id: score for m in members if (score := m.pareto_score) is not None},
        price_range=(min(prices), max(prices)),
        cost_per_1k_range=(round(min(costs), 2), round(max(costs), 2)),
        tier_ratios=tuple(round(prices[i] / prices[i + 1], 1) for i in range(len(prices) - 1)),
        relaxed_tier_rule=relaxed,
        measured=measured,
    )


def _no_reserve_reason(constraint: str) -> str:
    return (
        f'The frontier under "{constraint}" is exhausted by the seats: every other '
        'qualifying model is either Pareto-dominated by a seat or the same weights '
        'under a second host. There is no fourth distinct model to hold back.'
    )


def build_pools(scored: dict[str, ScoredModel] | None = None) -> tuple[dict[str, RouterPool], dict[str, str]]:
    """Derive every RES-1345 pool from a garden (the live one when none is given).

    Returns (pools, underivable): the pools that could be seated, paired with
    the specs whose frontier could not seat a trio, mapped to the reason. The
    register is returned with the pools it describes, never held as state a
    second call could overwrite: `build_pools(other_garden)` leaves what
    `get_pools()` returns untouched. A spec that cannot seat a trio degrades
    into the register instead of raising, so one thin constraint never takes
    the other pools down with it.
    """
    scored = scored if scored is not None else _load_scored()
    # enabled is False only when the catalog says the workspace switched the
    # model off: listed and scored, but not servable, so not seatable. None
    # (a card without the field) passes.
    # Seatability applies before any constraint: the floor removes the noise
    # tier and NEVER_SEAT removes families the platform will not recommend
    # regardless of score (model_frontier documents both).
    models = [m for m in scored.values() if m.enabled is not False and seatable(m)]
    pools: dict[str, RouterPool] = {}
    underivable: dict[str, str] = {}

    for spec in _SPECS:
        eligible = {m.router_id: m for m in models if spec.predicate(m)}
        # An old version is retired once a newer one is seatable in this same
        # pool; a successor outside the constraint (say, not EU) retires nothing.
        frontier = pareto_frontier([m for m in eligible.values() if not derived_successor(m.router_id, eligible)])
        # A garden that empties a constraint (say, every EU host delisted) must
        # not take the other pools down with it: catch the failure into the
        # returned register, and let the test that checks every spec answers
        # with a pool or a reason fail.
        try:
            members, relaxed = derive_trio(frontier, model_identity)
        except ValueError as error:
            underivable[spec.name] = f'{spec.constraint}: {error}'
            logger.warning('router pool {!r} is not derivable from this garden: {}', spec.name, error)
            continue
        # Not caught: derive_trio is the one call that signals "no trio" by
        # design, so a malformed pool here is a bug and must raise.
        pools[spec.name] = _build_pool(
            spec.name,
            spec.constraint,
            spec.use_when,
            members,
            frontier,
            relaxed=relaxed,
            reserve_gap_reason=_no_reserve_reason(spec.constraint),
        )
    return pools, underivable


@lru_cache(maxsize=1)
def _live_pools() -> tuple[dict[str, RouterPool], dict[str, str]]:
    return build_pools()


def get_pools() -> dict[str, RouterPool]:
    """Every pool derived from the live garden, built on first call and cached for the process.

    Returns a copy of the register, so adding or removing a pool cannot change what the next caller sees.
    """
    return dict(_live_pools()[0])


def get_underivable_pools() -> dict[str, str]:
    """Specs the live garden could not seat, mapped to the reason. Empty on a healthy garden."""
    return dict(_live_pools()[1])


def all_pool_model_ids() -> set[str]:
    """Every distinct model any pool can route to, reserves included."""
    return {model_id for pool in get_pools().values() for model_id in (*pool.models, *pool.reserve_models)}


def pool_is_frontier(pool: RouterPool, scored: dict[str, ScoredModel]) -> list[tuple[str, str]]:
    """Members beaten on both axes by another member: (loser, winner) pairs.

    Empty for every derived pool by construction.
    """
    members = [scored[m] for m in pool.models if m in scored]
    return [(loser.router_id, winner.router_id) for loser in members for winner in members if winner.dominates(loser)]
