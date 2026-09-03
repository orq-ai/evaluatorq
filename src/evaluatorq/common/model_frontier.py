"""Intelligence/price frontier over the model garden (RES-1347).

Ported from the research repo (`orq_shared.model_frontier`), which is where the
seating arithmetic is derived and where router pools, classifiers and evaluator
groups use it too. Only the part the jury presets need lives here: the trio
derivation for router pools stayed behind, since evaluatorq ships no pools.

Presets rot because models move, and the only computable axis used to be price,
so "is this successor better?" was a judgement call. Every autorouter-eligible
model card carries `metadata.autorouter.v2.intelligence_index` (an Artificial
Analysis score) alongside a 3:1 input/output blended `price`. That pair is
enough to decide seating arithmetically rather than by taste.

Dominance is the whole rule: B dominates A when B is at least as intelligent and
no more expensive, and strictly better on one of them. A dominated seat is dead
weight the user cannot see. A successor that dominates the incumbent is a
straight upgrade; one that does not is a trade the preset's `use_when` has to
justify in words.

The snapshot is committed rather than fetched. `common.model_catalogue` reads
live prices from `/v2/models`, which is right for costing a run and wrong for
deciding what a preset means: a preset ships to customers on their own
contracts, so it must not change because our gateway is down or our workspace
has a model switched off.
"""

import json
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field


class ScoredModel(BaseModel):
    """A garden model with the two axes seating decisions are made on."""

    router_id: str
    intelligence_index: float = Field(
        description=(
            'The headline Artificial Analysis index on the card, stamped at whichever '
            'rung AA published as the default. Kept for provenance; seating ranks on '
            '`ceiling_index` instead.'
        )
    )
    price: float = Field(description='USD per 1M tokens, 3:1 input/output blend, lower is better.')
    region: str | None = None
    provider: str | None = None
    # The reasoning effort the index was measured at, and the index at every
    # effort the card offers. Absent on snapshots captured before 2026-08-24,
    # which is how `gpt-5.4` came to be seated on its xhigh score against a
    # price billed at effort `none`.
    reasoning_effort: str | None = None
    effort_indices: dict[str, float] = {}
    # The effort the index above was actually measured at, resolved from the
    # benchmark slug, which is not always the effort the card is called at.
    # Null where the card's headline effort is not one it enumerates.
    scored_effort: str | None = None
    # False means the workspace has the model switched off: listed, scored,
    # not servable. None on older snapshots, which recorded only enabled cards.
    enabled: bool | None = None

    @property
    def ladder(self) -> dict[str, float]:
        """Every rung this card measures, including the one its headline came from.

        The capture writes the headline rung into `effort_indices`, because AA
        publishes it in a field apart from the ladder and seven Anthropic-lineage
        cards never repeat it there. This repeats the fold for cards built by hand
        and for older snapshots, and is a no-op on a current one. It never
        overwrites a published rung, so the one card whose headline conflicts with
        its own rung, `kimi-k2.6`, is still refused by `seatable`.

        Empty for a card that publishes no ladder at all: one number, no claim
        about how it breaks down.
        """
        if not self.effort_indices:
            return {}
        rungs = dict(self.effort_indices)
        rungs.setdefault(self.scored_effort or self.reasoning_effort or 'none', self.intelligence_index)
        return rungs

    @property
    def ceiling_index(self) -> float:
        """The best index this card measures, at whatever rung reaches it.

        This, not `intelligence_index`, is what every seating decision ranks on.
        The headline is stamped at whichever rung Artificial Analysis happened to
        publish as the card's default, so ranking on it compares a model measured
        at `none` against one measured at `max` and calls the result a comparison.
        It is also unstable in a way that has already moved seats: `gpt-5.4` left
        every seat on the 2026-08-25 re-capture because AA restamped its headline
        from `xhigh` to `none`, while the model itself did not change.

        Cards with no ladder publish one number and it stands as the ceiling.
        """
        return max(self.ladder.values(), default=self.intelligence_index)

    @property
    def ceiling_effort(self) -> str | None:
        """The rung the ceiling was measured at, cheapest first on ties.

        None where the card has no ladder, which means the effort to send is
        whatever the provider defaults to. `reasoning` sorts last because it is
        not on the graded scale, so a card that reaches the same index with and
        without thinking is reported as reaching it without.
        """
        rungs = self.ladder
        if not rungs:
            return None
        best = max(rungs.values())
        reached = [effort for effort, index in rungs.items() if index == best]
        return min(reached, key=lambda e: EFFORT_ORDER.index(e) if e in EFFORT_ORDER else len(EFFORT_ORDER))

    @property
    def priced_at_ceiling(self) -> bool:
        """Whether the captured price is the price at the rung this model is ranked at.

        False means the 3:1 blend was captured at a cheaper rung than the one the
        ceiling was measured at, so the seat's quality and its cost are read at
        different operating points. Recorded rather than corrected: correcting it
        needs a probe at the seated effort for every seat.
        """
        return self.ceiling_effort in (None, self.reasoning_effort or 'none')

    def dominates(self, other: 'ScoredModel') -> bool:
        """At least as good on both axes, strictly better on one."""
        at_least_as_good = self.ceiling_index >= other.ceiling_index and self.price <= other.price
        strictly_better = self.ceiling_index > other.ceiling_index or self.price < other.price
        return at_least_as_good and strictly_better


# Below this index nothing is recommended anywhere: the tier is noise, and a
# cheap seat filled from it makes a panel look complete while lowering what the
# user gets. The value sits under every seat a reviewer has accepted (Claude
# Haiku 4.5 at 23.7 is the platform's single-judge baseline) and above the band
# it exists to exclude, which on the 2026-08-25 garden is Llama-3.3-70B at 9.4,
# the 4o generation at 6.9 to 11.2, qwen3-coder at 13.6 and gpt-oss-20b at 14.9.
MIN_SEAT_INDEX = 15.0

# The index is a benchmark aggregate, and a floor cannot catch a model whose
# benchmarks say more than its behaviour: gpt-oss-120b scores 23.8, a hair above
# Haiku 4.5's 23.7, and does not perform beside it. Families here are never
# seated at any score or price, with the verdict on record.
NEVER_SEAT: dict[str, str] = {
    'gpt-oss': (
        'Benchmark-inflated relative to observed quality: scores beside Claude '
        'Haiku 4.5 on the index and does not deliver beside it. Not to be '
        'recommended at any price tier.'
    ),
}


# Listing is not callability. `GET /v2/models` advertises these router IDs and
# `client.models.list()` returns them, so every offline freshness test passes,
# but an actual completion call 404s: the host serves a catalog that does not
# include the model the router ID names. Only a live probe finds this, which is
# why entries carry the date and the error they were found by.
UNROUTABLE: dict[str, str] = {
    'wafer/MiniMax-M3': (
        'Listed by the models API and seated in two pools, but a completion call '
        "returns 404 'Model MiniMax-M3 is not available on this endpoint'. The "
        'wafer host serves DeepSeek-V4-Flash, GLM-5.2, Kimi-K2.6, Kimi-K3 and '
        'Qwen3.5-397B-A17B only (probed 2026-08-25). The same weights are '
        'callable as nvidia/minimaxai/minimax-m3, which the capture skips for '
        'pricing at zero. Drop this entry when either is fixed.'
    ),
}


# The two `gpt-5-mini` cards read `minimal 14.3, medium 30.9, high 25.3`: more
# effort scoring lower, which is measurement noise rather than a ladder. Their
# ceiling would be a rung the card cannot be trusted about, and this is the same
# model whose 1.1-point gap motivated `MIN_INDEX_GAIN`.
NON_MONOTONE_LADDERS: dict[str, str] = {
    'openai/gpt-5-mini': 'high (25.3) scores below medium (30.9) on its own card; not rankable.',
    'azure/gpt-5-mini': 'high (25.3) scores below medium (30.9) on its own card; not rankable.',
}


# The garden speaks two disjoint effort vocabularies, not one scale with gaps.
# Binary is {none, reasoning}: 21 cards across 10 providers where thinking is
# off or on and there are no levels. Graded is {minimal, low, medium, high,
# xhigh, max}, OpenAI-shaped, with `xhigh` azure and openai only. `reasoning`
# never co-occurs with a graded rung on any card, and `none` is the only rung
# the two dialects share.
#
# `reasoning` is deliberately absent below. Placing it at `high` would be a
# guess: the none-to-reasoning uplift averages 6.6 points across the 21 cards
# with a wide spread. This tuple orders the graded dialect only, and the order
# is monotone on 90 of 92 cards (the two exceptions are in
# `NON_MONOTONE_LADDERS`). Comparisons do not use it to pair rungs across cards;
# that is what `ceiling_index` is for.
EFFORT_ORDER = ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max')


MIN_INDEX_GAIN = 2.0
"""Index points a *different* model must gain before it counts as an upgrade.

The Artificial Analysis index is a benchmark aggregate, not a measurement with
an error bar, and sub-point gaps between neighbouring tiers are not signal. Left
ungated, the arithmetic proposes seating `gpt-5-mini` (index 30.9) over
`gpt-5.4-mini` (29.8) purely because it is cheaper, which trades a generation
and a knowledge cutoff for 1.1 points of benchmark noise.

The margin is deliberately asymmetric: a swap must clear it, keeping an
incumbent never does. A swap carries churn a retained seat does not - a new
knowledge cutoff, new failure modes, re-probing, a diff a customer sees - so the
burden of proof sits on the change. Defending an incumbent against the
arithmetic still requires a written reason at the call site, it just does not
require an index margin.
"""


def headline_contradicts_its_own_rung(model: ScoredModel) -> bool:
    """Whether the card publishes a different index for the very rung its headline came from.

    A card that scores its own called rung twice, differently, is not a
    measurement any seat can rest on. `kimi-k2.6` reads a headline of 44.2 taken
    at effort `none` beside a `none` rung of 34.6: one of the two is wrong and
    the card does not say which, so it holds no seat until the card is fixed.

    This is narrower than "the headline must appear on the ladder", which would
    also refuse seven Anthropic cards whose ladders merely omit the `high` rung
    they were scored at; `ladder` folds those back in.
    """
    rung = model.scored_effort or model.reasoning_effort or 'none'
    return rung in model.effort_indices and model.effort_indices[rung] != model.intelligence_index


SNAPSHOT_PATH = Path(__file__).parent / 'data' / 'model_garden.json'


@lru_cache(maxsize=1)
def load_snapshot() -> dict[str, object]:
    """The committed model-garden capture, as written by the research repo's refresh script."""
    return json.loads(SNAPSHOT_PATH.read_text())


@lru_cache(maxsize=1)
def load_garden() -> dict[str, ScoredModel]:
    """Every scored card in the committed garden snapshot, keyed by router ID."""
    scored = cast('dict[str, dict[str, Any]]', load_snapshot()['scored_models'])
    return {router_id: ScoredModel(router_id=router_id, **fields) for router_id, fields in scored.items()}


def load_judge_pricing() -> dict[str, dict[str, float]]:
    """Captured USD-per-1M input/output rates for every router ID a preset can call.

    This is what the published $/1k figures are recomputed from. Separate from
    `scored_models`, whose `price` is a 3:1 blend used for ranking and cannot be
    split back into the two rates a cost estimate needs.
    """
    return cast('dict[str, dict[str, float]]', load_snapshot()['preset_judge_pricing'])


def seated_effort(router_id: str) -> str | None:
    """The reasoning effort a seat is ranked at, and so the effort a caller must send.

    None where the card publishes no ladder, which means the provider default
    stands, or where the garden does not score the model at all.
    """
    model = load_garden().get(router_id)
    return model.ceiling_effort if model else None


def seatable(model: ScoredModel) -> bool:
    """Whether a model may hold any recommended seat, pool or jury."""
    if model.ceiling_index < MIN_SEAT_INDEX:
        return False
    if headline_contradicts_its_own_rung(model):
        return False
    if model.router_id in NON_MONOTONE_LADDERS:
        return False
    if model.router_id in UNROUTABLE:
        return False
    return not any(family in model.router_id for family in NEVER_SEAT)


def region_compatible(candidate: ScoredModel, incumbent: ScoredModel) -> bool:
    """Whether a candidate can stand in for an incumbent without moving regions.

    Without this guard the arithmetic gives bad advice. `azure/eu.gpt-5.6-luna`
    beats `openai/gpt-5.4-mini` on both axes, so raw dominance says swap, but
    Balanced Trio is general-purpose and the swap would silently pin it to the
    EU. A region-pinned model may only replace another region-pinned one.
    """
    pinned = {'europe'}
    if incumbent.region in pinned:
        return candidate.region == incumbent.region
    return candidate.region not in pinned


def upgrade_verdict(incumbent: ScoredModel, successor: ScoredModel) -> tuple[str, str]:
    """Whether a successor should take a seat, and why, in one computed answer.

    Deliberately three-valued. 'seat' and 'reject' are arithmetic. 'trade' is the
    honest middle: the successor is better on one axis and worse on the other,
    which no index can settle, so it goes back to a human with the numbers
    attached rather than being auto-applied or silently dropped.

    Both indices are ceilings, so the quality axis reads the same quantity for
    every card whichever effort dialect it speaks. The price axis does not: see
    `priced_at_ceiling`.
    """
    if headline_contradicts_its_own_rung(successor):
        return 'trade', (
            f'{successor.router_id} reads {successor.intelligence_index} at effort '
            f'{successor.reasoning_effort or "none"} while its own card scores that rung at '
            f'{successor.effort_indices.get(successor.reasoning_effort or "none")}. One of the '
            'two is wrong, so this goes to a human, not to the arithmetic.'
        )
    if successor.dominates(incumbent):
        return 'seat', (
            f'{successor.router_id} dominates {incumbent.router_id}: '
            f'index {successor.ceiling_index} at {successor.ceiling_effort} vs '
            f'{incumbent.ceiling_index} at {incumbent.ceiling_effort}, '
            f'price {successor.price} vs {incumbent.price}.'
        )
    if incumbent.dominates(successor):
        return 'reject', (
            f'{incumbent.router_id} still dominates {successor.router_id}: '
            f'index {incumbent.ceiling_index} at {incumbent.ceiling_effort} vs '
            f'{successor.ceiling_index} at {successor.ceiling_effort}, '
            f'price {incumbent.price} vs {successor.price}.'
        )
    return 'trade', (
        f'{successor.router_id} trades against {incumbent.router_id}: '
        f'index {successor.ceiling_index} vs {incumbent.ceiling_index}, '
        f'price {successor.price} vs {incumbent.price}. '
        'Neither dominates, so seating it is a stated preference, not an upgrade.'
    )


def in_family_upgrades(
    incumbent: ScoredModel,
    models: list[ScoredModel],
    family_of: Callable[[str], str],
    identity_of: Callable[[str], str] | None = None,
    min_index_gain: float = MIN_INDEX_GAIN,
) -> list[ScoredModel]:
    """Same-lineage models that beat this seat on both axes without moving region.

    This, not the global frontier, is the seating rule for any preset that buys
    diversity. A jury seats three lineages on purpose: judged globally, 19 of 24
    seats look dominated, and "fixing" them would collapse the panel onto
    whichever vendor is cheapest that month, which is the correlated-error
    failure the jury exists to prevent. The lineage is a requirement, so the only
    honest question is whether a seat is the best buy *within* it.
    """
    try:
        family = family_of(incumbent.router_id)
    except ValueError:
        return []
    better = []
    for m in models:
        if m.router_id == incumbent.router_id:
            continue
        try:
            if family_of(m.router_id) != family:
                continue
        except ValueError:
            continue
        if not region_compatible(m, incumbent):
            continue
        if headline_contradicts_its_own_rung(m):
            # The card contradicts itself, so there is no number to upgrade to.
            continue
        m_index, incumbent_index = m.ceiling_index, incumbent.ceiling_index
        beats = (
            m_index >= incumbent_index
            and m.price <= incumbent.price
            and (m_index > incumbent_index or m.price < incumbent.price)
        )
        if not beats:
            continue
        # Same weights served by another host is pure price arbitrage: there is
        # no quality question to answer, so no index margin is required.
        same_model = identity_of is not None and identity_of(m.router_id) == identity_of(incumbent.router_id)
        if same_model or m_index - incumbent_index >= min_index_gain:
            better.append(m)
    return sorted(better, key=lambda m: -m.ceiling_index)
