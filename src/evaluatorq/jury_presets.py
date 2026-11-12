"""LLM-as-a-jury preset definitions (RES-1171, RES-1346).

The single source for what a named jury preset means: which judges, which
aggregation mode, and which reserve judges replace them when one is retired,
deprecated or repriced. `llm_jury(preset=...)` builds a panel from here.

Nothing about a seat is committed except the decision to seat it. Rates and
default efforts are read from the live model garden (`common.model_frontier`,
which reads `/v2/models`), and the live drift suite recomputes every published
figure against it, so a repricing or a changed default shows up as a failing
check rather than in a refreshed file's diff. The freshness registers at the
bottom of this file are the written record that suite reads: every newer
model and stale seat is seated or argued for in words.

Seats are chosen here. Compare intelligence at the catalogue default reasoning
effort against price within each lineage. A card with no score at its default
effort ranks on its headline index, which has no reasoning level.
Seats are compared in-family, because the lineage diversity is the point:
judged against the whole catalogue most seats look dominated, and acting on that
would collapse every panel onto whichever vendor is cheapest this month and
recreate the correlated errors a panel exists to cancel.

A preset that is not in `PRESETS` still owes an explanation, so `WITHHELD_PRESETS`
and `DROPPED_PRESETS` carry one and `get_preset` hands it back by name instead of
a bare "unknown preset".

Reserves are a reviewed change to this file, not a grade-time substitution. A
panel that contains the generator warns and proceeds (`panel_composition_messages`
in `common.jury`, or `strict_panel=True` to make it fatal); it never silently
swaps a judge, because a user who picks a preset should get the panel they picked.

Router IDs are the literal strings the orq model garden returns from
`GET /v2/models`; they are what the router and `common.model_catalogue` expect.

They are never `orq/*` routers (RES-1573), even ones that derive the same
frontier server-side and stay current without a capture. A router resolves per
request and optimises each one alone, so a panel of them can seat three cards
from one vendor, which is the correlated-error case a panel exists to cancel.
The lineage diversity is what does not transfer to the platform, so seats stay
pinned here even where the arithmetic behind them has moved. A single judge is
free to be a router, since it has no panel to correlate.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from types import MappingProxyType
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, Field, model_validator

from evaluatorq.common.jury import AggregatorName, provider_family
from evaluatorq.common.llm_client import MissingLLMCredentialsError
from evaluatorq.common.model_catalogue import ModelCatalogueError
from evaluatorq.common.model_frontier import ScoredModel, load_garden, load_garden_sync, with_manual_cards

# Re-exported: model_identity moved to common.model_frontier so the frontier's
# lineage parsing can share it without importing this module.
from evaluatorq.common.model_frontier import model_identity as model_identity

if TYPE_CHECKING:
    from collections.abc import Mapping


def _reachable(garden: Mapping[str, ScoredModel] | None) -> dict[str, ScoredModel]:
    """The garden a seat is priced and ranked in: live unless passed, plus the hand-written MANUAL_MODEL_CARDS."""
    return with_manual_cards(dict(garden) if garden is not None else load_garden_sync())


async def _reachable_async(garden: Mapping[str, ScoredModel] | None) -> dict[str, ScoredModel]:
    """`_reachable` for async callers: the live garden is awaited, not fetched with a blocking call."""
    return with_manual_cards(dict(garden) if garden is not None else await load_garden())


class JudgeRates(BaseModel):
    """What a judge bills per 1M tokens and the effort it runs at. Deprecated; read `ScoredModel` from the garden."""

    model_config = {'frozen': True}

    input_rate: float
    output_rate: float
    default_reasoning_effort: str | None


def judge_rates(router_id: str, garden: Mapping[str, ScoredModel] | None = None) -> JudgeRates | None:
    """Deprecated: the live card rates for a judge, or None when the garden has no card for it.

    Rates used to come from a committed monthly snapshot; they are now read from
    `/v2/models`. Use `load_garden()` and read the `ScoredModel` instead.

    Without ``garden`` it reads the live catalogue through the ambient key. Where
    that is impossible (no key, a failed fetch, or a running event loop) it logs
    a warning and returns None, as the shim did when it had no rates for a judge.
    Pass ``garden=`` from async code.
    """
    warnings.warn(
        'judge_rates() is deprecated; rates are live now. Read the ScoredModel from common.model_frontier.load_garden().',
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        model = _reachable(garden).get(router_id)
    except (MissingLLMCredentialsError, ModelCatalogueError, RuntimeError) as exc:
        logger.warning('judge_rates({!r}) could not read the live catalogue: {}', router_id, exc)
        return None
    if model is None:
        return None
    return JudgeRates(
        input_rate=model.input_price, output_rate=model.output_price, default_reasoning_effort=model.default_effort
    )


def captured_at() -> str:
    """Deprecated: today's UTC date, since rates are read live on every call rather than captured."""
    warnings.warn(
        'captured_at() is deprecated; rates are read live from /v2/models, so there is no capture date.',
        DeprecationWarning,
        stacklevel=2,
    )
    return datetime.now(timezone.utc).date().isoformat()


def judge_family(router_id: str) -> str:
    """Training lineage of a router ID, the axis self-preference runs on.

    Delegates to `common.jury.provider_family`, which is what the runtime panel
    check uses: two lineage implementations would let a preset be composed on one
    reading and warned about on another. Raises rather than returning 'unknown',
    because an unclassified judge would silently never match a generator and the
    composition check would pass by accident.

    Lineage is who trained the model, not who serves it: `groq/openai/gpt-oss-120b`
    is OpenAI despite being open-weight on Groq, and `baseten/kimi-k3` is Moonshot.
    """
    family = provider_family(router_id)
    if family == 'unknown':
        raise ValueError(f'unknown lineage for {router_id!r}; add a marker to common.jury._FAMILY_MARKERS')
    return family


class JuryPreset(BaseModel):
    """A named panel: fixed judges, fixed aggregation, and reserves for retirement."""

    name: str
    judges: tuple[str, ...]
    reserve_judges: tuple[str, ...]
    numeric_aggregation: AggregatorName = 'mean_std'
    use_when: str
    estimated_cost_per_1k: float = Field(
        description='USD per 1,000 pointwise items at 1,500 input / 500 output tokens, uncached.'
    )

    @model_validator(mode='after')
    def _panel_is_well_formed(self) -> JuryPreset:
        """Even panels split, and a repeated judge is one voter casting two votes."""
        if len(self.judges) % 2 == 0:
            raise ValueError(f'{self.name}: panel size {len(self.judges)} is even')
        if len(set(self.judges)) != len(self.judges):
            raise ValueError(f'{self.name}: panel has a duplicate judge')
        if set(self.reserve_judges) & set(self.judges):
            raise ValueError(f'{self.name}: a reserve judge is already on the panel')
        if len(set(judge_family(r) for r in self.reserve_judges)) != len(self.reserve_judges):
            raise ValueError(f'{self.name}: reserves share a lineage with each other')
        return self

    def seated_efforts(self, garden: Mapping[str, ScoredModel] | None = None) -> dict[str, str | None]:
        """The reasoning effort each judge runs at: its catalogue default.

        A preset sends no effort, so each seat runs at the provider default,
        which is also the rung it is ranked at. None means the card exposes no
        effort parameter. Reads the live garden unless one is passed; async
        callers use `seated_efforts_async`.

        `llm_jury` takes one `reasoning_effort` for the whole panel, so passing
        one overrides every seat's default; per-judge call settings are a schema
        change and its own ticket (RES-1347).

        Raises:
            KeyError: A seat has no card in the garden and no MANUAL_MODEL_CARDS entry.
            MissingLLMCredentialsError: No garden was passed and no Orq key is set.
            ModelCatalogueError: No garden was passed and the live fetch failed.
            RuntimeError: No garden was passed and an event loop is running.
        """
        return self._efforts(_reachable(garden))

    async def seated_efforts_async(self, garden: Mapping[str, ScoredModel] | None = None) -> dict[str, str | None]:
        """`seated_efforts` for async callers: awaits the live garden instead of blocking on it."""
        return self._efforts(await _reachable_async(garden))

    def _seat(self, judge: str, reachable: Mapping[str, ScoredModel]) -> ScoredModel:
        model = reachable.get(judge)
        if model is None:
            raise KeyError(f'{self.name}: no live card for {judge!r}; reseat it or add a MANUAL_MODEL_CARDS entry')
        return model

    def _efforts(self, reachable: Mapping[str, ScoredModel]) -> dict[str, str | None]:
        return {judge: self._seat(judge, reachable).default_effort for judge in self.judges}

    def duplicated_lineages(self) -> dict[str, tuple[str, ...]]:
        """Lineages seated more than once, whose errors correlate.

        Not an error. A panel may repeat a lineage deliberately, as
        single-provider-trio does with three OpenAI judges, but the diversity
        claim weakens and the reviewer should see it rather than count IDs by
        hand.
        """
        seated: dict[str, list[str]] = {}
        for judge in self.judges:
            seated.setdefault(judge_family(judge), []).append(judge)
        return {family: tuple(js) for family, js in seated.items() if len(js) > 1}

    def cost_per_1k(self, garden: Mapping[str, ScoredModel] | None = None) -> float:
        """The panel's $/1k recomputed from live card rates.

        `estimated_cost_per_1k` is the figure the docs publish; this is where it
        has to come from. The live drift suite asserts the two agree, so a
        repricing fails that suite rather than quietly making a published table
        wrong. Reads the live garden unless one is passed; async callers use
        `cost_per_1k_async`.

        Raises:
            KeyError: A seat has no card in the garden and no MANUAL_MODEL_CARDS entry.
            MissingLLMCredentialsError: No garden was passed and no Orq key is set.
            ModelCatalogueError: No garden was passed and the live fetch failed.
            RuntimeError: No garden was passed and an event loop is running.
        """
        return self._cost(_reachable(garden))

    async def cost_per_1k_async(self, garden: Mapping[str, ScoredModel] | None = None) -> float:
        """`cost_per_1k` for async callers: awaits the live garden instead of blocking on it."""
        return self._cost(await _reachable_async(garden))

    def _cost(self, reachable: Mapping[str, ScoredModel]) -> float:
        total = Decimal(0)
        for judge in self.judges:
            model = self._seat(judge, reachable)
            total += Decimal(str(model.input_price)) * Decimal(ESTIMATED_PROMPT_TOKENS)
            total += Decimal(str(model.output_price)) * Decimal(ESTIMATED_COMPLETION_TOKENS)
        # Decimal, and one rounding at the end. A figure that lands exactly on a
        # half-cent would otherwise be decided by binary-float representation and
        # banker's rounding rather than by a price. Rates are per 1M tokens, so
        # one item costs total / 1e6 and a thousand cost total / 1e3.
        return float((total / Decimal(1_000_000) * Decimal(1_000)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


BALANCED_TRIO = JuryPreset(
    name='balanced-trio',
    judges=(
        # Held by claude-haiku-4-5 until the age check went in (2026-08-25) and
        # named what the frontier checks structurally cannot see: Anthropic has
        # shipped no small model since October 2025, so no in-family upgrade
        # was ever going to be found and the seat aged 314 days in silence. It
        # was also the weakest buy in the library, 23.7 at a $2.00 blend beside
        # luna on this same panel at 38.1 for $0.45. The only argument for
        # keeping it was an Anthropic vote, and this panel is sold on three
        # families rather than three brands, which deepseek satisfies.
        # Reseated 2026-11-10 from deepseek-v4-pro (headline 36.0, $0.544
        # blend): v4.1 Flash is 39.5 for $0.525. Neither card scores its high
        # default, so both rank on the headline index (no reasoning level).
        'baseten/deepseek-v4.1-flash',
        # Accepted evaluatorq reseat on coherent measured AA v4.3.2 medium:
        # GPT-6 Luna 29.9251773515762 vs GPT-5.6 Luna 25.0358741440784,
        # $0.20 vs $0.45 blended. Never compare the unversioned cached pair.
        'openai/gpt-6-luna',
        # Reseated 2026-11-10 from gemini-3.6-flash (headline 34.0, $3.00
        # blend; no medium score): 3.8 Flash scores 39.8 at its medium default
        # for $1.50.
        'google-ai/gemini-3.8-flash',
    ),
    # Same lineage as the seat it backs, on purpose: a retirement is usually a
    # version bump, and flash keeps the panel at three families where a
    # fourth-vendor reserve would quietly reshape it.
    reserve_judges=('deepseek/deepseek-flash',),
    use_when='Default subjective eval. Three families, three error surfaces.',
    estimated_cost_per_1k=4.45,
)

STRONG_JURY = JuryPreset(
    name='strong-jury',
    judges=(
        'anthropic/claude-opus-5-5',
        # Reseated 2026-11-10 from gpt-5.6-sol (medium 39.2, $8.00 blend):
        # GPT-6.1 Sol is 47.8 at medium for $4.00, clear of the 2-point gate.
        'openai/gpt-6.1-sol',
        # Reseated 2026-11-10 from gemini-3.6-flash, as on balanced-trio.
        'google-ai/gemini-3.8-flash',
    ),
    # Reseated 2026-11-10 from deepseek-v4-pro, as on balanced-trio.
    reserve_judges=('baseten/deepseek-v4.1-flash',),
    use_when='Customer-facing benchmarks, preference data, anything that compounds.',
    estimated_cost_per_1k=27.00,
)

CHEAP_AGGREGATE = JuryPreset(
    name='cheap-aggregate',
    # Derived, not hand-picked: the best default-effort index of each model
    # family whose 3:1 blend is under MAX_BLEND, from the live catalogue plus
    # the reviewed MANUAL_MODEL_CARDS and AA_FILLS registers, minus anything
    # superseded by a newer version of itself (model_frontier.rank_cheapest_top).
    # test_cheap_aggregate_matches_the_live_derivation fails with a readable
    # diff when the live ranking moves; a change here is a deliberate edit.
    # Ranked 2026-11-10, with headline fallback for cards unscored at their
    # default: glm-5.3-flash 41.8, gemini-3.8-flash 39.8, deepseek-v4.1-flash
    # 39.5 (headline); mistral-large-4 38.4 (headline) is next.
    judges=(
        # Default effort max; the live card has no ladder, so 41.8 is the AA
        # API value for the same slug (AA_FILLS). PRICE HOLD: the card says
        # $0.075/$0.25 per 1M ($0.119 blend), while judge-endpoint usage was
        # billed at $0.15/$0.50 ($0.2375 blend), twice the card. The live card
        # leads here and the estimate uses it; at the billed rate this seat
        # costs $0.475 per 1k instead of $0.2375 and the panel estimate is
        # $4.53 instead of $4.29. Replace the registry rate the day Zai billing
        # is confirmed.
        'zai/glm-5.3-flash',
        'google-ai/gemini-3.8-flash',
        'baseten/deepseek-v4.1-flash',
    ),
    # Next-ranked family under the ceiling.
    reserve_judges=('mistral/mistral-large-4',),
    use_when='High volume regression eval, nightly reruns, HITL triage feeders.',
    estimated_cost_per_1k=4.29,
)

OPEN_WEIGHT_PORTABLE = JuryPreset(
    name='open-weight-portable',
    judges=(
        # Held by gpt-oss-120b until the family entered NEVER_SEAT (PR #330
        # review, 2026-08-24), then deepseek-v4-pro; reseated 2026-11-10 to
        # v4.1 Flash, as on balanced-trio.
        'baseten/deepseek-v4.1-flash',
        # Back on Baseten at its max default (measured 43.6) from 2026-11-10.
        # The 2026-10-05 move to Wafer at an unbenchmarked `none` default was
        # bought with a $12.75 output rate; Wafer now bills $15.00 like
        # Baseten, so the trade kept the unmeasured default and lost the price.
        'baseten/kimi-k3',
        # Reseated 2026-11-10 from glm-5.2 (33.7, $2.15 blend), which
        # SUPERSEDED already retired: 41.8 at max on the card rate. Same
        # billing hold as cheap-aggregate's seat.
        'zai/glm-5.3-flash',
    ),
    reserve_judges=('greenpt/qwen3.6-35b-a3b',),
    use_when='No dependence on a closed frontier vendor; migration path to self-hosting.',
    estimated_cost_per_1k=13.29,
)

EU_REGION = JuryPreset(
    name='eu-region',
    judges=(
        # Reseated 2026-11-10 by the in-family frontier check: Haiku 5.5 medium
        # 34.5 at $0.11/$0.55 against Haiku 4.5 15.4 at $1.10/$5.50, and
        # Gemini 3.7 Flash medium 39.6 at $0.825/$4.125 against 3.5 Flash
        # 33.6 at $1.50/$9.00, both on the EU endpoints.
        'aws/eu.anthropic.claude-haiku-5-5',
        'google/eu.gemini-3.7-flash',
        # Same weights as openai/eu.gpt-5.6-luna, 10% cheaper. The OpenAI EU
        # endpoint bills 0.22/1.32 against Azure's 0.20/1.20.
        'azure/eu.gpt-5.6-luna',
    ),
    reserve_judges=('google/eu.claude-sonnet-5',),
    use_when=(
        'Data residency: every judge serves from an EU region. Anthropic, Google and OpenAI, which is what the EU catalog can field at the current generation: the DeepSeek seat carrying balanced-trio has no EU endpoint newer than v3.1, so this is not that panel relocated.'
    ),
    estimated_cost_per_1k=4.64,
)

SINGLE_PROVIDER_TRIO = JuryPreset(
    name='single-provider-trio',
    # The three tiers of the newest OpenAI generation (user decision
    # 2026-11-10): 6.1 Sol and 6.1 Terra, and GPT-6 Luna since no 6.1 Luna is
    # listed. 6.1 Sol medium 47.8 at $2/$10 (EU sibling ladder); 6.1 Terra has
    # no medium score and ranks on its v5.0.1 headline.
    judges=(
        'openai/gpt-6.1-sol',
        'openai/gpt-6.1-terra',
        'openai/gpt-6-luna',
    ),
    # Same generation, off the panel: GPT-6 Sol medium 39.8 at $2/$10.
    reserve_judges=('openai/gpt-6-sol',),
    use_when=(
        'Workspaces locked to one provider contract. The three tiers of the '
        'current OpenAI generation, so no family diversity: errors correlate '
        'and OpenAI-generated outputs face a self-preference risk the panel '
        'cannot vote away.'
    ),
    estimated_cost_per_1k=12.40,
)


PRESETS: MappingProxyType[str, JuryPreset] = MappingProxyType({
    preset.name: preset
    for preset in (
        BALANCED_TRIO,
        STRONG_JURY,
        CHEAP_AGGREGATE,
        OPEN_WEIGHT_PORTABLE,
        EU_REGION,
        SINGLE_PROVIDER_TRIO,
    )
})

# Flat estimate behind every published cost, the same for every seat whatever
# its reasoning effort: 3:1 input to output, the ratio the garden's blended
# price is ranked on. Measured usage runs above this when a judge reasons
# without being asked (RES-996), so treat it as a floor.
ESTIMATED_PROMPT_TOKENS = 1500
ESTIMATED_COMPLETION_TOKENS = 500


# Presets derived and costed, but not seated in PRESETS because a seat cannot
# do the job today. Distinct from DROPPED_PRESETS: nothing here was judged a bad
# panel, and each entry names the one thing that has to be true for it to ship.
WITHHELD_PRESETS: dict[str, str] = {}

# Days a seat may sit in the garden before it has to be argued for again.
# Generations have been turning over in roughly eight weeks, so a seat past
# this has watched two or three rounds of its own band ship without moving.
MAX_SEAT_AGE_DAYS = 180

# Waivers for seats that are older than MAX_SEAT_AGE_DAYS but kept on purpose,
# keyed by model identity, each with the written reason. Without a waiver the
# weekly drift check fails on a seat that old.
#
# The other registers all answer "something newer exists, why not it?".  This
# one answers the question nothing else asks: a lineage can simply stop
# shipping, and then no in-family upgrade is ever found, no successor is ever
# flagged, and a seat quietly ages out while every freshness test passes.
#
# Age here is time since first listing in the garden, taken across every host
# of the same weights, so a second host relisting a model does not reset it.
STALE_SEAT_WAIVERS: dict[str, str] = {
    'claude-haiku-4-5-20251001': (
        'Listed 2025-10, held only by the derived Anthropic Only pool, and only '
        'on a project-scoped key: anthropic/claude-haiku-5-5 is in the public '
        "catalogue but not in this project's /v2/models, and the pool "
        'derivation reads /v2/models alone, so it still reaches for 4.5. '
        'Retire this entry once the drift suite runs on a workspace key; '
        'test_stale_seat_waivers_are_dropped_once_the_seat_is_young_again says when.'
    ),
}

# Successors already looked at and deliberately not seated. A model reaching the
# garden is not a reason to put it in front of customers, but it is a reason to
# say why not: this dict is that record, and the freshness test fails on any
# successor missing from it. Seat the model or write the line; silence is the
# one thing that is not allowed.
# Shared reasons for the 2026-11-10 review of the public catalogue. Public
# `artificial_analysis_intelligence_index` values quoted below name no effort,
# so they explain a dismissal but never rank a seat (finding 10 of the research
# repo's LLM jury presets 2-pager).
_OLDER_GENERATION = (
    'An older generation than the seat in its lineage, first listed here late; '
    'a listing date is not a release date. Public catalogue 2026-11-10.'
)
_NOT_A_JUDGE = 'Not a general text judge (translation, vision, omni, image, speech or classify-only endpoint). Public catalogue 2026-11-10.'
_UNSCORED = (
    'No AA index on the public card and no card in the project-scoped /v2/models '
    '(2026-11-10), so there is nothing to rank it on. Review again once scored.'
)
_ALIBABA_RESERVE = (
    "Alibaba only holds Portable's reserve, chosen for cost (greenpt/qwen3.6-35b-a3b, "
    '$0.30/$1.80). This one is pricier per 1M or europe-pinned, so it is not a '
    'cheaper reserve. Public catalogue 2026-11-10.'
)

REVIEWED_SUCCESSORS: dict[str, str] = {
    'qwen3.7-flash': (
        'alibaba/qwen3.7-flash is europe-pinned and unscored in the live catalogue (2026-10-26); the Portable reserve cannot take a region-pinned or unscored seat.'
    ),
    'qwen3.8-flash': (
        'alibaba/qwen3.8-flash is europe-pinned and unscored in the live catalogue (2026-10-26), so it has no index to rank on and no non-EU route.'
    ),
    'qwen-3.8-27b': (
        "cerebras/qwen-3.8-27b scores 33.7 live but at $0.99/$1.49 per 1M; a reserve held for price, not strength, so it stays out. Revisit if Portable's reserve is ever chosen by pareto rather than by cost."
    ),
    'qwen3.8-flash-next': (
        'tensorix/qwen/qwen3.8-flash-next is europe-pinned and unscored in the live catalogue (2026-10-26).'
    ),
    'qwen-mt-flash': _NOT_A_JUDGE,
    'qwen-mt-lite': _NOT_A_JUDGE,
    'qwen3-vl-flash-2025-10-15': _NOT_A_JUDGE,
    'qwen3-vl-235b-a22b-instruct': _NOT_A_JUDGE,
    'qwen3.8-omni-flash': _NOT_A_JUDGE,
    'qwen3.5-2b': _NOT_A_JUDGE,
    'decision-model-preview': _NOT_A_JUDGE,
    'voxtral-mini-transcribe-3': _NOT_A_JUDGE,
    'voxtral-mini-transcribe-3-0': _NOT_A_JUDGE,
    'voxtral-mini-transcribe-realtime-3': _NOT_A_JUDGE,
    'voxtral-mini-transcribe-realtime-3-0': _NOT_A_JUDGE,
    'voxtral-mini-tts-3': _NOT_A_JUDGE,
    'voxtral-mini-tts-3-0': _NOT_A_JUDGE,
    'voxtral-small-transcribe-3': _NOT_A_JUDGE,
    'voxtral-small-transcribe-3-0': _NOT_A_JUDGE,
    'qwen-plus-2025-12-01': _OLDER_GENERATION,
    'qwen3.5-flash-2026-02-23': _OLDER_GENERATION,
    'qwen3-30b-a3b-instruct-2507': _OLDER_GENERATION,
    'qwen3.6-flash': _UNSCORED,
    'qwen3.6-flash-2026-04-16': _UNSCORED,
    'qwen3.7-flash-2026-07-15': _UNSCORED,
    'qwen3.6-27b': _ALIBABA_RESERVE,
    'qwen3.6-plus-2026-04-02': _ALIBABA_RESERVE,
    'qwen3.7-plus': _ALIBABA_RESERVE,
    'qwen3.7-plus-2026-05-26': _ALIBABA_RESERVE,
    'qwen3.7-max': _ALIBABA_RESERVE,
    'qwen3.7-max-2026-05-20': _ALIBABA_RESERVE,
    'qwen3.7-max-2026-06-08': _ALIBABA_RESERVE,
    'qwen3.8-27b': _ALIBABA_RESERVE,
    'qwen3.8-27b-fp8': _ALIBABA_RESERVE,
    'qwen3.8-2.4t-a95b': _ALIBABA_RESERVE,
    'qwen3.8-max': _ALIBABA_RESERVE,
    'glm-4-32b-0414-128k': _OLDER_GENERATION,
    'glm-4.5-airx': _OLDER_GENERATION,
    'glm-4.5-x': _OLDER_GENERATION,
    'claude-fable-5-2': '$8/$40 per 1M on every host, no AA index on the public card and no card in /v2/models (2026-11-10). Five times the Opus 5.5 rate it would compete with; review for strong-jury once scored.',
    'claude-mythos-5-2': 'Same as claude-fable-5-2: $8/$40 per 1M, unscored (2026-11-10). Review for strong-jury once scored.',
    'deepseek-v4.1-flash-1017': _UNSCORED,
    'kimi-k2.8-preview': _UNSCORED,
    'kimi-k3.1-pro-highspeed': 'A faster K3.1 Pro at $4/$20 per 1M, unscored (2026-11-10). If K3.1 Pro is seated, this is its premium variant, not a separate decision.',
    'kimi-k3.1-pro': '$2/$10 per 1M on moonshotai, cheaper than the seated K3 ($3/$15), but unscored on the public card and absent from /v2/models (2026-11-10). The likeliest Portable upgrade: re-review the moment it is scored.',
    'gemini-nano-banana-2.1': _NOT_A_JUDGE,
    'gemma-4-12b': 'munito EU host only, public AA index 9.4, under the 15.0 floor. 2026-11-10.',
    'qwen3.8-max-0902': "Public AA index 45.4 at $1.65/$4.95 per 1M, europe/us/apac. Alibaba holds only Portable's cost-chosen reserve ($0.30/$1.80), so a stronger, pricier Qwen is not a reserve upgrade; it would be a fourth-family judge, which no panel has a seat for. 2026-11-10.",
    'glm-5.3-fast': 'Public AA index 44.8 at $2.10/$6.60 per 1M against the seated glm-5.3-flash at $0.075/$0.25 per 1M on the card: stronger, far pricier. 2026-11-10.',
    'glm-5.3-flashx': 'Same 41.8 public index as the seated glm-5.3-flash at $0.37/$1.25 against $0.075/$0.25: same score, five times the price. 2026-11-10.',
    'deepseek-v4.1-flash-fast': 'Baseten fast variant at $0.60/$2.40 per 1M, public AA index 39.5; twice deepseek-v4.1-flash and six times the deepseek-flash reserve ($0.14/$0.28) at a 3:1 blend. 2026-11-10.',
    'kimi-k3-fast': 'The same K3 weights on Fireworks at $4.50/$22.50 per 1M against Baseten $3/$15: same score, pricier. 2026-11-10.',
    'gemini-4-argon': "Public AA index 52.6 at $2/$10 per 1M, against strong-jury's gemini-3.8-flash at a $1.50 3:1 blend. No card in the project-scoped /v2/models, so no default-effort rung to rank on (2026-11-10). The strongest unreviewed candidate in the catalogue: rank it for strong-jury once the drift suite has a workspace key.",
}


# Presets that shipped in a draft and were then retired, with the reason on
# record. A shape that disappears silently leaves its users guessing, so removal
# costs a written line here and `get_preset` hands it back by name.
DROPPED_PRESETS: dict[str, str] = {
    'value-trio': (
        'Retired. A budget panel whose measured cost ran far above its published '
        'figure, because two of its judges wrote long prose verdicts. Use '
        'balanced-trio for a default panel, or open-weight-portable for vendor '
        'independence.'
    ),
}


def all_router_ids() -> set[str]:
    """Every router ID a preset can send a call to, judges and reserves alike."""
    return {judge for preset in PRESETS.values() for judge in preset.judges} | {
        reserve for preset in PRESETS.values() for reserve in preset.reserve_judges
    }


# The display names presets carried before they were renamed to slugs. They still
# resolve, with a DeprecationWarning, so a caller written against the old name keeps
# working; new code uses the slug.
LEGACY_PRESET_NAMES: MappingProxyType[str, str] = MappingProxyType({
    'Balanced Trio': 'balanced-trio',
    'Strong Jury': 'strong-jury',
    'Cheap Aggregate': 'cheap-aggregate',
    'Open-Weight / Portable': 'open-weight-portable',
    'EU Region': 'eu-region',
    'Single-Provider Trio': 'single-provider-trio',
    'Value Trio': 'value-trio',
})


def get_preset(name: str) -> JuryPreset:
    """Look a preset up by name, listing the alternatives when there is no match.

    A pre-rename display name such as ``'Balanced Trio'`` resolves to its slug
    (``'balanced-trio'``) with a ``DeprecationWarning``.
    """
    name = name.strip()
    renamed = LEGACY_PRESET_NAMES.get(name)
    if renamed is not None:
        warnings.warn(
            f'jury preset {name!r} was renamed to {renamed!r}; the old name will stop working in a future release.',
            DeprecationWarning,
            stacklevel=2,
        )
        name = renamed
    try:
        return PRESETS[name]
    except KeyError:
        dropped = DROPPED_PRESETS.get(name)
        if dropped:
            raise ValueError(f'jury preset {name!r} was retired: {dropped}') from None
        withheld = WITHHELD_PRESETS.get(name)
        if withheld:
            raise ValueError(f'jury preset {name!r} is not shipped yet: {withheld}') from None
        raise ValueError(f'unknown jury preset {name!r}; available: {sorted(PRESETS)}') from None
