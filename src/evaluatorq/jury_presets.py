"""LLM-as-a-jury preset definitions (RES-1171, derived in RES-996/RES-1346).

The single source for what a named jury preset means: which judges, which
aggregation mode, and which reserve judges replace them when one is retired,
deprecated or repriced. `llm_jury(preset=...)` builds a panel from here, and a
test recomputes every published figure from `common/data/jury_judge_rates.json`,
so the table in the docs cannot drift away from what the code seats.

Seats are derived rather than hand-listed. The derivation itself is not here:
it reads a capture of the whole model garden, ranks each lineage on an
intelligence index and a blended price, and is rerun and reviewed in the
research repo that owns the capture. What ships to callers is its output, the
panels and the rates they were costed at. Seats are compared in-family, because
the lineage diversity is the point: judged against the whole garden most seats
look dominated, and acting on that would collapse every panel onto whichever
vendor is cheapest this month and recreate the correlated errors a panel exists
to cancel.

A preset that is not in `PRESETS` still owes an explanation, so `WITHHELD_PRESETS`
and `DROPPED_PRESETS` carry one and `get_preset` hands it back by name instead of
a bare "unknown preset". The registers behind the seating itself, which record a
successor deliberately passed over or a seat knowingly held past its age limit,
live with the derivation in the research repo where they can be checked against
the garden.

Reserves are a reviewed change to this file, not a grade-time substitution. A
panel that contains the generator warns and proceeds (`_panel_composition_messages`
in `common.jury`, or `strict_panel=True` to make it fatal); it never silently
swaps a judge, because a user who picks a preset should get the panel they picked.

Router IDs are the literal strings the orq model garden returns from
`GET /v2/models`; they are what the router and `common.model_catalogue` expect.
"""

import json
import re
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from pydantic import BaseModel, Field, model_validator

from evaluatorq.common.jury import AggregatorName, provider_family

_RATES_PATH = Path(__file__).parent / 'common' / 'data' / 'jury_judge_rates.json'


class JudgeRates(BaseModel):
    """What a seated judge bills and the rung it was ranked at, per 1M tokens."""

    model_config = {'frozen': True}

    input_rate: float
    output_rate: float
    seated_effort: str | None
    priced_at_ceiling: bool


@lru_cache(maxsize=1)
def _load_rates() -> MappingProxyType[str, JudgeRates]:
    """The committed rate table, read once and handed out read-only.

    Frozen on purpose. This is cached process-wide, so a caller who mutated the
    returned mapping would silently reprice every published preset for the life
    of the process.
    """
    raw = cast('dict[str, Any]', json.loads(_RATES_PATH.read_text()))
    return MappingProxyType({rid: JudgeRates(**fields) for rid, fields in raw['judges'].items()})


def judge_rates(router_id: str) -> JudgeRates | None:
    """The captured rates for a seated judge, or None for a model no preset seats."""
    return _load_rates().get(router_id)


def captured_at() -> str:
    """When the rates were captured, for anyone deciding whether to trust the table."""
    return cast('str', json.loads(_RATES_PATH.read_text())['captured_at'])


# Host, region and serving-variant noise that makes one model look like several.
# `openai/gpt-5.6-terra`, `azure/global.gpt-5.6-terra` and `openai/eu.gpt-5.6-terra`
# are one model; a successor check that counts them separately reports work that
# does not exist.
_REGION_PREFIXES: tuple[str, ...] = ('eu.', 'us.', 'apac.', 'global.')
# Vendor tokens `model_identity` strips when one leads a dotted model name, which
# is how Bedrock spells an id. A vendor whose name can open a legitimate model
# name (the way 'gpt-5.6-terra' opens with a version, not a vendor) would make
# `model_identity` truncate it, so this list is not the family list.
_VENDOR_TOKENS: frozenset[str] = frozenset({
    'openai',
    'anthropic',
    'google',
    'zhipu',
    'moonshot',
    'minimax',
    'deepseek',
    'alibaba',
    'meta',
    'mistral',
    'xai',
})


def model_identity(router_id: str) -> str:
    """The model behind a router ID, with host, region, vendor and version stripped.

    Two IDs sharing an identity are the same weights served twice, so seating one
    means the other needs no separate review. Bedrock is the awkward case:
    `aws/eu.anthropic.claude-haiku-4-5-20251001-v1:0` and
    `anthropic/claude-haiku-4-5-20251001` are one model, and EU Region seats the
    first while Balanced Trio used to seat the second.
    """
    name = router_id.split('/')[-1].lower()
    for prefix in _REGION_PREFIXES:
        name = name.removeprefix(prefix)
    head, _, tail = name.partition('.')
    if tail and head in _VENDOR_TOKENS:
        name = tail
    return re.sub(r'-v\d+:\d+$', '', name)


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
        description='USD per 1,000 pointwise items at 1,500 input / 150 output tokens, uncached.'
    )

    @model_validator(mode='after')
    def _panel_is_well_formed(self) -> 'JuryPreset':
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

    def seated_efforts(self) -> dict[str, str | None]:
        """The reasoning effort each judge is ranked at, and so must be called at.

        A preset names judges and a caller sends the call, so until the seated
        effort is stated the two can disagree silently: the panel is costed and
        ranked at one operating point and run at another. None means the card
        publishes no ladder and the provider default stands.

        `llm_jury` takes one `reasoning_effort` for the whole panel, so a preset
        whose seats disagree cannot express itself through it yet; per-judge call
        settings are a schema change and its own ticket (RES-1347).

        These are the rungs the cards were scored at, not values to send: a card
        that only distinguishes thinking from not thinking is scored at
        `reasoning` or `none`, which the EU Region haiku seat reads back as
        `reasoning` and no provider accepts as a `reasoning_effort`. Sending one
        is a 400 on the value, which the retry path used to read as the model
        refusing the parameter.
        """
        return {judge: (r.seated_effort if (r := judge_rates(judge)) else None) for judge in self.judges}

    def priced_below_seated_effort(self) -> tuple[str, ...]:
        """Judges whose captured price is the blend at a cheaper rung than the seated one.

        The published $/1k understates these until a probe measures them at the
        effort they are seated at. Disclosed rather than corrected: correcting it
        is a re-probe of every panel, not an arithmetic fix.
        """
        return tuple(j for j in self.judges if (r := judge_rates(j)) and not r.priced_at_ceiling)

    def duplicated_lineages(self) -> dict[str, tuple[str, ...]]:
        """Lineages seated more than once, whose errors correlate.

        Not an error. A panel may repeat a lineage deliberately, as
        Single-Provider Trio does with three OpenAI judges, but the diversity
        claim weakens and the reviewer should see it rather than count IDs by
        hand.
        """
        seated: dict[str, list[str]] = {}
        for judge in self.judges:
            seated.setdefault(judge_family(judge), []).append(judge)
        return {family: tuple(js) for family, js in seated.items() if len(js) > 1}

    def cost_per_1k(self) -> float:
        """The panel's $/1k recomputed from the captured billing rates.

        `estimated_cost_per_1k` is the figure the docs publish; this is where it
        has to come from. A test asserts the two agree, so a repricing in the
        snapshot fails CI rather than quietly making a published table wrong.
        """
        total = Decimal(0)
        for judge in self.judges:
            rates = judge_rates(judge)
            if rates is None:
                raise KeyError(f'{self.name}: no captured pricing for {judge!r}; re-run the rate capture')
            total += Decimal(str(rates.input_rate)) * Decimal(ESTIMATED_PROMPT_TOKENS)
            total += Decimal(str(rates.output_rate)) * Decimal(ESTIMATED_COMPLETION_TOKENS)
        # Decimal, and one rounding at the end. Two published figures land exactly
        # on a half-cent (Strong Jury 23.625, EU Region 6.555), so binary floats
        # and banker's rounding would decide them by representation rather than by
        # a price, and a recapture that moved nothing could flip the table.
        return float((total / Decimal(1_000)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


BALANCED_TRIO = JuryPreset(
    name='Balanced Trio',
    judges=(
        # Held by claude-haiku-4-5 until the age check went in (2026-08-25) and
        # named what the frontier checks structurally cannot see: Anthropic has
        # shipped no small model since October 2025, so no in-family upgrade
        # was ever going to be found and the seat aged 314 days in silence. It
        # was also the weakest buy in the library: at the rungs each card is
        # ranked on, 29.6 at a $2.00 blend beside luna on this same panel at
        # 51.2 for $0.45. The only argument for
        # keeping it was an Anthropic vote, and this panel is sold on three
        # families rather than three brands, which deepseek satisfies at 43.1
        # for $0.544 while taking the panel from $6.11 to $4.64.
        'deepseek/deepseek-v4-pro',
        # Held by gpt-5.4-mini until the general-purpose luna card landed
        # (2026-08): same OpenAI lineage, 21.4 index points stronger at their
        # ceilings (51.2 against 29.8) at a quarter of the price.
        'openai/gpt-5.6-luna',
        # Reseated from gemini-3.5-flash on the 2026-08-25 re-capture, and a
        # trade rather than an upgrade: at their ceilings the successor is a
        # tenth of a point behind (50.1 against 50.2), so neither dominates the
        # other. It is seated for the price, $3.00 against $3.375 blended, and
        # for reaching that index at one rung where its predecessor needs its
        # top one.
        'google/gemini-3.6-flash',
    ),
    # Same lineage as the seat it backs, on purpose: a retirement is usually a
    # version bump, and flash keeps the panel at three families where a
    # fourth-vendor reserve would quietly reshape it.
    reserve_judges=('deepseek/deepseek-v4-flash',),
    use_when='Default subjective eval. Three families, three error surfaces.',
    estimated_cost_per_1k=4.64,
)

STRONG_JURY = JuryPreset(
    name='Strong Jury',
    judges=(
        'anthropic/claude-opus-5',
        # Held by gpt-5.4 on a number that turned out to be measured at the
        # wrong operating point: the 51.4 this seat was defended with is
        # gpt-5.4's xhigh score, while its default effort is `none`, where it
        # scores 27.7 against the same price. The 2026-08-25 re-capture carries
        # per-effort indices, and at their ceilings sol is the strongest OpenAI
        # model in the garden, 58.9 against that same 51.4. This jury is bought
        # for judgment quality,
        # so it pays the $8.00 blend.
        'openai/gpt-5.6-sol',
        # Reseated from gemini-3.1-pro-preview by the in-family frontier check,
        # then from gemini-3.5-flash on the re-capture, on the same reading as
        # Balanced Trio: level at the ceiling, cheaper per call.
        'google/gemini-3.6-flash',
    ),
    reserve_judges=('deepseek/deepseek-v4-pro',),
    use_when='Customer-facing benchmarks, preference data, anything that compounds.',
    estimated_cost_per_1k=23.63,
)

OPEN_WEIGHT_PORTABLE = JuryPreset(
    name='Open-Weight / Portable',
    judges=(
        # Held by gpt-oss-120b until the family entered NEVER_SEAT (PR #330
        # review, 2026-08-24); the panel's own reserve steps up.
        'deepseek/deepseek-v4-pro',
        # Held by kimi-k2.6 until ranking moved to card ceilings (2026-09-03).
        # That card reads a headline of 44.2 taken at effort `none` beside a
        # `none` rung of 34.6: it scores the same rung twice, differently, so
        # nothing can be seated on it and `seatable` now refuses it. K3 is the
        # replacement rather than a cheaper open-weight lineage because the
        # only ones available are dominated here: qwen3.6-27b scores 37.1 at
        # $1.20 against deepseek-v4-pro's 43.1 at $0.544, and a dominated judge
        # is dead weight. It nearly doubles the panel, $5.57 to $10.29, which
        # is the price of not seating a self-contradicting card. Served from
        # baseten because the Moonshot account 429s.
        'baseten/kimi-k3',
        'zai/glm-5.2',
    ),
    reserve_judges=('minimax/MiniMax-M2.7',),
    use_when='No dependence on a closed frontier vendor; migration path to self-hosting.',
    estimated_cost_per_1k=10.29,
)

EU_REGION = JuryPreset(
    name='EU Region',
    judges=(
        'aws/eu.anthropic.claude-haiku-4-5-20251001-v1:0',
        'google/eu.gemini-3.5-flash',
        # Same weights as openai/eu.gpt-5.6-luna, 10% cheaper. The OpenAI EU
        # endpoint bills 0.22/1.32 against Azure's 0.20/1.20.
        'azure/eu.gpt-5.6-luna',
    ),
    reserve_judges=('google/eu.claude-sonnet-5',),
    use_when=(
        'Data residency: every judge serves from an EU region. Anthropic, Google and OpenAI, which is what the EU catalog can field at the current generation: the DeepSeek seat carrying Balanced Trio has no EU endpoint newer than v3.1, so this is not that panel relocated.'
    ),
    estimated_cost_per_1k=6.56,
)

SINGLE_PROVIDER_TRIO = JuryPreset(
    name='Single-Provider Trio',
    judges=(
        # Held by gpt-5.4 until the re-capture showed its 51.4 was an xhigh
        # score against a default-effort price (27.7 at the effort we are
        # billed for, below its own nano). sol is the real top of the lineup.
        'openai/gpt-5.6-sol',
        'openai/gpt-5.6-terra',
        # Held by gpt-5.4-mini and gpt-5.4-nano while luna only existed as
        # EU-pinned endpoints. The general-purpose luna card landed in the
        # garden 2026-08 (same weights as the EU twin, index 38.1, $0.45
        # blended) and dominates both minis outright.
        'openai/gpt-5.6-luna',
    ),
    # gpt-5-mini held the reserve until ranking moved to card ceilings
    # (2026-09-03): its card reads minimal 14.3, medium 30.9, high 25.3, so
    # more effort scores lower and there is no rung to rank it at. It is in
    # NON_MONOTONE_LADDERS and nano, the next cheap OpenAI card, takes the role.
    reserve_judges=('openai/gpt-5.4-nano',),
    use_when=(
        'Workspaces locked to one provider contract. The three real tiers of '
        'the current OpenAI lineup rather than one tier plus two minis, so no '
        'family diversity: errors correlate and OpenAI-generated outputs face a '
        'self-preference risk the panel cannot vote away.'
    ),
    estimated_cost_per_1k=14.28,
)

PRESETS: MappingProxyType[str, JuryPreset] = MappingProxyType({
    preset.name: preset
    for preset in (
        BALANCED_TRIO,
        STRONG_JURY,
        OPEN_WEIGHT_PORTABLE,
        EU_REGION,
        SINGLE_PROVIDER_TRIO,
    )
})

# Estimate behind every published cost. Measured usage runs above this when a
# judge reasons without being asked (RES-996), so treat it as a floor.
ESTIMATED_PROMPT_TOKENS = 1500
ESTIMATED_COMPLETION_TOKENS = 150


# Presets derived and costed, but not seated in PRESETS because a seat cannot
# do the job today. Distinct from DROPPED_PRESETS: nothing here was judged a bad
# panel, and each entry names the one thing that has to be true for it to ship.
WITHHELD_PRESETS: dict[str, str] = {
    'Cheap Aggregate': (
        'Held back 2026-09-07, not retired. Five small judges across five '
        'lineages concluding on three of five at $2.56 per 1k, and the panel '
        'the volume story is built on. One seat does not vote: '
        '`minimax/MiniMax-M2.7` answers a `json_schema` response format with '
        'prose, and the fallback that would resend the schema as instructions '
        'sits behind `except BadRequestError`, so it never fires on a 200 that '
        'simply has the wrong shape. The model itself complies when the schema '
        'reaches it as instructions. That leaves four voting seats, an even '
        'panel, which is the one shape `_panel_is_well_formed` exists to '
        'reject, passing only because the validator counts declared seats and '
        'not voting ones. Seating the named reserve is not a fix either: '
        'zai/glm-5.2 costs $2.76 per 1k for that seat alone and takes the panel '
        'to $4.69, past Balanced Trio, which is the whole reason the panel '
        'exists gone. Ships the day the judge path falls back on an '
        'unparseable 200 as well as on a rejected request.'
    ),
}

# Presets that shipped in a draft and were then retired, with the reason on
# record. A shape that disappears silently leaves its users guessing, so removal
# costs a written line here and `get_preset` hands it back by name.
DROPPED_PRESETS: dict[str, str] = {
    'Value Trio': (
        'Retired 2026-08-24. Sold as the budget panel, but the blend repricing '
        'left Cheap Aggregate cheaper on the table ($2.73 vs $2.86 per 1k) with '
        'five judges to its three, and the 2026-08-18 probe measured Value Trio '
        '83% over its own table. '
        'The overage was prose length, not reasoning: neither of the two '
        'expensive judges reports a reasoning token. glm-5-maas held 63% of '
        'the measured cost writing 556 to 567 tokens every time, and MiniMax '
        'M2.7 held 32% ranging 345 to 1,780 across repeats of one prompt. '
        'A budget preset that is neither cheapest on paper nor close to its paper '
        'in practice has no seat to hold: budget traffic goes to Cheap '
        'Aggregate, vendor independence to Open-Weight / Portable.'
    ),
}


def all_router_ids() -> set[str]:
    """Every router ID a preset can send a call to, judges and reserves alike."""
    return {judge for preset in PRESETS.values() for judge in preset.judges} | {
        reserve for preset in PRESETS.values() for reserve in preset.reserve_judges
    }


def get_preset(name: str) -> JuryPreset:
    """Look a preset up by name, listing the alternatives when there is no match."""
    name = name.strip()
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
