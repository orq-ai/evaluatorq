"""LLM-as-a-jury preset definitions (RES-1171, derived in RES-996/RES-1346).

The single source for what a named jury preset means: which judges, which
aggregation mode, and which reserve judges replace them when one is retired,
deprecated or repriced. `llm_jury(preset=...)` builds a panel from here, the
cost tests recompute every published figure from the committed garden snapshot,
and the freshness tests refuse a seat that has quietly gone stale, so a preset
cannot drift between the docs and the code.

Seats are derived, not hand-listed: each one is the best buy within its own
lineage on the two axes `common.model_frontier` reads off the model garden. The
lineage itself is the point, so seats are judged in-family rather than globally
- judged globally, 19 of 24 look dominated, and acting on that would collapse
every panel onto the cheapest vendor and recreate the correlated errors the
panel exists to cancel.

Anything the arithmetic cannot settle is written down rather than passing in
silence: `AGED_SEATS`, `REVIEWED_SUCCESSORS`, `DROPPED_PRESETS` here, and
`NEVER_SEAT`, `UNROUTABLE`, `NON_MONOTONE_LADDERS` in `common.model_frontier`.
An entry that stops being true fails a test.

Reserves are a reviewed change to this file, not a grade-time substitution. A
panel that contains the generator warns and proceeds (`_panel_composition_messages`
in `common.jury`, or `strict_panel=True` to make it fatal); it never silently
swaps a judge, because a user who picks a preset should get the panel they picked.

Router IDs are the literal strings the orq model garden returns from
`GET /v2/models`; they are what the router and `common.model_catalogue` expect.
"""

import math
import re

from loguru import logger
from pydantic import BaseModel, Field, model_validator

from evaluatorq.common.jury import AggregatorName, provider_family
from evaluatorq.common.model_frontier import load_judge_pricing, seated_effort
from evaluatorq.contracts import StrEnum  # 3.10 compat shim; `enum.StrEnum` is 3.11+


class Aggregation(StrEnum):
    """How a panel turns per-judge verdicts into one.

    Both map to evaluatorq's `majority` aggregator, which is a strict >50% of the
    decisive votes: on a five-seat panel that is exactly three of five. The two
    members are kept apart because the second one also encodes a panel size, and
    the validator refuses to let it outlive one.
    """

    MAJORITY = 'majority'
    MAJORITY_3_OF_5 = 'majority_3_of_5'


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
    aggregation: Aggregation
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
        if self.aggregation is Aggregation.MAJORITY_3_OF_5 and len(self.judges) != 5:
            raise ValueError(f'{self.name}: 3-of-5 majority needs 5 judges, not {len(self.judges)}')

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
        """
        return {judge: seated_effort(judge) for judge in self.judges}

    def priced_below_seated_effort(self) -> tuple[str, ...]:
        """Judges whose captured price is the blend at a cheaper rung than the seated one.

        The published $/1k understates these until a probe measures them at the
        effort they are seated at. Disclosed rather than corrected: correcting it
        is a re-probe of every panel, not an arithmetic fix.
        """
        from evaluatorq.common.model_frontier import load_garden

        garden = load_garden()
        return tuple(j for j in self.judges if (m := garden.get(j)) and not m.priced_at_ceiling)

    def duplicated_lineages(self) -> dict[str, tuple[str, ...]]:
        """Lineages seated more than once, whose errors correlate.

        Not an error. A panel may deliberately repeat a lineage on cost grounds,
        as Cheap Aggregate does with two OpenAI judges, but the diversity claim
        weakens and the reviewer should see it rather than count IDs by hand.
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
        pricing = load_judge_pricing()
        components = []
        for judge in self.judges:
            rates = pricing.get(judge)
            if rates is None:
                raise KeyError(f'{self.name}: no captured pricing for {judge!r}; re-run the garden capture')
            components.extend((
                ESTIMATED_PROMPT_TOKENS / 1_000_000 * rates['input_rate'],
                ESTIMATED_COMPLETION_TOKENS / 1_000_000 * rates['output_rate'],
            ))
        # fsum, and one rounding at the end: rounding each judge and accumulating
        # puts EU Region at 6.56 against a true 6.555, so the published table and
        # the code would disagree over floating-point noise rather than a price.
        return round(math.fsum(components) * 1000, 2)


BALANCED_TRIO = JuryPreset(
    name='Balanced Trio',
    judges=(
        # Held by claude-haiku-4-5 until the age check went in (2026-08-25) and
        # named what the frontier checks structurally cannot see: Anthropic has
        # shipped no small model since October 2025, so no in-family upgrade
        # was ever going to be found and the seat aged 314 days in silence. It
        # was also the weakest buy in the library, 23.7 at a $2.00 blend beside
        # luna on this same panel at 38.1 for $0.45. The only argument for
        # keeping it was an Anthropic vote, and this panel is sold on three
        # families rather than three brands, which deepseek satisfies at 43.1
        # for $0.544 while taking the panel from $6.11 to $4.64.
        'deepseek/deepseek-v4-pro',
        # Held by gpt-5.4-mini until the general-purpose luna card landed
        # (2026-08): same OpenAI lineage, 8.3 index points stronger at a
        # quarter of the price.
        'openai/gpt-5.6-luna',
        # Reseated from gemini-3.5-flash on the 2026-08-25 re-capture: the
        # successor is stronger at its default effort (50.1 against 45.4) and
        # cheaper ($3.00 against $3.375 blended).
        'google/gemini-3.6-flash',
    ),
    aggregation=Aggregation.MAJORITY,
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
        # per-effort indices, and at matched effort sol is the strongest OpenAI
        # model in the garden (53.6). This jury is bought for judgment quality,
        # so it pays the $8.00 blend.
        'openai/gpt-5.6-sol',
        # Reseated from gemini-3.1-pro-preview by the in-family frontier check,
        # then from gemini-3.5-flash on the re-capture: stronger at default
        # effort and cheaper.
        'google/gemini-3.6-flash',
    ),
    aggregation=Aggregation.MAJORITY,
    reserve_judges=('deepseek/deepseek-v4-pro',),
    use_when='Customer-facing benchmarks, preference data, anything that compounds.',
    estimated_cost_per_1k=23.62,
)

CHEAP_AGGREGATE = JuryPreset(
    name='Cheap Aggregate',
    judges=(
        # Held by gpt-5.4-nano until the general-purpose luna card landed
        # (2026-08): dominated on both axes (38.1 vs 30.2 at a lower blend).
        'openai/gpt-5.6-luna',
        # gemini-3.1-flash-lite-preview is listed by the garden but Vertex
        # answers 404 for it (probe 2026-08-18) — the RES-1285 reachability
        # trap. gemini-3-flash-preview took the seat, and the 2026-08-25
        # re-capture replaced it with the released flash-lite, which is both
        # stronger and cheaper and drops the last preview dependency.
        'google/gemini-3.5-flash-lite',
        # Held this seat as gpt-oss-20b until the family entered NEVER_SEAT
        # (PR #330 review, 2026-08-24). Qwen keeps the panel cheap where the
        # in-house reserves cannot ($0.69 blended against Kimi's $1.71), and
        # takes it from four lineages to five.
        # Held by alibaba/qwen3.5-35b-a3b until a probe measured what the
        # index and price could not see (probe-candidates-20260825T102031Z,
        # three repeats of one rubric): a mean 1,249 reasoning tokens to return
        # a one-sentence verdict, ranging 581 to 2,523, which costs $2.96 per
        # 1k for this seat alone, more than the whole five-judge panel's
        # published figure at the time ($2.56). It returned the same verdict on
        # every repeat while the panel spread across 1, 2 and 3, so the spend
        # bought no agreement either. grok-4-1-fast gives that verdict in 347
        # billable tokens for $0.47 per 1k.
        #
        # This seat was defended as the weakest on the panel, 16.9 against
        # qwen's 29.3, on the argument that a 3-of-5 majority is built to
        # absorb one. Both halves of that were wrong. 16.9 is grok's
        # non-reasoning score against qwen's reasoning score, which is not a
        # comparison, and grok is not the weakest seat: at their ceilings it
        # reaches 30.6 against qwen's 29.3, and the probe shows it reasoning by
        # default (269 to 313 tokens), so the ceiling is where it operates. The
        # seat is cheaper and stronger; only the argument for it was wrong.
        'xai/grok-4-1-fast',
        'deepseek/deepseek-v4-flash',
        'minimax/MiniMax-M2.7',
    ),
    aggregation=Aggregation.MAJORITY_3_OF_5,
    # glm-5-maas was the first reserve until Google delisted it from the
    # garden (2026-08-25 price audit), and kimi-k2.6 carried the role alone
    # until its card was found scoring its own `none` rung twice (2026-09-03).
    # glm-5.2 is the strongest reserve on a lineage the panel does not already
    # seat, at $2.15 against the panel's $2.56 for all five, which is what a
    # reserve is for: it stands in once, it does not run every night.
    reserve_judges=('zai/glm-5.2',),
    use_when='High volume regression eval, nightly reruns, HITL triage feeders.',
    estimated_cost_per_1k=2.56,
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
    aggregation=Aggregation.MAJORITY,
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
    aggregation=Aggregation.MAJORITY,
    reserve_judges=('google/eu.claude-sonnet-5',),
    use_when=(
        'Data residency: every judge serves from an EU region. Anthropic, Google and OpenAI, which is what the EU catalog can field at the current generation: the DeepSeek seat carrying Balanced Trio has no EU endpoint newer than v3.1, so this is not that panel relocated.'
    ),
    estimated_cost_per_1k=6.55,
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
    aggregation=Aggregation.MAJORITY,
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

PRESETS: dict[str, JuryPreset] = {
    preset.name: preset
    for preset in (
        BALANCED_TRIO,
        STRONG_JURY,
        CHEAP_AGGREGATE,
        OPEN_WEIGHT_PORTABLE,
        EU_REGION,
        SINGLE_PROVIDER_TRIO,
    )
}

DEFAULT_PRESET = BALANCED_TRIO

# Estimate behind every published cost. Measured usage runs above this when a
# judge reasons without being asked (RES-996), so treat it as a floor.
ESTIMATED_PROMPT_TOKENS = 1500
ESTIMATED_COMPLETION_TOKENS = 150


# How each provider counts reasoning tokens, keyed by the provider half of the
# router ID.
#
# 'inside'     reasoning is part of `completion_tokens`; adding it double-bills.
# 'alongside'  reasoning is reported separately; ignoring it under-bills, by an
#              order of magnitude on a model that thinks and answers briefly.
#
# Read off the probe artifacts rather than assumed. Subtracting reasoning from
# the completion total leaves 36 to 82 tokens for every provider here except
# xai, which is the length of the one-sentence verdict the rubric asks for, so
# the visible answer is all that remains and the reasoning was inside. xai
# returns a mean 34 completion tokens beside 297 reasoning: the completion
# total is the verdict alone.
#
# Only providers a probe has actually measured belong here. Seven entries were
# written from the shape of the others, covering providers that have never
# returned a reasoning token in any run, and a guessed 'inside' is not a
# harmless default: it drops a seated judge's reasoning spend in silence, which
# is the exact failure keying on the provider was meant to end. They are gone,
# and a test refuses to let an unmeasured entry back in. An unregistered
# provider falls back and logs, which is a state you can see.
REASONING_ACCOUNTING: dict[str, str] = {
    'alibaba': 'inside',
    'anthropic': 'inside',
    'azure': 'inside',
    'baseten': 'inside',
    'deepseek': 'inside',
    'google': 'inside',
    'groq': 'inside',
    'moonshotai': 'inside',
    'openai': 'inside',
    'xai': 'alongside',
    'zai': 'inside',
}


def billable_completion(usage: dict[str, int], model: str) -> int:
    """Output tokens the provider bills, which is not always `completion_tokens`.

    Providers disagree about whether reasoning tokens are counted inside the
    completion total or reported alongside it. qwen3.5-35b-a3b returns 618
    completion with 581 reasoning, so reasoning is inside; grok-4-1-fast
    returns 34 completion with 301 reasoning, so it is not. Costing on
    `completion_tokens` alone bills the second kind for its visible answer only
    and understates it by an order of magnitude, which is exactly the
    comparison a reseat decision turns on.

    This used to decide by size, adding reasoning whenever it exceeded the
    completion total. That reads the right answer off the wrong thing: it is
    true of grok because grok answers in one sentence, and a provider that
    reports alongside while writing a longer answer than it thinks satisfies
    `reasoning <= completion` and has its reasoning dropped in silence. The
    accounting is a property of the provider, so key on the provider and keep
    the size relation as a check that the register still matches what arrives.

    Lives here rather than in the probe script because the probe and the
    artifact test both recost the same runs, and two implementations of this
    rule would silently disagree about what a panel costs.
    """
    completion = usage.get('completion_tokens', 0)
    reasoning = usage.get('reasoning_tokens') or 0
    if not reasoning:
        return completion

    accounting = REASONING_ACCOUNTING.get(model.split('/')[0])
    if accounting is None:
        # Unregistered provider: fall back to the old reading rather than fail
        # a live probe, and say so. A test refuses to let a seat stay here.
        logger.warning(
            f'{model}: provider is not in REASONING_ACCOUNTING, costing it by '
            f'the size of its reasoning count. Read the artifact and register it.'
        )
        return completion + reasoning if reasoning > completion else completion

    if accounting == 'inside' and reasoning > completion:
        raise ValueError(
            f'{model} reports {reasoning} reasoning tokens inside {completion} '
            f'completion tokens, which cannot be. REASONING_ACCOUNTING says its '
            f'provider counts reasoning inside the completion total; if that has '
            f'changed, change the register.'
        )
    return completion + reasoning if accounting == 'alongside' else completion


# Days a seat may sit in the garden before it has to be argued for again.
# Generations have been turning over in roughly eight weeks, so a seat past
# this has watched two or three rounds of its own band ship without moving.
MAX_SEAT_AGE_DAYS = 180

# Seats knowingly held past MAX_SEAT_AGE_DAYS, keyed by model identity.
#
# The other registers all answer "something newer exists, why not it?".  This
# one answers the question nothing else asks: a lineage can simply stop
# shipping, and then no in-family upgrade is ever found, no successor is ever
# flagged, and a seat quietly ages out while every freshness test passes.
#
# Age here is time since first listing in the garden, taken across every host
# of the same weights, so a second host relisting a model does not reset it.
AGED_SEATS: dict[str, str] = {
    'grok-4-1-fast': (
        'Listed 2026-03 and 182 days old on 2026-09-03, holding the cheap '
        'thinking seat on Cheap Aggregate at a $0.275 blend. xai has shipped '
        'one model since, grok-4.5, which is stronger at a 53.8 ceiling '
        'against 30.6 and eleven times the price at $3.00, on the one panel '
        'whose whole promise is volume. The alternative is not a newer xai '
        'model but no xai model, which would take the panel from five '
        'lineages to four and put two OpenAI-shaped judges in a five-judge '
        'majority. Retire this entry the day xai ships a small model.'
    ),
    'claude-haiku-4-5-20251001': (
        'Listed 2025-10 and the oldest seat in the library, held now only where '
        'a constraint holds it: the bottom seat of the Anthropic Only pool and '
        'the cheap seat of EU Region. Anthropic has shipped no small model '
        'since, so the in-family check finds nothing and the alternative in '
        'both places is not a newer Anthropic model but no Anthropic model, '
        'which is the one thing those two presets promise. It lost its third '
        'seat, on Balanced Trio, when this register was added: nothing was '
        'constraining it there and it was the weakest buy in the library, 23.7 '
        'at a $2.00 blend beside luna on the same panel at 38.1 for $0.45. '
        'Retire this entry the day Anthropic ships a small model.'
    ),
}


# Successors already looked at and deliberately not seated. A model reaching the
# garden is not a reason to put it in front of customers, but it is a reason to
# say why not: this dict is that record, and the freshness test fails on any
# successor missing from it. Seat the model or write the line; silence is the
# one thing that is not allowed.
REVIEWED_SUCCESSORS: dict[str, str] = {
    'grok-4.5': (
        'Newer than the grok-4-1-fast seat Cheap Aggregate now holds, and '
        'stronger (a 53.8 ceiling against 30.6), but at a $3.00 blend against $0.275 it '
        'is eleven times the price of the seat it would take, in the one panel '
        'whose whole purpose is volume. It already holds the middle seat of the '
        'Balanced Pareto Trio router pool, which is where that price buys '
        'something. Revisit if a jury preset ever wants an xai seat on quality.'
    ),
    'gemini-3.7-flash': (
        'Garden lists it EU-only (google/eu.gemini-3.7-flash) with no pricing '
        'entry, so it cannot be costed. Revisit for EU Region once priced.'
    ),
    'kimi-k3-fast': 'Latency variant of the seated baseten/kimi-k3, a serving variant rather than a generation.',
    'glm-5.2-fast': 'Baseten serving variant of the seated zai/glm-5.2, not a new generation.',
    'minimax-m3': (
        'Unseatable, not merely unseated: wafer/MiniMax-M3 is listed and priced '
        'but 404s on a completion call, so it carries an UNROUTABLE entry and '
        'held a pool seat on nothing but a catalog listing until a probe caught '
        'it. Would stay off the juries regardless, since Cheap Aggregate seats '
        'M2.7 for the MiniMax vote and the probe measured that lineage spending '
        '308 to 1,780 reasoning tokens across repeats of one prompt: a second '
        'MiniMax seat buys correlated cost variance, not a voice.'
    ),
    'gpt-transcribe': 'Speech-to-text, not a text judge. Wrong model type for any panel.',
    'gpt-live-transcribe': 'Realtime speech-to-text. Same reasoning as gpt-transcribe.',
}


# Presets that shipped in a draft and were then retired, with the reason on
# record. Mirrors DROPPED_POOLS in router_pools: a shape that disappears
# silently leaves its users guessing, so removal costs a written line here.
DROPPED_PRESETS: dict[str, str] = {
    'Value Trio': (
        'Retired 2026-08-24. Sold as the budget panel, but the blend repricing '
        'left Cheap Aggregate cheaper on the table ($2.73 vs $2.86 per 1k) with '
        'five judges to its three, and the 2026-08-18 probe measured Value Trio '
        '83% over its own table (probes/archive/probe-value-trio-20260818T084331Z). '
        'The overage was prose length, not reasoning: neither of the two '
        'expensive judges reports a reasoning token. glm-5-maas held 63% of '
        'the measured cost writing 556 to 567 tokens every time, and MiniMax '
        'M2.7 held 32% ranging 345 to 1,780 across repeats of one prompt. '
        '(Those two shares were first written as 58.7 and 39.6, recosted '
        'before `billable_completion` keyed on the provider.) A '
        'budget preset that is neither cheapest on paper nor close to its paper '
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
    try:
        return PRESETS[name]
    except KeyError:
        dropped = DROPPED_PRESETS.get(name)
        if dropped:
            raise ValueError(f'jury preset {name!r} was retired: {dropped}') from None
        raise ValueError(f'unknown jury preset {name!r}; available: {sorted(PRESETS)}') from None
