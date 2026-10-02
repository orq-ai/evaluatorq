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

They are never `orq/*` routers (RES-1573), even ones that derive the same
frontier server-side and stay current without a capture. A router resolves per
request and optimises each one alone, so a panel of them can seat three cards
from one vendor, which is the correlated-error case a panel exists to cancel.
The lineage diversity is what does not transfer to the platform, so seats stay
pinned here even where the arithmetic behind them has moved. A single judge is
free to be a router, since it has no panel to correlate.
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
    """What a seated judge bills per 1M tokens and the effort it runs at."""

    model_config = {'frozen': True}

    input_rate: float
    output_rate: float
    default_reasoning_effort: str | None


@lru_cache(maxsize=1)
def _load_rates() -> MappingProxyType[str, JudgeRates]:
    """The committed rate table, read once and handed out read-only.

    Frozen on purpose. This is cached process-wide, so a caller who mutated the
    returned mapping would silently reprice every published preset for the life
    of the process.
    """
    raw = cast('dict[str, Any]', json.loads(_RATES_PATH.read_text(encoding='utf-8')))
    return MappingProxyType({rid: JudgeRates(**fields) for rid, fields in raw['judges'].items()})


def judge_rates(router_id: str) -> JudgeRates | None:
    """The captured rates for a seated judge, or None for a model no preset seats."""
    return _load_rates().get(router_id)


def captured_at() -> str:
    """When the rates were captured, for anyone deciding whether to trust the table."""
    return cast('str', json.loads(_RATES_PATH.read_text(encoding='utf-8'))['captured_at'])


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
        description='USD per 1,000 pointwise items at 1,500 input / 1,500 output tokens, uncached.'
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
        """The reasoning effort each judge runs at: its catalogue default.

        A preset sends no effort, so each seat runs at the provider default. None
        means the catalogue names no effort parameter for that model.

        `llm_jury` takes one `reasoning_effort` for the whole panel, so passing
        one overrides every seat's default; per-judge call settings are a schema
        change and its own ticket (RES-1347).
        """
        return {judge: (r.default_reasoning_effort if (r := judge_rates(judge)) else None) for judge in self.judges}

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
        # Decimal, and one rounding at the end. A figure that lands exactly on a
        # half-cent would otherwise be decided by binary-float representation and
        # banker's rounding rather than by a price, and a recapture that moved
        # nothing could flip the table.
        return float((total / Decimal(1_000)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


BALANCED_TRIO = JuryPreset(
    name='Balanced Trio',
    judges=(
        'deepseek/deepseek-v4-pro',
        'openai/gpt-5.6-luna',
        'google/gemini-3.6-flash',
    ),
    # Same lineage as the seat it backs, so a promotion keeps the panel at three
    # families.
    reserve_judges=('deepseek/deepseek-flash',),
    use_when='Default subjective eval. Three families, three error surfaces.',
    estimated_cost_per_1k=17.56,
)

STRONG_JURY = JuryPreset(
    name='Strong Jury',
    judges=(
        'anthropic/claude-opus-5-5',
        'openai/gpt-5.6-sol',
        'google/gemini-3.6-flash',
    ),
    reserve_judges=('deepseek/deepseek-v4-pro',),
    use_when='Customer-facing benchmarks, preference data, anything that compounds.',
    estimated_cost_per_1k=85.5,
)

OPEN_WEIGHT_PORTABLE = JuryPreset(
    name='Open-Weight / Portable',
    judges=(
        'deepseek/deepseek-v4-pro',
        # Served from baseten because the Moonshot account 429s.
        'baseten/kimi-k3',
        'zai/glm-5.2',
    ),
    reserve_judges=('minimax/MiniMax-M2.7',),
    use_when='No dependence on a closed frontier vendor; migration path to self-hosting.',
    estimated_cost_per_1k=37.66,
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
    estimated_cost_per_1k=27.75,
)

SINGLE_PROVIDER_TRIO = JuryPreset(
    name='Single-Provider Trio',
    judges=(
        'openai/gpt-5.6-sol',
        'openai/gpt-5.6-terra',
        'openai/gpt-5.6-luna',
    ),
    reserve_judges=('openai/gpt-5.4-nano',),
    use_when=(
        'Workspaces locked to one provider contract. The three real tiers of '
        'the current OpenAI lineup rather than one tier plus two minis, so no '
        'family diversity: errors correlate and OpenAI-generated outputs face a '
        'self-preference risk the panel cannot vote away.'
    ),
    estimated_cost_per_1k=59.1,
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

# Flat estimate behind every published cost, the same for every seat whatever
# its reasoning effort.
ESTIMATED_PROMPT_TOKENS = 1500
ESTIMATED_COMPLETION_TOKENS = 1500


# Presets derived and costed, but not seated in PRESETS because a seat cannot
# do the job today. Distinct from DROPPED_PRESETS: nothing here was judged a bad
# panel, and each entry names the one thing that has to be true for it to ship.
WITHHELD_PRESETS: dict[str, str] = {
    'Cheap Aggregate': (
        'Not shipped yet. Five small judges across five lineages, concluding on '
        'three of five. One seat does not vote: `minimax/MiniMax-M2.7` answers a '
        '`json_schema` response format with prose, and the fallback that would '
        'resend the schema as instructions sits behind `except BadRequestError`, '
        'so it never fires on a 200 that simply has the wrong shape. The model '
        'itself complies when the schema reaches it as instructions. That leaves '
        'four voting seats, an even panel, which `_panel_is_well_formed` exists '
        'to reject and passes only because it counts declared seats rather than '
        'voting ones. Seating the reserve, `zai/glm-5.2`, is not a fix: at $8.70 '
        'per 1k for that seat alone it defeats a budget panel. Ships once the '
        'judge path falls back on an unparseable 200 as well as on a rejected '
        'request.'
    ),
}

# Presets that shipped in a draft and were then retired, with the reason on
# record. A shape that disappears silently leaves its users guessing, so removal
# costs a written line here and `get_preset` hands it back by name.
DROPPED_PRESETS: dict[str, str] = {
    'Value Trio': (
        'Retired. A budget panel whose measured cost ran far above its published '
        'figure, because two of its judges wrote long prose verdicts. Use '
        'Balanced Trio for a default panel, or Open-Weight / Portable for vendor '
        'independence.'
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
