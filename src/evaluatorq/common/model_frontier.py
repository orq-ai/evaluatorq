"""Intelligence/price frontier over the model garden (RES-1347, RES-1346).

Presets rot because models move, and until now the only computable axis was
price, so "is this successor better?" was a judgement call. Every scored card
in the public model catalogue carries Artificial Analysis intelligence scores,
per reasoning effort where the model has one, beside its own input and output
rates. That is the pair RES-1285 used to show 19 of 24 models on a live workspace are dominated and structurally unselectable, and it
is enough to decide seating arithmetically rather than by taste.

Both axes are read where a judge actually runs. Judges are called with no
reasoning effort, so the provider applies the catalogue default
(`default_effort` on the card), and each card is ranked on its score at that
rung. Price is a 3:1 input/output blend.

Dominance is the whole rule: B dominates A when B is at least as intelligent and
no more expensive, and strictly better on one of them. A dominated seat is dead
weight the user cannot see. A successor that dominates the incumbent is a
straight upgrade; one that does not is a trade the preset's `use_when` has to
justify in words.

This is shared by all four preset sets in RES-1347, since router pools, juries,
classifiers and evaluator groups all pick models out of the same garden.

Nothing here is committed: the garden is derived from the keyless public
`GET /v2/model-catalog` on every process (`load_garden`). That endpoint lists
the whole platform catalogue whatever key the caller holds, where `/v2/models`
filters to what one workspace or project enabled. The registers below
(MANUAL_MODEL_CARDS, AA_FILLS, SUPERSEDED, UNROUTABLE, NEVER_SEAT) are reviewed
judgements about that live data, each with its source.
"""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from evaluatorq.common.env_config import env_float
from evaluatorq.common.llm_client import orq_base_url
from evaluatorq.common.model_catalogue import ModelCatalogueError, _finite_nonneg

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


class ScoredModel(BaseModel):
    """A garden model with the two axes seating decisions are made on."""

    # A misspelled field in a hand-written MANUAL_MODEL_CARDS entry must fail, not fall back to a default.
    model_config = {'extra': 'forbid'}

    router_id: str
    intelligence_index: float = Field(
        description=(
            'The headline Artificial Analysis index on the card, with no reasoning level. '
            'Ranked on when the card exposes no reasoning effort, or has no score at its '
            'default effort; otherwise `default_index` reads the ladder.'
        )
    )
    input_price: float = Field(description='USD per 1M input tokens.')
    output_price: float = Field(description='USD per 1M output tokens.')
    region: str | None = None
    provider: str | None = None
    # The rung the headline was measured at (see `_headline_rung`), when known.
    # Only used to catch a card contradicting itself.
    headline_effort: str | None = None
    effort_indices: dict[str, float] = {}
    effort_parameter: bool = Field(
        default=False, description='Whether the card lists reasoning_effort in supported_parameters.'
    )
    default_effort: str | None = Field(
        default=None,
        description='The catalogue default effort. None on a card with an effort parameter '
        'means the card states no default.',
    )

    @property
    def price(self) -> float:
        """USD per 1M tokens, 3:1 input/output blend, lower is better."""
        return (3 * self.input_price + self.output_price) / 4

    @property
    def free(self) -> bool:
        """A zero rate on either side. Free cards are never ranked, seated or pooled."""
        return self.input_price <= 0 or self.output_price <= 0

    @property
    def default_index(self) -> float:
        """The index a model is ranked on: its score at the catalogue default effort, else its headline.

        The default rung is where the judge runs, so its ladder score wins when the
        card has one. A card that never benchmarked its default (deepseek-v4-pro
        runs at high while Artificial Analysis measures max and none) ranks on its
        headline index instead, which carries no reasoning level;
        `ranked_on_headline` says when that happened.
        """
        if self.effort_parameter and self.default_effort is not None and self.default_effort in self.effort_indices:
            return self.effort_indices[self.default_effort]
        return self.intelligence_index

    @property
    def ranked_on_headline(self) -> bool:
        """Whether a card with an effort parameter is ranked on its headline for want of a default-rung score."""
        if not self.effort_parameter:
            return False
        if self.default_effort is None:
            return True
        return self.default_effort not in self.effort_indices and self.headline_effort != self.default_effort

    @property
    def pareto_score(self) -> float | None:
        """Ranked intelligence index per blended USD per 1M tokens, higher is better. None for a free card."""
        if self.free:
            return None
        return self.default_index / self.price

    def dominates(self, other: ScoredModel) -> bool:
        """At least as good on both axes, strictly better on one."""
        if self.free or other.free:
            return False
        mine, theirs = self.default_index, other.default_index
        at_least_as_good = mine >= theirs and self.price <= other.price
        strictly_better = mine > theirs or self.price < other.price
        return at_least_as_good and strictly_better


# Below this index nothing is recommended anywhere: the tier is noise, and a
# cheap seat filled from it makes a pool look complete while lowering what the
# user gets. The value sits under every seat a reviewer has accepted (Claude
# Haiku 4.5 at 5.7 is the platform's single-judge baseline) and above the band
# it exists to exclude, which on the AA index v5 garden (staging, 2026-11-17) is
# Llama-3.3-70B at 1.9, gpt-4o at 2.2, gpt-oss-20b at 2.4, qwen3-coder at 2.5
# to 2.7 and gpt-oss-120b at 3.8. Re-derived from 15.0 on the v4.x garden for
# the v5 switch; gpt-5.4-mini (3.3 at its none default) now falls under it.
MIN_SEAT_INDEX = 5.0

# MIN_SEAT_INDEX, NEVER_SEAT verdicts and the preset seats are calibrated on AA
# index v5. A catalogue with no v5 rung still serves the old version (prod until
# orquesta-web cc09670ca and 8851da930 ship) and would rank against the wrong
# floor, so load_garden refuses it. A new major version needs a re-baseline too.
REQUIRED_INDEX_VERSION = 'v5'

# The index is a benchmark aggregate, and a floor cannot catch a model whose
# benchmarks say more than its behaviour: on index v4.3, when the verdict was
# made, gpt-oss-120b scored 23.8 beside Haiku 4.5's 23.7 and did not perform beside it. Families here are
# never seated at any score or price, with the verdict on record. Review
# verdict from PR #330 (2026-08-24).
NEVER_SEAT: dict[str, str] = {
    'gpt-oss': (
        'Benchmark-inflated relative to observed quality: scores beside Claude '
        'Haiku 4.5 on the index and does not deliver beside it. Not to be '
        'recommended at any price tier.'
    ),
    '-contributor': (
        "Meta's Contributor tier (e.g. meta/muse-spark-1.3-contributor, $0.10/$0.20 "
        'against $1.25/$4.25 standard) is discounted in exchange for Meta training on '
        "prompts and completions, so seating it would send users' eval data to Meta "
        'unasked. The standard tier ranks on its own card. Reported by TechCrunch '
        '(2026-09-03).'
    ),
}


# Listing is not callability. The model catalogue advertises these router IDs and
# `client.models.list()` returns them, so every listing-based freshness test passes,
# but an actual completion call 404s: the host serves a catalog that does not
# include the model the router ID names. Only a live probe can find this, which
# is why entries carry the date and the error they were found by.
UNROUTABLE: dict[str, str] = {
    'minimax/MiniMax-M2.7': (
        "Listed and priced, but a completion call returns 404 'The model "
        '"MiniMax-M2.7" is no longer available\' (probed 2026-10-26, '
        'archive/probe-cheap-aggregate-20261026T100004Z). Drop this entry when '
        'it answers again.'
    ),
    'wafer/MiniMax-M3': (
        'Listed by the models API and seated in two pools, but a completion call '
        "returns 404 'Model MiniMax-M3 is not available on this endpoint'. The "
        'wafer host serves DeepSeek-V4-Flash, GLM-5.2, Kimi-K2.6, Kimi-K3 and '
        'Qwen3.5-397B-A17B only (probed 2026-08-25). The same weights are '
        'callable as nvidia/minimaxai/minimax-m3, which prices at zero and so is '
        'excluded as free. minimax/MiniMax-M3 answers but is unlisted, unpriced and '
        'unscored, so it cannot be seated either. Verdict shape on that route over '
        'eight repeats (probe-candidates-20261026T101750Z): 8 of 8 bare parseable '
        'JSON, 40 to 48 completion tokens, but scores spread 1, 2 and 3 on one '
        'identical prompt. Drop this entry when either is fixed.'
    ),
}


# Answers a `json_schema` response format with prose. The judge falls back to
# json_object only when a provider rejects the request, not on a 200 of the
# wrong shape, so such a seat never votes. Binds the cheap-aggregate derivation
# only: a router pool member is not asked for a verdict. Keyed by exact router
# ID, with the probe that found it.
NO_STRUCTURED_VERDICT: dict[str, str] = {
    'minimax/MiniMax-M3.1-Flash': (
        'Ten judge prompts through the real structured-output path on the prod '
        'router: 0 of 10 parsed, each answered in prose ("True. 7 x 8 = 56."). '
        'The same weights on wafer/MiniMax-M3.1-Flash (same price, global) parsed '
        '10 of 10, so the fault is this host. 2026-11-17.'
    ),
    'alibaba/qwen3.6-35b-a3b': (
        'Ten judge prompts through run_judge on the prod router: 0 of 10 verdicts. '
        'Responses json_schema is rejected with a 400 asking for the word "json", '
        'and the json_object fallback returns the injected schema instead of a '
        'verdict. greenpt (9/10) and scaleway (8/10) pass but are EUR-priced. 2026-11-20.'
    ),
}

# Old models with a newer version in the live catalogue, keyed by exact router
# ID. `unseatable_reason` refuses these. Name-derived successors are applied by
# `rank_cheapest_top` and `router_pools.build_pools`. This register carries only what
# `derived_successor` cannot find from the names (same stem and tier, higher
# version), such as a successor that changed tier word (glm-5.2 -> glm-5.3-flash)
# or an old model the derivation skips because its successor is not rankable. Models that
# `derived_successor` already finds (gpt-5.6-luna, haiku-4.5, gemini-3.5-flash-lite) are not
# repeated here.
# Every entry names its successor, the reason and where it was checked.
SUPERSEDED: dict[str, str] = {
    'moonshotai/kimi-k2.6': (
        'superseded by kimi-k3 (baseten/kimi-k3, nebius/moonshotai/Kimi-K3, 43.6 max); '
        'checked on the live catalogue 2026-10-26. K3 blends $6.00, so it is not a '
        'cheap-tier successor, but the old model is not a seat either.'
    ),
    'zai/glm-5.2': (
        'superseded by zai/glm-5.3-flash (41.8 at max, $0.119 blended on the card, '
        '$0.2375 at actual Zai billing); live catalogue 2026-10-26.'
    ),
    'alibaba/qwen3.5-35b-a3b': (
        'superseded by qwen3.6-35b-a3b (greenpt/qwen3.6-35b-a3b rest, scaleway/qwen3.6-35b-a3b '
        'europe); live catalogue 2026-10-26.'
    ),
    'tensorix/deepseek/deepseek-chat-v3.1': (
        'superseded by the deepseek-v4 generation (baseten/deepseek-v4.1-flash, '
        'deepseek/deepseek-v4-pro); live catalogue 2026-10-26.'
    ),
    # Rankable once a headline measured at the default rung counts (2026-10-26).
    'anthropic/claude-opus-4-6': 'superseded by claude-opus-5 and claude-opus-5-5 on the same route.',
    'anthropic/claude-sonnet-4-6': 'superseded by claude-sonnet-5 on the same route.',
    # Recorded by hand on 2026-10-27, when gpt-5.4-mini had no score at its default (none) and could not be derived.
    'openai/gpt-5-mini': 'superseded by openai/gpt-5.4-mini (listed; 3.3 at its default none on v5, under the seat floor); 2026-11-17.',
    'azure/gpt-5-mini': 'superseded by openai/gpt-5.4-mini (listed; 3.3 at its default none on v5, under the seat floor); 2026-11-17.',
}


# Host, region and serving-variant noise that makes one model look like several.
# `openai/gpt-5.6-terra`, `azure/global.gpt-5.6-terra` and `openai/eu.gpt-5.6-terra`
# are one model; a successor check that counts them separately reports work that
# does not exist.
_REGION_PREFIXES: tuple[str, ...] = ('eu.', 'us.', 'apac.', 'global.', 'au.', 'jp.')
# Bedrock in-region routes spell the AWS region out: `aws/us-east-1.openai.gpt-5.4`.
_AWS_REGION_PREFIX = re.compile(r'^[a-z]{2}(?:-[a-z]+)+-\d+\.')
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
    `anthropic/claude-haiku-4-5-20251001` are one model, and eu-region seats the
    first while balanced-trio used to seat the second.
    In-region Bedrock routes (`aws/us-east-1.openai.gpt-5.4`) and host region
    tags (`@eu`) are stripped the same way.
    """
    name = _AWS_REGION_PREFIX.sub('', router_id.split('/')[-1].lower())
    for prefix in _REGION_PREFIXES:
        name = name.removeprefix(prefix)
    head, _, tail = name.partition('.')
    if tail and head in _VENDOR_TOKENS:
        name = tail
    # Bedrock version tags and host region tags (`munito/.../glm-5.3-flash@eu`).
    return re.sub(r'(-v\d+:\d+|@[a-z-]+)$', '', name)


def _lineage(router_id: str) -> tuple[str, tuple[int, ...], str] | None:
    """(stem, version, tier) from a router ID: `gpt-5.6-luna` is ('gpt-', (5, 6), '-luna').

    Read off `model_identity`, so hosts, regions (including in-region Bedrock
    routes), dotted vendors and Bedrock suffixes are stripped, then build dates; dashed
    Claude versions (`haiku-5-5`) read as dotted. None where no version parses.
    """
    name = model_identity(router_id)
    name = re.sub(r'-?\d{8}$', '', name)
    name = re.sub(r'-\d{4}$', '', name)
    if name.startswith('claude'):
        name = re.sub(r'(?<=\d)-(?=\d)', '.', name)
    match = re.match(r'^([a-z][a-z0-9]*(?:-[a-z]+)*?)-?(\d+(?:\.\d+)*)((?:-[a-z0-9.]+)*)$', name)
    if not match:
        return None
    return match[1], tuple(int(part) for part in match[2].split('.')), match[3]


def derived_successor(router_id: str, garden: Mapping[str, ScoredModel]) -> str | None:
    """A seatable router ID in the garden with the same stem and tier and a higher version, if any.

    The successor must itself be comparable (priced, not free) and
    seatable (not NEVER_SEAT, UNROUTABLE, SUPERSEDED or under the floor),
    or a newer card nothing may be seated on would retire a model that still works.

    The highest version wins; ties go to the lowest router ID. Cross-tier
    successors (`glm-5.2` to `glm-5.3-flash`) are not derivable and live in
    SUPERSEDED.
    """
    mine = _lineage(router_id)
    if mine is None:
        return None
    newer = []
    for other, candidate in garden.items():
        theirs = _lineage(other)
        same_lineage = theirs is not None and theirs[0] == mine[0] and theirs[2] == mine[2]
        if same_lineage and theirs[1] > mine[1] and not candidate.free and seatable(candidate):
            newer.append((theirs[1], other))
    return min(newer, key=lambda pair: (tuple(-v for v in pair[0]), pair[1]))[1] if newer else None


# A card ranked on its headline (no effort parameter) can still publish a
# ladder, and if that ladder scores the headline's own rung differently the
# card is contradicting itself about the one number it is ranked on.
# `kimi-k2.6` read a headline of 44.2 taken at `none` beside a `none` rung of
# 34.6: one of the two is wrong and the card does not say which, so it holds no
# seat until the card is fixed. Cards with an effort parameter rank on the
# ladder rung directly and never on the headline, so this cannot affect them.
# `test_no_card_scores_its_own_rung_twice` fires on any new one.
def headline_contradicts_its_own_rung(model: ScoredModel) -> bool:
    """Whether a headline-ranked card publishes a different index for the rung its headline came from."""
    if model.effort_parameter:
        return False
    rung = model.headline_effort or 'none'
    return rung in model.effort_indices and not math.isclose(
        model.effort_indices[rung], model.intelligence_index, rel_tol=0.0, abs_tol=1e-6
    )


def _effort_ladder(card: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """The card's `reasoning_efforts` rungs, keeping only well-formed ones.

    External data: a ladder that is not a mapping, or a rung that is not one,
    reads as absent rather than raising and taking the whole garden down.
    """
    ladder = card.get('reasoning_efforts')
    if not isinstance(ladder, Mapping):
        return {}
    return {effort: entry for effort, entry in ladder.items() if isinstance(entry, Mapping)}


def _score(value: object) -> float | None:
    """A benchmark index as a float, or None unless it is a finite real number (not a bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _rung_version(entry: Mapping[str, Any]) -> str:
    """The major AA index version a rung names, e.g. 'v5' for 'v5.1', or '' when it names none."""
    return str(entry.get('index_version', '')).split('.')[0]


def _rung_indices(router_id: str, card: Mapping[str, Any]) -> dict[str, float]:
    """Each REQUIRED_INDEX_VERSION rung's index, dropping (with a warning) any that is present but not a finite number.

    A rung on another version is left out whatever its score, so a partly
    migrated card never ranks a v4 number against the v5 floor and seats.
    """
    indices: dict[str, float] = {}
    for effort, entry in _effort_ladder(card).items():
        if _rung_version(entry) != REQUIRED_INDEX_VERSION:
            continue
        raw = entry.get('intelligence_index')
        if raw is None:
            continue
        score = _score(raw)
        if score is None:
            logger.warning('{}: ignoring effort rung {!r} with a non-numeric index {!r}', router_id, effort, raw)
            continue
        indices[effort] = score
    return indices


def _price_per_million(card: Mapping[str, Any], side: Literal['input', 'output']) -> float | None:
    """A public card's base USD rate per 1M tokens, or None when it is missing or in another unit or currency."""
    pricing = card.get('pricing')
    rate = pricing.get(side) if isinstance(pricing, Mapping) else None
    if not isinstance(rate, Mapping):
        return None
    cost, per = _finite_nonneg(rate.get('cost')), _finite_nonneg(rate.get('per'))
    currency = rate.get('currency')
    if cost is None or not per or rate.get('unit') != 'tokens':
        return None
    if not isinstance(currency, str) or currency.upper() != 'USD':
        return None
    return round(cost / per * 1_000_000, 6)


def _price_gap(card: Mapping[str, Any]) -> str | None:
    """Why a card has no USD per-token price, or None when it has one."""
    if _price_per_million(card, 'input') is not None and _price_per_million(card, 'output') is not None:
        return None
    pricing = card.get('pricing')
    rates = [pricing.get(side) for side in ('input', 'output')] if isinstance(pricing, Mapping) else []
    currencies = {rate.get('currency') for rate in rates if isinstance(rate, Mapping)}
    if any(isinstance(currency, str) and currency.upper() != 'USD' for currency in currencies):
        return 'non-USD price'
    return 'no USD per-token price'


def _headline_rung(
    card: Mapping[str, Any], headline: float, rungs: Mapping[str, float], default: str | None
) -> str | None:
    """The rung the headline was measured at.

    The default when the card says it was scored, else the one rung with the
    same index.

    The public card names no Artificial Analysis slug per rung, so an equal
    index is the only other evidence. Two rungs on the headline's index prove
    nothing, and neither does a host default on its own.
    """
    if default is not None and card.get('default_effort_scored') is True:
        return default
    matches = [effort for effort, index in rungs.items() if math.isclose(index, headline, rel_tol=0.0, abs_tol=1e-6)]
    return matches[0] if len(matches) == 1 else None


def scored_model_from_card(card: Mapping[str, Any]) -> ScoredModel | None:
    """A ScoredModel from one public `/v2/model-catalog` card, or None if it has no index or rates.

    A card whose default effort rung names another AA index version is None
    too, with a warning: its headline carries no version, so it cannot stand
    in for the off-version rung the card would be ranked on.

    The headline is `artificial_intelligence.artificial_analysis_intelligence_index`;
    `_headline_rung` says which rung it was measured at, when that is known. A
    card whose `location` is only `eu` is EU-pinned (`region='europe'`); every
    other card is unpinned, since only `europe` constrains a seat.
    """
    router_id = card.get('id')
    scores = card.get('artificial_intelligence')
    headline = _score(scores.get('artificial_analysis_intelligence_index')) if isinstance(scores, Mapping) else None
    input_price, output_price = _price_per_million(card, 'input'), _price_per_million(card, 'output')
    if not isinstance(router_id, str) or headline is None or input_price is None or output_price is None:
        return None
    parameters = card.get('supported_parameters')
    effort_parameter = isinstance(parameters, list) and 'reasoning_effort' in parameters
    default = card.get('default_effort')
    default_effort = default if effort_parameter and isinstance(default, str) and default else None
    rungs = _rung_indices(router_id, card)
    default_rung = _effort_ladder(card).get(default_effort) if default_effort is not None else None
    if default_rung is not None and _rung_version(default_rung) != REQUIRED_INDEX_VERSION:
        # The headline carries no version, so falling back to it would rank the
        # off-version default rung's score after all.
        logger.warning(
            '{}: default effort rung {!r} is on AA index {!r}, not {}; the card is left out',
            router_id,
            default_effort,
            default_rung.get('index_version'),
            REQUIRED_INDEX_VERSION,
        )
        return None
    location = card.get('location')
    provider = card.get('provider')
    return ScoredModel(
        router_id=router_id,
        intelligence_index=headline,
        input_price=input_price,
        output_price=output_price,
        region='europe' if location == ['eu'] else None,
        provider=provider.get('id') if isinstance(provider, Mapping) else router_id.split('/')[0],
        headline_effort=_headline_rung(card, headline, rungs, default_effort),
        effort_indices=rungs,
        effort_parameter=effort_parameter,
        default_effort=default_effort,
    )


def garden_from_cards(cards: Iterable[Mapping[str, Any]]) -> dict[str, ScoredModel]:
    """Every scored card, keyed by router ID. Unscored or unpriced cards are left out."""
    models = (scored_model_from_card(card) for card in cards)
    return {m.router_id: m for m in models if m is not None}


# Base URL -> the non-deprecated public cards, fetched once per process.
_public_cards: dict[str, tuple[dict[str, Any], ...]] = {}


async def public_cards(*, timeout_s: float | None = None) -> tuple[dict[str, Any], ...]:
    """Every non-deprecated card in the keyless public `GET /v2/model-catalog`, all pages.

    Cached per host for the process. A failed fetch is not cached, so the next
    call retries. `timeout_s` defaults to EVALUATORQ_CATALOGUE_TIMEOUT_S.

    Raises:
        ModelCatalogueError: The fetch failed, a page was unreadable, or the
            catalogue reported more pages but returned none or repeated a cursor.
    """
    base = orq_base_url()
    cached = _public_cards.get(base)
    if cached is not None:
        return cached
    url = f'{base}/v2/model-catalog'
    cards: list[dict[str, Any]] = []
    cursor: str | None = None
    try:
        async with httpx.AsyncClient(timeout=timeout_s or env_float('EVALUATORQ_CATALOGUE_TIMEOUT_S')) as http:
            while True:
                response = await http.get(url, params={'starting_after': cursor} if cursor else None)
                response.raise_for_status()
                page = response.json()
                raw = page.get('data') if isinstance(page, dict) else None
                if not isinstance(raw, list):
                    raise ModelCatalogueError(f'model catalogue page has no card list: {str(page)[:200]}')
                data = [card for card in raw if isinstance(card, dict) and isinstance(card.get('id'), str)]
                cards.extend(card for card in data if not card.get('deprecated'))
                if not page.get('has_more'):
                    break
                if not data:
                    raise ModelCatalogueError(
                        'model catalogue reports more pages but returned none; the roster would be truncated'
                    )
                if data[-1]['id'] == cursor:
                    raise ModelCatalogueError(
                        f'model catalogue repeated the page after {cursor}; pagination does not advance'
                    )
                cursor = data[-1]['id']
    except (httpx.HTTPError, ValueError) as exc:
        status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
        raise ModelCatalogueError(f'public model catalogue fetch failed: {exc}', status_code=status) from exc
    if not cards:
        raise ModelCatalogueError('public model catalogue returned no cards')
    _public_cards[base] = result = tuple(cards)
    return result


def reset_public_cards() -> None:
    """Clear the process-lifetime public card cache; exists for test isolation."""
    _public_cards.clear()


def _index_versions(card: Mapping[str, Any]) -> set[str]:
    """The major AA index versions the card's effort rungs name, e.g. {'v5'}."""
    return {_rung_version(rung) for rung in _effort_ladder(card).values()}


async def load_garden() -> dict[str, ScoredModel]:
    """The live model garden: every scored public catalogue card, keyed by router ID.

    Needs no key. Scores and prices come from one response, so they share a
    benchmark version.

    A card whose ladder names no REQUIRED_INDEX_VERSION rung is left out with
    a warning, since its scores sit on another scale, and so is one whose
    default effort rung names another version. A card with no ladder
    carries no version and is kept, with one warning naming them. A scored card
    left out for its price, which is not a USD per-token one, is named in a
    warning too.

    Raises:
        ModelCatalogueError: The catalogue fetch failed or was unreadable, no
            card on it carried a USD price and a score, or no card has a rung
            on REQUIRED_INDEX_VERSION.
    """
    cards = await public_cards()
    versions = {card['id']: _index_versions(card) for card in cards}
    if not any(REQUIRED_INDEX_VERSION in found for found in versions.values()):
        found_any = sorted(set().union(*versions.values()))
        raise ModelCatalogueError(
            f'public model catalogue has no AA index {REQUIRED_INDEX_VERSION} rung '
            f'(found {found_any or "none"}); the seat floor and presets are calibrated on '
            f'{REQUIRED_INDEX_VERSION}'
        )
    off_scale = sorted(
        router_id for router_id, found in versions.items() if found and REQUIRED_INDEX_VERSION not in found
    )
    if off_scale:
        logger.warning(
            '{} catalogue cards have no AA index {} rung and are left out: {}',
            len(off_scale),
            REQUIRED_INDEX_VERSION,
            ', '.join(off_scale),
        )
    kept = [card for card in cards if card['id'] not in off_scale]
    garden = garden_from_cards(kept)
    unpriced: dict[str, list[str]] = {}
    for card in kept:
        scores = card.get('artificial_intelligence')
        scored = (
            isinstance(scores, Mapping) and _score(scores.get('artificial_analysis_intelligence_index')) is not None
        )
        reason = _price_gap(card)
        if scored and reason:
            unpriced.setdefault(reason, []).append(card['id'])
    for reason, ids in sorted(unpriced.items()):
        logger.warning(
            '{} scored catalogue cards have a {} and are left out: {}', len(ids), reason, ', '.join(sorted(ids))
        )
    unversioned = sorted(router_id for router_id in garden if not versions[router_id])
    if unversioned:
        logger.warning(
            '{} seatable catalogue cards have no effort ladder, so their index version is unknown and is taken as {}: {}',
            len(unversioned),
            REQUIRED_INDEX_VERSION,
            ', '.join(unversioned),
        )
    if not garden:
        raise ModelCatalogueError('public model catalogue has cards but none with a USD price and a score')
    return garden


def load_garden_sync() -> dict[str, ScoredModel]:
    """`load_garden` for synchronous callers.

    Refuses inside a running event loop, where blocking on the fetch would stall
    the loop: async callers `await load_garden()` and pass the garden in.

    Raises:
        RuntimeError: Called from inside a running event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(load_garden())
    raise RuntimeError('load_garden_sync() inside a running event loop: await load_garden() and pass the garden in')


def _created_ms(created: object) -> int | None:
    """A catalogue `created` stamp as epoch milliseconds; ISO strings, seconds and milliseconds all occur."""
    if isinstance(created, str):
        try:
            return int(datetime.fromisoformat(created.replace('Z', '+00:00')).timestamp() * 1000)
        except ValueError:
            return None
    if isinstance(created, (int, float)) and not isinstance(created, bool):
        return int(created if created > 1e11 else created * 1000)
    return None


async def public_listings(*, timeout_s: float | None = None) -> dict[str, int | None]:
    """Router ID to first-listed epoch milliseconds, for every non-deprecated public card.

    The roster the successor and age checks date models by, scored or not.

    Raises:
        ModelCatalogueError: The catalogue fetch failed or was unreadable.
    """
    return {card['id']: _created_ms(card.get('created')) for card in await public_cards(timeout_s=timeout_s)}


# The cheap-tier ceiling: 3:1 blended USD per 1M tokens, strictly under. The
# user's rule is "input price under $1.00"; this is the same rule on the axis
# the library ranks on. A model priced just under $1.00 in at the usual 1:5
# input:output ratio blends to (3*1 + 5)/4 = $2.00. On the 2026-10-26 live
# catalogue the two rules pick the same candidates, and $1.00 would leave only
# three families, so no reserve (derivation in the 2-pager).
MAX_BLEND = 2.0

# Hand-written cards for models the router can call but the public catalogue
# does not score, so there is no live card to read their score from. Each
# value carries its own source and date. The live catalogue wins: an entry is
# used only while the router id is absent from the live garden, so once a
# scored card appears the entry is dead and should be deleted. Never add a
# value that has no source.
MANUAL_MODEL_CARDS: dict[str, dict[str, Any]] = {
    'openai/gpt-6.1-terra': {
        # Callable 2026-11-10 (chat completion 200). The public card lists it
        # and its rates but carries no score, default effort or ladder, nor does
        # its EU sibling (prod and staging, 2026-11-17).
        'input_price': 1.00,  # public /v2/model-catalog card, 2026-11-17
        'output_price': 5.00,  # same
        'effort_parameter': True,
        # Inferred, not read: every scored GPT-6 generation card (Sol, Luna,
        # 6.1 Sol) defaults to medium.
        'default_effort': 'medium',
        'effort_indices': {},
        # AA v5.0.1 max (artificialanalysis.ai/models/gpt-6-1-terra, 2026-11-10).
        # No medium rung, so it ranks on this headline.
        'headline_effort': 'max',
        'intelligence_index': 51.0,
    },
}


# Scores the live card lacks at its catalogue default effort, taken from the
# Artificial Analysis API. Accepted only where the AA-measured rung is the
# catalogue default; the live ladder wins whenever it has the rung. Fill
# nothing from another rung: DeepSeek flash and pro default to high or medium
# and AA measures only max and none, so they rank on their headline index.
# Empty since the AA index v5 switch (2026-11-17): the glm-5.3-flash entries
# held a v4.3 value (41.8) and the v5 card now carries its own headline (27.8).
AA_FILLS: dict[str, dict[str, Any]] = {}


def with_aa_fills(garden: dict[str, ScoredModel]) -> dict[str, ScoredModel]:
    """The garden with AA_FILLS applied where the live ladder lacks the filled rung."""
    filled = {}
    for router_id, fill in AA_FILLS.items():
        model = garden.get(router_id)
        if model is not None and model.default_effort == fill['effort'] and fill['effort'] not in model.effort_indices:
            filled[router_id] = model.model_copy(
                update={'effort_indices': {**model.effort_indices, fill['effort']: fill['index']}}
            )
    return {**garden, **filled}


def with_manual_cards(garden: dict[str, ScoredModel]) -> dict[str, ScoredModel]:
    """The garden plus every hand-written MANUAL_MODEL_CARDS entry the live garden has no card for."""
    extra = {
        router_id: ScoredModel(router_id=router_id, provider=router_id.split('/')[0], **fields)
        for router_id, fields in MANUAL_MODEL_CARDS.items()
        if router_id not in garden
    }
    return {**garden, **extra}


def rank_cheapest_top(
    garden: dict[str, ScoredModel],
    family_of: Callable[[str], str],
    max_blend: float = MAX_BLEND,
) -> list[ScoredModel]:
    """The best cheap model of each family, strongest first: the cheap-aggregate derivation.

    Eligible: comparable (not free), seatable (not
    NEVER_SEAT, UNROUTABLE, SUPERSEDED or under the floor), able to return a
    structured verdict (not NO_STRUCTURED_VERDICT), not EU-pinned, without a newer eligible version of itself
    (`derived_successor` over the eligible set, so a successor that is itself EU-pinned or over the ceiling retires nothing), and a 3:1 blend strictly under `max_blend`. Prices are the garden's own, so pass
    the live garden. One model per family (`family_of`), the highest default
    index in it; ties break to the lower price, then router_id. Seats are the
    first n entries and the reserve is the next, which is by construction a
    different family from every seat.
    """
    eligible = {
        router_id: m
        for router_id, m in garden.items()
        if not m.free
        and seatable(m)
        and router_id not in NO_STRUCTURED_VERDICT
        and m.region != 'europe'
        and m.price < max_blend
    }
    best: dict[str, ScoredModel] = {}
    for model in sorted(eligible.values(), key=lambda m: (-(m.default_index or 0), m.price, m.router_id)):
        if derived_successor(model.router_id, eligible):
            continue
        try:
            family = family_of(model.router_id)
        except ValueError:
            logger.warning('{}: unknown lineage, left out of the cheap ranking', model.router_id)
            continue
        best.setdefault(family, model)
    return sorted(best.values(), key=lambda m: (-m.default_index, m.price, m.router_id))


def unseatable_reason(model: ScoredModel) -> str | None:
    """Why a model may not hold any recommended seat, pool or jury; None if it may."""
    if model.free:
        return f'free (input {model.input_price}, output {model.output_price} per 1M)'
    if model.default_index < MIN_SEAT_INDEX:
        return f'index {model.default_index} is under the {MIN_SEAT_INDEX} floor'
    if headline_contradicts_its_own_rung(model):
        return 'card scores its own headline rung twice, differently'
    if model.router_id in UNROUTABLE:
        return f'unroutable: {UNROUTABLE[model.router_id]}'
    if model.router_id in SUPERSEDED:
        return f'superseded: {SUPERSEDED[model.router_id]}'
    family = next((f for f in NEVER_SEAT if f in model.router_id), None)
    return f'never seated: {NEVER_SEAT[family]}' if family else None


def seatable(model: ScoredModel) -> bool:
    """Whether a model may hold any recommended seat, pool or jury."""
    return unseatable_reason(model) is None


def pareto_frontier(models: list[ScoredModel]) -> list[ScoredModel]:
    """Comparable models nothing else beats on both axes, strongest first.

    Everything off the frontier is a worse deal than something else in the same
    garden, which is why a preset containing one is dead weight. Free cards
    have no position on the price axis and are left out.
    """
    comparable = [m for m in models if not m.free]
    frontier = [m for m in comparable if not any(other.dominates(m) for other in comparable)]
    return sorted(frontier, key=lambda m: -m.default_index)


def region_compatible(candidate: ScoredModel, incumbent: ScoredModel) -> bool:
    """Whether a candidate can stand in for an incumbent without moving regions.

    Without this guard the arithmetic gives bad advice. `azure/eu.gpt-5.6-luna`
    beats `openai/gpt-5.4-mini` on both axes, so raw dominance says swap, but
    balanced-trio is general-purpose and the swap would silently pin it to the
    EU. A region-pinned model may only replace another region-pinned one.
    """
    pinned = {'europe'}
    if incumbent.region in pinned:
        return candidate.region == incumbent.region
    return candidate.region not in pinned


MIN_INDEX_GAIN = 2.0
"""Index points a *different* model must gain before it counts as an upgrade.

The Artificial Analysis index is a benchmark aggregate, not a measurement with
an error bar, and sub-point gaps between neighbouring tiers are not signal. Left
ungated, the arithmetic proposes seating `gpt-5-mini` (index 30.9) over
`gpt-5.4-mini` (29.8) purely because it is cheaper, which trades a generation
and a knowledge cutoff for 1.1 points of benchmark noise. It likewise proposes
`gpt-5.4-nano` over `gpt-5.4-mini` on a 0.4 point gap, collapsing the tier
structure the preset chose deliberately.

The margin is deliberately asymmetric: a swap must clear it, keeping an
incumbent never does. A swap carries churn a retained seat does not - a new
knowledge cutoff, new failure modes, re-probing, a diff a customer sees - so
the burden of proof sits on the change. Defending an incumbent against the
arithmetic still requires a written reason at the call site (the strong-jury
OpenAI seat is the example), it just does not require an index margin.
"""


def in_family_upgrades(
    incumbent: ScoredModel,
    models: list[ScoredModel],
    family_of: Callable[[str], str],
    identity_of: Callable[[str], str] | None = None,
) -> list[ScoredModel]:
    """Same-lineage models that beat this seat on both axes without moving region.

    This, not the global frontier, is the seating rule for any preset that buys
    diversity. A jury seats three lineages on purpose: judged globally, 19 of 24
    seats look dominated, and "fixing" them would collapse the panel onto
    whichever vendor is cheapest that month, which is the correlated-error
    failure the jury exists to prevent. The lineage is a requirement, so the
    only honest question is whether a seat is the best buy *within* it.
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
        if not seatable(m):
            # A card nothing may be seated on cannot be an upgrade to a seat.
            continue
        if not m.dominates(incumbent):
            continue
        m_index, incumbent_index = m.default_index, incumbent.default_index
        # Same weights served by another host is pure price arbitrage: there is
        # no quality question to answer, so no index margin is required.
        same_model = identity_of is not None and identity_of(m.router_id) == identity_of(incumbent.router_id)
        if same_model or m_index - incumbent_index >= MIN_INDEX_GAIN:
            better.append(m)
    return sorted(better, key=lambda m: -m.default_index)


# The tier rule was authored at 3.0, which was a judgement about how far apart
# price bands must sit before routing down is worth anything, not a figure
# RES-1285 measured. On the open-weight frontier it cost more than it bought.
# glm-5.2 sits 2.79x under kimi-k3, just inside the rule, so at 3.0 the only
# trio that passes is kimi-k3 with both DeepSeek models: one vendor holding two
# of three seats, which is the correlated-failure shape that retired the Max
# Savings pool. 2.75 seats glm-5.2 in the middle band instead, at a 51.1
# index against deepseek-v4-flash's 37.5, and changes no other pool.
MIN_TIER_RATIO = 2.75


def derive_trio(
    frontier: list[ScoredModel],
    identity_of: Callable[[str], str],
) -> tuple[list[ScoredModel], bool]:
    """Pick a router pool from a constrained frontier, by the RES-1285 criteria.

    The four criteria, each traceable to what that benchmark measured:

    1. Every member on the frontier — when the band extremes are all
       Pareto-optimal, every selection the router can make is a good one.
    2. Exactly three members — the configuration measured clean, every model
       used and no dead weight. Three is also the only size trivially safe
       under positional banding: one model per band, so both routing profiles
       select the same model per band and no seat is unreachable.
    3. Adjacent price tiers at least `MIN_TIER_RATIO` apart, or the cost
       profile has nothing to win by routing down; the measured six-model pool
       failed partly because its tiers collapsed into each other.
    4. Maximise the weakest seat's intelligence subject to the above,
       tie-broken on total — the guard against the other measured failure
       mode, a mediocre model pushed into the easy band.

    Members must be distinct models, not the same weights under two hosts,
    which is what `identity_of` decides. Models that are not `comparable`
    (unranked at their default, or free) are never candidates. Returns (members, relaxed): when no
    trio satisfies the tier rule — single-vendor lineups may span only 5x end
    to end — the fallback takes the widest spread the constraint can actually
    reach and flags the pool, because a silent relaxation would let a caller
    read constrained pools as equivalent to open ones.
    """
    from itertools import combinations

    def distinct_trios():
        by_price = sorted((m for m in frontier if not m.free), key=lambda m: -m.price)
        for trio in combinations(by_price, 3):
            if len({identity_of(m.router_id) for m in trio}) == 3:
                yield trio

    best = None
    for trio in distinct_trios():
        prices = [m.price for m in trio]
        if prices[0] / prices[1] < MIN_TIER_RATIO or prices[1] / prices[2] < MIN_TIER_RATIO:
            continue
        rank = (min(m.default_index for m in trio), sum(m.default_index for m in trio))
        if best is None or rank > best[0]:
            best = (rank, trio)
    if best is not None:
        return list(best[1]), False

    widest = None
    for trio in distinct_trios():
        prices = [m.price for m in trio]
        rank = (min(prices[0] / prices[1], prices[1] / prices[2]), min(m.default_index for m in trio))
        if widest is None or rank > widest[0]:
            widest = (rank, trio)
    if widest is None:
        raise ValueError('fewer than three distinct models on the frontier')
    return list(widest[1]), True
