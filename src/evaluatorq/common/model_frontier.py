"""Intelligence/price frontier over the model garden (RES-1347, RES-1346).

Presets rot because models move, and until now the only computable axis was
price, so "is this successor better?" was a judgement call. Every autorouter
eligible model card carries Artificial Analysis intelligence scores in
`metadata.autorouter.v2`, per reasoning effort where the model has one, beside
its own input and output rates. That is the pair RES-1285 used to show 19 of 24
models on a live workspace are dominated and structurally unselectable, and it
is enough to decide seating arithmetically rather than by taste.

Both axes are read where a judge actually runs. Judges are called with no
reasoning effort, so the provider applies the catalogue default
(`/v2/models` `parameters[reasoningEffort].config.default`), and each card is
ranked on its score at that rung. Price is a 3:1 input/output blend.

Dominance is the whole rule: B dominates A when B is at least as intelligent and
no more expensive, and strictly better on one of them. A dominated seat is dead
weight the user cannot see. A successor that dominates the incumbent is a
straight upgrade; one that does not is a trade the preset's `use_when` has to
justify in words.

This is shared by all four preset sets in RES-1347, since router pools, juries,
classifiers and evaluator groups all pick models out of the same garden.

Nothing here is committed: the garden is derived from the live `/v2/models`
cards on every process (`load_garden`), read through `common.model_catalogue`
so the pricing table and the garden can never describe different catalogues.
The registers below (UNLISTED_ROUTES, AA_FILLS, SUPERSEDED, UNROUTABLE,
NEVER_SEAT) are reviewed judgements about that live data, each with its source.
"""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from evaluatorq.common.llm_client import orq_base_url
from evaluatorq.common.model_catalogue import (
    _entry_metadata,
    _parse_default_reasoning_effort,
    _parse_reasoning_efforts,
    _reasoning_effort_config,
    _usd_price,
    model_cards,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from openai import AsyncOpenAI


class ScoredModel(BaseModel):
    """A garden model with the two axes seating decisions are made on."""

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
    # The rung the headline was measured at, resolved through the benchmark
    # slug where possible. Only used to catch a card contradicting itself.
    headline_effort: str | None = None
    effort_indices: dict[str, float] = {}
    effort_parameter: bool = Field(
        default=False, description='Whether the card exposes a reasoningEffort parameter at all.'
    )
    default_effort: str | None = Field(
        default=None,
        description='The catalogue default effort. None on a card with an effort parameter '
        'means the default is unknown (missing, or not among the parameter options).',
    )
    # False means the workspace has the model switched off: listed, scored,
    # not servable.
    enabled: bool | None = None

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
    def comparable(self) -> bool:
        """Whether the card has a position on the price axis: free cards are never ranked."""
        return not self.free

    @property
    def ranked_index(self) -> float:
        """Alias of `default_index`, kept for the sort keys and margins that read it."""
        return self.default_index

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
# Haiku 4.5 at 23.7 is the platform's single-judge baseline) and above the band
# it exists to exclude, which on the 2026-08-25 garden is Llama-3.3-70B at 9.4,
# the 4o generation at 6.9 to 11.2, qwen3-coder at 13.6 and gpt-oss-20b at 14.9.
MIN_SEAT_INDEX = 15.0

# The index is a benchmark aggregate, and a floor cannot catch a model whose
# benchmarks say more than its behaviour: gpt-oss-120b scores 23.8, a hair
# above Haiku 4.5's 23.7, and does not perform beside it. Families here are
# never seated at any score or price, with the verdict on record. Review
# verdict from PR #330 (2026-08-24).
NEVER_SEAT: dict[str, str] = {
    'gpt-oss': (
        'Benchmark-inflated relative to observed quality: scores beside Claude '
        'Haiku 4.5 on the index and does not deliver beside it. Not to be '
        'recommended at any price tier.'
    ),
}


# Listing is not callability. `GET /v2/models` advertises these router IDs and
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
        'superseded by zai/glm-5.3-flash (41.8 at max, $0.10 blended on the card, '
        '$0.20 at actual Zai billing); live catalogue 2026-10-26. Held on Open-Weight / '
        'Portable only (HELD_SUPERSEDED in evaluatorq.jury_presets).'
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
    # Recorded by hand when gpt-5.4-mini had no score at its default (none) and could not be derived.
    'openai/gpt-5-mini': 'superseded by openai/gpt-5.4-mini (listed, unscored at its default none); 2026-10-27.',
    'azure/gpt-5-mini': 'superseded by openai/gpt-5.4-mini (listed, unscored at its default none); 2026-10-27.',
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
    `anthropic/claude-haiku-4-5-20251001` are one model, and EU Region seats the
    first while Balanced Trio used to seat the second.
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
    seatable (not disabled, NEVER_SEAT, UNROUTABLE, SUPERSEDED or under the floor),
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
        if same_lineage and theirs[1] > mine[1] and candidate.comparable and seatable(candidate):
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


def _effort_ladder(v2: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """The card's `reasoning_efforts` rungs, keeping only well-formed ones.

    External data: a ladder that is not a mapping, or a rung that is not one,
    reads as absent rather than raising and taking the whole garden down.
    """
    ladder = v2.get('reasoning_efforts')
    if not isinstance(ladder, Mapping):
        return {}
    return {effort: entry for effort, entry in ladder.items() if isinstance(entry, Mapping)}


def _scored_effort(v2: Mapping[str, Any]) -> str | None:
    """The effort the card's headline index was measured at, via its benchmark slug."""
    slug, index = v2.get('aa_slug'), v2.get('intelligence_index')
    return next(
        (
            e
            for e, entry in _effort_ladder(v2).items()
            if entry.get('aa_slug') == slug and entry.get('intelligence_index') == index
        ),
        None,
    )


def _catalogue_effort(card: dict[str, object]) -> tuple[bool, str | None]:
    """Whether the card exposes a reasoningEffort parameter, and its default (None if not among the options).

    Read through the catalogue's own parsers, so `ModelInfo.default_reasoning_effort`
    and the rung a seat is ranked at come from one reading of the card.
    """
    if _reasoning_effort_config(card) is None:
        return False, None
    options = _parse_reasoning_efforts(card) or frozenset()
    default = _parse_default_reasoning_effort(card)
    return True, default if default in options else None


def scored_model_from_card(card: dict[str, object]) -> ScoredModel | None:
    """A ScoredModel from one `/v2/models` card, or None if it has no index or rates.

    Card rates are USD per 1k tokens; ScoredModel prices are per 1M.
    """
    metadata = _entry_metadata(card)
    autorouter = metadata.get('autorouter')
    v2 = (autorouter.get('v2') if isinstance(autorouter, dict) else None) or {}
    index = v2.get('intelligence_index') if isinstance(v2, dict) else None
    input_cost, output_cost = _usd_price(card, 'input'), _usd_price(card, 'output')
    if not isinstance(index, (int, float)) or isinstance(index, bool) or input_cost is None or output_cost is None:
        return None
    effort_parameter, default_effort = _catalogue_effort(card)
    region = metadata.get('region')
    enabled = card.get('enabled')
    return ScoredModel(
        router_id=f'{card["provider"]}/{card["model_id"]}',
        intelligence_index=float(index),
        input_price=round(input_cost * 1000, 6),
        output_price=round(output_cost * 1000, 6),
        region=region if isinstance(region, str) else None,
        provider=str(card.get('provider')),
        headline_effort=_scored_effort(v2),
        effort_indices={
            e: entry['intelligence_index']
            for e, entry in _effort_ladder(v2).items()
            if entry.get('intelligence_index') is not None
        },
        effort_parameter=effort_parameter,
        default_effort=default_effort,
        enabled=enabled if isinstance(enabled, bool) else None,
    )


def catalog_scope(cards: Iterable[Mapping[str, object]]) -> str:
    """'workspace' when the response carries disabled cards, else 'project'.

    A project-scoped key filters out everything the workspace has not enabled,
    so zero disabled cards is its signature, and the frontier then derives from
    what one workspace switched on rather than from the platform catalogue.
    """
    return 'workspace' if any(card.get('enabled') is False for card in cards) else 'project'


def garden_from_cards(cards: Iterable[dict[str, object]]) -> dict[str, ScoredModel]:
    """Every scored card, keyed by router ID. Unscored or unpriced cards are left out."""
    models = (scored_model_from_card(card) for card in cards if 'provider' in card and 'model_id' in card)
    return {m.router_id: m for m in models if m is not None}


_scope_warned: set[int] = set()


async def load_garden(client: AsyncOpenAI | None = None) -> dict[str, ScoredModel]:
    """The live model garden: every scored `/v2/models` card, keyed by router ID.

    Scores and prices come from one response, so they share a benchmark version.
    The cards are fetched once per process and credential by
    `common.model_catalogue.model_cards`; this only ranks them. A
    project-scoped catalogue is used with a warning rather than refused, since
    it is the only key most callers hold; the live drift suite fails on it.

    Raises:
        MissingLLMCredentialsError: No Orq key or OAuth session is available.
        ModelCatalogueError: The catalogue fetch failed or was unreadable.
    """
    cards = await model_cards(client)
    if catalog_scope(cards) == 'project' and id(cards) not in _scope_warned:
        _scope_warned.add(id(cards))
        logger.warning(
            '/v2/models returned no disabled cards: this looks like a project-scoped key, so the '
            'garden holds only what this workspace enabled. Use a workspace-scoped ORQ_API_KEY for '
            'the full catalogue.'
        )
    return garden_from_cards(cards)


def load_garden_sync() -> dict[str, ScoredModel]:
    """`load_garden` for synchronous callers, using the ambient ``ORQ_API_KEY``.

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
        return int(datetime.fromisoformat(created.replace('Z', '+00:00')).timestamp() * 1000)
    if isinstance(created, (int, float)) and not isinstance(created, bool):
        return int(created if created > 1e11 else created * 1000)
    return None


async def public_listings(*, timeout_s: float = 60.0) -> dict[str, int | None]:
    """Router ID to first-listed epoch milliseconds, for every non-deprecated public card.

    `GET /v2/model-catalog` needs no key, so the roster and listing dates can be
    read by anyone and in CI. It carries no reasoning-effort ladder, so scores
    stay on the authenticated `/v2/models` path; this is the roster the
    successor and age checks date models by.

    Raises:
        RuntimeError: The catalogue reports more pages but returns none.
    """
    url = f'{orq_base_url()}/v2/model-catalog'
    listed: dict[str, int | None] = {}
    cursor: str | None = None
    async with httpx.AsyncClient(timeout=timeout_s) as http:
        while True:
            response = await http.get(url, params={'starting_after': cursor} if cursor else None)
            response.raise_for_status()
            page = response.json()
            data = [card for card in page.get('data') or [] if isinstance(card, dict)]
            listed.update({card['id']: _created_ms(card.get('created')) for card in data if not card.get('deprecated')})
            if not page.get('has_more'):
                return listed
            if not data:
                raise RuntimeError(
                    'model catalogue reports more pages but returned none; the roster would be truncated'
                )
            cursor = str(data[-1]['id'])


# The cheap-tier ceiling: 3:1 blended USD per 1M tokens, strictly under. The
# user's rule is "input price under $1.00"; this is the same rule on the axis
# the library ranks on. A model priced just under $1.00 in at the usual 1:5
# input:output ratio blends to (3*1 + 5)/4 = $2.00. On the 2026-10-26 live
# catalogue the two rules pick the same candidates, and $1.00 would leave only
# three families, so no reserve (derivation in the 2-pager).
MAX_BLEND = 2.0

# Routes the router serves but the project-scoped `/v2/models` listing omits,
# so they have no card to read. Each value carries its own source and date;
# the live catalogue leads, so an entry is used only while the router id is
# absent from the live garden. Never add a value that has no source.
UNLISTED_ROUTES: dict[str, dict[str, Any]] = {
    'anthropic/claude-haiku-5-5': {
        # Callable 2026-10-26 (chat completion 200); `aws/eu.anthropic.claude-haiku-5-5`
        # is the listed EU sibling and its card carries the same ladder.
        'input_price': 0.10,  # AA API data/llms/models, claude-haiku-5-5, 2026-10-26
        'output_price': 0.50,  # same
        'region': 'us',  # direct Anthropic route
        'effort_parameter': True,
        # Default effort read from the listed EU sibling's reasoningEffort parameter.
        'default_effort': 'medium',
        # AA API slugs claude-haiku-5-5-{low,medium,high,xhigh} and claude-haiku-5-5 (max),
        # 2026-10-26; identical to the EU sibling's live ladder, so v4.3.x coherent.
        'effort_indices': {'low': 29.5, 'medium': 34.5, 'high': 37.8, 'xhigh': 41.2, 'max': 43.4},
        'headline_effort': 'max',
        'intelligence_index': 43.4,
    },
    'openai/gpt-6.1-sol': {
        # Callable 2026-11-10 (chat completion 200); listed in the public
        # catalogue but absent from the project-scoped /v2/models.
        'input_price': 2.00,  # public /v2/model-catalog card, 2026-11-10
        'output_price': 10.00,  # same
        'region': 'us',
        'effort_parameter': True,
        # Ladder and default read from the listed EU sibling `openai/eu.gpt-6.1-sol`
        # (aa_slug gpt-6-1-sol-medium, aa_index_version v4.3), 2026-11-10.
        'default_effort': 'medium',
        'effort_indices': {'low': 42.1, 'medium': 47.8, 'high': 50.2, 'xhigh': 51.0, 'max': 51.8},
        'headline_effort': 'medium',
        'intelligence_index': 47.8,
    },
    'openai/gpt-6.1-terra': {
        # Callable 2026-11-10 (chat completion 200); listed in the public
        # catalogue but absent from the project-scoped /v2/models, and neither
        # it nor its EU sibling carries a v4.3 score.
        'input_price': 1.00,  # public /v2/model-catalog card, 2026-11-10
        'output_price': 5.00,  # same
        'region': 'us',
        'effort_parameter': True,
        # Inferred, not read: no Terra 6.1 card is visible to this key. Every
        # GPT-6 generation card that is (Sol, Luna, EU 6.1 Sol) defaults to medium.
        'default_effort': 'medium',
        'effort_indices': {},
        # AA v5.0.1 max (artificialanalysis.ai/models/gpt-6-1-terra, 2026-11-10).
        # No medium rung, so it ranks on this headline, which is a different
        # index version from the v4.3 garden.
        'headline_effort': 'max',
        'intelligence_index': 51.0,
    },
}


# Scores the live card lacks at its catalogue default effort, taken from the
# Artificial Analysis API. Accepted only where the AA-measured rung is the
# catalogue default; the live ladder wins whenever it has the rung. Fill
# nothing from another rung: DeepSeek flash and pro default to high or medium
# and AA measures only max and none, so they rank on their headline index.
AA_FILLS: dict[str, dict[str, Any]] = {
    'zai/glm-5.3-flash': {
        'effort': 'max',
        'index': 41.8,
        'source': (
            'AA API data/llms/models slug glm-5-3-flash, 2026-10-26, index v4.3 (the card names the same slug). '
            'The AA entry carries no rung label; the max rung was confirmed on the AA page by the user.'
        ),
    },
    'tencent/glm-5.3-flash': {
        'effort': 'max',
        'index': 41.8,
        'source': 'same AA entry as zai/glm-5.3-flash (host differs, weights are the same), 2026-10-26.',
    },
}


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


def with_unlisted_routes(garden: dict[str, ScoredModel]) -> dict[str, ScoredModel]:
    """The garden plus every UNLISTED_ROUTES entry the garden does not already carry."""
    extra = {
        router_id: ScoredModel(router_id=router_id, provider=router_id.split('/')[0], **fields)
        for router_id, fields in UNLISTED_ROUTES.items()
        if router_id not in garden
    }
    return {**garden, **extra}


def rank_cheapest_top(
    garden: dict[str, ScoredModel],
    family_of: Callable[[str], str],
    max_blend: float = MAX_BLEND,
) -> list[ScoredModel]:
    """The best cheap model of each family, strongest first: the Cheap Aggregate derivation.

    Eligible: comparable (not free), seatable (not
    NEVER_SEAT, UNROUTABLE, SUPERSEDED or under the floor), enabled, not
    EU-pinned, without a newer version of itself in the garden
    (`derived_successor`), and a 3:1 blend strictly under `max_blend`. Prices are the garden's own, so pass
    the live garden. One model per family (`family_of`), the highest default
    index in it; ties break to the lower price, then router_id. Seats are the
    first n entries and the reserve is the next, which is by construction a
    different family from every seat.
    """
    best: dict[str, ScoredModel] = {}
    for model in sorted(garden.values(), key=lambda m: (-(m.default_index or 0), m.price, m.router_id)):
        if not (model.comparable and seatable(model) and model.enabled is not False):
            continue
        if model.region == 'europe' or model.price >= max_blend:
            continue
        if derived_successor(model.router_id, garden):
            continue
        best.setdefault(family_of(model.router_id), model)
    return sorted(best.values(), key=lambda m: (-m.ranked_index, m.price, m.router_id))


def unseatable_reason(model: ScoredModel) -> str | None:
    """Why a model may not hold any recommended seat, pool or jury; None if it may."""
    if model.enabled is False:
        return 'disabled in the model catalogue'
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
    comparable = [m for m in models if m.comparable]
    frontier = [m for m in comparable if not any(other.dominates(m) for other in comparable)]
    return sorted(frontier, key=lambda m: -m.ranked_index)


def dominated_by(model: ScoredModel, models: list[ScoredModel]) -> list[ScoredModel]:
    """Everything in the garden that beats this model outright."""
    return sorted(
        (m for m in models if m.dominates(model)),
        key=lambda m: -m.ranked_index,
    )


def upgrade_verdict(incumbent: ScoredModel, successor: ScoredModel) -> tuple[str, str]:
    """Whether a successor should take a seat, and why, in one computed answer.

    Deliberately three-valued. 'seat' and 'reject' are arithmetic. 'trade' is
    the honest middle: the successor is better on one axis and worse on the
    other, which no index can settle, so it goes back to a human with the
    numbers attached rather than being auto-applied or silently dropped.
    Both indices are read at each card's catalogue default effort.
    """
    if (reason := unseatable_reason(successor)) is not None:
        return 'trade', f'{successor.router_id} cannot hold a seat: {reason}. This goes to a human.'
    if not region_compatible(successor, incumbent):
        return 'trade', (
            f'{successor.router_id} ({successor.region}) would move {incumbent.router_id} '
            f'({incumbent.region}) to another region. This goes to a human.'
        )
    # A dead incumbent is replaced by any seatable successor; dominance is moot.
    if (reason := unseatable_reason(incumbent)) is not None:
        return 'seat', f'{incumbent.router_id} cannot hold a seat ({reason}); {successor.router_id} can.'
    if successor.dominates(incumbent):
        return 'seat', (
            f'{successor.router_id} dominates {incumbent.router_id}: '
            f'index {successor.default_index} at {successor.default_effort} vs '
            f'{incumbent.default_index} at {incumbent.default_effort}, '
            f'price {successor.price} vs {incumbent.price}.'
        )
    if incumbent.dominates(successor):
        return 'reject', (
            f'{incumbent.router_id} still dominates {successor.router_id}: '
            f'index {incumbent.default_index} at {incumbent.default_effort} vs '
            f'{successor.default_index} at {successor.default_effort}, '
            f'price {incumbent.price} vs {successor.price}.'
        )
    return 'trade', (
        f'{successor.router_id} trades against {incumbent.router_id}: '
        f'index {successor.default_index} vs {incumbent.default_index}, '
        f'price {successor.price} vs {incumbent.price}. '
        'Neither dominates, so seating it is a stated preference, not an upgrade.'
    )


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
arithmetic still requires a written reason at the call site (the Strong Jury
OpenAI seat is the example), it just does not require an index margin.
"""


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
        m_index, incumbent_index = m.ranked_index, incumbent.ranked_index
        # Same weights served by another host is pure price arbitrage: there is
        # no quality question to answer, so no index margin is required.
        same_model = identity_of is not None and identity_of(m.router_id) == identity_of(incumbent.router_id)
        if same_model or m_index - incumbent_index >= min_index_gain:
            better.append(m)
    return sorted(better, key=lambda m: -m.ranked_index)


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
    min_tier_ratio: float = MIN_TIER_RATIO,
) -> tuple[list[ScoredModel], bool]:
    """Pick a router pool from a constrained frontier, by the RES-1285 criteria.

    The four criteria, each traceable to what that benchmark measured:

    1. Every member on the frontier — when the band extremes are all
       Pareto-optimal, every selection the router can make is a good one.
    2. Exactly three members — the configuration measured clean, every model
       used and no dead weight. Three is also the only size trivially safe
       under positional banding: one model per band, so both routing profiles
       select the same model per band and no seat is unreachable.
    3. Adjacent price tiers at least `min_tier_ratio` apart, or the cost
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
        by_price = sorted((m for m in frontier if m.comparable), key=lambda m: -m.price)
        for trio in combinations(by_price, 3):
            if len({identity_of(m.router_id) for m in trio}) == 3:
                yield trio

    best = None
    for trio in distinct_trios():
        prices = [m.price for m in trio]
        if prices[0] / prices[1] < min_tier_ratio or prices[1] / prices[2] < min_tier_ratio:
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
