"""Model catalogue: prices, Artificial Analysis scores and the Pareto frontier.

Every model the Orq router serves carries a list price and, for most chat
models, Artificial Analysis (AA) scores: an intelligence index (per reasoning
effort where the model has one), domain indices such as `legal`, and median
speed and latency. This module reads those cards and answers "which models are
the best deal": a model is **dominated** when another is at least as good on
both axes and strictly better on one, and the **Pareto frontier** is every
model nothing dominates.

Example:
```python
from evaluatorq.models import load_models_sync, pareto

result = pareto(load_models_sync())
for point in result.frontier:
    print(point.model.router_id, point.score, point.x)
for point in result.dominated:
    print(point.model.router_id, 'switch to', point.dominated_by)
```

Price is a blend of the card's rates per 1M tokens, chosen with `mix`. The
default, `agentic`, is Artificial Analysis' cache-aware blend: 7 parts cache
read, 2 parts fresh input, 1 part output, which is what an agent loop that
re-sends its history looks like. `list` is the 3:1 input/output blend the Orq
smart router and preset seating rank on, and a `READ:WRITE:FRESH:OUTPUT` ratio
such as `7:1:1:1` sets the shares yourself. A card with no cache-read rate
prices reads at the Artificial Analysis cache price on the same card, else at
its input rate, and each point says which (`cache_read_source`).

`pareto` and `dominated_by` default to `agentic`. The re-exported seating
helpers `dominates`, `pareto_frontier` and `x_of` default to `list`, as preset
seating does; pass `mix=` to change either.

Every blend is a list price: a model that writes more tokens per task costs
more than its blend suggests, and the catalogue does not say how many tokens a
model spends per task.
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from evaluatorq.common.model_catalogue import (
    CallCost,
    ModelCatalogueError,
    ModelInfo,
    Pricing,
    cost_of,
    get_model_info,
    model_cards,
    models_by_provider,
    price_usage,
    register_model,
)
from evaluatorq.common.model_frontier import (
    CONTRIBUTOR_REASON,
    INDICES,
    MIN_SEAT_INDEX,
    PRICE_MIXES,
    X_AXES,
    AxisSpec,
    CacheSource,
    Index,
    PriceMix,
    ScoredModel,
    XAxis,
    check_axes,
    dominates,
    garden_from_cards,
    one_per_model,
    pareto_frontier,
    parse_mix,
    placement,
    point_dominates,
    score_of,
    x_axis_spec,
    x_of,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from openai import AsyncOpenAI


class ParetoPoint(BaseModel):
    """One model placed on the two axes of a Pareto analysis."""

    model: ScoredModel
    score: float = Field(description='The y value: the chosen index at the chosen effort.')
    x: float = Field(description='The x value in its own unit (USD per 1M tokens, tokens/s or seconds).')
    pareto_score: float | None = Field(
        default=None,
        description="Points above the minimum score (0 when there is none) per blended USD per 1M tokens (under the result's mix), higher is better. Set only when x is price.",
    )
    cache_read_source: CacheSource | None = Field(
        default=None,
        description="Where the cache-read rate in `x` came from: 'host' (this host's card), 'artificial_analysis' (the "
        "Artificial Analysis price on the same card, because the host lists none) or 'input' (no cache price on the "
        'card, so reads bill at the input rate). None when x is not price or the mix has no cache-read share.',
    )
    cache_read_rate: float | None = Field(
        default=None,
        description='The cache-read rate (USD per 1M tokens) behind `cache_read_source`. None when that is None.',
    )
    cache_write_source: CacheSource | None = Field(
        default=None,
        description="'host' or 'input' (no cache-write price on the card). None when x is not price or the mix has no "
        'cache-write share.',
    )
    on_frontier: bool
    dominated_by: str | None = Field(
        default=None,
        description='Router id of the nearest frontier model that dominates this one: the one with the best x '
        'among those at least as good on score. None on the frontier.',
    )
    dominators: list[str] = Field(
        default_factory=list, description='Router ids of every plotted model that dominates this one, best x first.'
    )

    @property
    def price_fallback(self) -> bool:
        """Whether `x` uses a cache rate that is not this host's own."""
        return self.cache_read_source not in {None, 'host'} or self.cache_write_source not in {None, 'host'}

    def summary(self) -> dict[str, object]:
        """A flat, JSON-ready row: every model field and its blended `price`, then the point's own fields."""
        return {**self.model.model_dump(), 'price': self.model.price, **self.model_dump(exclude={'model'})}


class ExcludedModel(BaseModel):
    """A model left off the chart, and why."""

    router_id: str
    reason: str
    below_min_score: bool = Field(
        default=False,
        description='True when the model has a value on both axes and was left out only for scoring below `min_score`.',
    )
    contributor: bool = Field(
        default=False,
        description='True for a contributor listing, left out because its discount is paid for with training rights.',
    )


class ParetoResult(BaseModel):
    """A Pareto analysis over a set of models on one index and one x axis."""

    index: str
    effort: str | None
    x: str
    mix: PriceMix | None = Field(default=None, description='The price blend on the x axis. None when x is not price.')
    min_score: float | None = Field(
        default=None, description='The score floor: models below it are in `excluded`. None when there is no floor.'
    )
    points: list[ParetoPoint] = Field(description='Every plotted model, frontier first, then by score descending.')
    excluded: list[ExcludedModel] = Field(default_factory=list)

    @property
    def frontier(self) -> list[ParetoPoint]:
        """Frontier points, strongest first."""
        return [p for p in self.points if p.on_frontier]

    @property
    def dominated(self) -> list[ParetoPoint]:
        """Dominated points, strongest first."""
        return [p for p in self.points if not p.on_frontier]

    @property
    def x_spec(self) -> AxisSpec:
        """The x axis's labels and direction, with the price blend named in the labels."""
        return x_axis_spec(self.x, self.mix)

    @property
    def price_fallbacks(self) -> list[ParetoPoint]:
        """Plotted points whose price uses a cache rate that is not their host's own."""
        return [p for p in self.points if p.price_fallback]

    @property
    def contributors(self) -> list[ExcludedModel]:
        """Excluded contributor listings."""
        return [e for e in self.excluded if e.contributor]

    @property
    def below_min_score(self) -> list[ExcludedModel]:
        """Excluded models that were left out only for scoring below `min_score`."""
        return [e for e in self.excluded if e.below_min_score]


def pareto(
    models: Iterable[ScoredModel],
    *,
    index: Index = 'intelligence',
    effort: str | None = None,
    x: XAxis = 'price',
    mix: str | PriceMix = 'agentic',
    exclude_estimated: bool = False,
    min_score: float | None = MIN_SEAT_INDEX,
) -> ParetoResult:
    """Place each model on (`x`, `index`), mark the frontier and name what beats each dominated model.

    On the price axis, `mix` sets the blend: `agentic` (the default, 7 cache
    read : 2 fresh input : 1 output), `list` (3 input : 1 output, what preset
    seating uses), a `READ:WRITE:FRESH:OUTPUT` ratio string, or a `PriceMix`.
    It is ignored on the other axes.

    Contributor listings (`ScoredModel.is_contributor`), models without a value
    on either axis, estimated scores when
    `exclude_estimated` is set, and models scoring below `min_score` on the chosen
    index are listed in `excluded` with the reason. The floor defaults to
    `MIN_SEAT_INDEX`, the same floor preset seating uses, because without it the
    cheap end of the frontier holds models nothing is cheaper than but that no
    one would ship. `min_score=None` keeps every model. The floor never changes
    which of the remaining models are on the frontier: anything that beats a
    model scores at least as high, so it clears the floor too.

    A router id that appears more than once is placed once, from its first
    occurrence, so a repeated model never sits on the frontier twice.

    Raises:
        ValueError: An unknown index, x axis or mix, an effort with a domain index,
            or a `min_score` that is not a finite number.
    """
    check_axes(index=index, effort=effort, x=x)
    if min_score is not None and not math.isfinite(min_score):
        raise ValueError(f'min_score must be a finite number or None, got {min_score!r}')
    price_mix = parse_mix(mix)
    spec = x_axis_spec(x)
    plotted: list[tuple[ScoredModel, tuple[float, float]]] = []
    excluded: list[ExcludedModel] = []
    seen: set[str] = set()
    for model in models:
        if model.router_id in seen:
            continue
        seen.add(model.router_id)
        if model.is_contributor:
            excluded.append(ExcludedModel(router_id=model.router_id, reason=CONTRIBUTOR_REASON, contributor=True))
            continue
        placed = placement(model, index=index, effort=effort, x=x, exclude_estimated=exclude_estimated, mix=price_mix)
        if isinstance(placed, str):
            excluded.append(ExcludedModel(router_id=model.router_id, reason=placed))
        elif min_score is not None and placed[0] < min_score:
            excluded.append(
                ExcludedModel(
                    router_id=model.router_id,
                    reason=f'score {placed[0]:g} is below min_score {min_score:g}',
                    below_min_score=True,
                )
            )
        else:
            plotted.append((model, placed))

    beaten_by = [
        sorted(
            (other for other in plotted if other[0] is not model and point_dominates(other[1], point, x=x)),
            key=lambda item: (spec.best_first(item[1][1]), -item[1][0]),
        )
        for model, point in plotted
    ]
    frontier_ids = {id(model) for (model, _), beaters in zip(plotted, beaten_by, strict=True) if not beaters}
    floor = min_score if min_score is not None else 0.0
    points: list[ParetoPoint] = []
    for (model, (score, xv)), beaters in zip(plotted, beaten_by, strict=True):
        # Every dominated model is dominated by a frontier one (dominance is transitive), so this is set whenever beaters is.
        nearest = next((b for b, _ in beaters if id(b) in frontier_ids), None)
        read_source, write_source = price_mix.sources(model) if x == 'price' else (None, None)
        points.append(
            ParetoPoint(
                model=model,
                score=score,
                x=xv,
                pareto_score=(score - floor) / xv if x == 'price' and xv > 0 else None,
                cache_read_source=read_source,
                cache_read_rate=model.cache_read_rate()[0] if read_source else None,
                cache_write_source=write_source,
                on_frontier=not beaters,
                dominated_by=nearest.router_id if nearest else None,
                dominators=[b.router_id for b, _ in beaters],
            )
        )
    points.sort(key=lambda p: (not p.on_frontier, -p.score, p.model.router_id))
    return ParetoResult(
        index=index,
        effort=effort,
        x=x,
        mix=price_mix if x == 'price' else None,
        min_score=min_score,
        points=points,
        excluded=excluded,
    )


def dominated_by(
    model: ScoredModel,
    models: Iterable[ScoredModel],
    *,
    index: Index = 'intelligence',
    effort: str | None = None,
    x: XAxis = 'price',
    mix: str | PriceMix = 'agentic',
    exclude_estimated: bool = False,
) -> ScoredModel | None:
    """The nearest frontier model that dominates `model`, or None when nothing does.

    "Nearest" is the frontier model with the best x among those at least as
    good on score: on price, the cheapest model that is at least as smart.
    No score floor applies: this answers what beats `model` whatever it scores.
    """
    pool = {m.router_id: m for m in models}
    pool[model.router_id] = model
    result = pareto(
        pool.values(), index=index, effort=effort, x=x, mix=mix, exclude_estimated=exclude_estimated, min_score=None
    )
    point = next((p for p in result.points if p.model.router_id == model.router_id), None)
    return pool[point.dominated_by] if point is not None and point.dominated_by else None


async def load_models(
    client: AsyncOpenAI | None = None,
    *,
    include_disabled: bool = False,
    all_hosts: bool = False,
) -> list[ScoredModel]:
    """Every catalogue model with an Artificial Analysis score and a USD list price.

    The cards come from the live Orq `/v2/models` endpoint, cached per process and
    credential. Free models are kept (with `free` set); models the workspace
    switched off are dropped unless `include_disabled` is set. By default one
    card per underlying model is kept (`one_per_model`); `all_hosts=True` keeps
    every host. Contributor listings (`ScoredModel.is_contributor`) are always
    kept beside the one listing per model, never in place of it; `pareto()` lists
    them in `excluded`.

    Raises:
        ValueError: No Orq key or OAuth session is available (``MissingLLMCredentialsError``).
        ModelCatalogueError: The catalogue fetch failed or was unreadable.
    """
    garden = garden_from_cards(await model_cards(client))
    models = [m for m in garden.values() if include_disabled or m.enabled is not False]
    if all_hosts:
        return sorted(models, key=lambda m: m.router_id)
    # A contributor listing is cheaper than its standard one, so grouping it in would make it the kept listing.
    contributors = sorted((m for m in models if m.is_contributor), key=lambda m: m.router_id)
    return one_per_model(m for m in models if not m.is_contributor) + contributors


def load_models_sync(*, include_disabled: bool = False, all_hosts: bool = False) -> list[ScoredModel]:
    """`load_models` for synchronous callers, using the ambient `ORQ_API_KEY`.

    Raises:
        RuntimeError: Called from inside a running event loop; `await load_models()` there instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(load_models(include_disabled=include_disabled, all_hosts=all_hosts))
    raise RuntimeError('load_models_sync() inside a running event loop: await load_models() instead')


def find_model(models: Sequence[ScoredModel], *, model_id: str) -> ScoredModel | None:
    """The model matching `model_id`, or None when nothing or more than one model matches.

    Tried in order, the first step with any match deciding: the exact router id;
    the id after the provider (`qwen/qwen3-coder` for
    `tensorix/qwen/qwen3-coder`); for an id without a `/`, the last path
    segment (`qwen3-coder`). A step that matches more than one model is
    ambiguous and returns None rather than falling through.
    """
    exact = next((m for m in models if m.router_id == model_id), None)
    if exact is not None:
        return exact
    after_provider = [m for m in models if m.router_id.split('/', 1)[-1] == model_id]
    if after_provider or '/' in model_id:
        return after_provider[0] if len(after_provider) == 1 else None
    last_segment = [m for m in models if m.router_id.rsplit('/', 1)[-1] == model_id]
    return last_segment[0] if len(last_segment) == 1 else None


__all__ = [
    'INDICES',
    'MIN_SEAT_INDEX',
    'PRICE_MIXES',
    'X_AXES',
    'CallCost',
    'ExcludedModel',
    'Index',
    'ModelCatalogueError',
    'ModelInfo',
    'ParetoPoint',
    'ParetoResult',
    'PriceMix',
    'Pricing',
    'ScoredModel',
    'XAxis',
    'cost_of',
    'dominated_by',
    'dominates',
    'find_model',
    'get_model_info',
    'load_models',
    'load_models_sync',
    'models_by_provider',
    'one_per_model',
    'pareto',
    'pareto_frontier',
    'parse_mix',
    'price_usage',
    'register_model',
    'score_of',
    'x_of',
]
