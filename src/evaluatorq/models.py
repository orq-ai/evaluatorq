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

Price is the 3:1 input/output blend per 1M tokens, the same blend the Orq
smart router uses. It is a list price: a model that writes more tokens per task
costs more than its list price suggests, and the catalogue does not say how
many tokens a model spends per task.
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
    INDICES,
    MIN_SEAT_INDEX,
    X_AXES,
    X_AXIS_SPECS,
    Index,
    ScoredModel,
    XAxis,
    check_axes,
    dominates,
    garden_from_cards,
    one_per_model,
    pareto_frontier,
    placement,
    point_dominates,
    score_of,
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
        description='Score per blended USD per 1M tokens, higher is better. Set only when x is price.',
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


class ParetoResult(BaseModel):
    """A Pareto analysis over a set of models on one index and one x axis."""

    index: str
    effort: str | None
    x: str
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
    def below_min_score(self) -> list[ExcludedModel]:
        """Excluded models that were left out only for scoring below `min_score`."""
        return [e for e in self.excluded if e.below_min_score]


def pareto(
    models: Iterable[ScoredModel],
    *,
    index: Index = 'intelligence',
    effort: str | None = None,
    x: XAxis = 'price',
    exclude_estimated: bool = False,
    min_score: float | None = MIN_SEAT_INDEX,
) -> ParetoResult:
    """Place each model on (`x`, `index`), mark the frontier and name what beats each dominated model.

    Models without a value on either axis, estimated scores when
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
        ValueError: An unknown index or x axis, an effort with a domain index,
            or a `min_score` that is not a finite number.
    """
    check_axes(index=index, effort=effort, x=x)
    if min_score is not None and not math.isfinite(min_score):
        raise ValueError(f'min_score must be a finite number or None, got {min_score!r}')
    spec = X_AXIS_SPECS[x]
    plotted: list[tuple[ScoredModel, tuple[float, float]]] = []
    excluded: list[ExcludedModel] = []
    seen: set[str] = set()
    for model in models:
        if model.router_id in seen:
            continue
        seen.add(model.router_id)
        placed = placement(model, index=index, effort=effort, x=x, exclude_estimated=exclude_estimated)
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
    points: list[ParetoPoint] = []
    for (model, (score, xv)), beaters in zip(plotted, beaten_by, strict=True):
        # Every dominated model is dominated by a frontier one (dominance is transitive), so this is set whenever beaters is.
        nearest = next((b for b, _ in beaters if id(b) in frontier_ids), None)
        points.append(
            ParetoPoint(
                model=model,
                score=score,
                x=xv,
                pareto_score=score / xv if x == 'price' and xv > 0 else None,
                on_frontier=not beaters,
                dominated_by=nearest.router_id if nearest else None,
                dominators=[b.router_id for b, _ in beaters],
            )
        )
    points.sort(key=lambda p: (not p.on_frontier, -p.score, p.model.router_id))
    return ParetoResult(index=index, effort=effort, x=x, min_score=min_score, points=points, excluded=excluded)


def dominated_by(
    model: ScoredModel,
    models: Iterable[ScoredModel],
    *,
    index: Index = 'intelligence',
    effort: str | None = None,
    x: XAxis = 'price',
    exclude_estimated: bool = False,
) -> ScoredModel | None:
    """The nearest frontier model that dominates `model`, or None when nothing does.

    "Nearest" is the frontier model with the best x among those at least as
    good on score: on price, the cheapest model that is at least as smart.
    No score floor applies: this answers what beats `model` whatever it scores.
    """
    pool = {m.router_id: m for m in models}
    pool[model.router_id] = model
    result = pareto(pool.values(), index=index, effort=effort, x=x, exclude_estimated=exclude_estimated, min_score=None)
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
    every host.

    Raises:
        ValueError: No Orq key or OAuth session is available (``MissingLLMCredentialsError``).
        ModelCatalogueError: The catalogue fetch failed or was unreadable.
    """
    garden = garden_from_cards(await model_cards(client))
    models = [m for m in garden.values() if include_disabled or m.enabled is not False]
    return sorted(models, key=lambda m: m.router_id) if all_hosts else one_per_model(models)


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
    """The model matching `model_id`: an exact `provider/model` id, else a bare id held by exactly one model."""
    exact = next((m for m in models if m.router_id == model_id), None)
    if exact is not None or '/' in model_id:
        return exact
    bare = [m for m in models if m.router_id.split('/', 1)[-1] == model_id]
    return bare[0] if len(bare) == 1 else None


__all__ = [
    'INDICES',
    'MIN_SEAT_INDEX',
    'X_AXES',
    'CallCost',
    'ExcludedModel',
    'Index',
    'ModelCatalogueError',
    'ModelInfo',
    'ParetoPoint',
    'ParetoResult',
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
    'price_usage',
    'register_model',
    'score_of',
    'x_of',
]
