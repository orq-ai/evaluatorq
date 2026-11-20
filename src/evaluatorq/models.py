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
from typing import TYPE_CHECKING, Literal, get_args

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
from evaluatorq.common.model_frontier import ScoredModel, garden_from_cards, model_identity

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from openai import AsyncOpenAI

Index = Literal[
    'intelligence',
    'economics',
    'engineering',
    'finance_and_accounting',
    'healthcare_and_medical',
    'legal',
    'strategy_and_ops',
    'coding',
    'agentic',
]
"""The score on the y axis: the AA intelligence index, or one AA domain index.

`coding` and `agentic` are accepted but empty in the live catalogue at the time
of writing, so every model is excluded under them.
"""

XAxis = Literal['price', 'output_speed', 'time_to_first_token']
"""The second axis: blended list price (lower is better), median output tokens
per second (higher is better), or median time to first token (lower is better)."""

INDICES: tuple[str, ...] = get_args(Index)
X_AXES: tuple[str, ...] = get_args(XAxis)
_LOWER_IS_BETTER: frozenset[str] = frozenset({'price', 'time_to_first_token'})


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
        """A flat, JSON-ready row: the point plus the model fields a table shows."""
        m = self.model
        return {
            'router_id': m.router_id,
            'display_name': m.display_name,
            'provider': m.provider,
            'developer': m.developer,
            'score': self.score,
            'x': self.x,
            'input_price': m.input_price,
            'output_price': m.output_price,
            'price': m.price,
            'pareto_score': self.pareto_score,
            'on_frontier': self.on_frontier,
            'dominated_by': self.dominated_by,
            'dominators': self.dominators,
            'is_estimated': m.is_estimated,
            'default_effort': m.default_effort,
            'effort_indices': m.effort_indices,
            'output_tokens_per_second': m.output_tokens_per_second,
            'time_to_first_token_seconds': m.time_to_first_token_seconds,
            'enabled': m.enabled,
        }


class ExcludedModel(BaseModel):
    """A model left off the chart, and why."""

    router_id: str
    reason: str


class ParetoResult(BaseModel):
    """A Pareto analysis over a set of models on one index and one x axis."""

    index: str
    effort: str | None
    x: str
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


def _check_axes(index: str, effort: str | None, x: str) -> None:
    if index not in INDICES:
        raise ValueError(f'unknown index {index!r}; choose one of {", ".join(INDICES)}')
    if x not in X_AXES:
        raise ValueError(f'unknown x axis {x!r}; choose one of {", ".join(X_AXES)}')
    if effort is not None and index != 'intelligence':
        raise ValueError('domain indices exist only at the default reasoning effort: pass effort=None')


def score_of(model: ScoredModel, *, index: Index = 'intelligence', effort: str | None = None) -> float | None:
    """A model's score on `index`, or None when the catalogue has none.

    With `effort=None` the intelligence index is read at the catalogue's default
    reasoning effort (the effort a call without one runs at), falling back to the
    headline index. A named effort reads that rung only.
    """
    _check_axes(index, effort, 'price')
    if index != 'intelligence':
        return model.category_indices.get(index)
    if effort is None:
        return model.default_index
    if effort in model.effort_indices:
        return model.effort_indices[effort]
    # A card without an effort ladder was measured at one effort only; its
    # headline counts for the rung it was measured at.
    if not model.effort_indices and model.headline_effort == effort:
        return model.intelligence_index
    return None


def x_of(model: ScoredModel, x: XAxis = 'price') -> float | None:
    """A model's value on the x axis, or None when it has none (a free model has no place on price)."""
    _check_axes('intelligence', None, x)
    if x == 'price':
        return None if model.free else model.price
    value = model.output_tokens_per_second if x == 'output_speed' else model.time_to_first_token_seconds
    return value if value is not None and value > 0 else None


def _x_better_or_equal(a: float, b: float, x: str) -> bool:
    return a <= b if x in _LOWER_IS_BETTER else a >= b


def _x_strictly_better(a: float, b: float, x: str) -> bool:
    return a < b if x in _LOWER_IS_BETTER else a > b


def _dominates(a: tuple[float, float], b: tuple[float, float], x: str) -> bool:
    (sa, xa), (sb, xb) = a, b
    return sa >= sb and _x_better_or_equal(xa, xb, x) and (sa > sb or _x_strictly_better(xa, xb, x))


def dominates(
    a: ScoredModel,
    b: ScoredModel,
    *,
    index: Index = 'intelligence',
    effort: str | None = None,
    x: XAxis = 'price',
) -> bool:
    """Whether `a` is at least as good as `b` on both axes and strictly better on one.

    False when either model has no value on an axis.
    """
    sa, sb = score_of(a, index=index, effort=effort), score_of(b, index=index, effort=effort)
    xa, xb = x_of(a, x), x_of(b, x)
    if sa is None or sb is None or xa is None or xb is None:
        return False
    return _dominates((sa, xa), (sb, xb), x)


def _exclusion(model: ScoredModel, *, index: Index, effort: str | None, x: XAxis, measured_only: bool) -> str | None:
    if measured_only and model.is_estimated:
        return 'estimated score'
    if score_of(model, index=index, effort=effort) is None:
        return f'no {index} score at effort {effort}' if effort else f'no {index} score'
    if x_of(model, x) is None:
        return 'free (no list price)' if x == 'price' else f'no {x.replace("_", " ")} data'
    return None


def pareto(
    models: Iterable[ScoredModel],
    *,
    index: Index = 'intelligence',
    effort: str | None = None,
    x: XAxis = 'price',
    measured_only: bool = False,
) -> ParetoResult:
    """Place each model on (`x`, `index`), mark the frontier and name what beats each dominated model.

    Models without a value on either axis, and estimated scores when
    `measured_only` is set, are listed in `excluded` with the reason.
    """
    _check_axes(index, effort, x)
    plotted: list[tuple[ScoredModel, float, float]] = []
    excluded: list[ExcludedModel] = []
    for model in models:
        reason = _exclusion(model, index=index, effort=effort, x=x, measured_only=measured_only)
        if reason is not None:
            excluded.append(ExcludedModel(router_id=model.router_id, reason=reason))
            continue
        score, xv = score_of(model, index=index, effort=effort), x_of(model, x)
        if score is not None and xv is not None:  # always true: _exclusion checked both
            plotted.append((model, score, xv))

    def best_x_first(item: tuple[ScoredModel, float, float]) -> tuple[float, float]:
        _, score, xv = item
        return (xv if x in _LOWER_IS_BETTER else -xv, -score)

    beaten_by = [
        sorted(
            (other for other in plotted if other[0] is not model and _dominates((other[1], other[2]), (score, xv), x)),
            key=best_x_first,
        )
        for model, score, xv in plotted
    ]
    frontier_ids = {id(item[0]) for item, beaters in zip(plotted, beaten_by, strict=True) if not beaters}
    points: list[ParetoPoint] = []
    for (model, score, xv), beaters in zip(plotted, beaten_by, strict=True):
        # Every dominated model is dominated by a frontier one (dominance is transitive), so this is set whenever beaters is.
        nearest = next((b[0] for b in beaters if id(b[0]) in frontier_ids), None)
        points.append(
            ParetoPoint(
                model=model,
                score=score,
                x=xv,
                pareto_score=score / xv if x == 'price' and xv > 0 else None,
                on_frontier=not beaters,
                dominated_by=nearest.router_id if nearest else None,
                dominators=[b[0].router_id for b in beaters],
            )
        )
    points.sort(key=lambda p: (not p.on_frontier, -p.score, p.model.router_id))
    return ParetoResult(index=index, effort=effort, x=x, points=points, excluded=excluded)


def pareto_frontier(
    models: Iterable[ScoredModel],
    *,
    index: Index = 'intelligence',
    effort: str | None = None,
    x: XAxis = 'price',
    measured_only: bool = False,
) -> list[ScoredModel]:
    """The models nothing else beats on both axes, strongest first."""
    result = pareto(models, index=index, effort=effort, x=x, measured_only=measured_only)
    return [p.model for p in result.frontier]


def dominated_by(
    model: ScoredModel,
    models: Iterable[ScoredModel],
    *,
    index: Index = 'intelligence',
    effort: str | None = None,
    x: XAxis = 'price',
    measured_only: bool = False,
) -> ScoredModel | None:
    """The nearest frontier model that dominates `model`, or None when nothing does.

    "Nearest" is the frontier model with the best x among those at least as
    good on score: on price, the cheapest model that is at least as smart.
    """
    pool = {m.router_id: m for m in models}
    pool[model.router_id] = model
    result = pareto(pool.values(), index=index, effort=effort, x=x, measured_only=measured_only)
    point = next((p for p in result.points if p.model.router_id == model.router_id), None)
    return pool[point.dominated_by] if point is not None and point.dominated_by else None


def _vendor(name: str) -> str:
    return name.lower().removesuffix('-ai').replace('-', '')


def _same_host_family(provider: str | None, developer: str | None) -> bool:
    return bool(provider and developer) and _vendor(provider or '') == _vendor(developer or '')


def one_per_model(models: Iterable[ScoredModel]) -> list[ScoredModel]:
    """Keep one card per underlying model: the developer's own listing, else the cheapest host.

    Cards are grouped by the Artificial Analysis entry they are scored from,
    falling back to `model_identity` for a card without one, so the same weights
    on OpenAI, Azure and an EU region become one point.
    """
    groups: dict[str, list[ScoredModel]] = {}
    for model in models:
        groups.setdefault(model.aa_slug or model_identity(model.router_id), []).append(model)

    def preference(m: ScoredModel) -> tuple[bool, bool, float, str]:
        price = m.price if not m.free else math.inf
        return (not _same_host_family(m.provider, m.developer), m.region is not None, price, m.router_id)

    return sorted((min(group, key=preference) for group in groups.values()), key=lambda m: m.router_id)


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


def find_model(models: Sequence[ScoredModel], model_id: str) -> ScoredModel | None:
    """The model matching `model_id`: an exact `provider/model` id, else a bare id held by exactly one model."""
    exact = next((m for m in models if m.router_id == model_id), None)
    if exact is not None or '/' in model_id:
        return exact
    bare = [m for m in models if m.router_id.split('/', 1)[-1] == model_id]
    return bare[0] if len(bare) == 1 else None


__all__ = [
    'INDICES',
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
