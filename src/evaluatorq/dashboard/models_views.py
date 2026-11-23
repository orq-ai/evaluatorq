"""``/models``: the Pareto view over the Orq model catalogue.

One dot per model: Artificial Analysis score (y) against blended price (a
cache-aware blend by default, the 3:1 list blend on request), output speed or
time to first token (x). Frontier models are joined by a step
line, dominated ones draw grey, estimated scores draw hollow. The table under the
chart names, for each dominated model, the nearest frontier model that beats it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from operator import itemgetter
from typing import TYPE_CHECKING, Any, NamedTuple, cast
from urllib.parse import urlencode

import httpx
from loguru import logger
from starlette.responses import HTMLResponse
from starlette.routing import Route

from evaluatorq.common.llm_client import MissingLLMCredentialsError
from evaluatorq.common.model_frontier import (
    INDEX_AXES,
    MIN_SEAT_INDEX,
    PRICE_MIXES,
    X_AXIS_SPECS,
    available_efforts,
    available_indices,
    matches_vendor,
    model_vendor,
    score_of,
    x_of,
)
from evaluatorq.common.reports import esc
from evaluatorq.common.reports.vega import render_embed, vl_frontier_scatter
from evaluatorq.dashboard import model_choices
from evaluatorq.dashboard.auth import auth_or_reason
from evaluatorq.dashboard.controls import FILTER_CHEVRON, chip, dropdown, pick_row, range_control, rich_pick, stat_tile
from evaluatorq.dashboard.report_kit import tag
from evaluatorq.dashboard.shell import page
from evaluatorq.dashboard.tables import flipped, sortable_th
from evaluatorq.models import X_AXES, ModelCatalogueError, PriceMix, load_models, pareto, parse_mix

if TYPE_CHECKING:
    from collections.abc import Callable

    from starlette.requests import Request

    from evaluatorq.common.model_frontier import AxisSpec
    from evaluatorq.models import Index, ParetoPoint, ParetoResult, ScoredModel, XAxis

ROUTE = '/models'


def _query(req: Request) -> dict[str, str]:
    return {k: v for k, v in req.query_params.items() if v}


def _choice(value: str | None, allowed: tuple[str, ...], default: str) -> str:
    return value if value in allowed else default


def _min_score(value: str | None) -> float:
    """The score floor from the query string; anything that is not a non-negative number falls back to the default."""
    try:
        floor = float(value) if value is not None else MIN_SEAT_INDEX
    except ValueError:
        return MIN_SEAT_INDEX
    return floor if math.isfinite(floor) and floor >= 0 else MIN_SEAT_INDEX


def _mix(value: str | None) -> PriceMix:
    """The price mix from the query string: a name or a READ:WRITE:FRESH:OUTPUT ratio; anything else is the default."""
    try:
        return parse_mix(value or 'agentic')
    except ValueError:
        return PRICE_MIXES['agentic']


_MIX_OPTION_LABELS = {'agentic': 'Agentic: 7:2:1 cached', 'list': 'List: 3:1 input/output'}
# What the table and tooltip append to a cache rate that is not the host's own.
_SOURCE_NOTE = {
    'artificial_analysis': ('AA', 'Not listed by this host: the Artificial Analysis cache price on the same card.'),
    'input': ('', 'No cache price on this card: cached tokens are billed at the input rate.'),
}

_CONNECT_HINT = '<a href="/settings">Open Settings</a> to connect an Orq account.'
_RETRY_HINT = (
    'Reload the page to try again. If it keeps failing, check the Orq account in <a href="/settings">Settings</a>.'
)
_LOG_HINT = 'The dashboard log has the details.'
# Failures outside the catalogue's own error types: a missing optional dependency, a client that refuses to
# start, or a card shape the parser did not expect. Named rather than a blind `except Exception`.
_UNEXPECTED_LOAD_ERRORS = (ImportError, RuntimeError, ValueError, LookupError, TypeError, AttributeError)


def _alert(message: str, *, hint: str) -> str:
    """An alert box: the escaped `message`, then `hint`, which is trusted markup."""
    return f'<div class="models-alert" role="alert"><p>{esc(message)}</p><p>{hint}</p></div>'


@dataclass(frozen=True)
class _State:
    """The filter state a request resolves to, which the controls show and the chart and table read."""

    index: str
    effort: str
    x: str
    mix: PriceMix
    min_score: float
    developers: list[str]
    unknown_x: str
    indices: list[str]
    efforts: list[str]
    vendors: list[str]

    @property
    def effort_allowed(self) -> bool:
        """Domain indices exist only at the default effort."""
        return self.index == 'intelligence'


def _field(label: str, name: str, control: str, *, off: str = '', title: str = '') -> str:
    """A labelled toolbar control; a non-empty `off` is the reason the control is greyed out and inert.

    An inert control still submits its hidden input, so a choice survives while it does not apply.
    """
    tip = off or title
    tip_attr = f' title="{esc(tip)}"' if tip else ''
    if off:
        control = f'<span class="models-field-off" inert>{control}</span>'
    return (
        f'<span class="models-field{" is-off" if off else ""}"{tip_attr}>'
        f'<label class="models-field-label" for="{esc(name)}">{esc(label)}</label>{control}</span>'
    )


# The More filters toggles: query name and chip label. Each submits `1` when on.
_TOGGLES = {'frontier': 'Frontier only', 'disabled': 'Include disabled models', 'no_estimate': 'Hide estimated scores'}


def _more_active(q: dict[str, str], st: _State) -> int:
    """How many filters inside More filters differ from the page's defaults, for the badge on its closed trigger."""
    return sum((bool(st.effort), st.min_score != MIN_SEAT_INDEX, *(bool(q.get(k)) for k in _TOGGLES)))


def _filters_html(q: dict[str, str], st: _State) -> str:
    mix_options = [(name, pick_row(_MIX_OPTION_LABELS[name])) for name in PRICE_MIXES]
    if st.mix.name is None:
        mix_options.append((st.mix.ratio, pick_row(f'Custom: {st.mix.ratio}')))
    effort_options = [('', pick_row('Default level')), *((e, pick_row(e)) for e in st.efforts)]
    more_rows = (
        _field(
            label='Reasoning level',
            name='effort',
            control=rich_pick('effort', st.effort, effort_options, current=effort_options[0][1]),
            off=''
            if st.effort_allowed
            else 'Domain indices exist only at the one effort Artificial Analysis benchmarked',
        )
        + '<div title="Models scoring below this on the chosen index are not plotted. 0 shows every model.">'
        + range_control(
            dim='min_score',
            label='Minimum score',
            min_val=0,
            max_val=100,
            step='1',
            sel=[f'{st.min_score:g}'],
            floor=True,
            default=MIN_SEAT_INDEX,
        )
        + '</div><div class="filter-chip-row">'
        + ''.join(
            chip(name, '1', checked=bool(q.get(name)), dot_cls='chip-dot-jade', label=label)
            for name, label in _TOGGLES.items()
        )
        + '</div>'
    )
    active = _more_active(q, st)
    badge = f'<span class="filter-dd-more-badge">{active}</span>' if active else ''
    more = (
        '<details id="filter-dd-more" class="filter-dd filter-dd-more">'
        f'<summary class="filter-dd-trigger"><span class="filter-dd-name">More filters</span>{badge}{FILTER_CHEVRON}'
        f'</summary><div class="filter-dd-more-body">{more_rows}</div></details>'
    )
    # `change` bubbles, so one handler on the form submits for every control. A search box also fires `change`
    # when it loses focus; submitting then would swap the page and drop the typed text.
    return (
        f'<form class="models-filters" method="get" action="{ROUTE}"'
        """ onchange="if (!event.target.matches('.facet-search')) this.requestSubmit()">"""
        + _field(
            label='Score',
            name='index',
            control=rich_pick(
                'index', st.index, [(i, pick_row(INDEX_AXES[i].label)) for i in st.indices], filterable=True
            ),
        )
        + _field(
            label='X axis',
            name='x',
            control=rich_pick('x', st.x, [(v, pick_row(X_AXIS_SPECS[v].short_label)) for v in X_AXES]),
        )
        + _field(
            label='Price mix',
            name='mix',
            control=rich_pick('mix', st.mix.name or st.mix.ratio, mix_options),
            off='The mix applies only to the price axis' if st.x != 'price' else '',
            title='How the price axis blends token prices, as cache read : cache write : fresh input : output. Add '
            '?mix=READ:WRITE:FRESH:OUTPUT to the address for your own ratio.',
        )
        + dropdown('developer', 'Developer', st.vendors, st.developers, searchable=True)
        + more
        + '<noscript><button type="submit">Apply</button></noscript></form>'
    )


def _usd(value: float) -> str:
    """A price in dollars with enough decimals to tell small rates apart: $4.00, $0.051, $0.0070; zero is `free`."""
    if value == 0:
        return 'free'
    return f'${value:.2f}' if value >= 1 else f'${value:.3f}' if value >= 0.01 else f'${value:.4f}'


class _AxisText(NamedTuple):
    """Everything the page says that depends on the x axis, so no function branches on `x == 'price'`."""

    gain: str  # what a model that beats another is, in the gap text
    same: str  # no difference on this axis
    cost: str  # what the next model up on score costs on this axis
    best: str  # the answer card's label for the best x value
    chart_format: str  # the chart's tick format
    fmt: Callable[[float], str]
    unit: str  # appended where a bare number would be ambiguous


_AXIS_TEXT = {
    'price': _AxisText('cheaper', 'same price', 'pricier', 'Cheapest', '$~g', _usd, ''),
    'output_speed': _AxisText(
        'faster', 'same speed', 'slower', 'Fastest', '~g', X_AXIS_SPECS['output_speed'].format, ' tok/s'
    ),
    'time_to_first_token': _AxisText(
        'lower TTFT',
        'same TTFT',
        'slower to first token',
        'Quickest first token',
        '~g',
        X_AXIS_SPECS['time_to_first_token'].format,
        ' s',
    ),
}


def _name(m: ScoredModel) -> str:
    return m.display_name or m.router_id


def _times(ratio: float) -> str:
    """A ratio as a multiple: 2.1x, 29x."""
    return f'{ratio:.0f}x' if ratio >= 10 else f'{ratio:.1f}x'


class _CacheRead(NamedTuple):
    text: str
    mark: str
    why: str

    @property
    def label(self) -> str:
        return f'{self.text} ({self.mark})' if self.mark else self.text


def _cache_read(point: ParetoPoint) -> _CacheRead | None:
    """The cache-read rate the point's price uses as text, its short source mark and the mark's explanation; None if unused."""
    if point.cache_read_source is None or point.cache_read_rate is None:
        return None
    mark, why = _SOURCE_NOTE.get(point.cache_read_source, ('', ''))
    # Reads billed at the input rate would repeat the Input column, so the cell says so instead.
    return _CacheRead('= input' if point.cache_read_source == 'input' else _usd(point.cache_read_rate), mark, why)


def _cache_cell(point: ParetoPoint) -> str:
    cache = _cache_read(point)
    if cache is None:
        return '<td class="num num-secondary">—</td>'
    title = f' title="{esc(cache.why)}"' if cache.why else ''
    return f'<td class="num num-secondary"{title}>{esc(cache.text)}{tag(cache.mark, small=True, title=cache.why)}</td>'


def _label_rows(frontier: list[dict[str, Any]]) -> None:
    """Name the best-scoring and the cheapest frontier points, then others 2 or more score points from every name.

    Walking from the highest score down means the corner that matters most is always named, and labels do not
    pile up. `frontier` is sorted by `x` ascending, so its first row is the lowest-x anchor.
    """
    for row in frontier:
        row['label'] = ''
    if not frontier:
        return
    named = [max(frontier, key=itemgetter('score')), frontier[0]]
    for row in sorted(frontier, key=itemgetter('score'), reverse=True):
        if row not in named and all(abs(row['score'] - n['score']) >= 2 for n in named):
            named.append(row)
    for row in named:
        row['label'] = row['model']


def _winner(point: ParetoPoint, by_id: dict[str, ParetoPoint]) -> ParetoPoint | None:
    return by_id.get(point.dominated_by or '')


_GUIDE_SAMPLES = 40
_MAX_GUIDES = 4


def _guide_ks(lo: float, hi: float) -> list[float]:
    """The 1-2-5 steps strictly inside (lo, hi), at most four spread across it: every other step when more fit."""
    if lo <= 0 or hi <= lo:
        return []
    decade = 10.0 ** math.floor(math.log10(lo))
    ks: list[float] = []
    while decade <= hi:
        ks += [s * decade for s in (1, 2, 5) if lo < s * decade < hi]
        decade *= 10
    if len(ks) > _MAX_GUIDES:
        ks = ks[::2]
    if len(ks) > _MAX_GUIDES:
        ks = [ks[round(i * (len(ks) - 1) / (_MAX_GUIDES - 1))] for i in range(_MAX_GUIDES)]
    return ks


def _guide_label(k: float) -> str:
    """A 1-2-5 step as plain text, never in exponent form: 1,000 and 0.05, not 1e+03 and 5e-02."""
    return f'{k:,.0f}' if k >= 1 else f'{k:f}'.rstrip('0')


def _guide_rows(result: ParetoResult, floor: float) -> list[dict[str, Any]]:
    """Samples of the curves `score = floor + k * x` the chart draws as Score per $ guides; none off the price axis.

    `k` comes from `_guide_ks` over the plotted Score per $ range. Each curve is sampled at 40 log-spaced prices
    across the plotted x range, keeping the samples whose score stays inside the plotted score range.
    """
    if result.x != 'price':
        return []
    # A model scoring exactly at the floor has Score per $ 0; it would stretch the steps down to flat lines on the floor.
    values = [p.pareto_score for p in result.points if p.pareto_score is not None and p.pareto_score > 0]
    xs = [p.x for p in result.points if p.x > 0]
    if not values or not xs:
        return []
    lo_x, hi_x = min(xs), max(xs)
    lo_s, hi_s = min(p.score for p in result.points), max(p.score for p in result.points)
    step = math.log(hi_x / lo_x) / (_GUIDE_SAMPLES - 1)
    samples = [lo_x * math.exp(i * step) for i in range(_GUIDE_SAMPLES - 1)] + [hi_x]
    return [
        {'x': x, 'score': score, 'k': _guide_label(k)}
        for k in _guide_ks(min(values), max(values))
        for x in samples
        if lo_s <= (score := floor + k * x) <= hi_s
    ]


def _chart(result: ParetoResult, by_id: dict[str, ParetoPoint], floor: float) -> str:
    spec = result.x_spec
    axis = _AXIS_TEXT[result.x]
    rows = [
        {
            'model': _name(p.model),
            'router_id': p.model.router_id,
            'score': round(p.score, 2),
            'x': p.x,
            'x_text': axis.fmt(p.x),
            'status': 'Frontier' if p.on_frontier else 'Dominated',
            'estimated': 'estimated' if p.model.is_estimated else 'measured',
            'beaten_by': _beaten_by_text(p, result, by_id) if p.dominated_by else '',
            'efforts': ', '.join(f'{k} {v:g}' for k, v in sorted(p.model.effort_indices.items())) or '—',
            'cache_read': c.label if (c := _cache_read(p)) else '—',
            'label': '',
        }
        for p in result.points
    ]
    _label_rows(sorted((r for r in rows if r['status'] == 'Frontier'), key=itemgetter('x')))
    tooltip = [
        {'field': 'model', 'title': 'Model'},
        {'field': 'router_id', 'title': 'Router id'},
        {'field': 'score', 'title': 'Score'},
        {'field': 'x_text', 'title': spec.short_label},
        {'field': 'cache_read', 'title': 'Cache read $/1M'},
        {'field': 'estimated', 'title': 'Measured or estimated'},
        {'field': 'beaten_by', 'title': 'Beaten by'},
        {'field': 'efforts', 'title': 'Score per level'},
    ]
    chart = vl_frontier_scatter(
        rows=rows,
        x_title=spec.label,
        y_title=f'{INDEX_AXES[result.index].label} index' + (f' ({result.effort})' if result.effort else ''),
        lower_is_better=spec.lower_is_better,
        tooltip=tooltip,
        x_format=axis.chart_format,
        guides=_guide_rows(result, floor),
    )
    return render_embed(chart, 'models-pareto') if chart else ''


def _sort_key(column: str, x: str, mix: PriceMix) -> Callable[[ParetoPoint], Any]:
    spec = X_AXIS_SPECS[x]
    keys: dict[str, Callable[[ParetoPoint], Any]] = {
        'model': lambda p: p.model.router_id,
        'score': lambda p: -p.score,
        'x': lambda p: spec.best_first(p.x),
        'price': lambda p: mix.price(p.model),
        'rates': lambda p: (p.model.input_price, p.model.output_price),
        'cache': lambda p: p.cache_read_rate if p.cache_read_rate is not None else math.inf,
        'value': lambda p: -p.pareto_score if p.pareto_score is not None else math.inf,
        'dominated_by': lambda p: (p.dominated_by or '', -p.score),
    }
    return keys.get(column, keys['score'])


def _ratio_text(better: ParetoPoint, worse: ParetoPoint, spec: AxisSpec, word: str) -> str:
    """How much better `better` is than `worse` on the x axis, as `word` with a multiple (2x or more) or a percentage."""
    if better.x <= 0 or worse.x <= 0 or better.x == worse.x:
        return ''
    ratio = worse.x / better.x if spec.lower_is_better else better.x / worse.x
    # Percent flattens everything past 90%, so a gap of 2x or more is said as a multiple.
    return f'{_times(ratio)} {word}' if ratio >= 2 else f'{1 - 1 / ratio:.0%} {word}'


def _gap(winner: ParetoPoint, loser: ParetoPoint, result: ParetoResult) -> str:
    """What `winner` gains over `loser`: its score lead and how much better it is on the x axis."""
    axis = _AXIS_TEXT[result.x]
    lead = winner.score - loser.score
    parts = [f'+{lead:.1f} score' if lead >= 0.05 else 'same score']
    if winner.x == loser.x:
        parts.append(axis.same)
    elif text := _ratio_text(winner, loser, result.x_spec, axis.gain):
        parts.append(text)
    return ', '.join(parts)


def _step_up(point: ParetoPoint, higher: ParetoPoint | None, result: ParetoResult) -> str:
    """What the next frontier model up on score costs over `point`, for the frontier rows' Beaten by cell."""
    if higher is None:
        return 'highest score'
    cost = _ratio_text(point, higher, result.x_spec, _AXIS_TEXT[result.x].cost)
    return f'next: {_name(higher.model)}, +{higher.score - point.score:.1f} score' + (f', {cost}' if cost else '')


def _frontier_dominators(point: ParetoPoint, by_id: dict[str, ParetoPoint]) -> list[ParetoPoint]:
    """The frontier models that beat `point`: a dominated model that beats it is never the right switch."""
    return [w for rid in point.dominators if (w := by_id.get(rid)) is not None and w.on_frontier]


def _best_dominator(point: ParetoPoint, by_id: dict[str, ParetoPoint]) -> ParetoPoint | None:
    """The highest-scoring frontier model that beats `point`, when it scores above the nearest one."""
    winner = _winner(point, by_id)
    best = max(_frontier_dominators(point, by_id), key=lambda w: w.score, default=None)
    return best if winner is not None and best is not None and best.score - winner.score >= 0.05 else None


def _beaten_by_text(point: ParetoPoint, result: ParetoResult, by_id: dict[str, ParetoPoint]) -> str:
    """The nearest frontier model that beats `point`, with what it gains, as plain text."""
    winner = _winner(point, by_id)
    if winner is None:
        return point.dominated_by or ''
    return f'{_name(winner.model)}: {_gap(winner, point, result)}'


def _beaten_by_cell(
    point: ParetoPoint, result: ParetoResult, by_id: dict[str, ParetoPoint], steps: dict[str, str]
) -> str:
    winner = _winner(point, by_id)
    if winner is None:
        step = steps.get(point.model.router_id, '')
        return f'<td><span class="models-frontier">frontier</span><span class="models-gap">{esc(step)}</span></td>'
    best = _best_dominator(point, by_id)
    rest = [w for w in _frontier_dominators(point, by_id) if w is not winner and w is not best]
    more = ''
    if rest:
        listing = '; '.join(f'{_name(w.model)} ({_gap(w, point, result)})' for w in rest)
        more = tag(f'+{len(rest)} more', small=True, title=listing)
    upgrade = (
        f'<span class="models-gap">best: {esc(_name(best.model))}, {esc(_gap(best, point, result))}</span>'
        if best
        else ''
    )
    return (
        f'<td><span class="models-name" title="{esc(winner.model.router_id)}">{esc(_name(winner.model))}</span>'
        f'<span class="models-gap">{esc(_gap(winner, point, result))}{more}</span>{upgrade}</td>'
    )


@dataclass(frozen=True)
class _Col:
    """One table column: its sort key and header, the direction it sorts in first, and whether it holds numbers."""

    key: str
    label: str
    natural: str | None = None  # None: the x axis's own best-first order
    numeric: bool = False


_COLUMNS = (
    _Col('model', 'Model', 'asc'),
    _Col('score', 'Score', 'desc', numeric=True),
    _Col('dominated_by', 'Beaten by', 'asc'),
    _Col('x', 'X', None, numeric=True),
    _Col('value', 'Score per $', 'desc', numeric=True),
    _Col('price', 'Price $/1M', 'asc', numeric=True),
    _Col('rates', 'Input / output $/1M', 'asc', numeric=True),
    _Col('cache', 'Cache read $/1M', 'asc', numeric=True),
)


def _link_params(params: dict[str, str], developers: list[str]) -> list[tuple[str, str]]:
    """A link's query pairs: the non-empty single values, then one `developer` pair per chosen developer."""
    return [(k, v) for k, v in params.items() if v and k != 'developer'] + [('developer', d) for d in developers]


def _head(columns: list[_Col], q: dict[str, str], result: ParetoResult, mix: PriceMix, developers: list[str]) -> str:
    """The sortable header row: each link sorts by its column, and clicking the active one reverses the order."""
    spec = result.x_spec
    sort = q.get('sort', 'score')
    reverse = q.get('dir') == 'rev'
    labels = {
        'x': spec.short_label,
        'price': f'Price $/1M ({mix.short_title})',
        'score': INDEX_AXES[result.index].label,
    }
    cells = []
    for col in columns:
        natural = col.natural or ('asc' if spec.lower_is_better else 'desc')
        active = col.key == sort
        params = {**q, 'sort': col.key, 'dir': 'rev' if active and not reverse else ''}
        cells.append(
            sortable_th(
                labels.get(col.key, col.label),
                control=f'href="{ROUTE}?{esc(urlencode(_link_params(params, developers)))}"',
                active=active,
                direction=flipped(natural) if reverse else natural,
                numeric=col.numeric,
                tag='a',
            )
        )
    return f'<thead><tr>{"".join(cells)}</tr></thead>'


def _row(
    p: ParetoPoint,
    result: ParetoResult,
    by_id: dict[str, ParetoPoint],
    steps: dict[str, str],
    *,
    mix: PriceMix,
    hidden: set[str],
) -> str:
    m = p.model
    host = m.router_id.split('/', 1)[0]
    via = tag(f'via {host}', small=True, title='Listed by a host, not its developer') if host != model_vendor(m) else ''
    est = tag('est', small=True, title='Estimated from an older benchmark version') if m.is_estimated else ''
    cells = [
        (
            f'<td><span class="models-name">{esc(_name(m))}{via}</span>'
            f'<span class="models-id" title="{esc(m.router_id)}">{esc(m.router_id)}</span></td>'
        ),
        f'<td class="num">{p.score:.1f}{est}</td>',
        _beaten_by_cell(p, result, by_id, steps),
        f'<td class="num">{_AXIS_TEXT[result.x].fmt(p.x)}</td>',
    ]
    if 'value' not in hidden:
        cells.append(f'<td class="num">{"—" if p.pareto_score is None else f"{p.pareto_score:,.1f}"}</td>')
    if 'price' not in hidden:
        cells.append(f'<td class="num">{_usd(mix.price(m))}</td>')
    cells.append(f'<td class="num num-secondary">{_usd(m.input_price)} / {_usd(m.output_price)}</td>')
    if 'cache' not in hidden:
        cells.append(_cache_cell(p))
    cls = [
        'models-row',
        *(['is-frontier'] if p.on_frontier else []),
    ]
    return f'<tr class="{" ".join(cls)}">{"".join(cells)}</tr>'


def _contributor_row(m: ScoredModel, reason: str, result: ParetoResult, *, mix: PriceMix, hidden: set[str]) -> str:
    """A contributor listing: shown with its numbers for reference, tagged as excluded with the reason on hover."""
    score = score_of(m, index=cast('Index', result.index), effort=result.effort)
    xv = x_of(m, cast('XAxis', result.x), mix=mix)
    cells = [
        (
            f'<td><span class="models-name">{esc(_name(m))}{tag("excluded", tone="amber", small=True, title=reason)}</span>'
            f'<span class="models-id" title="{esc(m.router_id)}">{esc(m.router_id)}</span></td>'
        ),
        f'<td class="num">{"—" if score is None else f"{score:.1f}"}</td>',
        f'<td title="{esc(reason)}">not compared</td>',
        f'<td class="num">{"—" if xv is None else _AXIS_TEXT[result.x].fmt(xv)}</td>',
    ]
    if 'value' not in hidden:
        cells.append('<td class="num">—</td>')
    if 'price' not in hidden:
        cells.append(f'<td class="num">{_usd(mix.price(m))}</td>')
    cells.append(f'<td class="num num-secondary">{_usd(m.input_price)} / {_usd(m.output_price)}</td>')
    if 'cache' not in hidden:
        cells.append('<td class="num num-secondary">—</td>')
    return f'<tr class="models-row is-excluded">{"".join(cells)}</tr>'


def _group(title: str, rows: list[str], ncols: int) -> str:
    """A `<tbody>` of `rows` under a named group row; nothing when there are none."""
    if not rows:
        return ''
    return (
        f'<tbody><tr class="models-group"><th colspan="{ncols}">{title} · {len(rows)}</th></tr>{"".join(rows)}</tbody>'
    )


def _table(
    result: ParetoResult,
    q: dict[str, str],
    by_id: dict[str, ParetoPoint],
    mix: PriceMix,
    pool: dict[str, ScoredModel],
    developers: list[str],
) -> str:
    # `result.frontier` is strongest first, so the model one score step up sits just before each point.
    frontier = result.frontier
    steps = {p.model.router_id: _step_up(p, frontier[i - 1] if i else None, result) for i, p in enumerate(frontier)}
    # On a speed axis the price column is otherwise gone, so it comes back as its own column.
    hidden = {'cache'} if not (result.mix and result.mix.read) else set()
    # Score per $ is the score over the price, so it only means something when price is the x axis.
    hidden.add('price' if result.x == 'price' else 'value')
    points = [p for p in result.points if p.on_frontier or not q.get('frontier')]
    points.sort(key=_sort_key(q.get('sort', 'score'), result.x, mix), reverse=q.get('dir') == 'rev')
    columns = [c for c in _COLUMNS if c.key not in hidden]

    def rows(members: list[ParetoPoint]) -> list[str]:
        return [_row(p, result, by_id, steps, mix=mix, hidden=hidden) for p in members]

    if 'sort' in q:
        body = f'<tbody>{"".join(rows(points))}</tbody>'
    else:
        # Frontier rows come first by default, so a named group row says why the score column is not in order.
        body = _group('On the frontier', rows([p for p in points if p.on_frontier]), len(columns)) + _group(
            'Dominated', rows([p for p in points if not p.on_frontier]), len(columns)
        )
    excluded = [
        _contributor_row(pool[e.router_id], e.reason, result, mix=mix, hidden=hidden)
        for e in result.contributors
        if e.router_id in pool and not q.get('frontier')
    ]
    body += _group('Excluded', excluded, len(columns))
    return (
        '<div class="models-table-wrap"><table class="models-table no-stack">'
        f'{_head(columns, q, result, mix, developers)}{body}</table></div>'
    )


def _popover(key: str, label: str, body: str) -> str:
    """A text button that opens `body` in a native popover under it; a click elsewhere closes it, no JS."""
    pid = f'models-pop-{key}'
    return (
        f'<button type="button" class="models-pop-btn" popovertarget="{pid}" style="anchor-name: --{pid}">'
        f'{label} <span aria-hidden="true">ⓘ</span></button>'
        f'<div popover id="{pid}" class="models-pop" style="position-anchor: --{pid}">{body}</div>'
    )


def _excluded(result: ParetoResult, min_score: float) -> str:
    """The not-plotted count, opening the missing models grouped by reason: one line per reason, not per model."""
    if not result.excluded:
        return ''
    groups: dict[str, list[str]] = {}
    for e in result.excluded:
        groups.setdefault(f'below score {min_score:g}' if e.below_min_score else e.reason, []).append(e.router_id)
    body = ''.join(f'<p><b>{len(ids)} {esc(reason)}</b><br>{esc(", ".join(ids))}</p>' for reason, ids in groups.items())
    return _popover('excluded', f'{len(result.excluded)} not plotted', body)


async def _load(req: Request, *, include_disabled: bool) -> list[ScoredModel] | str:
    """The catalogue models, or the alert HTML the page shows instead.

    Each failure kind gets its own next step: missing credentials point at
    Settings, a catalogue or network failure asks for a retry, and anything
    else is logged with its traceback and named on the page instead of
    surfacing as a 500.
    """
    from evaluatorq.dashboard.trace_finder.routes import selected_dashboard_auth

    auth, reason = auth_or_reason(lambda: selected_dashboard_auth(req.app))
    if auth is None:
        return _alert(reason or 'No Orq account is connected.', hint=_CONNECT_HINT)
    try:
        async with model_choices.catalogue_client(auth) as client:
            return await load_models(client, include_disabled=include_disabled)
    except MissingLLMCredentialsError as exc:
        logger.warning('No Orq credentials for /models: {}', exc)
        return _alert(str(exc), hint=_CONNECT_HINT)
    except (ModelCatalogueError, httpx.HTTPError, OSError) as exc:
        logger.warning('Orq model catalogue unavailable for /models ({}): {}', type(exc).__name__, exc)
        return _alert(f'The Orq model catalogue could not be loaded: {exc}', hint=_RETRY_HINT)
    except _UNEXPECTED_LOAD_ERRORS as exc:
        logger.exception('The /models page failed to load the catalogue')
        return _alert(f'The Pareto Frontier page failed with an unexpected {type(exc).__name__}.', hint=_LOG_HINT)


def _intro(x_label: str) -> str:
    return (
        '<p class="models-intro">Each dot is one model, at its developer\'s own listing when there is one, else at the cheapest '
        'listing, preferring one without a region. The '
        f'orange line joins the models no other model beats on both score and {esc(x_label)}; every other row names '
        'the model that does, and by how much.</p>'
    )


_MIX_PARTS = (('read', 'cache read'), ('write', 'cache write'), ('fresh', 'fresh input'), ('output', 'output'))
_PRICE_DETAIL = (
    'The default is the Artificial Analysis blend, close to an agent loop that re-sends its history; the Price mix '
    'control switches to the 3:1 input/output list blend. A cache rate marked <b>AA</b> is the Artificial Analysis '
    "price on the same card, and <b>= input</b> means reads are billed at the input rate; neither is the host's own. "
    'Either way a model that writes more tokens per task costs more than it shows here.'
)


def _price_caption(mix: PriceMix) -> str:
    """The one line under the chart saying how the price axis blends rates, with the detail in a popover."""
    parts = ', '.join(
        f'{getattr(mix, k):g} part{"" if getattr(mix, k) == 1 else "s"} {label}'
        for k, label in _MIX_PARTS
        if getattr(mix, k)
    )
    return (
        f'<p class="models-caption">Price blends the rates per 1M tokens as {esc(parts)}. '
        f'{_popover("price", "How prices are computed", f"<p>{_PRICE_DETAIL}</p>")}</p>'
    )


def _shell(inner: str) -> HTMLResponse:
    """The page around `inner`: boost swaps `.models-page` inside the stable #models-root on a filter change.

    An outerHTML swap would detach the node dashboard.js scans for charts, and the server still fetches the
    catalogue per request; cache `load_models()` if that round trip is the slow part.
    """
    body = (
        '<div id="models-root" hx-boost="true" hx-target="#models-root" hx-select=".models-page" '
        f'hx-swap="innerHTML" hx-push-url="true"><div class="models-page">{inner}</div></div>'
    )
    return HTMLResponse(page('Pareto Frontier', body, active_nav='models'))


def _x_text(p: ParetoPoint, result: ParetoResult) -> str:
    axis = _AXIS_TEXT[result.x]
    return f'{axis.fmt(p.x)}{axis.unit}'


def _answer_cards(result: ParetoResult, min_score: float) -> str:
    """Three quick answers: the best score, the best score per dollar (price axis) and the best x at the floor."""
    if not result.frontier:
        return ''
    best = max(result.frontier, key=lambda p: p.score)
    cards = [
        stat_tile(label='Best score', value=f'{best.score:.1f}', sub=f'{_name(best.model)} · {_x_text(best, result)}')
    ]
    valued = [p for p in result.frontier if p.pareto_score is not None]
    value = max(valued, key=lambda p: cast('float', p.pareto_score), default=None)
    if value is not None:
        per_dollar = cast('float', value.pareto_score)
        cards.append(
            stat_tile(
                label='Best score per $',
                # Whole numbers would print a score per $ under 0.5 as 0, so small values keep one decimal.
                value=f'{per_dollar:,.0f}' if per_dollar >= 10 else f'{per_dollar:.1f}',
                unit=' per $',
                sub=f'{_name(value.model)} · score {value.score:.1f} · {_x_text(value, result)}',
            )
        )
    cheapest = min(result.frontier, key=lambda p: result.x_spec.best_first(p.x))
    # A card naming the same model as another one adds nothing: one frontier model would fill every card.
    if cheapest is not value and cheapest is not best:
        label = _AXIS_TEXT[result.x].best + (f' at score ≥ {min_score:g}' if min_score > 0 else '')
        cards.append(
            stat_tile(
                label=label,
                value=_x_text(cheapest, result),
                sub=f'{_name(cheapest.model)} · score {cheapest.score:.1f}',
            )
        )
    return f'<div class="stat-band models-cards">{"".join(cards)}</div>'


def _summary(result: ParetoResult, state: _State) -> str:
    """The one status strip: plotted and dominated counts, what is not plotted, estimates and fallback-priced models."""
    estimated = sum(p.model.is_estimated for p in result.points)
    fallbacks = len(result.price_fallbacks)
    chips = [
        f'{len(result.frontier)} on the frontier',
        f'{len(result.dominated)} dominated',
        _excluded(result, state.min_score),
        f'{estimated} estimated {"score" if estimated == 1 else "scores"} (hollow dots)' if estimated else '',
        f'{fallbacks} priced with a cache rate their host does not list' if fallbacks else '',
        f'Unknown x axis "{esc(state.unknown_x)}", showing price' if state.unknown_x else '',
    ]
    return '<div class="models-summary">' + ''.join(f'<span>{c}</span>' for c in chips if c) + '</div>'


async def models_page(req: Request) -> HTMLResponse:
    """Render the Pareto view over the live catalogue."""
    q = _query(req)
    x = _choice(q.get('x'), X_AXES, 'price')
    mix = _mix(q.get('mix'))
    models = await _load(req, include_disabled=bool(q.get('disabled')))
    if isinstance(models, str):
        return _shell(f'{_intro(X_AXIS_SPECS[x].short_label)}{models}')
    indices = available_indices(models)
    index = _choice(q.get('index'), tuple(indices), 'intelligence')
    efforts = available_efforts(models)
    vendors = sorted({model_vendor(m) for m in models} - {''})
    # The developer list submits every ticked box, so all of them ticked is the same as no developer filter.
    developers = [d for d in req.query_params.getlist('developer') if d]
    if set(vendors) <= set(developers):
        developers = []
    state = _State(
        index=index,
        effort=_choice(q.get('effort'), tuple(efforts), '') if index == 'intelligence' else '',
        x=x,
        mix=mix,
        min_score=_min_score(q.get('min_score')),
        developers=developers,
        unknown_x=q['x'] if q.get('x') and q['x'] not in X_AXES else '',
        indices=indices,
        efforts=efforts,
        vendors=vendors,
    )
    pool = [m for m in models if not state.developers or matches_vendor(m, state.developers)]
    result = pareto(
        pool,
        index=cast('Index', index),
        effort=state.effort or None,
        x=cast('XAxis', x),
        mix=mix,
        exclude_estimated=bool(q.get('no_estimate')),
        min_score=state.min_score,
    )
    by_id = {p.model.router_id: p for p in result.points}
    return _shell(
        f'{_answer_cards(result, state.min_score)}{_intro(result.x_spec.short_label)}'
        f'{_filters_html(q, state)}{_summary(result, state)}'
        f'<div class="models-chart">{_chart(result, by_id, state.min_score)}</div>'
        f'{_price_caption(mix) if x == "price" else ""}'
        f'{_table(result, q, by_id, mix, {m.router_id: m for m in pool}, state.developers)}'
    )


def register_models_routes(app: Any) -> None:
    """Register ``GET /models``."""
    app.add_route(Route(ROUTE, endpoint=models_page, methods=['GET']))
