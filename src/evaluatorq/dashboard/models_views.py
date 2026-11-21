"""``/models``: the Pareto view over the Orq model catalogue.

One dot per model: Artificial Analysis score (y) against blended list price,
output speed or time to first token (x). Frontier models are joined by a step
line, dominated ones fade, estimated scores draw hollow. The table under the
chart names, for each dominated model, the nearest frontier model that beats it.
"""

from __future__ import annotations

import math
from operator import itemgetter
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlencode

import httpx
from loguru import logger
from starlette.responses import HTMLResponse
from starlette.routing import Route

from evaluatorq.common.llm_client import MissingLLMCredentialsError
from evaluatorq.common.model_frontier import (
    INDEX_AXES,
    MIN_SEAT_INDEX,
    X_AXIS_SPECS,
    available_efforts,
    available_indices,
    matches_vendor,
    model_vendor,
)
from evaluatorq.common.reports import esc
from evaluatorq.common.reports.palette import COLORS
from evaluatorq.common.reports.vega import render_embed
from evaluatorq.dashboard import model_choices
from evaluatorq.dashboard.auth import auth_or_reason
from evaluatorq.dashboard.shell import page
from evaluatorq.models import X_AXES, ModelCatalogueError, load_models, pareto

if TYPE_CHECKING:
    from collections.abc import Callable

    from starlette.requests import Request

    from evaluatorq.models import Index, ParetoPoint, ParetoResult, ScoredModel, XAxis

ROUTE = '/models'

_COLUMNS: tuple[tuple[str, str], ...] = (
    ('model', 'Model'),
    ('developer', 'Developer'),
    ('score', 'Score'),
    ('input', 'Input $/1M'),
    ('output', 'Output $/1M'),
    ('x', 'X'),
    ('pareto', 'Score per $'),
    ('estimated', 'Estimated'),
    ('dominated_by', 'Dominated by'),
)


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


def _filters_html(
    q: dict[str, str],
    *,
    indices: list[str],
    efforts: list[str],
    vendors: list[str],
    index: str,
    effort: str,
    x: str,
    min_score: float,
) -> str:
    def options(values: list[tuple[str, str]], current: str) -> str:
        return ''.join(
            f'<option value="{esc(v)}"{" selected" if v == current else ""}>{esc(label)}</option>'
            for v, label in values
        )

    effort_disabled = (
        ' disabled title="Domain indices exist only at the default effort"' if index != 'intelligence' else ''
    )

    def checkbox(name: str, label: str) -> str:
        return (
            f'<label class="models-check"><input type="checkbox" name="{name}" value="1"'
            f'{" checked" if q.get(name) else ""} onchange="this.form.submit()"> {esc(label)}</label>'
        )

    return (
        f'<form class="models-filters" method="get" action="{ROUTE}">'
        f'<label>Score <select name="index" onchange="this.form.submit()">'
        f'{options([(i, INDEX_AXES[i].label) for i in indices], index)}</select></label>'
        f'<label>Reasoning level <select name="effort" onchange="this.form.submit()"{effort_disabled}>'
        f'{options([("", "Default level"), *((e, e) for e in efforts)], effort)}</select></label>'
        f'<label>X axis <select name="x" onchange="this.form.submit()">'
        f'{options([(v, X_AXIS_SPECS[v].short_label) for v in X_AXES], x)}</select></label>'
        f'<label>Developer <select name="developer" onchange="this.form.submit()">'
        f'{options([("", "All developers"), *((v, v) for v in vendors)], q.get("developer", ""))}</select></label>'
        f'<label title="Models scoring below this on the chosen index are not plotted. 0 shows every model.">'
        f'Minimum score <input type="number" name="min_score" min="0" step="any" value="{min_score:g}" '
        'class="models-min-score" onchange="this.form.submit()"></label>'
        f'{checkbox("disabled", "Include disabled models")}'
        f'{checkbox("no_estimate", "Hide estimated scores")}'
        '<noscript><button type="submit">Apply</button></noscript>'
        '</form>'
    )


def _chart(result: ParetoResult) -> str:
    rows = [
        {
            'model': p.model.display_name or p.model.router_id,
            'router_id': p.model.router_id,
            'score': round(p.score, 2),
            'x': p.x,
            'status': 'Frontier' if p.on_frontier else 'Dominated',
            'estimated': 'estimated' if p.model.is_estimated else 'measured',
            'dominated_by': p.dominated_by or '',
            'pareto_score': round(p.pareto_score, 2) if p.pareto_score is not None else None,
            'efforts': ', '.join(f'{k} {v:g}' for k, v in sorted(p.model.effort_indices.items())) or '—',
        }
        for p in result.points
    ]
    if not rows:
        return ''
    spec = X_AXIS_SPECS[result.x]
    frontier = sorted((r for r in rows if r['status'] == 'Frontier'), key=itemgetter('x'))
    x_enc: dict[str, Any] = {'field': 'x', 'type': 'quantitative', 'title': spec.label}
    if result.x == 'price':
        x_enc['scale'] = {'type': 'log'}
    y_enc = {
        'field': 'score',
        'type': 'quantitative',
        'title': f'{INDEX_AXES[result.index].label} index' + (f' ({result.effort})' if result.effort else ''),
        'scale': {'zero': False},
    }
    color = {
        'field': 'status',
        'type': 'nominal',
        'scale': {'domain': ['Frontier', 'Dominated'], 'range': [COLORS['orange_300'], COLORS['ink_700']]},
        'legend': {'title': None},
    }
    tooltip = [
        {'field': 'model', 'title': 'Model'},
        {'field': 'router_id', 'title': 'Router id'},
        {'field': 'score', 'title': 'Score'},
        {'field': 'x', 'title': spec.short_label, 'format': f'.{spec.digits}~f'},
        {'field': 'pareto_score', 'title': 'Score per $'},
        {'field': 'estimated', 'title': 'Measured or estimated'},
        {'field': 'dominated_by', 'title': 'Dominated by'},
        {'field': 'efforts', 'title': 'Score per level'},
    ]
    opacity = {'condition': {'test': "datum.status === 'Frontier'", 'value': 0.95}, 'value': 0.35}

    def points(*, estimated: bool) -> dict[str, Any]:
        return {
            'transform': [{'filter': f"datum.estimated === '{'estimated' if estimated else 'measured'}'"}],
            'mark': {'type': 'point', 'filled': not estimated, 'size': 90, 'strokeWidth': 2},
            'encoding': {'x': x_enc, 'y': y_enc, 'color': color, 'opacity': opacity, 'tooltip': tooltip},
        }

    spec: dict[str, Any] = {
        'data': {'values': rows},
        'layer': [
            {
                'data': {'values': frontier},
                'mark': {
                    'type': 'line',
                    # Lower-is-better x: the frontier holds its score until the next, dearer step up.
                    'interpolate': 'step-after' if spec.lower_is_better else 'step-before',
                    'color': COLORS['orange_300'],
                    'strokeWidth': 1.5,
                    'opacity': 0.7,
                },
                'encoding': {'x': x_enc, 'y': y_enc, 'order': {'field': 'x', 'type': 'quantitative'}},
            },
            points(estimated=False),
            points(estimated=True),
            {
                'data': {'values': frontier},
                'mark': {'type': 'text', 'align': 'left', 'dx': 8, 'dy': -6, 'fontSize': 11},
                'encoding': {'x': x_enc, 'y': y_enc, 'text': {'field': 'model'}},
            },
        ],
        'width': 'container',
        'height': 420,
    }
    return render_embed(spec, 'models-pareto')


def _sort_key(column: str, x: str) -> Callable[[ParetoPoint], Any]:
    spec = X_AXIS_SPECS[x]
    keys: dict[str, Callable[[ParetoPoint], Any]] = {
        'model': lambda p: p.model.router_id,
        'developer': lambda p: (model_vendor(p.model), p.model.router_id),
        'score': lambda p: -p.score,
        'input': lambda p: p.model.input_price,
        'output': lambda p: p.model.output_price,
        'x': lambda p: spec.best_first(p.x),
        'pareto': lambda p: -(p.pareto_score or 0.0),
        'estimated': lambda p: (not p.model.is_estimated, -p.score),
        'dominated_by': lambda p: (p.dominated_by or '', -p.score),
    }
    return keys.get(column, keys['score'])


def _table(result: ParetoResult, q: dict[str, str]) -> str:
    sort = q.get('sort', 'score')
    reverse = q.get('dir') == 'rev'
    spec, price = X_AXIS_SPECS[result.x], X_AXIS_SPECS['price']
    points: list[ParetoPoint] = sorted(result.points, key=_sort_key(sort, result.x), reverse=reverse)
    head: list[str] = []
    for key, label in _COLUMNS:
        if key == 'pareto' and result.x != 'price':
            continue
        label = spec.short_label if key == 'x' else label
        active = key == sort
        params = {**q, 'sort': key, 'dir': 'rev' if active and not reverse else ''}
        caret = ('▲' if reverse else '▼') if active else '⇅'
        aria = ('ascending' if reverse else 'descending') if active else 'none'
        head.append(
            f'<th scope="col" aria-sort="{aria}"><a href="{ROUTE}?{esc(urlencode({k: v for k, v in params.items() if v}))}">'
            f'{esc(label)} <span class="models-caret">{caret}</span></a></th>'
        )
    body: list[str] = []
    for p in points:
        m = p.model
        cells = [
            (
                f'<td><span class="models-name">{esc(m.display_name or m.router_id)}</span>'
                f'<span class="models-id">{esc(m.router_id)}</span></td>'
            ),
            f'<td>{esc(m.developer or m.provider or "—")}</td>',
            f'<td class="num">{p.score:.1f}</td>',
            f'<td class="num">{price.format(m.input_price)}</td>',
            f'<td class="num">{price.format(m.output_price)}</td>',
            f'<td class="num">{spec.format(p.x)}</td>',
        ]
        if result.x == 'price':
            pareto_cell = '—' if p.pareto_score is None else f'{p.pareto_score:.1f}'
            cells.append(f'<td class="num">{pareto_cell}</td>')
        cells.extend((
            f'<td>{"estimated" if m.is_estimated else ""}</td>',
            f'<td>{esc(p.dominated_by) if p.dominated_by else "<span class=models-frontier>frontier</span>"}</td>',
        ))
        row_cls = 'models-row is-frontier' if p.on_frontier else 'models-row'
        body.append(f'<tr class="{row_cls}">{"".join(cells)}</tr>')
    return (
        '<div class="models-table-wrap"><table class="models-table">'
        f'<thead><tr>{"".join(head)}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
    )


def _excluded(result: ParetoResult) -> str:
    if not result.excluded:
        return ''
    items = ''.join(f'<li><code>{esc(e.router_id)}</code>: {esc(e.reason)}</li>' for e in result.excluded)
    return (
        f'<details class="models-excluded"><summary>{len(result.excluded)} model(s) not plotted</summary>'
        f'<ul>{items}</ul></details>'
    )


async def _load(req: Request, *, include_disabled: bool) -> tuple[list[ScoredModel] | None, str]:
    """The catalogue models and an empty string, or None and the alert the page shows instead.

    Each failure kind gets its own next step: missing credentials point at
    Settings, a catalogue or network failure asks for a retry, and anything
    else is logged with its traceback and named on the page instead of
    surfacing as a 500.
    """
    from evaluatorq.dashboard.trace_finder.routes import selected_dashboard_auth

    auth, reason = auth_or_reason(lambda: selected_dashboard_auth(req.app))
    if auth is None:
        return None, _alert(reason or 'No Orq account is connected.', hint=_CONNECT_HINT)
    try:
        async with model_choices.catalogue_client(auth) as client:
            return await load_models(client, include_disabled=include_disabled), ''
    except MissingLLMCredentialsError as exc:
        logger.warning('No Orq credentials for /models: {}', exc)
        return None, _alert(str(exc), hint=_CONNECT_HINT)
    except (ModelCatalogueError, httpx.HTTPError, OSError) as exc:
        logger.warning('Orq model catalogue unavailable for /models ({}): {}', type(exc).__name__, exc)
        return None, _alert(f'The Orq model catalogue could not be loaded: {exc}', hint=_RETRY_HINT)
    except _UNEXPECTED_LOAD_ERRORS as exc:
        logger.exception('The /models page failed to load the catalogue')
        return None, _alert(f'The Models page failed with an unexpected {type(exc).__name__}.', hint=_LOG_HINT)


async def models_page(req: Request) -> HTMLResponse:
    """Render the Pareto view over the live catalogue."""
    q = _query(req)
    x = _choice(q.get('x'), X_AXES, 'price')
    models, alert = await _load(req, include_disabled=bool(q.get('disabled')))
    intro = (
        '<p class="models-intro">Each dot is one model, at its developer\'s own listing. A model is '
        '<b>dominated</b> when another is at least as good on both axes and strictly better on one; the orange '
        'line joins the models nothing dominates. Hollow dots are estimated scores. Price is the list price, so a '
        'model that writes more tokens per task costs more than it shows here.</p>'
    )
    if models is None:
        body = f'<div class="models-page">{intro}{alert}</div>'
        return HTMLResponse(page('Models', body, active_nav='models'))
    indices = available_indices(models)
    index = _choice(q.get('index'), tuple(indices), 'intelligence')
    efforts = available_efforts(models)
    effort = _choice(q.get('effort'), tuple(efforts), '') if index == 'intelligence' else ''
    vendors = sorted({model_vendor(m) for m in models} - {''})
    developer = q.get('developer', '')
    pool = [m for m in models if not developer or matches_vendor(m, [developer])]
    min_score = _min_score(q.get('min_score'))
    result = pareto(
        pool,
        index=cast('Index', index),
        effort=effort or None,
        x=cast('XAxis', x),
        exclude_estimated=bool(q.get('no_estimate')),
        min_score=min_score,
    )
    hidden = len(result.below_min_score)
    below = f' ({hidden} below the minimum score of {min_score:g})' if hidden else ''
    summary = (
        f'<p class="models-summary">{len(result.frontier)} on the frontier, {len(result.dominated)} dominated, '
        f'{len(result.excluded)} not plotted{below}.</p>'
    )
    body = (
        '<div class="models-page">'
        f'{intro}{_filters_html(q, indices=indices, efforts=efforts, vendors=vendors, index=index, effort=effort, x=x, min_score=min_score)}{summary}'
        f'<div class="models-chart">{_chart(result)}</div>{_table(result, q)}{_excluded(result)}'
        '</div>'
    )
    return HTMLResponse(page('Models', body, active_nav='models'))


def register_models_routes(app: Any) -> None:
    """Register ``GET /models``."""
    app.add_route(Route(ROUTE, endpoint=models_page, methods=['GET']))
