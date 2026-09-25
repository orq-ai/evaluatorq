"""HTML rendering for the read-only Insights run review page."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import quote, urlencode

from evaluatorq.common.reports import esc
from evaluatorq.common.reports.palette import COLORS, ORQ_SCALE_GOOD_BAD, ORQ_SCALE_HEAT, QUALITATIVE
from evaluatorq.common.reports.vega import render_embed
from evaluatorq.dashboard.shell import page
from evaluatorq.dashboard.trace_links import trace_link_button, trace_span_url

if TYPE_CHECKING:
    from evaluatorq.insights.models import Cluster, InsightsRun, TraceInsight

TABS = ('dimensions', 'labels', 'crosstab', 'traces', 'priority', 'map')
TAB_LABELS = {
    'dimensions': 'Dimensions',
    'labels': 'Labels',
    'crosstab': 'Crosstab',
    'traces': 'Traces',
    'priority': 'Priority matrix',
    'map': 'Map',
}


def run_rail(entries: list[tuple[str, str, str]], selected: str | None) -> str:
    """Render run links and unreadable entries in the left rail."""
    rows = []
    for run_id, label, status in entries:
        classes = 'insights-run'
        if run_id == selected:
            classes += ' selected'
        if status == 'error' or status == 'unreadable':
            classes += ' failed'
        icon = {'running': '◌', 'error': '✕', 'unreadable': '!', 'completed': '●'}.get(status, '●')
        href = f'/insights/{quote(run_id, safe="")}'
        rows.append(
            f'<a class="{classes}" href="{href}"><span class="insights-run-name">{icon} {esc(label)}</span>'
            f'<span class="insights-run-status">{esc(status)}</span></a>'
        )
    body = ''.join(rows) or '<p class="insights-empty">No Insights runs yet.</p>'
    return f'<aside class="insights-rail"><h2>Runs</h2>{body}</aside>'


def _chip(label: str, value: object) -> str:
    if value in (None, '', [], {}, ()):
        return ''
    return f'<span class="insights-chip"><b>{esc(label)}</b> {esc(str(value))}</span>'


def header(run: InsightsRun) -> str:
    population = run.population
    query = population.get('query')
    facets = population.get('facets', {})
    facets = facets if isinstance(facets, dict) else {}
    numeric = population.get('numeric', {})
    numeric = numeric if isinstance(numeric, dict) else {}
    chips = [_chip('query', query)]
    for name, values in facets.items():
        if isinstance(values, (list, tuple, set, frozenset)):
            chips.extend(_chip(name, value) for value in values)
        elif values:
            chips.append(_chip(name, values))
    numeric_labels = {
        'tokens_min': 'tokens ≥',
        'tokens_max': 'tokens ≤',
        'duration_ms_min': 'duration ms ≥',
        'duration_ms_max': 'duration ms ≤',
    }
    for name, value in numeric.items():
        label = numeric_labels.get(name, name.replace('_', ' '))
        chips.append(_chip(label, value))
    chips.extend((_chip('start', population.get('start')), _chip('end', population.get('end'))))
    chips.append(_chip('limit', population.get('limit')))
    label_chips = [_chip('label', label.name) for label in run.config.labels]
    models = f'{run.config.summary_model} · {run.config.classifier_model} · {run.config.embedding_model}'
    finder = ''
    if query:
        finder = '<a class="insights-action" href="/find">Open in finder ↗</a>'
    population_chips = ''.join(chips) or '<span class="insights-muted">All traces</span>'
    label_chip_html = ''.join(label_chips) or '<span class="insights-muted">No labels</span>'
    run_url = quote(run.run_id, safe='')
    return (
        '<header class="insights-header">'
        f'<div><h2>{esc(run.run_name)}</h2><p class="insights-subtitle">{run.counts.get("n_traces", len(run.traces))} traces · '
        f'{esc(run.created_at.strftime("%Y-%m-%d %H:%M UTC"))} · {esc(models)}</p>'
        f'<div class="insights-chip-group"><span class="insights-group-label">Population</span>{population_chips}</div>'
        f'<div class="insights-chip-group"><span class="insights-group-label">Labels</span>{label_chip_html}</div></div>'
        '<div class="insights-actions">'
        f'{finder}<a class="insights-action" href="/insights/{run_url}/export.json">Export JSON</a>'
        '</div></header>'
    )


def failures(run: InsightsRun) -> str:
    banners = []
    if run.status == 'error':
        items = (
            ''.join(f'<li><b>{esc(item.stage)}</b>: {esc(item.message)}</li>' for item in run.stage_failures)
            or '<li>The run ended with an error.</li>'
        )
        banners.append(
            f'<section class="insights-error" role="alert"><b>Run failed; results may be partial.</b><ul>{items}</ul></section>'
        )
    n_failed = run.counts.get('n_failed_traces', 0)
    if n_failed:
        banners.append(
            f'<p class="insights-warning">{n_failed} traces had a label, summary, match, or dimension error.</p>'
        )
    if run.warnings:
        items = ''.join(f'<li>{esc(warning)}</li>' for warning in run.warnings)
        banners.append(f'<section class="insights-warning" role="status"><b>Run warnings</b><ul>{items}</ul></section>')
    return ''.join(banners)


def tabs(run: InsightsRun, active: str) -> str:
    links = []
    for name in TABS:
        cls = 'insights-tab active' if name == active else 'insights-tab'
        href = f'/insights/{quote(run.run_id, safe="")}/tab/{name}'
        links.append(
            f'<a class="{cls}" href="{href}" hx-get="{href}" hx-target="#insights-content" hx-push-url="true">{TAB_LABELS[name]}</a>'
        )
    return f'<nav class="insights-tabs" aria-label="Insights views">{"".join(links)}</nav>'


def _bar(size: int, total: int) -> str:
    pct = min(100, round(size / total * 100)) if total else 0
    return f'<span class="insights-bar"><i style="width:{pct}%"></i></span>'


def dimensions(run: InsightsRun, selected_cluster: str | None = None) -> str:
    if not run.dimensions:
        return '<section class="insights-empty-state"><h3>No dimensions available</h3><p>This run did not produce any cluster dimensions.</p></section>'
    controls = ''.join(f'<option value="{esc(name)}">{esc(name)}</option>' for name in run.dimensions)
    parts = [
        (
            '<div class="insights-dimension-control"><label for="insights-dimension">Dimension</label>'
            f'<select id="insights-dimension" name="dimension" onchange="document.querySelectorAll(\'.insights-dimension\').forEach(function(s){{s.hidden=s.dataset.dimension!==this.value}}.bind(this))">{controls}</select>'
            '<div class="insights-view-toggle" role="group" aria-label="Dimension view">'
            '<button type="button" class="active" data-insights-view="tree">Tree</button>'
            '<button type="button" data-insights-view="map">Map</button></div></div>'
        )
    ]
    for index, (name, dimension) in enumerate(run.dimensions.items()):
        total = max(1, len(run.traces))
        top_clusters = [cluster for cluster in dimension.clusters if cluster.level == 'top']
        rows = []
        for top in top_clusters:
            rows.append(
                f'<div class="insights-cluster-row top"><span class="insights-tree-marker">▾</span><b>{esc(top.name)}</b>'
                f'<span class="insights-count">{top.size} · {round(top.size / total * 100)}%</span>{_bar(top.size, total)}</div>'
            )
            children = [c for c in dimension.clusters if c.level == 'base' and c.parent_id == top.id]
            for child in children:
                href = f'/insights/{quote(run.run_id, safe="")}/cluster/{quote(child.id, safe="")}'
                selected = ' selected' if child.id == selected_cluster else ''
                rows.append(
                    f'<button class="insights-cluster-row child{selected}" hx-get="{href}" hx-target="#insights-cluster-detail" '
                    f'aria-label="Show cluster {esc(child.name)}"><span class="insights-tree-marker">↳</span><span>{esc(child.name)}</span>'
                    f'<span class="insights-count">{child.size}</span>{_bar(child.size, total)}</button>'
                )
        if not rows:
            rows.append('<p class="insights-empty">No clusters in this dimension.</p>')
        if dimension.n_noise:
            rows.append(
                f'<div class="insights-cluster-row muted"><span>Outliers (noise)</span><span class="insights-count">{dimension.n_noise}</span></div>'
            )
        if dimension.n_failed:
            rows.append(
                f'<div class="insights-cluster-row failed"><span>Failed / unclassified</span><span class="insights-count">{dimension.n_failed}</span></div>'
            )
        if dimension.n_no_signal:
            rows.append(
                f'<div class="insights-cluster-row muted"><span>No signal</span><span class="insights-count">{dimension.n_no_signal}</span></div>'
            )
        hidden = ' hidden' if index else ''
        tree = f'<div class="insights-tree" data-insights-mode="tree">{"".join(rows)}</div>'
        map_html = _dimension_map(run, name)
        parts.append(
            f'<section class="insights-dimension" data-dimension="{esc(name)}"{hidden}><h3>{esc(name.title())}</h3>{tree}{map_html}</section>'
        )
    side = '<section id="insights-cluster-detail" class="insights-detail"><p class="insights-empty">Select a cluster to see its details and example traces.</p></section>'
    return f'<div class="insights-dimensions"><div class="insights-tree-column">{"".join(parts)}</div>{side}</div>'


def _find_cluster(run: InsightsRun, cluster_id: str) -> tuple[str, Cluster] | None:
    for dimension_name, dimension in run.dimensions.items():
        for cluster in dimension.clusters:
            if cluster.id == cluster_id:
                return dimension_name, cluster
    return None


def cluster_detail(run: InsightsRun, cluster_id: str) -> str:
    found = _find_cluster(run, cluster_id)
    if found is None:
        return '<section class="insights-detail"><p class="insights-empty">Cluster not found.</p></section>'
    dimension, cluster = found
    members = [trace for trace in run.traces if trace.trace_id in set(cluster.trace_ids)]
    label_rows = []
    for label_name in run.labels:
        values: dict[str, int] = {}
        for trace in members:
            answer = trace.labels.get(label_name)
            if answer is not None and answer.error is None and answer.value is not None:
                key = str(answer.value)
                values[key] = values.get(key, 0) + 1
        if values:
            summary = ', '.join(f'{esc(value)} {count}/{len(members)}' for value, count in sorted(values.items()))
            label_rows.append(
                f'<div class="insights-label-summary"><b>{esc(label_name)}</b><span>{summary}</span></div>'
            )
    examples = []
    by_id = {trace.trace_id: trace for trace in members}
    for trace_id in cluster.example_trace_ids:
        trace = by_id.get(trace_id)
        if trace is None:
            continue
        excerpt = trace.summary.summary if trace.summary else ''
        link = trace_link_button(trace_span_url(trace.trace_id, trace.span_id), 'Open in Orq ↗')
        examples.append(
            f'<div class="insights-example"><code>{esc(trace.trace_id)}</code><span>{esc(excerpt)}</span>{link}</div>'
        )
    traces_href = (
        f'/insights/{quote(run.run_id, safe="")}/tab/traces?dimension={quote(dimension)}&cluster={quote(cluster.id)}'
    )
    label_html = ''.join(label_rows) or '<p class="insights-empty">No label answers for this cluster.</p>'
    example_html = ''.join(examples) or '<p class="insights-empty">No example traces available.</p>'
    return (
        '<section class="insights-detail"><div class="insights-detail-kicker">Cluster · '
        f'{esc(dimension)}</div><h3>{esc(cluster.name)}</h3><p>{esc(cluster.description) or "No description available."}</p>'
        f'<div class="insights-detail-kicker">Labels in this cluster</div>{label_html}'
        f'<div class="insights-detail-kicker">Example traces</div>{example_html}'
        f'<a class="insights-action" href="{traces_href}" hx-get="{traces_href}" hx-target="#insights-content" hx-push-url="true">Show all {cluster.size} in Traces →</a></section>'
    )


def labels(run: InsightsRun) -> str:
    if not run.labels:
        return '<section class="insights-empty-state"><h3>No labels in this run</h3><p>This run has no classifier labels to review.</p></section>'
    cards = []
    for name, result in run.labels.items():
        total = max(1, sum(result.counts.values()))
        rows = []
        for value, count in result.counts.items():
            pct = round(count / total * 100)
            href = f'/insights/{quote(run.run_id, safe="")}/tab/traces?label={quote(name)}&value={quote(value)}'
            rows.append(
                f'<a class="insights-label-row" href="{href}" hx-get="{href}" hx-target="#insights-content" hx-push-url="true"><span>{esc(value)}</span><b>{pct}%</b>{_bar(count, total)}</a>'
            )
        row_html = ''.join(rows) or '<p class="insights-empty">No label values were recorded.</p>'
        confidence = f'{result.mean_confidence:.2f}' if result.mean_confidence is not None else 'unavailable'
        card = (
            f'<article class="insights-label-card"><div class="insights-card-head"><h3>{esc(name)}</h3><span>{esc(result.spec.kind)}</span></div>'
            f'<p class="insights-label-instructions">{esc(result.spec.instructions)}</p>{row_html}'
            f'<p class="insights-muted">Mean confidence: {confidence}</p>'
        )
        cards.append(
            card
            + f'<p class="insights-muted">{result.n_low_confidence} below confidence 0.6 · {result.n_failed} failed</p></article>'
        )
    return f'<div class="insights-label-grid">{"".join(cards)}</div>'


def _trace_row(run: InsightsRun, trace: TraceInsight) -> str:
    cells = [
        f'<td class="id">{esc(trace.trace_id)}</td>',
        f'<td>{esc(trace.timestamp.strftime("%Y-%m-%d %H:%M"))}</td>',
    ]
    for name in run.config.dimensions:
        assignment = trace.assignments.get(name)
        cluster = assignment.base if assignment else '—'
        cells.append(f'<td>{esc(cluster)}</td>')
    for name in run.labels:
        answer = trace.labels.get(name)
        if answer is None or answer.error:
            value = 'failed'
        elif answer.value is None:
            value = '—'
        else:
            value = str(answer.value)
            if answer.confidence is not None:
                value += f' · {answer.confidence:.2f}'
        cells.append(f'<td>{esc(value)}</td>')
    link = trace_link_button(trace_span_url(trace.trace_id, trace.span_id), '↗ Orq')
    cells.append(f'<td>{link}</td>')
    return f'<tr>{"".join(cells)}</tr>'


def traces(
    run: InsightsRun,
    *,
    dimension: str | None = None,
    cluster: str | None = None,
    label: str | None = None,
    value: str | None = None,
    row: str | None = None,
    row_value: str | None = None,
    column: str | None = None,
    column_value: str | None = None,
) -> str:
    active_filters: list[tuple[str, str]] = []
    if dimension and cluster:
        selected = (
            next(
                (item for item in run.dimensions.get(dimension, ()).clusters if item.id == cluster),
                None,
            )
            if dimension in run.dimensions
            else None
        )
        label_text = selected.name if selected is not None else cluster
        active_filters.append((f'{dimension} = {label_text}', 'cluster'))
    if label is not None:
        active_filters.append((f'{label} = {value}' if value is not None else label, 'label'))
    if row and row_value is not None and column and column_value is not None:
        active_filters.append((f'{row_value} x {column_value}', 'crosstab'))
    chips = []
    for text, field in active_filters:
        remaining = {
            'dimension': dimension,
            'cluster': cluster,
            'label': label,
            'value': value,
        }
        if field == 'cluster':
            remaining.pop('dimension', None)
            remaining.pop('cluster', None)
        elif field == 'label':
            remaining.pop('label', None)
            remaining.pop('value', None)
        else:
            remaining = {}
        remaining = {key: item for key, item in remaining.items() if item is not None}
        href = f'/insights/{quote(run.run_id, safe="")}/tab/traces'
        if remaining:
            href += '?' + urlencode(remaining)
        chips.append(
            f'<a class="insights-filter-chip" href="{esc(href)}" hx-get="{esc(href)}" '
            f'hx-target="#insights-content" hx-push-url="true">{esc(text)} <span aria-label="Clear filter">&#215;</span></a>'
        )
    if active_filters:
        clear_href = f'/insights/{quote(run.run_id, safe="")}/tab/traces'
        active_filter_bar = (
            f'<div class="insights-filter-bar"><span class="insights-group-label">Filters</span>{"".join(chips)}'
            f'<a class="insights-clear-filters" href="{esc(clear_href)}" hx-get="{esc(clear_href)}" '
            'hx-target="#insights-content" hx-push-url="true">Clear filters</a></div>'
        )
    else:
        active_filter_bar = ''
    filtered = list(run.traces)
    if dimension and cluster:
        filtered = [
            trace
            for trace in filtered
            if (assignment := trace.assignments.get(dimension)) is not None
            and cluster in {assignment.base, assignment.top}
        ]
    if label is not None:
        filtered = [
            trace
            for trace in filtered
            if (answer := trace.labels.get(label)) is not None and (value is None or str(answer.value) == value)
        ]
    if row and row_value is not None:
        filtered = [trace for trace in filtered if _cross_value(run, trace, row) == row_value]
    if column and column_value is not None:
        filtered = [trace for trace in filtered if _cross_value(run, trace, column) == column_value]
    if not filtered:
        return (
            f'{active_filter_bar}<section class="insights-empty-state"><h3>No traces match these filters</h3>'
            '<p>Clear a filter or choose a different cluster or label value.</p></section>'
        )
    headings = ['Trace', 'Time', *run.config.dimensions, *run.labels, 'Orq']
    head = ''.join(f'<th>{esc(item)}</th>' for item in headings)
    return (
        f'{active_filter_bar}<div class="insights-table-wrap"><table class="insights-table"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(_trace_row(run, trace) for trace in filtered)}</tbody></table></div>'
    )


def _cross_value(run: InsightsRun, trace: TraceInsight, field: str) -> str | None:
    """Return a top-level dimension or successful label value for a crosstab field."""
    kind, _, name = field.partition(':')
    if kind == 'dimension':
        assignment = trace.assignments.get(name)
        if assignment is None:
            return None
        dimension = run.dimensions.get(name)
        cluster = (
            next(
                (item for item in dimension.clusters if item.id == assignment.top),
                None,
            )
            if dimension
            else None
        )
        return cluster.name if cluster else assignment.top
    if kind == 'label':
        answer = trace.labels.get(name)
        return str(answer.value) if answer is not None and answer.error is None and answer.value is not None else None
    return None


def _cross_fields(run: InsightsRun) -> list[tuple[str, str]]:
    return [
        *((f'dimension:{name}', f'{name.title()} cluster') for name in run.dimensions),
        *((f'label:{name}', name) for name in run.labels),
    ]


def crosstab(run: InsightsRun, query: dict[str, str]) -> str:
    fields = _cross_fields(run)
    if len(fields) < 2:
        return '<section class="insights-empty-state"><h3>Crosstab unavailable</h3><p>Add at least two dimensions or labels to compare trace counts.</p></section>'
    default_row, default_column = fields[0][0], fields[1][0]
    row = query.get('row') if query.get('row') in dict(fields) else default_row
    fallback_column = next((key for key, _ in fields if key != row), default_column)
    column = (
        query.get('column') if query.get('column') in dict(fields) and query.get('column') != row else fallback_column
    )
    row_name, column_name = dict(fields)[row], dict(fields)[column]
    rows = sorted({v for trace in run.traces if (v := _cross_value(run, trace, row)) is not None})
    columns = sorted({v for trace in run.traces if (v := _cross_value(run, trace, column)) is not None})
    if not rows or not columns:
        return '<section class="insights-empty-state"><h3>No crosstab values</h3><p>No traces have readable values for both selected fields.</p></section>'
    counts: dict[tuple[str, str], int] = {}
    for trace in run.traces:
        y = _cross_value(run, trace, row)
        x = _cross_value(run, trace, column)
        if x is not None and y is not None:
            counts[y, x] = counts.get((y, x), 0) + 1
    maximum = max(counts.values(), default=1)
    cell_rows = []
    run_url = quote(run.run_id, safe='')
    for y in rows:
        for x in columns:
            count = counts.get((y, x), 0)
            href = f'/insights/{run_url}/tab/traces?' + urlencode({
                'row': row,
                'row_value': y,
                'column': column,
                'column_value': x,
            })
            cell_rows.append({'x': x, 'y': y, 'count': count, 'text': str(count), 'href': href})
    spec = {
        'data': {'values': cell_rows},
        'layer': [
            {
                'mark': {'type': 'rect', 'tooltip': True, 'cursor': 'pointer'},
                'encoding': {
                    'x': {'field': 'x', 'type': 'nominal', 'title': column_name},
                    'y': {'field': 'y', 'type': 'nominal', 'title': row_name},
                    'color': {
                        'field': 'count',
                        'type': 'quantitative',
                        'scale': {'domain': [0, maximum], 'range': [color for _, color in ORQ_SCALE_HEAT]},
                        'legend': {'title': 'Trace count'},
                    },
                    'href': {'field': 'href', 'type': 'nominal'},
                },
            },
            {
                'mark': {'type': 'text', 'fontSize': 11, 'color': COLORS['ink_700']},
                'encoding': {
                    'x': {'field': 'x', 'type': 'nominal'},
                    'y': {'field': 'y', 'type': 'nominal'},
                    'text': {'field': 'text', 'type': 'nominal'},
                },
            },
        ],
        'width': {'step': 100},
        'height': {'step': 32},
    }
    row_options = ''.join(
        f'<option value="{esc(key)}"{" selected" if key == row else ""}>{esc(label)}</option>' for key, label in fields
    )
    col_options = ''.join(
        f'<option value="{esc(key)}"{" selected" if key == column else ""}>{esc(label)}</option>'
        for key, label in fields
        if key != row
    )
    form = (
        f'<form class="insights-chart-controls" hx-get="/insights/{run_url}/tab/crosstab" hx-target="#insights-content" hx-push-url="true">'
        f'<label>Rows <select name="row">{row_options}</select></label>'
        f'<label>Columns <select name="column">{col_options}</select></label>'
        '<button type="submit" class="insights-action">Update</button></form>'
    )
    return f'{form}{render_embed(spec, "insights-crosstab")}'


def priority_matrix(run: InsightsRun) -> str:
    points = run.priority
    if not points:
        reason = run.priority_reason or 'Priority data is not available for this run.'
        return (
            f'<section class="insights-empty-state"><h3>Priority matrix unavailable</h3><p>{esc(reason)}</p></section>'
        )
    max_volume = max(point.volume for point in points)
    volumes = sorted(point.volume for point in points)
    median_volume = (
        volumes[len(volumes) // 2]
        if len(volumes) % 2
        else sum(volumes[len(volumes) // 2 - 1 : len(volumes) // 2 + 1]) / 2
    )
    x_max = max_volume * 1.1
    x_left = median_volume / 2
    x_right = (median_volume + x_max) / 2
    labels = [
        {'x': x_right, 'y': 0.25, 'text': 'fix first'},
        {'x': x_right, 'y': 0.78, 'text': 'watch'},
        {'x': x_left, 'y': 0.78, 'text': 'fine'},
        {'x': x_left, 'y': 0.25, 'text': 'niche'},
    ]
    spec = {
        'data': {
            'values': [
                {
                    'name': p.name,
                    'volume': p.volume,
                    'satisfaction': p.mean_satisfaction,
                    'error_share': p.error_share,
                    'cluster_id': p.cluster_id,
                }
                for p in points
            ]
        },
        'layer': [
            {
                'mark': {'type': 'circle', 'opacity': 0.82},
                'encoding': {
                    'x': {
                        'field': 'volume',
                        'type': 'quantitative',
                        'title': 'Trace volume',
                        'scale': {'domain': [0, x_max]},
                    },
                    'y': {
                        'field': 'satisfaction',
                        'type': 'quantitative',
                        'title': 'Mean satisfaction',
                        'scale': {'domain': [0, 1]},
                    },
                    'size': {
                        'field': 'error_share',
                        'type': 'quantitative',
                        'title': 'Error share',
                        'scale': {'range': [60, 900]},
                    },
                    'color': {
                        'field': 'error_share',
                        'type': 'quantitative',
                        'scale': {'domain': [0, 1], 'range': [color for _, color in ORQ_SCALE_HEAT]},
                        'legend': {'title': 'Error share'},
                    },
                    'tooltip': [
                        {'field': 'name'},
                        {'field': 'volume'},
                        {'field': 'satisfaction'},
                        {'field': 'error_share'},
                    ],
                },
            },
            {
                'data': {'values': [{'volume': median_volume}]},
                'mark': {'type': 'rule', 'strokeDash': [5, 4], 'color': COLORS['ink_700']},
                'encoding': {'x': {'field': 'volume', 'type': 'quantitative'}},
            },
            {
                'data': {'values': [{'satisfaction': 0.5}]},
                'mark': {'type': 'rule', 'strokeDash': [5, 4], 'color': COLORS['ink_700']},
                'encoding': {'y': {'field': 'satisfaction', 'type': 'quantitative'}},
            },
            {
                'data': {'values': labels},
                'mark': {'type': 'text', 'fontSize': 12, 'fontWeight': 'bold', 'color': COLORS['teal_400']},
                'encoding': {
                    'x': {'field': 'x', 'type': 'quantitative'},
                    'y': {'field': 'y', 'type': 'quantitative'},
                    'text': {'field': 'text'},
                },
            },
        ],
        'width': 'container',
        'height': 360,
    }
    return render_embed(spec, 'insights-priority')


def _map_empty(reason: str) -> str:
    return f'<div class="insights-map-empty"><h4>Map unavailable: {esc(reason)}</h4></div>'


def _dimension_map(
    run: InsightsRun,
    dimension_name: str,
    *,
    initially_visible: bool = False,
    include_detail: bool = False,
) -> str:
    hidden = '' if initially_visible else ' hidden'
    dimension = run.dimensions.get(dimension_name)
    if dimension is None:
        return f'<div data-insights-mode="map"{hidden}>{_map_empty("This dimension is not part of the run.")}</div>'
    if not any(dimension_name in trace.coords for trace in run.traces):
        reason = (
            dimension.warnings[0] if dimension.warnings else 'UMAP coordinates were not produced for this dimension.'
        )
        return f'<div data-insights-mode="map"{hidden}>{_map_empty(reason)}</div>'
    color_options = ['<option value="cluster">Clusters</option>']
    color_options.extend(f'<option value="label:{esc(name)}">{esc(name)}</option>' for name in run.labels)
    chart = (
        f'<div id="insights-map-{quote(dimension_name, safe="")}" class="insights-map-chart" '
        f'data-run-id="{esc(run.run_id)}" data-map-dimension="{esc(dimension_name)}" data-map-url="/insights/{quote(run.run_id, safe="")}/map.json?dimension={quote(dimension_name, safe="")}&amp;color_by=cluster"></div>'
    )
    detail = (
        '<section class="insights-detail" data-map-detail><p class="insights-empty">Select a point to see its cluster details and example traces.</p></section>'
        if include_detail
        else ''
    )
    layout = f'<div class="insights-map-layout">{chart}{detail}</div>' if include_detail else chart
    return (
        f'<div class="insights-map-view" data-insights-mode="map"{hidden}><div class="insights-map-toolbar">'
        f'<label>Colour by <select data-map-color data-dimension="{esc(dimension_name)}">{"".join(color_options)}</select></label>'
        f'<span>{len(run.traces)} traces · UMAP 3D</span></div>'
        f'{layout}</div>'
    )


def map_payload(run: InsightsRun, dimension_name: str, color_by: str = 'cluster') -> dict[str, object]:
    """Build the browser map's coordinates, cluster palette, and optional label colours."""
    dimension = run.dimensions.get(dimension_name)
    if dimension is None:
        return {
            'points': [],
            'legend': [],
            'color_scale': None,
            'color_mode': 'cluster',
            'grid_color': COLORS['sand_400'],
            'background_color': COLORS['sand_100'],
        }
    base_clusters = [item for item in dimension.clusters if item.level == 'base']
    cluster_by_id = {item.id: item for item in base_clusters}
    label_name = color_by.removeprefix('label:') if color_by.startswith('label:') else None
    label_spec = next((item.spec for key, item in run.labels.items() if key == label_name), None)
    is_continuous = label_spec is not None and label_spec.kind == 'score'
    categories = (
        sorted({
            str(trace.labels[label_name].value)
            for trace in run.traces
            if label_name in trace.labels
            and trace.labels[label_name].error is None
            and trace.labels[label_name].value is not None
        })
        if label_name and not is_continuous
        else []
    )
    color_scale: list[list[float | str]] | None = (
        (ORQ_SCALE_GOOD_BAD if 'satisfaction' in (label_name or '') else ORQ_SCALE_HEAT) if is_continuous else None
    )
    points: list[dict[str, object]] = []
    legend: dict[str, dict[str, str]] = {}
    for trace in run.traces:
        coords = trace.coords.get(dimension_name)
        if coords is None:
            continue
        assignment = trace.assignments.get(dimension_name)
        cluster_id = assignment.base if assignment is not None else 'noise'
        cluster = cluster_by_id.get(cluster_id)
        cluster_index = next((index for index, item in enumerate(base_clusters) if item.id == cluster_id), -1)
        color = _cluster_color(cluster_index) if cluster_index >= 0 else COLORS['sand_400']
        symbol = 'diamond' if cluster_index >= 8 else 'circle'
        color_value: object = cluster.name if cluster else 'noise'
        marker_color = color
        label_category: str | None = None
        if label_name:
            answer = trace.labels.get(label_name)
            if answer is None or answer.error is not None or answer.value is None:
                continue
            if is_continuous:
                if not isinstance(answer.value, (int, float)) or isinstance(answer.value, bool):
                    continue
                value = min(1.0, max(0.0, float(answer.value)))
                color_value = 1.0 - value if 'satisfaction' in label_name else value
                marker_color = ''
            else:
                category = str(answer.value)
                label_category = category
                category_index = categories.index(category)
                color_value = category_index
                marker_color = QUALITATIVE[category_index % len(QUALITATIVE)]
        point = {
            'trace_id': trace.trace_id,
            'x': coords[0],
            'y': coords[1],
            'z': coords[2],
            'cluster_id': cluster_id,
            'cluster_name': cluster.name if cluster else 'noise',
            'color_value': color_value,
            'color': marker_color,
            'symbol': symbol,
        }
        if label_category is not None:
            point['label_value'] = label_category
        points.append(point)
        if not label_name:
            legend.setdefault(
                cluster_id,
                {
                    'cluster_id': cluster_id,
                    'name': cluster.name if cluster else 'noise',
                    'color': color,
                    'symbol': symbol,
                },
            )
        elif label_category is not None:
            legend.setdefault(
                label_category,
                {'cluster_id': label_category, 'name': label_category, 'color': marker_color, 'symbol': symbol},
            )
    if is_continuous:
        legend_items: list[dict[str, str]] = []
    else:
        legend_items = list(legend.values())
    return {
        'points': points,
        'legend': legend_items,
        'color_scale': color_scale,
        'color_mode': 'cluster' if not label_name else 'continuous' if is_continuous else 'category',
        'grid_color': COLORS['sand_400'],
        'background_color': COLORS['sand_100'],
    }


def _cluster_color(index: int) -> str:
    base = QUALITATIVE[index % len(QUALITATIVE)]
    if index < len(QUALITATIVE):
        return base
    target = COLORS['sand_100']
    blend = 0.45
    channels = [
        round(int(base[offset : offset + 2], 16) * (1 - blend) + int(target[offset : offset + 2], 16) * blend)
        for offset in (1, 3, 5)
    ]
    return '#' + ''.join(f'{channel:02x}' for channel in channels)


def tab_content(run: InsightsRun, tab: str, *, query: dict[str, str] | None = None) -> str:
    query = query or {}
    if tab == 'dimensions':
        return dimensions(run)
    if tab == 'labels':
        return labels(run)
    if tab == 'traces':
        return traces(
            run,
            dimension=query.get('dimension'),
            cluster=query.get('cluster'),
            label=query.get('label'),
            value=query.get('value'),
            row=query.get('row'),
            row_value=query.get('row_value'),
            column=query.get('column'),
            column_value=query.get('column_value'),
        )
    if tab == 'map':
        name = query.get('dimension') or next(iter(run.dimensions), '')
        return (
            _dimension_map(run, name, initially_visible=True, include_detail=True)
            if name
            else _map_empty('No dimensions are available for mapping.')
        )
    if tab == 'crosstab':
        return crosstab(run, query or {})
    return priority_matrix(run)


def full_page(
    run: InsightsRun,
    entries: list[tuple[str, str, str]],
    active_tab: str = 'dimensions',
    *,
    query: dict[str, str] | None = None,
) -> str:
    body = (
        '<div class="insights-layout">'
        f'{run_rail(entries, run.run_id)}'
        '<div class="insights-main">'
        f'{header(run)}{failures(run)}{tabs(run, active_tab)}'
        f'<div id="insights-content">{tab_content(run, active_tab, query=query)}</div>'
        '</div></div>'
    )
    return page('Insights', body + '<script src="/static/plotly-gl3d.min.js" defer></script>', active_nav='insights')


def landing(entries: list[tuple[str, str, str]]) -> str:
    body = (
        '<div class="insights-layout">'
        f'{run_rail(entries, None)}'
        '<div class="insights-main"><section class="insights-empty-state"><h2>Insights</h2>'
        '<p>Run an Insights analysis from the Python API or command line, then return here to review its clusters, labels and traces.</p>'
        '<p>This page is for reviewing completed runs. It does not start an analysis.</p></section></div></div>'
    )
    return page('Insights', body, active_nav='insights')


def running_page(run_id: str, label: str, entries: list[tuple[str, str, str]]) -> str:
    body = (
        '<div class="insights-layout">'
        f'{run_rail(entries, run_id)}'
        f'<div class="insights-main"><section class="insights-empty-state"><h2>{esc(label)}</h2>'
        '<p>This Insights run is still running. Its results will appear when the run file is written.</p></section></div></div>'
    )
    return page('Insights', body, active_nav='insights')


def unreadable_page(run_id: str, error: str, entries: list[tuple[str, str, str]]) -> str:
    body = (
        '<div class="insights-layout">'
        f'{run_rail(entries, run_id)}'
        '<div class="insights-main"><section class="insights-error"><h2>Run file is unreadable</h2>'
        f'<p>{esc(error)}</p></section></div></div>'
    )
    return page('Insights', body, active_nav='insights')
