"""HTML rendering for the read-only Insights run review page."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import quote

from evaluatorq.common.reports import esc
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
    request = population.get('request')
    request = request if isinstance(request, dict) else {}
    query = population.get('query')
    facets = request.get('facets', {}) if isinstance(request.get('facets', {}), dict) else {}
    chips = [_chip('query', query)]
    for name, values in facets.items():
        if isinstance(values, (list, tuple, set, frozenset)):
            chips.extend(_chip(name, value) for value in values)
        elif values:
            chips.append(_chip(name, values))
    chips.extend((
        _chip('window', request.get('window_days') or request.get('window')),
        _chip('limit', request.get('limit')),
    ))
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
            f'<p class="insights-warning">{n_failed} traces could not be fully classified or summarized.</p>'
        )
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
            '<span>Tree view</span></div>'
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
        parts.append(
            f'<section class="insights-dimension" data-dimension="{esc(name)}"{hidden}><h3>{esc(name.title())}</h3><div class="insights-tree">{"".join(rows)}</div></section>'
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
            href = f'/insights/{quote(run.run_id, safe="")}/traces?label={quote(name)}&value={quote(value)}'
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
) -> str:
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
    if not filtered:
        return '<section class="insights-empty-state"><h3>No traces match these filters</h3><p>Clear a filter or choose a different cluster or label value.</p></section>'
    headings = ['Trace', 'Time', *run.config.dimensions, *run.labels, 'Orq']
    head = ''.join(f'<th>{esc(item)}</th>' for item in headings)
    return f'<div class="insights-table-wrap"><table class="insights-table"><thead><tr>{head}</tr></thead><tbody>{"".join(_trace_row(run, trace) for trace in filtered)}</tbody></table></div>'


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
        )
    if tab == 'map':
        return '<section class="insights-empty-state"><h3>Map view is being prepared</h3><p>Map rendering will be available in a later dashboard update.</p></section>'
    if tab == 'crosstab':
        return '<section class="insights-empty-state"><h3>Crosstab view is being prepared</h3><p>Cross-tabulated counts will be available in a later dashboard update.</p></section>'
    return f'<section class="insights-empty-state"><h3>Priority matrix unavailable</h3><p>{esc(run.priority_reason or "Priority data is not available for this run.")}</p></section>'


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
    return page('Insights', body, active_nav='insights')


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
