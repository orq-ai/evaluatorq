"""HTML rendering for the read-only Insights run review page."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from math import isfinite
from typing import TYPE_CHECKING
from urllib.parse import quote, urlencode

from evaluatorq.common.reports import esc
from evaluatorq.common.reports.palette import COLORS, ORQ_SCALE_GOOD_BAD, ORQ_SCALE_HEAT, QUALITATIVE
from evaluatorq.common.reports.vega import render_embed
from evaluatorq.common.structured_output import sum_structured_usage
from evaluatorq.dashboard.shell import page
from evaluatorq.dashboard.trace_links import trace_link_button, trace_span_url
from evaluatorq.insights.models import label_key, label_order
from evaluatorq.insights.population import describe_projection_coverage

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from evaluatorq.contracts import RunManifest
    from evaluatorq.insights.models import Cluster, InsightsRun, TraceInsight

TABS = ('dimensions', 'labels', 'crosstab', 'traces', 'priority', 'map')
TAB_LABELS = {
    'dimensions': 'Dimensions',
    'labels': 'Labels',
    'crosstab': 'Crosstab',
    'traces': 'Traces',
    'priority': 'Priority matrix',
    'map': '3D Map',
}

_SIGNALS_STYLESHEET = '<link rel="stylesheet" href="/static/insights-signals.css">\n'
_SIGNALS_SCRIPT = '<script src="/static/insights-signals.js" defer></script>\n'


def _stage_name(manifest: RunManifest) -> str:
    return manifest.stage_labels.get(manifest.stage or '', (manifest.stage or 'Starting').replace('_', ' ').title())


def _stage_rows(manifest: RunManifest) -> list[tuple[str, str, str]]:
    records = {stage.name: stage for stage in manifest.stages}
    planned = [*manifest.planned_stages, *(name for name in records if name not in manifest.planned_stages)]
    if manifest.status == 'error' and manifest.stage and manifest.stage not in planned:
        planned.append(manifest.stage)
    rows = []
    for name in planned:
        record = records.get(name)
        status = (
            record.status.value
            if record is not None
            else 'error'
            if manifest.status == 'error' and name == manifest.stage
            else 'skipped'
            if manifest.status != 'running'
            else 'pending'
        )
        label = manifest.stage_labels.get(name, name.replace('dimension:', 'Cluster ').replace('_', ' ').title())
        rows.append((name, label, status))
    return rows


def _overview_stages(manifest: RunManifest | None) -> str:
    if manifest is None:
        return '<span class="insights-overview-no-stages">Stage history unavailable</span>'
    rows = _stage_rows(manifest)
    if not rows:
        return '<span class="insights-overview-no-stages">No stages recorded</span>'
    marks = ''.join(
        f'<li class="{esc(status)}" title="{esc(label)}: {esc(status)}" aria-label="{esc(label)}: {esc(status)}"></li>'
        for _, label, status in rows
    )
    done = sum(status == 'completed' for _, _, status in rows)
    return (
        '<span class="insights-overview-stage-wrap"><span>Stages</span>'
        f'<ol class="insights-overview-stages" aria-label="Run stages">{marks}</ol>'
        f'<span>{done}/{len(rows)}</span></span>'
    )


def run_overview(
    entries: list[tuple[str, str, str]], runs: dict[str, InsightsRun], manifests: dict[str, RunManifest]
) -> str:
    """Render the saved runs as one navigable overview."""
    if not entries:
        return (
            '<section class="insights-overview-empty"><h2>No Insights runs yet</h2>'
            '<p>Start a run to find patterns across traces. Its progress and results will appear here.</p>'
            '<a class="insights-overview-new" href="/insights/new">+ New Run</a></section>'
        )
    rows = []
    for run_id, label, status in entries:
        run = runs.get(run_id)
        manifest = manifests.get(run_id)
        stage = _stage_name(manifest) if manifest is not None and status == 'running' else ''
        started_at = run.created_at if run is not None else manifest.started_at if manifest is not None else None
        created = started_at.strftime('%d %b %Y, %H:%M UTC') if started_at is not None else ''
        count = run.counts.get('n_traces', len(run.traces)) if run is not None else None
        count_html = str(count) if count is not None else '—'
        dimensions = ', '.join(name.title() for name in run.dimensions) if run is not None else '—'
        if run is not None and not run.dimensions:
            dimensions = 'None'
        status_class = status if status in {'running', 'completed', 'error', 'cancelled', 'unreadable'} else 'other'
        rows.append(
            f'<a class="insights-overview-row" href="/insights/{quote(run_id, safe="")}">'
            f'<span class="insights-overview-name"><strong>{esc(label)}</strong>'
            f'<small>{esc(stage or created or "Run file could not be read")}</small></span>'
            f'{_overview_stages(manifest)}'
            f'<span class="insights-overview-status {status_class}">{esc(status)}</span>'
            f'<span class="insights-overview-number">{esc(count_html)}</span>'
            f'<span class="insights-overview-dimensions">{esc(dimensions)}</span>'
            '<span class="insights-overview-arrow" aria-hidden="true">→</span></a>'
        )
    return (
        '<section class="insights-overview"><div class="insights-overview-head">'
        '<div><h2>Recent runs</h2><p>Explore clusters, labels, traces, and 3D maps from each run.</p></div>'
        f'<span>{len(entries)} run{"s" if len(entries) != 1 else ""}</span></div>'
        '<div class="insights-overview-table"><div class="insights-overview-columns">'
        '<span>Run</span><span>Stages</span><span>Status</span><span>Traces</span><span>Dimensions</span><span></span></div>'
        f'{"".join(rows)}</div></section>'
    )


def progress(manifest: RunManifest | None) -> str:
    """Render the persisted stage plan and actual stage outcomes."""
    if manifest is None:
        return ''
    records = {stage.name: stage for stage in manifest.stages}
    rows = _stage_rows(manifest)
    items = []
    for name, label, status in rows:
        record = records.get(name)
        short_label = (
            name.split(':', 1)[1].title()
            if name.startswith('dimension:')
            else {
                'Load local traces': 'Load local',
                'Load Finder matches': 'Load Finder',
                'Find matching traces': 'Find matches',
                'Load recent traces': 'Load recent',
                'Classify traces': 'Classify',
                'Match traces': 'Match',
                'Match and classify traces': 'Match + classify',
                'Prepare traces': 'Prepare',
                'Summarize traces': 'Summarize',
                'Build priority matrix': 'Priority',
                'Check priority matrix': 'Priority',
                'Save run': 'Save',
            }.get(label, label)
        )
        flag = ''
        if status == 'running':
            elapsed = _elapsed(record.started_at, None) if record is not None else ''
            flag = f'<span class="insights-stage-flag">Running{f" · {elapsed}" if elapsed else ""}</span>'
        elif status == 'error':
            flag = '<span class="insights-stage-flag">Failed</span>'
        stage_count = (
            f'<small>{record.completed}/{record.total}</small>'
            if record is not None and record.completed is not None and record.total is not None
            else ''
        )
        items.append(
            f'<li class="insights-stage {esc(status)}" title="{esc(label)}: {esc(status)}">{flag}'
            f'<span class="insights-stage-mark" aria-hidden="true"></span>'
            f'<span class="insights-stage-label">{esc(short_label)}</span>{stage_count}'
            f'<span class="sr-only">{esc(status)}</span></li>'
        )
    done = sum(status == 'completed' for _, _, status in rows)
    current = _stage_name(manifest) if manifest.status == 'running' else manifest.status.value.title()
    count = f' · {done} of {len(rows)}' if rows else ''
    list_html = ''.join(items) or '<li class="insights-stage pending">Preparing run</li>'
    return (
        '<section class="insights-progress" aria-label="Run progress">'
        f'<div class="insights-progress-head"><h3>Run progress</h3><span>{esc(current)}{count}</span></div>'
        f'<ol>{list_html}</ol></section>'
    )


def _elapsed(start: datetime, end: datetime | None) -> str:
    seconds = int(((end or datetime.now(timezone.utc)) - start).total_seconds())
    if seconds < 0:
        return ''
    return f'{seconds // 60}m {seconds % 60:02d}s' if seconds >= 60 else f'{seconds}s'


def _chip(label: str, value: object) -> str:
    if value in (None, '', [], {}, ()):
        return ''
    return f'<span class="insights-chip"><b>{esc(label)}</b> {esc(str(value))}</span>'


def _window_chip(label: str, value: object) -> str:
    if not value:
        return ''
    try:
        date = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return _chip(label, value)
    if date.tzinfo is not None:
        date = date.astimezone(timezone.utc)
        formatted = date.strftime('%d %b %Y, %H:%M UTC')
    else:
        formatted = date.strftime('%d %b %Y, %H:%M')
    return f'<span class="insights-chip" title="{esc(str(value))}"><b>{esc(label)}</b> {esc(formatted)}</span>'


def _cost(run: InsightsRun) -> str:
    if not run.cost_by_stage:
        return ''
    total = sum_structured_usage(list(run.cost_by_stage.values()))
    if total is None or total.total_cost is None:
        return ' · cost unknown'
    text = f' · ${total.total_cost:.4f}'
    if total.cost_is_partial:
        text += f' (priced for {total.priced_calls} of {total.calls} calls)'
    # Population and filter-selection model calls never reach the usage ledger, so the total is a lower bound.
    return f'{text} · excludes trace selection'


def header(run: InsightsRun) -> str:
    population = run.population
    query = population.get('query')
    facets = population.get('facets', {})
    facets = facets if isinstance(facets, dict) else {}
    numeric = population.get('numeric', {})
    numeric = numeric if isinstance(numeric, dict) else {}
    chips = [_chip('source', 'local snapshot' if population.get('mode') == 'snapshot' else None), _chip('query', query)]
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
    chips.extend((_window_chip('from', population.get('start')), _window_chip('to', population.get('end'))))
    chips.append(_chip('limit', population.get('limit')))
    label_chips = [_chip('label', label.name) for label in run.config.labels]
    models = f'{run.config.summary_model} · {run.config.classifier_model} · {run.config.embedding_model}'
    finder = ''
    if query:
        finder = '<a class="insights-action" href="/find">Open in finder ↗</a>'
    population_chips = ''.join(chips) or '<span class="insights-muted">All traces</span>'
    label_chip_html = ''.join(label_chips)
    label_group = (
        f'<div class="insights-chip-group"><span class="insights-group-label">Labels</span>{label_chip_html}</div>'
        if label_chip_html
        else ''
    )
    run_url = quote(run.run_id, safe='')
    trace_count = run.counts.get('n_traces', len(run.traces))
    trace_text = f'{trace_count} trace' if trace_count == 1 else f'{trace_count} traces'
    return (
        '<header class="insights-header">'
        f'<div class="insights-header-copy"><h2>{esc(run.run_name)}</h2><p class="insights-subtitle">'
        f'<span class="insights-header-count">{trace_text}{esc(_cost(run))}</span>'
        f'<span>{esc(run.created_at.strftime("%d %b %Y, %H:%M UTC"))}</span></p>'
        f'<div class="insights-chip-group"><span class="insights-group-label">Population</span>{population_chips}</div>'
        f'{label_group}'
        f'<details class="insights-models"><summary>Models used</summary><span>{esc(models)}</span></details></div>'
        '<div class="insights-actions">'
        f'{finder}<a class="insights-action" href="/insights/{run_url}/export.json">Export JSON</a>'
        '</div></header>'
    )


def failures(run: InsightsRun) -> str:
    banners = []
    if 'n_projection_truncated' in run.population:
        banners.append(projection_notice(run.population))
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
        noun = 'trace' if n_failed == 1 else 'traces'
        banners.append(
            f'<p class="insights-warning">{n_failed} {noun} had a label, summary, match, or dimension error.</p>'
        )
    warnings = [
        warning
        for warning in run.warnings
        if not (
            'n_projection_truncated' in run.population
            and (
                warning.startswith('Projection budget: ')
                or (
                    warning.endswith('whole messages were omitted from classification and summaries.')
                    and ' exceeded the model projection budget; ' in warning
                )
            )
        )
    ]
    if warnings:
        items = ''.join(f'<li>{esc(warning)}</li>' for warning in warnings)
        banners.append(f'<section class="insights-warning" role="status"><b>Run warnings</b><ul>{items}</ul></section>')
    return ''.join(banners)


def projection_notice(coverage: dict[str, Any]) -> str:
    """Show budget omissions separately from all source-to-projection compression."""
    traces = int(coverage.get('n_traces', coverage.get('n_scanned', 0)))
    trimmed = int(coverage.get('n_projection_truncated', 0))
    if 'n_source_messages' not in coverage:
        return (
            '<section class="insights-projection" role="status"><b>Model input projection</b>'
            f'<p>{trimmed:,} of {traces:,} traces exceeded the 50,000-byte budget. '
            'Whole-message counts were not saved for this earlier run.</p></section>'
        )
    return (
        '<section class="insights-projection" role="status"><b>Model input projection</b>'
        f'<p>{esc(describe_projection_coverage(coverage))}</p>'
        '<p>Text inside kept messages may also be shortened. Tool results use 1,024-byte excerpts; source traces are not changed.</p>'
        '</section>'
    )


def tabs(run: InsightsRun, active: str, *, oob: bool = False) -> str:
    """Render the tab bar; ``oob=True`` marks it for an htmx out-of-band swap so a tab fragment moves the indicator."""
    links = []
    for name in TABS:
        cls = 'insights-tab active' if name == active else 'insights-tab'
        current_attr = ' aria-current="page"' if name == active else ''
        href = f'/insights/{quote(run.run_id, safe="")}/tab/{name}'
        # hx-sync aborts an in-flight tab request, so a slow response (the map) cannot land over a newer click.
        links.append(
            f'<a class="{cls}" href="{href}" hx-get="{href}" hx-target="#insights-content" hx-push-url="true"'
            f'{current_attr} hx-sync="closest nav:replace">{TAB_LABELS[name]}</a>'
        )
    swap = ' hx-swap-oob="true"' if oob else ''
    return f'<nav id="insights-tabs" class="insights-tabs" aria-label="Insights views"{swap}>{"".join(links)}</nav>'


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
            '</div>'
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
        tree = f'<div class="insights-tree">{"".join(rows)}</div>'
        parts.append(
            f'<section class="insights-dimension" data-dimension="{esc(name)}"{hidden}><h3>{esc(name.title())}</h3>{tree}</section>'
        )
    side = '<section id="insights-cluster-detail" class="insights-detail"><p class="insights-empty">Select a cluster to see its details and example traces.</p></section>'
    return f'<div class="insights-dimensions"><div class="insights-tree-column">{"".join(parts)}</div>{side}</div>'


def _find_cluster(run: InsightsRun, cluster_id: str) -> tuple[str, Cluster] | None:
    for dimension_name, dimension in run.dimensions.items():
        for cluster in dimension.clusters:
            if cluster.id == cluster_id:
                return dimension_name, cluster
    return None


def _trace_href(run: InsightsRun, trace: TraceInsight) -> str:
    locator = urlencode({'trace_id': trace.trace_id, 'span_id': trace.span_id or ''})
    return f'/insights/{quote(run.run_id, safe="")}/trace?{locator}'


def _trace_id_link(run: InsightsRun, trace: TraceInsight) -> str:
    return f'<a class="insights-trace-id" href="{esc(_trace_href(run, trace))}">{esc(trace.trace_id)}</a>'


def cluster_detail(run: InsightsRun, cluster_id: str) -> str:
    found = _find_cluster(run, cluster_id)
    if found is None:
        return '<p class="insights-empty">Cluster not found.</p>'
    dimension, cluster = found
    members = [trace for trace in run.traces if trace.trace_id in set(cluster.trace_ids)]
    label_rows = []
    for label_name in run.labels:
        values: dict[str, int] = {}
        for trace in members:
            answer = trace.labels.get(label_name)
            if answer is not None and answer.error is None and answer.value is not None:
                key = _label_value_key(run, label_name, answer.value)
                values[key] = values.get(key, 0) + 1
        if values:
            summary = ', '.join(
                f'{esc(_label_value_name(run, label_name, value))} {count}/{len(members)}'
                for value, count in sorted(values.items())
            )
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
        link = (
            ''
            if run.population.get('mode') == 'snapshot'
            else trace_link_button(trace_span_url(trace.trace_id, trace.span_id), 'Open in Orq ↗')
        )
        examples.append(
            f'<div class="insights-example"><code>{_trace_id_link(run, trace)}</code><span>{esc(excerpt)}</span>{link}</div>'
        )
    traces_href = (
        f'/insights/{quote(run.run_id, safe="")}/tab/traces?dimension={quote(dimension)}&cluster={quote(cluster.id)}'
    )
    label_html = ''.join(label_rows) or '<p class="insights-empty">No label answers for this cluster.</p>'
    example_html = ''.join(examples) or '<p class="insights-empty">No example traces available.</p>'
    return (
        '<div class="insights-detail-kicker">Cluster · '
        f'{esc(dimension)}</div><h3>{esc(cluster.name)}</h3><p>{esc(cluster.description) or "No description available."}</p>'
        f'<div class="insights-detail-kicker">Labels in this cluster</div>{label_html}'
        f'<div class="insights-detail-kicker">Example traces</div>{example_html}'
        f'<a class="insights-action" href="{traces_href}" hx-get="{traces_href}" hx-target="#insights-content" hx-push-url="true">Show all {cluster.size} in Traces →</a>'
    )


def _label_key(run: InsightsRun, name: str, value: object) -> str:
    """Show an answer under the same text its label card counts it by, so a card row filters to its traces."""
    result = run.labels.get(name)
    return label_key(result.spec, value) if result is not None else str(value)


def _ordered(run: InsightsRun, name: str | None, values: Iterable[str]) -> list[str]:
    """Sort label values in their label's own order (positive before negative, 1 before 5); other values follow A-Z."""
    result = run.labels.get(name) if name else None
    rank = {value: index for index, value in enumerate(label_order(result.spec))} if result is not None else {}
    return sorted(values, key=lambda value: (rank.get(value, len(rank)), value))


def labels(run: InsightsRun) -> str:
    if not run.labels:
        return '<section class="insights-empty-state"><h3>No labels in this run</h3><p>This run has no classifier labels to review.</p></section>'
    cards = []
    for name, result in run.labels.items():
        total = max(1, sum(result.counts.values()))
        rows = []
        counts_by_value = result.counts
        counts = [(value, counts_by_value[value]) for value in _ordered(run, name, counts_by_value)]
        for value, count in counts:
            pct = round(count / total * 100)
            href = f'/insights/{quote(run.run_id, safe="")}/tab/traces?label={quote(name)}&value={quote(value)}'
            rows.append(
                f'<a class="insights-label-row" href="{href}" hx-get="{href}" hx-target="#insights-content" hx-push-url="true"><span>{esc(_label_value_name(run, name, value))}</span><b>{pct}%</b>{_bar(count, total)}</a>'
            )
        row_html = ''.join(rows) or '<p class="insights-empty">No label values were recorded.</p>'
        card = (
            f'<article class="insights-label-card"><div class="insights-card-head"><h3>{esc(name)}</h3><span>{esc(result.spec.kind)}</span></div>'
            f'<p class="insights-label-instructions">{esc(result.spec.instructions)}</p>{row_html}'
        )
        failed = f'<p class="insights-muted">{result.n_failed} failed</p>' if result.n_failed else ''
        cards.append(card + failed + '</article>')
    return f'<div class="insights-label-grid">{"".join(cards)}</div>'


def _label_value_key(run: InsightsRun, label_name: str, value: bool | float | str) -> str:  # noqa: FBT001
    result = run.labels.get(label_name)
    if result is None or result.spec.kind != 'score' or not isinstance(result.spec.criteria, list):
        return str(value)
    levels = len(result.spec.criteria)
    if levels < 2:
        return str(value)
    return str(round(float(value) * (levels - 1)))


def _label_value_name(run: InsightsRun, label_name: str, value: str) -> str:
    result = run.labels.get(label_name)
    if result is None or result.spec.kind != 'score' or not isinstance(result.spec.criteria, list):
        return value
    try:
        index = int(value)
    except ValueError:
        return value
    if 0 <= index < len(result.spec.criteria):
        return result.spec.criteria[index]
    return value


def _trace_row(run: InsightsRun, trace: TraceInsight) -> str:
    cells = [
        f'<td class="id">{_trace_id_link(run, trace)}</td>',
        f'<td>{esc(trace.timestamp.strftime("%Y-%m-%d %H:%M") if trace.timestamp else "Time unavailable")}</td>',
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
            value = _label_key(run, name, answer.value)
        cells.append(f'<td>{esc(value)}</td>')
    if run.population.get('mode') != 'snapshot':
        link = trace_link_button(trace_span_url(trace.trace_id, trace.span_id), '↗ Orq')
        cells.append(f'<td>{link}</td>')
    return f'<tr>{"".join(cells)}</tr>'


def _tool_counts(title: str, counts: dict[str, int], empty: str) -> str:
    items = ''.join(f'<li>{esc(name)} <b>{count}</b></li>' for name, count in counts.items())
    body = f'<ul class="insights-tool-counts">{items}</ul>' if items else f'<p class="insights-muted">{esc(empty)}</p>'
    return f'<h4>{esc(title)}</h4>{body}'


def trace_detail_page(run: InsightsRun, trace: TraceInsight) -> str:
    """Show the saved analysis for one trace, including traces from a local snapshot."""

    def field(name: str, value: str | None) -> str:
        return f'<dt>{esc(name)}</dt><dd>{esc(value)}</dd>' if value else ''

    summary = trace.summary
    context = (
        field('Request', summary.request) + field('Task', summary.task) + field('Topic', summary.topic)
        if summary is not None
        else ''
    )
    labels = ''.join(
        field(
            name.replace('_', ' ').title(),
            f'Failed: {answer.error}'
            if answer.error
            else _label_key(run, name, answer.value)
            if answer.value is not None
            else 'No answer',
        )
        for name, answer in trace.labels.items()
    )
    cluster_names = {
        name: {cluster.id: cluster.name for cluster in dimension.clusters} for name, dimension in run.dimensions.items()
    }
    dimensions = ''.join(
        field(
            name.replace('_', ' ').title(),
            cluster_names.get(name, {}).get(assignment.base, assignment.base),
        )
        for name, assignment in trace.assignments.items()
    )
    errors = ''.join(f'<li><b>{esc(name)}</b>: {esc(message)}</li>' for name, message in trace.errors.items())
    stats = trace.tool_stats
    tools_html = (
        _tool_counts('Tools', stats.tools, 'No tool calls.')
        + _tool_counts('Shell commands', stats.commands, 'No shell commands.')
        + _tool_counts('Skills', stats.skills, 'No skills loaded.')
        if stats is not None
        else '<p>Tool use was not recorded for this run.</p>'
    )
    orq_link = (
        ''
        if run.population.get('mode') == 'snapshot'
        else trace_link_button(trace_span_url(trace.trace_id, trace.span_id), 'Open full trace in Orq ↗')
    )
    context_html = f'<dl>{context}</dl>' if context else ''
    labels_html = f'<dl>{labels}</dl>' if labels else '<p>No labels were recorded.</p>'
    dimensions_html = f'<dl>{dimensions}</dl>' if dimensions else '<p>No dimension assignments were recorded.</p>'
    errors_html = f'<section><h3>Errors</h3><ul>{errors}</ul></section>' if errors else ''
    signal_report = trace.signals.model_dump(mode='json') if trace.signals is not None else None
    signal_options = {
        'report': signal_report,
        'detail': {'signals': signal_report, 'source_coverage': trace.source_coverage},
        'level': 'L4',
    }
    signal_html = (
        f'<div data-saved-signals="{esc(json.dumps(signal_options, ensure_ascii=False, separators=(",", ":")))}"></div>'
    )
    body = (
        '<div class="insights-layout"><div class="insights-main insights-trace-detail">'
        '<div class="insights-detail-kicker">Trace analysis</div>'
        f'<h2><code>{esc(trace.trace_id)}</code></h2>'
        f'<p class="insights-muted">{esc(trace.timestamp.isoformat(timespec="minutes") if trace.timestamp else "Time unavailable")}'
        f'{" · span " + esc(trace.span_id) if trace.span_id else ""}'
        f'{" · " + esc(trace.agent_name) if trace.agent_name else ""}'
        f'{" · " + esc(trace.project) if trace.project else ""}</p>{orq_link}'
        '<section><h3>Summary</h3>'
        f'<p>{esc(summary.summary) if summary is not None else "No summary was saved for this trace."}</p>'
        f'{context_html}</section>'
        f'<section><h3>Labels</h3>{labels_html}</section>'
        f'<section><h3>Dimensions</h3>{dimensions_html}</section>'
        f'<section><h3>Tool use</h3>{tools_html}</section>'
        f'{signal_html}'
        f'{errors_html}'
        '</div></div>'
    )
    back = f'<a class="report-back" href="/insights/{quote(run.run_id, safe="")}/tab/traces">← Traces in this run</a>'
    return page(
        'Insights trace',
        body,
        active_nav='insights',
        back_html=back,
        head_html=f'{_SIGNALS_STYLESHEET}{_SIGNALS_SCRIPT}',
    )


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
            if (answer := trace.labels.get(label)) is not None
            and answer.error is None
            and answer.value is not None
            and (value is None or _label_value_key(run, label, answer.value) == value)
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
    headings = ['Trace', 'Time', *run.config.dimensions, *run.labels]
    if run.population.get('mode') != 'snapshot':
        headings.append('Orq')
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
        return (
            _label_key(run, name, answer.value)
            if answer is not None and answer.error is None and answer.value is not None
            else None
        )
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
    rows = _ordered(
        run,
        (row or '').removeprefix('label:'),
        {v for trace in run.traces if (v := _cross_value(run, trace, row)) is not None},
    )
    columns = _ordered(
        run,
        (column or '').removeprefix('label:'),
        {v for trace in run.traces if (v := _cross_value(run, trace, column)) is not None},
    )
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
    x_max = max(1.0, max_volume * 1.1)
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


def _dimension_map(run: InsightsRun, dimension_name: str) -> str:
    dimension = run.dimensions.get(dimension_name)
    if dimension is None:
        return _map_empty('This dimension is not part of the run.')
    if not any(dimension_name in trace.coords for trace in run.traces):
        reason = (
            dimension.warnings[0] if dimension.warnings else 'UMAP coordinates were not produced for this dimension.'
        )
        return _map_empty(reason)
    color_options = ['<option value="cluster">Clusters</option>']
    color_options.extend(
        f'<option value="dimension:{esc(name)}">{esc(name.title())} clusters</option>'
        for name in run.dimensions
        if name != dimension_name
    )
    color_options.extend(f'<option value="label:{esc(name)}">{esc(name)}</option>' for name in run.labels)
    color_options.extend(('<option value="agent">Agent</option>', '<option value="project">Project</option>'))
    chart = (
        f'<div id="insights-map-{quote(dimension_name, safe="")}" class="insights-map-chart" '
        f'data-run-id="{esc(run.run_id)}" data-map-dimension="{esc(dimension_name)}" '
        f'data-cluster-detail-url-template="/insights/{quote(run.run_id, safe="")}/cluster/{{cluster_id}}" '
        f'data-map-url="/insights/{quote(run.run_id, safe="")}/map.json?dimension={quote(dimension_name, safe="")}&amp;color_by=cluster"></div>'
    )
    detail = '<section class="insights-detail" data-map-detail><p class="insights-empty">Select a point to see its trace and cluster details.</p></section>'
    layout = f'<div class="insights-map-layout">{chart}{detail}</div>'
    return (
        f'<div class="insights-map-view"><div class="insights-map-toolbar">'
        f'<label>Colour by <select data-map-color data-dimension="{esc(dimension_name)}">{"".join(color_options)}</select></label>'
        f'<span>{sum(dimension_name in trace.coords for trace in run.traces)} of {len(run.traces)} traces mapped · UMAP 3D</span></div>'
        f'{layout}</div>'
    )


def map_tab(run: InsightsRun, selected: str | None = None) -> str:
    """Render every dimension's persisted UMAP projection in one expandable viewer."""
    if not run.dimensions:
        return _map_empty('This run has no discovered dimensions.')
    current = selected if selected in run.dimensions else next(iter(run.dimensions))
    options = ''.join(
        f'<option value="{esc(name)}"{" selected" if name == current else ""}>{esc(name.title())}</option>'
        for name in run.dimensions
    )
    panels = ''.join(
        f'<div data-map-projection-panel="{esc(name)}"{" hidden" if name != current else ""}>'
        f'{_dimension_map(run, name)}</div>'
        for name in run.dimensions
    )
    return (
        '<section class="insights-map-fullscreen-viewer"><div class="insights-map-head">'
        '<div><h3>Trace map</h3><p>Each projection uses the embeddings for one discovered dimension. '
        'Colour by clusters, another dimension, a label, agent, or project.</p></div>'
        '<button type="button" class="insights-action" data-map-fullscreen>Full screen</button></div>'
        f'<div class="insights-map-projection"><label>Projection <select data-map-projection>{options}</select></label></div>'
        f'{panels}</section>'
    )


def _map_category(run: InsightsRun, trace: TraceInsight, label_name: str | None, color_by: str) -> str:
    if label_name:
        answer = trace.labels.get(label_name)
        return (
            _label_key(run, label_name, answer.value)
            if answer and answer.error is None and answer.value is not None
            else 'No value'
        )
    if color_by == 'agent':
        return trace.agent_name or 'Unknown agent'
    return trace.project or 'Unknown project'


def map_payload(run: InsightsRun, dimension_name: str, color_by: str = 'cluster') -> dict[str, object]:
    """Keep one dimension's coordinates fixed while colouring by any saved axis."""
    dimension = run.dimensions.get(dimension_name)
    if dimension is None:
        return {
            'points': [],
            'missing_points': [],
            'legend': [],
            'color_scale': None,
            'color_mode': 'cluster',
            'grid_color': COLORS['sand_400'],
            'background_color': COLORS['sand_100'],
        }
    color_dimension = color_by.removeprefix('dimension:') if color_by.startswith('dimension:') else dimension_name
    label_name = color_by.removeprefix('label:') if color_by.startswith('label:') else None
    label_spec = run.labels[label_name].spec if label_name in run.labels else None
    if color_by.startswith('label:') and label_spec is None:
        return {'error': f'Unknown or unsupported colour label: {label_name}', 'points': []}
    if color_by.startswith('dimension:') and color_dimension not in run.dimensions:
        return {'error': f'Unknown colour dimension: {color_dimension}', 'points': []}
    if color_by not in {'cluster', 'agent', 'project'} and not color_by.startswith(('label:', 'dimension:')):
        return {'error': f'Unknown colour axis: {color_by}', 'points': []}
    categorical = color_by in {'agent', 'project'} or (label_spec is not None and label_spec.kind != 'score')
    continuous = label_spec is not None and label_spec.kind == 'score'
    base_clusters = [item for item in run.dimensions[color_dimension].clusters if item.level == 'base']
    cluster_by_id = {item.id: item for item in base_clusters}
    cluster_indexes = {item.id: index for index, item in enumerate(base_clusters)}
    shape_clusters = [item for item in dimension.clusters if item.level == 'base']
    shape_indexes = {item.id: index for index, item in enumerate(shape_clusters)}
    categories = {_map_category(run, trace, label_name, color_by) for trace in run.traces} if categorical else set()
    ordered = _ordered(run, label_name, (category for category in categories if category != 'No value'))
    category_colors = {category: QUALITATIVE[index % len(QUALITATIVE)] for index, category in enumerate(ordered)}
    points: list[dict[str, object]] = []
    missing_points: list[dict[str, object]] = []
    legend: dict[str, dict[str, str]] = {}
    for trace in run.traces:
        coords = trace.coords.get(dimension_name)
        if coords is None:
            continue
        shape_assignment = trace.assignments.get(dimension_name)
        shape_index = shape_indexes.get(shape_assignment.base if shape_assignment else '', -1)
        assignment = trace.assignments.get(color_dimension)
        cluster_id = assignment.base if assignment is not None else 'Unclassified'
        symbol = 'diamond' if shape_index >= 8 else 'circle'
        cluster = cluster_by_id.get(cluster_id)
        cluster_name = cluster.name if cluster else cluster_id
        point: dict[str, object] = {
            'trace_id': trace.trace_id,
            'span_id': trace.span_id,
            'x': coords[0],
            'y': coords[1],
            'z': coords[2],
            'cluster_id': cluster_id,
            'cluster_name': cluster_name,
            'symbol': symbol,
            'summary': trace.summary.summary if trace.summary else '',
            'agent': trace.agent_name,
            'project': trace.project,
        }
        if continuous:
            answer = trace.labels.get(label_name or '')
            if (
                answer is None
                or answer.error is not None
                or isinstance(answer.value, bool)
                or not isinstance(answer.value, (int, float))
                or not isfinite(float(answer.value))
            ):
                point['color'] = COLORS['sand_400']
                missing_points.append(point)
                continue
            value = min(1.0, max(0.0, float(answer.value)))
            point['color_value'] = 1.0 - value if 'satisfaction' in (label_name or '') else value
            point['color'] = ''
        elif categorical:
            category = _map_category(run, trace, label_name, color_by)
            color = category_colors.get(category, COLORS['sand_400'])
            point.update({
                'color_value': ordered.index(category) if category in ordered else -1,
                'color': color,
                'label_value': category,
            })
            legend.setdefault(category, {'cluster_id': category, 'name': category, 'color': color, 'symbol': symbol})
        else:
            cluster_index = cluster_indexes.get(cluster_id, -1)
            color = _cluster_color(cluster_index) if cluster_index >= 0 else COLORS['sand_400']
            point.update({'color_value': cluster_name, 'color': color})
            legend.setdefault(
                cluster_id,
                {'cluster_id': cluster_id, 'name': cluster_name, 'color': color, 'symbol': symbol},
            )
        points.append(point)
    color_scale: list[list[float | str]] | None = (
        (ORQ_SCALE_GOOD_BAD if 'satisfaction' in (label_name or '') else ORQ_SCALE_HEAT) if continuous else None
    )
    return {
        'points': points,
        'missing_points': missing_points,
        'legend': [] if continuous else list(legend.values()),
        'color_scale': color_scale,
        'color_mode': 'continuous' if continuous else 'category' if categorical else 'cluster',
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
        return map_tab(run, query.get('dimension'))
    if tab == 'crosstab':
        return crosstab(run, query or {})
    return priority_matrix(run)


def full_page(
    run: InsightsRun,
    active_tab: str = 'dimensions',
    *,
    query: dict[str, str] | None = None,
    manifest: RunManifest | None = None,
) -> str:
    body = (
        '<div class="insights-layout">'
        '<div class="insights-main">'
        f'{header(run)}{failures(run)}{progress(manifest)}{tabs(run, active_tab)}'
        f'<div id="insights-content">{tab_content(run, active_tab, query=query)}</div>'
        '</div></div>'
    )
    return page(
        'Insights',
        body + '<script src="/static/plotly-gl3d.min.js" defer></script>',
        active_nav='insights',
        back_html=_back_to_runs(),
    )


def _back_to_runs() -> str:
    return '<a class="report-back" href="/insights">← All Insights runs</a>'


def overview_page(
    entries: list[tuple[str, str, str]], runs: dict[str, InsightsRun], manifests: dict[str, RunManifest]
) -> str:
    body = f'<div class="insights-layout">{run_overview(entries, runs, manifests)}</div>'
    return page(
        'Insights',
        body + _refresh_if_running(entries),
        active_nav='insights',
        actions_html='<a class="insights-overview-new" href="/insights/new">+ New Run</a>',
    )


def _refresh_if_running(entries: list[tuple[str, str, str]]) -> str:
    return (
        '<script>setTimeout(function(){location.reload()},3000)</script>'
        if any(row[2] == 'running' for row in entries)
        else ''
    )


def running_page(
    run_id: str,
    label: str,
    manifest: RunManifest,
) -> str:
    if manifest.status == 'running':
        notice = (
            f'<section class="insights-empty-state"><h2>{esc(label)}</h2>'
            '<p>This run updates every few seconds. Results appear when it finishes.</p></section>'
        )
    elif manifest.status == 'error':
        notice = (
            f'<section class="insights-error" role="alert"><h2>{esc(label)} failed</h2>'
            f'<p>{esc(manifest.error or "The run stopped before a report was saved.")}</p></section>'
        )
    else:
        notice = (
            f'<section class="insights-empty-state"><h2>{esc(label)}</h2>'
            '<p>This run ended without a report file.</p></section>'
        )
    body = f'<div class="insights-layout"><div class="insights-main">{notice}{progress(manifest)}</div></div>'
    refresh = _refresh_if_running([(run_id, label, 'running')]) if manifest.status == 'running' else ''
    return page('Insights', body + refresh, active_nav='insights', back_html=_back_to_runs())


def unreadable_page(error: str) -> str:
    body = (
        '<div class="insights-layout">'
        '<div class="insights-main"><section class="insights-error"><h2>Run file is unreadable</h2>'
        f'<p>{esc(error)}</p></section></div></div>'
    )
    return page('Insights', body, active_nav='insights', back_html=_back_to_runs())
