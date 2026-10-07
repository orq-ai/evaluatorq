"""JSON payload adapter for the Insights review dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from urllib.parse import quote, urlencode

from evaluatorq.common.structured_output import sum_structured_usage
from evaluatorq.dashboard.insights_review_projection import review_trace_key, review_xy, review_xy_state
from evaluatorq.dashboard.insights_views import _signal_summary, _trace_href
from evaluatorq.dashboard.trace_links import trace_span_url
from evaluatorq.insights.models import label_key, reads_snapshot, real_assistant_errors
from evaluatorq.insights.presets import CODING_LABELS
from evaluatorq.insights.transcript import SHELL_TOOLS

if TYPE_CHECKING:
    from evaluatorq.insights.models import InsightsRun


def build_signal_detail_payload(run: InsightsRun, trace_id: str, span_id: str | None) -> dict[str, object] | None:
    """Return one saved trace's full signal report and coverage, or None when the key is not in this run."""
    trace = next((item for item in run.traces if item.trace_id == trace_id and item.span_id == span_id), None)
    if trace is None:
        return None
    report = getattr(trace, 'signals', None)
    return {
        'trace_id': trace.trace_id,
        'span_id': trace.span_id,
        'has_signals': report is not None,
        'signals': report.model_dump(mode='json') if report is not None else None,
        'source_coverage': dict(getattr(trace, 'source_coverage', {}) or {}),
    }


def _timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def build_review_payload(run: InsightsRun) -> dict[str, object]:
    """Map a saved Insights run into the complete, JSON-safe dashboard contract."""
    projections = {name: review_xy(run, name) for name in run.dimensions}
    projection_states = {name: review_xy_state(run, name, points) for name, points in projections.items()}
    label_specs = {spec.name: spec for spec in run.config.labels}
    if run.config.coding_analysis:
        coding_specs = CODING_LABELS
    elif run.config.coding_labels:
        coding_specs = (CODING_LABELS[0], *run.config.coding_labels)
    else:
        coding_specs = ()
    label_specs.update({spec.name: spec for spec in coding_specs})
    labels = [
        {
            'name': result.spec.name,
            'kind': result.spec.kind,
            'criteria': result.spec.criteria,
            'instructions': result.spec.instructions,
            'counts': dict(result.counts),
            'n_failed': result.n_failed,
            'n_not_asked': result.n_not_asked,
            'mean_confidence': result.mean_confidence,
            'n_low_confidence': result.n_low_confidence,
        }
        for result in run.labels.values()
    ]
    traces: list[dict[str, object]] = []
    for trace in run.traces:
        answers: dict[str, list[object]] = {}
        question_errors: dict[str, str] = {}
        for name, answer in trace.labels.items():
            spec = label_specs.get(name)
            if answer.error:
                question_errors[name] = answer.error
            elif answer.value is not None and spec is not None:
                answers[name] = [label_key(spec, answer.value)]
            elif answer.value is not None:
                answers[name] = [answer.value]

        summary = trace.summary
        stats = trace.tool_stats
        signal_summary = _signal_summary(trace)
        assignment = {name: value.base for name, value in trace.assignments.items()}
        traces.append({
            'id': review_trace_key(trace),
            'trace_id': trace.trace_id,
            'span_id': trace.span_id,
            'trace_url': _trace_href(run, trace),
            'orq_url': (trace_span_url(trace.trace_id, trace.span_id) if not reads_snapshot(run.population) else None),
            'ts': _timestamp(getattr(trace, 'timestamp', None)),
            'agent': trace.agent_name,
            'l': answers,
            'label_errors': question_errors,
            'a': assignment,
            'xy': {
                name: list(points[review_trace_key(trace)])
                for name, points in projections.items()
                if review_trace_key(trace) in points
            },
            'errors': real_assistant_errors(summary) if summary else [],
            'summary': summary.summary if summary else None,
            'request': summary.request if summary else None,
            'topic': summary.topic if summary else None,
            'has_summary': summary is not None,
            'has_tool_stats': stats is not None,
            'tools': dict(stats.tools) if stats else {},
            'skills': dict(stats.skills) if stats else {},
            'commands': dict(stats.commands) if stats else {},
            'has_signals': signal_summary is not None,
            'has_signal_details': signal_summary is not None,
            'signals': signal_summary,
            'signal_detail_url': (
                f'/insights/{quote(run.run_id, safe="")}/trace-signals.json?'
                f'{urlencode({"trace_id": trace.trace_id, "span_id": trace.span_id or ""})}'
                if signal_summary is not None
                else None
            ),
        })

    dimensions = {
        name: {
            'source_field': dimension.source_field,
            'clusters': [
                {
                    'id': cluster.id,
                    'parent_id': cluster.parent_id,
                    'level': cluster.level,
                    'name': cluster.name,
                    'description': cluster.description,
                    'size': cluster.size,
                    'trace_ids': list(cluster.trace_ids),
                    'example_trace_ids': list(cluster.example_trace_ids),
                    'group': cluster.group,
                }
                for cluster in dimension.clusters
            ],
            'n_noise': dimension.n_noise,
            'n_no_signal': dimension.n_no_signal,
            'n_failed': dimension.n_failed,
            'warnings': list(dimension.warnings),
        }
        for name, dimension in run.dimensions.items()
    }
    total_usage = sum_structured_usage(list(run.cost_by_stage.values()))
    cost = total_usage.total_cost if total_usage is not None else None
    population = dict(run.population)
    return {
        'run': {
            'id': run.run_id,
            'name': run.run_name,
            'created': _timestamp(run.created_at),
            'status': run.status,
            'population': population,
            'cost': cost,
            'cost_is_partial': total_usage.cost_is_partial if total_usage is not None else None,
            'cost_by_stage': {
                name: usage.total_cost if usage is not None else None for name, usage in run.cost_by_stage.items()
            },
            'cost_by_stage_details': {
                name: {
                    'cost': usage.total_cost,
                    'is_partial': usage.cost_is_partial,
                    'priced_calls': usage.priced_calls,
                    'calls': usage.calls,
                }
                if usage is not None
                else None
                for name, usage in run.cost_by_stage.items()
            },
            'summary_model': run.config.summary_model,
            'classifier_model': run.config.classifier_model,
            'embedding_model': run.config.embedding_model,
            'warnings': list(run.warnings),
            'priority': [point.model_dump(mode='json') for point in run.priority] if run.priority is not None else None,
            'priority_reason': run.priority_reason,
            'priority_dimension': run.config.priority_dimension,
            'counts': dict(run.counts),
            'stage_failures': [failure.model_dump(mode='json') for failure in run.stage_failures],
        },
        'dims': dimensions,
        'labels': labels,
        'traces': traces,
        'map_states': projection_states,
        'activity': {'shell_tools': sorted(SHELL_TOOLS)},
    }
