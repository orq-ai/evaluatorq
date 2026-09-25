"""Unit tests for `evaluatorq.insights.priority` — the priority matrix per cluster."""

from __future__ import annotations

from datetime import datetime, timezone

from evaluatorq.insights.models import Cluster, DimensionResult, LabelAnswer, TraceInsight
from evaluatorq.insights.presets import CUSTOMER_SATISFACTION, MADE_ERRORS, score_to_unit
from evaluatorq.insights.priority import priority_points


def _trace(
    trace_id: str,
    *,
    satisfaction: float | None = None,
    satisfaction_error: str | None = None,
    made_errors: bool | None = None,
    made_errors_error: str | None = None,
    include_satisfaction: bool = True,
    include_made_errors: bool = True,
) -> TraceInsight:
    labels: dict[str, LabelAnswer] = {}
    if include_satisfaction:
        labels['customer_satisfaction'] = LabelAnswer(
            value=satisfaction, confidence=0.9, probabilities=None, error=satisfaction_error
        )
    if include_made_errors:
        labels['made_errors'] = LabelAnswer(
            value=made_errors,
            confidence=0.9,
            probabilities={'true': 0.9, 'false': 0.1} if made_errors is not None else None,
            error=made_errors_error,
        )
    return TraceInsight(
        trace_id=trace_id,
        span_id=f'{trace_id}-span',
        timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
        labels=labels,
    )


def _base_cluster(cluster_id: str, trace_ids: list[str]) -> Cluster:
    return Cluster(
        id=cluster_id,
        parent_id='top-1',
        level='base',
        name=f'cluster {cluster_id}',
        description='a base cluster',
        size=len(trace_ids),
        trace_ids=trace_ids,
        example_trace_ids=trace_ids[:1],
    )


def _dimension(clusters: list[Cluster]) -> DimensionResult:
    return DimensionResult(name='intent', source_field='request', clusters=clusters)


def test_happy_path_computes_volume_satisfaction_and_error_share() -> None:
    traces = [
        _trace('t1', satisfaction=4.0, made_errors=True),
        _trace('t2', satisfaction=0.0, made_errors=False),
        _trace('t3', satisfaction=2.0, made_errors=False),
    ]
    cluster = _base_cluster('base-1', ['t1', 't2', 't3'])
    dimension = _dimension([cluster])

    points, reason = priority_points(traces, dimension, satisfaction_spec=CUSTOMER_SATISFACTION)

    assert reason is None
    assert points is not None
    assert len(points) == 1
    point = points[0]
    assert point.cluster_id == 'base-1'
    assert point.volume == 3
    expected_mean = sum(
        score_to_unit(CUSTOMER_SATISFACTION, v) for v in (4.0, 0.0, 2.0)
    ) / 3
    assert point.mean_satisfaction == expected_mean
    assert point.error_share == 1 / 3


def test_missing_satisfaction_label_returns_none_with_reason() -> None:
    traces = [
        _trace('t1', include_satisfaction=False, made_errors=True),
        _trace('t2', include_satisfaction=False, made_errors=False),
    ]
    cluster = _base_cluster('base-1', ['t1', 't2'])
    dimension = _dimension([cluster])

    points, reason = priority_points(traces, dimension, satisfaction_spec=CUSTOMER_SATISFACTION)

    assert points is None
    assert reason is not None
    assert 'customer_satisfaction' in reason


def test_failed_satisfaction_answers_excluded_from_mean() -> None:
    traces = [
        _trace('t1', satisfaction=4.0, made_errors=False),
        _trace('t2', satisfaction_error='classify timed out', made_errors=False),
        _trace('t3', satisfaction=None, made_errors=False),
    ]
    cluster = _base_cluster('base-1', ['t1', 't2', 't3'])
    dimension = _dimension([cluster])

    points, reason = priority_points(traces, dimension, satisfaction_spec=CUSTOMER_SATISFACTION)

    assert reason is None
    assert points is not None
    assert len(points) == 1
    point = points[0]
    # volume still counts every member; only t1 contributed a valid satisfaction answer.
    assert point.volume == 3
    assert point.mean_satisfaction == score_to_unit(CUSTOMER_SATISFACTION, 4.0)


def test_no_cluster_has_satisfaction_answer_returns_none() -> None:
    traces = [
        _trace('t1', satisfaction_error='classify failed', made_errors=True),
        _trace('t2', satisfaction=None, made_errors=False),
    ]
    cluster = _base_cluster('base-1', ['t1', 't2'])
    dimension = _dimension([cluster])

    points, reason = priority_points(traces, dimension, satisfaction_spec=CUSTOMER_SATISFACTION)

    assert points is None
    assert reason is not None
    assert 'no base cluster' in reason


def test_errors_label_absent_defaults_error_share_to_zero_with_reason() -> None:
    traces = [
        _trace('t1', satisfaction=4.0, include_made_errors=False),
        _trace('t2', satisfaction=2.0, include_made_errors=False),
    ]
    cluster = _base_cluster('base-1', ['t1', 't2'])
    dimension = _dimension([cluster])

    points, reason = priority_points(traces, dimension, satisfaction_spec=CUSTOMER_SATISFACTION)

    assert points is not None
    assert points[0].error_share == 0.0
    assert reason is not None
    assert 'made_errors' in reason


def test_only_base_clusters_are_scored() -> None:
    traces = [_trace('t1', satisfaction=4.0, made_errors=False)]
    top = Cluster(
        id='top-1',
        parent_id=None,
        level='top',
        name='top',
        description='top cluster',
        size=1,
        trace_ids=['t1'],
        example_trace_ids=['t1'],
    )
    base = _base_cluster('base-1', ['t1'])
    dimension = _dimension([top, base])

    points, _reason = priority_points(traces, dimension, satisfaction_spec=CUSTOMER_SATISFACTION)

    assert points is not None
    assert [p.cluster_id for p in points] == ['base-1']


def test_made_errors_value_is_true_not_truthy() -> None:
    """A `noul` label's `value` is already the thresholded bool; a non-True value never counts as an error."""
    traces = [
        _trace('t1', satisfaction=4.0, made_errors=False),
        _trace('t2', satisfaction=4.0, made_errors=True),
    ]
    cluster = _base_cluster('base-1', ['t1', 't2'])
    dimension = _dimension([cluster])

    points, _reason = priority_points(traces, dimension, satisfaction_spec=CUSTOMER_SATISFACTION)

    assert points is not None
    assert points[0].error_share == 0.5
    assert MADE_ERRORS.kind == 'noul'
