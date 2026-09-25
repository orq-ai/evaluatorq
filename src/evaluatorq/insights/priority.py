"""Priority matrix: one point per base cluster of a discovered dimension, plotting volume against satisfaction and error share.

Pure computation over an already-run `InsightsRun`'s traces and one dimension's
clusters — no LLM or network calls, so there is no retry layer here. Per the
"no optimistic defaults" house rule, a trace whose `satisfaction_label` or
`errors_label` answer failed (or is missing) is excluded from that cluster's
mean/share rather than counted as 0 or a pass; only the *label being absent
from the whole run* falls back to a documented 0.0 default, and that default
is always named in the returned reason string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.insights.models import PriorityPoint

if TYPE_CHECKING:
    from evaluatorq.insights.models import DimensionResult, LabelSpec, TraceInsight


def priority_points(
    run_traces: list[TraceInsight],
    dimension: DimensionResult,
    *,
    satisfaction_label: str = 'customer_satisfaction',
    errors_label: str = 'made_errors',
    satisfaction_spec: LabelSpec,
) -> tuple[list[PriorityPoint] | None, str | None]:
    """Compute one `PriorityPoint` per base cluster in `dimension`.

    volume = the cluster's member count. mean_satisfaction = the mean of
    `answer.value` over members whose `satisfaction_label` answer succeeded
    (`error is None` and `value is not None`); `labeling._map_answer` already
    normalizes a `score`-kind answer to `[0, 1]` (`value = score / top`), so
    this reads it directly rather than re-normalizing it a second time.
    `satisfaction_spec` is accepted for interface stability (the caller
    already has it at hand) but is not otherwise read here. Members with a
    failed or missing answer are excluded from the mean, never counted as 0.
    error_share = the share of members with a non-failed `errors_label`
    answer whose `value is True` (already the threshold-derived boolean —
    never re-derive it from `probabilities`).

    Returns `(None, reason)` when `satisfaction_label` was never requested
    for this run (no trace carries that key in `labels`) or when every base
    cluster ends up with zero valid `satisfaction_label` answers. A base
    cluster with zero valid `satisfaction_label` answers is skipped (logged),
    and likewise a base cluster is skipped when `errors_label` was requested
    for the run but every member's answer failed or is missing for that one
    cluster — neither ever falls back to a guessed value. Only when
    `errors_label` was never requested for the *whole run* does every
    produced point's `error_share` default to `0.0`, and that default is
    always named in the returned `reason`.
    """
    satisfaction_requested = any(satisfaction_label in trace.labels for trace in run_traces)
    if not satisfaction_requested:
        reason = f'the {satisfaction_label!r} label was not requested for this run'
        logger.warning('Insights priority matrix for dimension {!r} skipped: {}', dimension.name, reason)
        return None, reason

    errors_requested = any(errors_label in trace.labels for trace in run_traces)
    degraded_reason: str | None = None
    if not errors_requested:
        degraded_reason = (
            f'the {errors_label!r} label was not requested for this run; error share defaults to 0.0 for every cluster'
        )
        logger.warning('Insights priority matrix for dimension {!r}: {}', dimension.name, degraded_reason)

    by_id = {trace.trace_id: trace for trace in run_traces}

    points: list[PriorityPoint] = []
    for cluster in dimension.clusters:
        if cluster.level != 'base':
            continue

        members = [by_id[trace_id] for trace_id in cluster.trace_ids if trace_id in by_id]

        satisfaction_values: list[float] = []
        for member in members:
            answer = member.labels.get(satisfaction_label)
            if answer is None or answer.error is not None or answer.value is None:
                continue
            satisfaction_values.append(float(answer.value))  # pyright: ignore[reportArgumentType]

        if not satisfaction_values:
            logger.warning(
                'Insights priority matrix: cluster {!r} in dimension {!r} has no {!r} answer and was skipped',
                cluster.id,
                dimension.name,
                satisfaction_label,
            )
            continue

        error_denominator = 0
        error_numerator = 0
        for member in members:
            answer = member.labels.get(errors_label)
            if answer is None or answer.error is not None:
                continue
            error_denominator += 1
            if answer.value is True:
                error_numerator += 1

        if error_denominator == 0:
            if errors_requested:
                logger.warning(
                    'Insights priority matrix: cluster {!r} in dimension {!r} has no valid {!r} answer and was skipped',
                    cluster.id,
                    dimension.name,
                    errors_label,
                )
                continue
            error_share = 0.0
        else:
            error_share = error_numerator / error_denominator

        points.append(
            PriorityPoint(
                cluster_id=cluster.id,
                name=cluster.name,
                volume=cluster.size,
                mean_satisfaction=sum(satisfaction_values) / len(satisfaction_values),
                error_share=error_share,
            )
        )

    if not points:
        reason = f'no base cluster in dimension {dimension.name!r} has a {satisfaction_label!r} answer'
        logger.warning('Insights priority matrix skipped: {}', reason)
        return None, reason

    return points, degraded_reason
