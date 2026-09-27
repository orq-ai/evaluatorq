"""Priority matrix: one point per base cluster of a discovered dimension, plotting volume against satisfaction and error share.

Pure computation over an already-run `InsightsRun`'s traces and one dimension's
clusters — no LLM or network calls, so there is no retry layer here. Per the
"no optimistic defaults" house rule, a trace whose `satisfaction_label` or
`errors_label` answer failed (or is missing) is excluded from that cluster's
mean/share rather than counted as 0 or a pass. When `errors_label` was not
requested, error share comes from summaries with `assistant_errors`; traces
without summaries are excluded, and the fallback source is named in the reason.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.insights.models import PriorityPoint, real_assistant_errors

if TYPE_CHECKING:
    from evaluatorq.insights.models import DimensionResult, LabelSpec, TraceInsight


def _summary_error_share(members: list[TraceInsight], cluster_id: str, dimension_name: str) -> float | None:
    summarized = [member for member in members if member.summary is not None]
    if not summarized:
        logger.warning(
            'Insights priority matrix: cluster {!r} in dimension {!r} has no summaries and was skipped',
            cluster_id,
            dimension_name,
        )
        return None
    return sum(bool(real_assistant_errors(member.summary)) for member in summarized if member.summary) / len(summarized)


def _label_error_share(
    members: list[TraceInsight], errors_label: str, cluster_id: str, dimension_name: str
) -> float | None:
    answers = [
        answer for member in members if (answer := member.labels.get(errors_label)) is not None and answer.error is None
    ]
    if not answers:
        logger.warning(
            'Insights priority matrix: cluster {!r} in dimension {!r} has no valid {!r} answer and was skipped',
            cluster_id,
            dimension_name,
            errors_label,
        )
        return None
    return sum(answer.value is True for answer in answers) / len(answers)


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
    cluster — neither ever falls back to a guessed value. When `errors_label`
    was never requested for the *whole run*, `error_share` comes from the
    share of members with summaries whose `assistant_errors` contains a real
    error. Members without summaries are excluded; a cluster without any
    summaries is skipped. The fallback source is named in the returned
    `reason`.
    """
    if satisfaction_spec.kind != 'score':
        reason = f'the {satisfaction_label!r} label must be a score label for the priority matrix'
        logger.warning('Insights priority matrix for dimension {!r} skipped: {}', dimension.name, reason)
        return None, reason

    satisfaction_requested = any(satisfaction_label in trace.labels for trace in run_traces)
    if not satisfaction_requested:
        reason = f'the {satisfaction_label!r} label was not requested for this run'
        logger.warning('Insights priority matrix for dimension {!r} skipped: {}', dimension.name, reason)
        return None, reason

    errors_requested = any(errors_label in trace.labels for trace in run_traces)
    degraded_reason: str | None = None
    if not errors_requested:
        degraded_reason = (
            f'error share taken from summary assistant_errors; the {errors_label!r} label was not requested'
        )
        logger.warning('Insights priority matrix for dimension {!r}: {}', dimension.name, degraded_reason)

    by_id = {trace.trace_id: trace for trace in run_traces}

    points: list[PriorityPoint] = []
    clusters_with_satisfaction = 0
    clusters_without_summaries = 0
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

        clusters_with_satisfaction += 1
        error_share = (
            _label_error_share(members, errors_label, cluster.id, dimension.name)
            if errors_requested
            else _summary_error_share(members, cluster.id, dimension.name)
        )
        if error_share is None:
            if not errors_requested:
                clusters_without_summaries += 1
            continue

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
        if (
            not errors_requested
            and clusters_with_satisfaction > 0
            and clusters_without_summaries == clusters_with_satisfaction
        ):
            reason = (
                f'no base cluster in dimension {dimension.name!r} with a {satisfaction_label!r} answer '
                'has a summary assistant_errors value'
            )
        else:
            reason = f'no base cluster in dimension {dimension.name!r} has a {satisfaction_label!r} answer'
        logger.warning('Insights priority matrix skipped: {}', reason)
        return None, reason

    return points, degraded_reason
