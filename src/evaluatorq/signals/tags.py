"""Group D trajectory tags, evaluated from already-computed signal metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from evaluatorq.signals.models import Group, Precondition, SignalFn, SignalResult, result

if TYPE_CHECKING:
    from evaluatorq.signals.walk import SignalContext


@dataclass(frozen=True)
class Clause:
    """A metric comparison against a cohort percentile or a fixed threshold."""

    metric: str
    op: Literal['>', '<']
    quantile: float | None = None
    fixed: float | None = None


@dataclass(frozen=True)
class TagRule:
    """A tag rule with required gates and a minimum number of alternative clauses."""

    name: str
    doc: str
    gate: tuple[Clause, ...] = ()
    any_of: tuple[Clause, ...] = ()
    need: int = 1


TAG_RULES: tuple[TagRule, ...] = (
    TagRule(
        'long_autonomous_run',
        'Extended execution without user input.',
        any_of=(Clause('max_autonomous_steps', '>', 0.8), Clause('max_autonomous_duration_ms', '>', 0.8)),
    ),
    TagRule(
        'delegation_heavy',
        'Disproportionate delegation to subagents for the cohort.',
        gate=(Clause('subagent_step_share', '>', 0.9),),
    ),
    TagRule(
        'error_heavy',
        'Errors dominate tool execution.',
        gate=(Clause('tool_error_rate', '>', 0.9), Clause('tool_error_count', '>', fixed=2)),
    ),
    TagRule(
        'tool_churn',
        'High tool activity with low argument diversity or frequent retries.',
        gate=(Clause('tool_call_count', '>', 0.75),),
        any_of=(Clause('distinct_tool_arg_ratio', '<', 0.1), Clause('tool_retry_count', '>', 0.9)),
    ),
    TagRule(
        'tool_loop',
        'Exact repetition or oscillation, parameter drift, or a retry storm.',
        any_of=(
            Clause('tool_loop_count', '>', 0.9),
            Clause('consecutive_same_tool_max', '>', 0.95),
            Clause('tool_retry_count', '>', 0.95),
        ),
    ),
    TagRule(
        'stalled',
        'The run ends without an answer after looping or repeated failure.',
        gate=(Clause('terminal_answer_present', '<', fixed=1),),
        any_of=(Clause('tool_loop_count', '>', 0.9), Clause('tool_retry_count', '>', 0.9)),
    ),
    TagRule(
        'output_heavy',
        'Unusually large tool-result volume.',
        any_of=(Clause('total_tool_result_bytes', '>', 0.95), Clause('max_tool_result_bytes', '>', 0.95)),
    ),
    TagRule(
        'inefficient_execution',
        'At least two structural inefficiency indicators fire together.',
        need=2,
        any_of=(
            Clause('duplicate_tool_call_count', '>', 0.9),
            Clause('tool_retry_count', '>', 0.9),
            Clause('distinct_tool_arg_ratio', '<', 0.1),
            Clause('empty_tool_result_count', '>', 0.9),
            Clause('tool_loop_count', '>', 0.9),
            Clause('tool_error_rate', '>', 0.9),
        ),
    ),
)

TAG_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    rule.name: tuple(dict.fromkeys(c.metric for c in rule.gate + rule.any_of)) for rule in TAG_RULES
}


def _clauses(rule: TagRule) -> tuple[Clause, ...]:
    return rule.gate + rule.any_of


def _tag_signal(rule: TagRule) -> SignalFn:
    def signal(ctx: SignalContext) -> SignalResult:
        thresholds_config = ctx.config.resolved_tag_thresholds()
        rows: list[tuple[Clause, float, float | None, bool | None]] = []
        for clause in _clauses(rule):
            key = f'{rule.name}.{clause.metric}'
            if clause.fixed is not None:
                threshold = clause.fixed
                percentile = None
            else:
                percentile = ctx.config.tag_percentiles.get(key)
                if percentile is None:
                    threshold = thresholds_config.thresholds[key]
                else:
                    threshold = thresholds_config.cohort_threshold(clause.metric, percentile)
            metric = ctx.results.get(clause.metric)
            value = metric.value if metric is not None and metric.no_basis is None else None
            hit = None if value is None else (value > threshold if clause.op == '>' else value < threshold)
            rows.append((clause, threshold, value, hit))

        gates = rows[: len(rule.gate)]
        options = rows[len(rule.gate) :]
        gate_hits = [hit for _, _, _, hit in gates]
        option_hits = [hit for _, _, _, hit in options]
        gates_met = all(hit is True for hit in gate_hits)
        gates_failed = any(hit is False for hit in gate_hits)
        gates_unknown = any(hit is None for hit in gate_hits)
        hits = sum(hit is True for hit in option_hits)
        possible_hits = hits + sum(hit is None for hit in option_hits)
        if gates_failed:
            # A required gate makes the tag impossible regardless of unknown alternatives.
            fired = False
        elif not options:
            fired = None if gates_unknown else gates_met
        elif possible_hits < rule.need:
            # Even every unknown alternative firing cannot meet the minimum.
            fired = False
        elif gates_unknown:
            # Enough alternatives may fire, but an unknown required gate may still block the tag.
            fired = None
        elif hits >= rule.need:
            fired = True
        else:
            fired = None

        available = sum(hit is not None for _, _, _, hit in rows)
        required_met = fired is not None
        preconditions = [
            Precondition(
                name='required metrics',
                met=required_met,
                detail=f'{available} of {len(rows)} clause metrics available',
            )
        ]
        labels: dict[Clause, str] = {}
        for clause, threshold, value, hit in rows:
            suffix = ''
            if clause.quantile is not None:
                percentile = ctx.config.tag_percentiles.get(f'{rule.name}.{clause.metric}', clause.quantile * 100)
                suffix = f' (cohort p{percentile:g})'
            label = f'{clause.metric} {clause.op} {_format_threshold(threshold)}{suffix}'
            labels[clause] = label
            preconditions.append(
                Precondition(
                    name=label,
                    met=bool(hit),
                    detail=f'{clause.metric} = {value}' if hit is not None else f'missing: {clause.metric}',
                    required=False,
                )
            )

        fired_clauses = [clause for clause, _, _, hit in rows if hit]
        evidence = [
            item.model_copy(update={'reason': f'{clause.metric}: {item.reason}' if item.reason else clause.metric})
            for clause in (fired_clauses if fired else [])
            if (metric := ctx.results.get(clause.metric)) is not None
            for item in metric.evidence
        ]
        return result(
            rule.name,
            'D',
            fired,
            evidence,
            preconditions,
            reason='; '.join(labels[clause] for clause in fired_clauses) if fired else None,
            rule_version=ctx.config.version,
        )

    signal.__name__ = rule.name
    signal.__doc__ = rule.doc
    return signal


def _format_threshold(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f'{value:g}'


SIGNALS: dict[str, tuple[Group, SignalFn]] = {rule.name: ('D', _tag_signal(rule)) for rule in TAG_RULES}
