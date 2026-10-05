"""The signal registry and `compute_signals`.

Each group module (`structure`, `tools`, `autonomy`, `tags`) exports one explicit dict of `name -> (group, fn)`.
Adding a group is one import and one entry in the `_merge(...)` call below; there is no decorator registration.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.signals import autonomy, structure, tags, tools
from evaluatorq.signals import preconditions as pre
from evaluatorq.signals.config import SignalsConfig
from evaluatorq.signals.models import Group, Precondition, SignalFn, SignalReport, SignalResult
from evaluatorq.signals.walk import SignalContext

if TYPE_CHECKING:
    from evaluatorq.formats.atif import AtifTrajectory

SignalTable = Mapping[str, tuple[Group, SignalFn]]


def _merge(*tables: SignalTable) -> SignalTable:
    """Combine the group tables, refusing a name two groups both define, and freeze the result."""
    merged: dict[str, tuple[Group, SignalFn]] = {}
    for table in tables:
        for name, entry in table.items():
            if name in merged:
                msg = f'Signal {name!r} is registered twice'
                raise ValueError(msg)
            merged[name] = entry
    return MappingProxyType(merged)


SIGNALS: SignalTable = _merge(structure.SIGNALS, tools.SIGNALS, autonomy.SIGNALS, tags.SIGNALS)
SIGNAL_NAMES: tuple[str, ...] = tuple(SIGNALS)

# These values depend on complete tool activity. Keep this list explicit: the registry test checks that every
# registered signal whose implementation reads tool calls is declared here, and that every name is registered.
TOOL_ACTIVITY_DEPENDENT_SIGNALS = frozenset({
    'tool_call_count',
    'unique_tools_used',
    'tool_call_value_count',
    'bash_command_value_count',
    'webview_value_count',
    'loaded_skill_value_count',
    'tool_error_count',
    'tool_error_rate',
    'duplicate_tool_call_count',
    'tool_retry_count',
    'tool_succeeded_after_retry_count',
    'invalid_schema_tool_call_count',
    'consecutive_same_tool_max',
    'consecutive_command_family_max',
    'identical_tool_call_run_count',
    'tool_oscillation_count',
    'tool_loop_count',
    'distinct_tool_arg_ratio',
    'empty_tool_result_count',
    'max_tool_result_bytes',
    'total_tool_result_bytes',
    'max_autonomous_steps',
    'avg_autonomous_steps',
    'max_llm_tool_cycles',
    'terminal_answer_present',
    'human_interruption_count',
    'subagent_step_share',
    'parallel_tool_batch_count',
    'max_parallel_tool_calls',
    'wall_time_ms',
    'active_time_ms',
    'tool_time_ms',
    'max_autonomous_duration_ms',
})
_TAG_TOOL_DEPENDENCIES = frozenset(
    name
    for rule_name, dependencies in tags.TAG_DEPENDENCIES.items()
    if any(dependency in TOOL_ACTIVITY_DEPENDENT_SIGNALS for dependency in dependencies)
    for name in (rule_name,)
)
_UNKNOWN_DEPENDENCIES = (TOOL_ACTIVITY_DEPENDENT_SIGNALS | _TAG_TOOL_DEPENDENCIES) - set(SIGNALS)
if _UNKNOWN_DEPENDENCIES:
    raise ValueError(f'Tool-activity dependency names are not registered: {sorted(_UNKNOWN_DEPENDENCIES)}')


def compute_signals(
    trajectory: AtifTrajectory, config: SignalsConfig | None = None, only: Iterable[str] | None = None
) -> SignalReport:
    """Run the signals on a trajectory, in registry order.

    A signal that raises is reported as no-basis with the exception type and message (and logged), so one broken
    rule does not hide the others. The trajectory is walked once into a `SignalContext` that every signal shares;
    `ctx.results` fills in registry order as signals finish.

    Args:
        trajectory: The trajectory to measure.
        config: Rule knobs; defaults to `SignalsConfig()`.
        only: Signal names to run; defaults to all of them.

    Raises:
        ValueError: `only` names a signal that is not registered.
    """
    config = config or SignalsConfig()
    wanted = None if only is None else set(only)
    if wanted is not None and (unknown := sorted(wanted - set(SIGNALS))):
        msg = f'Unknown signal names: {", ".join(unknown)}'
        raise ValueError(msg)
    if wanted is not None:
        pending = list(wanted)
        while pending:
            name = pending.pop()
            pending.extend(dependency for dependency in tags.TAG_DEPENDENCIES.get(name, ()) if dependency not in wanted)
            wanted.update(tags.TAG_DEPENDENCIES.get(name, ()))
    selected = {name: entry for name, entry in SIGNALS.items() if wanted is None or name in wanted}
    report = partial(SignalReport, trajectory_id=trajectory.trajectory_id, config_version=config.version)
    results: dict[str, SignalResult] = {}
    try:
        ctx = SignalContext.build(trajectory, config)
    except Exception as exc:  # noqa: BLE001 - failure isolation: a walk that fails must still yield a report
        logger.warning('Walking the trajectory failed: {}: {}', type(exc).__name__, exc)
        why = f'walk failed: {type(exc).__name__}: {exc}'
        return report(results={n: SignalResult(name=n, group=g, no_basis=why) for n, (g, _) in selected.items()})
    for name, (group, fn) in selected.items():
        try:
            computed = fn(ctx)
            if name in TOOL_ACTIVITY_DEPENDENT_SIGNALS:
                computed = _require_source_tool_coverage(computed, ctx)
            elif name in _TAG_TOOL_DEPENDENCIES:
                computed = _inherit_source_tool_coverage(computed, name, ctx.results)
            results[name] = computed
        except Exception as exc:  # noqa: BLE001 - failure isolation: one broken signal must not hide the others
            logger.warning('Signal {} failed: {}: {}', name, type(exc).__name__, exc)
            results[name] = SignalResult(name=name, group=group, no_basis=f'signal raised {type(exc).__name__}: {exc}')
        ctx.results[name] = results[name]
    return report(results=results)


def _require_source_tool_coverage(computed: SignalResult, ctx: SignalContext) -> SignalResult:
    coverage = pre.source_tool_coverage(ctx)
    pcs = [*computed.preconditions, coverage]
    if coverage.met is False:
        return SignalResult(
            name=computed.name,
            group=computed.group,
            approximate=computed.approximate,
            no_basis=coverage.detail,
            preconditions=pcs,
        )
    return computed.model_copy(update={'preconditions': pcs})


def _inherit_source_tool_coverage(computed: SignalResult, name: str, prior: dict[str, SignalResult]) -> SignalResult:
    unavailable = [
        dependency
        for dependency in tags.TAG_DEPENDENCIES.get(name, ())
        if (item := prior.get(dependency)) is not None
        and any(pc.name == 'source tool activity represented' and pc.met is False for pc in item.preconditions)
    ]
    if not unavailable:
        return computed
    detail = f'tool-dependent metrics unavailable: {", ".join(unavailable)}'
    coverage = Precondition(name='source tool activity represented', met=False, detail=detail)
    pcs = [*computed.preconditions, coverage]
    return SignalResult(
        name=computed.name,
        group=computed.group,
        approximate=computed.approximate,
        no_basis=detail,
        preconditions=pcs,
    )
