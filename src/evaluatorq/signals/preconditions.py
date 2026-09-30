"""Precondition checks: what a signal needs from a trajectory, and how much of it is there.

Each check returns a `Precondition`. Signals list the checks they depend on; the `result` builder turns a failed
*required* check into `no_basis`. The checks read the walked view (see `walk`), never the raw trajectory tree.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Literal

from evaluatorq.formats._shared import RAW_ARGUMENTS_KEY
from evaluatorq.signals.models import Precondition
from evaluatorq.signals.walk import (
    CallRecord,
    WalkedStep,
    finish_reasons,
    infer_provider,
    is_llm_step,
    llm_times,
    step_model,
    tool_schemas,
)

if TYPE_CHECKING:
    from evaluatorq.formats.atif import AtifTrajectory
    from evaluatorq.signals.config import SignalsConfig

_MAX_LISTED = 8
TRUNCATED_EXTRA_KEY = 'content_truncated'
"""Result `extra` key a converter sets when it kept only a preview of a tool result."""


def _listing(items: list[str]) -> str:
    counts = Counter(items)
    shown = [f'{name} x{n}' if n > 1 else name for name, n in counts.most_common(_MAX_LISTED)]
    more = len(counts) - len(shown)
    return ', '.join(shown) + (f', +{more} more' if more > 0 else '')


def _where(entry: WalkedStep) -> str:
    path = '/'.join(entry.agent_path)
    return f'step {entry.step.step_id}' + (f' of {path}' if path else '')


def _at(record: CallRecord) -> str:
    return f'{record.call.function_name} @{_where(record.step)}'


def _coverage(name: str, have: int, total: int, detail: str, *, required: bool, missing: str = '') -> Precondition:
    """True when all qualify, False when none do, `'partial'` otherwise."""
    if total == 0:
        return Precondition(name=name, met=False, detail=f'nothing to check: {detail}', required=required)
    met: bool | Literal['partial'] = True if have == total else ('partial' if have else False)
    text = f'{have} of {total} {detail}'
    if missing:
        text += f'; missing: {missing}'
    return Precondition(name=name, met=met, detail=text, required=required)


def has_tool_calls(calls: list[CallRecord]) -> Precondition:
    """At least one tool call (rates and ratios are undefined otherwise)."""
    return Precondition(name='has tool calls', met=bool(calls), detail=f'{len(calls)} tool calls')


def has_tool_results(calls: list[CallRecord]) -> Precondition:
    """At least one tool call has a result to measure."""
    n = sum(r.result is not None for r in calls)
    return Precondition(name='has tool results', met=n > 0, detail=f'{n} tool results')


def results_matched(calls: list[CallRecord]) -> Precondition:
    """Every tool call has a matching result. Orphans are listed and can never count as failed."""
    if not calls:
        return Precondition(name='calls matched to results', met=True, detail='no tool calls', required=False)
    orphans = [r for r in calls if r.result is None]
    return _coverage(
        'calls matched to results',
        len(calls) - len(orphans),
        len(calls),
        'tool calls have a result',
        required=False,
        missing=_listing([_at(r) for r in orphans]),
    )


def explicit_error_status(calls: list[CallRecord], config: SignalsConfig) -> Precondition:
    """Tool results carry an explicit error status (vs. only content sniffing possible).

    Required only when error detection is status-only: without statuses, nothing can fail.
    """
    results = [r for r in calls if r.result is not None]
    required = config.error_detection == 'status'
    if not results:
        return Precondition(name='explicit error status', met=True, detail='no tool results', required=False)
    with_status = [r for r in results if r.is_error is not None]
    pc = _coverage(
        'explicit error status', len(with_status), len(results), 'tool results carry an error status', required=required
    )
    if pc.met is False and not required:
        return pc.model_copy(update={'detail': pc.detail + ' (content sniffing only)'})
    return pc


def args_parseable(calls: list[CallRecord]) -> Precondition:
    """Arguments parse as a JSON object; unparseable ones (`{'_raw': text}`) are compared as raw strings."""
    if not calls:
        return Precondition(name='arguments parse as JSON', met=True, detail='no tool calls', required=False)
    bad = [r for r in calls if set(r.call.arguments) == {RAW_ARGUMENTS_KEY}]
    return _coverage(
        'arguments parse as JSON',
        len(calls) - len(bad),
        len(calls),
        'calls have JSON arguments',
        required=False,
        missing=_listing([_at(r) for r in bad]),
    )


def result_content_available(calls: list[CallRecord]) -> Precondition:
    """Result content is complete (not a truncated preview)."""
    results = [r for r in calls if r.result is not None]
    if not results:
        return Precondition(name='result content available', met=True, detail='no tool results', required=False)
    truncated = [r for r in results if r.result is not None and (r.result.extra or {}).get(TRUNCATED_EXTRA_KEY)]
    return _coverage(
        'result content available',
        len(results) - len(truncated),
        len(results),
        'results have full content',
        required=False,
        missing=_listing([f'{_at(r)} (truncated preview)' for r in truncated]),
    )


def tool_schemas_coverage(trajectory: AtifTrajectory, calls: list[CallRecord]) -> list[Precondition]:
    """Tool schemas exist, and how many calls have one. A schema is active for the whole trajectory.

    A call is covered by its own trajectory's `tool_definitions`, else the root's.
    """
    root = tool_schemas(trajectory)
    own = {id(r.step.trajectory): tool_schemas(r.step.trajectory) for r in calls}
    if not root and not any(own.values()):
        return [Precondition(name='tool schemas present', met=False, detail='the trajectory carries no tool schemas')]
    known = set(root).union(*own.values())
    present = Precondition(name='tool schemas present', met=True, detail=f'schemas for {len(known)} tools')
    if not calls:
        return [present]
    uncovered = [r for r in calls if r.call.function_name not in own[id(r.step.trajectory)] | root.keys()]
    coverage = _coverage(
        'schema active at call time',
        len(calls) - len(uncovered),
        len(calls),
        'calls have a schema active at call time (the rest are excluded)',
        required=True,
        missing=_listing([r.call.function_name for r in uncovered]),
    )
    return [present, coverage]


def model_present(walked: list[WalkedStep]) -> Precondition:
    """LLM steps carry a model name (their own, or their trajectory's agent)."""
    llm = [w for w in walked if is_llm_step(w)]
    have = sum(bool(step_model(w)) for w in llm)
    return _coverage('model on LLM steps', have, len(llm), 'agent steps carry a model', required=True)


def token_usage_complete(walked: list[WalkedStep]) -> Precondition:
    """Session totals need prompt and completion tokens on every LLM step: all or nothing."""
    llm = [w for w in walked if is_llm_step(w)]
    have = sum(
        w.step.metrics is not None
        and w.step.metrics.prompt_tokens is not None
        and w.step.metrics.completion_tokens is not None
        for w in llm
    )
    return Precondition(
        name='token usage complete',
        met=bool(llm) and have == len(llm),
        detail=f'{have} of {len(llm)} agent steps carry token usage',
        required=True,
    )


def provider_derivable(walked: list[WalkedStep]) -> Precondition:
    """Provider comes from the model name: a `provider/` prefix is read, a known name prefix is inferred (flagged)."""
    llm = [w for w in walked if is_llm_step(w)]
    models = [step_model(w) for w in llm]
    have = sum(infer_provider(m) is not None for m in models)
    guessed = sum(infer_provider(m) is not None and '/' not in (m or '') for m in models)
    pc = _coverage('provider derivable', have, len(llm), 'agent steps have a provider', required=True)
    if guessed:
        detail = pc.detail + f' ({guessed} inferred from the model name)'
        return pc.model_copy(update={'detail': detail, 'met': 'partial' if pc.met is True else pc.met})
    return pc


def finish_reason_present(walked: list[WalkedStep]) -> Precondition:
    """LLM steps carry a finish reason."""
    llm = [w for w in walked if is_llm_step(w)]
    have = sum(bool(finish_reasons(w.step)) for w in llm)
    return _coverage('finish_reason present', have, len(llm), 'agent steps have a finish_reason', required=True)


def subagent_linkage(walked: list[WalkedStep], calls: list[CallRecord], config: SignalsConfig) -> Precondition:
    """Subagent-spawning calls (by tool role, or any call whose result references a subagent) are linked to one."""

    def linked(record: CallRecord) -> bool:
        return bool(record.result is not None and record.result.subagent_trajectory_ref)

    spawns = [r for r in calls if config.tool_role(r.call.function_name) == 'subagent' or linked(r)]
    unlinked_agents = sorted({w.agent_path[-1] for w in walked if w.unlinked and w.agent_path})
    if not spawns:
        detail = 'no subagent-spawning calls'
        if unlinked_agents:
            detail += f'; subagents without a spawning call: {_listing(unlinked_agents)}'
        return Precondition(name='subagent linkage', met=not unlinked_agents, detail=detail, required=False)
    unlinked = [r for r in spawns if not linked(r)]
    pc = _coverage(
        'subagent linkage',
        len(spawns) - len(unlinked),
        len(spawns),
        'subagent-spawning calls linked to a subagent trajectory',
        required=False,
        missing=_listing([_at(r) for r in unlinked]),
    )
    if unlinked_agents:
        detail = pc.detail + f'; subagents without a spawning call: {_listing(unlinked_agents)}'
        return pc.model_copy(update={'detail': detail, 'met': 'partial' if pc.met is True else pc.met})
    return pc


def has_subagents(walked: list[WalkedStep]) -> Precondition:
    """At least one subagent invocation (averages are undefined otherwise)."""
    n = len({w.agent_path for w in walked if w.agent_path})
    return Precondition(name='has subagent invocations', met=n > 0, detail=f'{n} subagent invocations')


def timestamps(walked: list[WalkedStep]) -> Precondition:
    """LLM steps carry timestamps (timing metrics are omitted without them).

    Invocation start and end timestamps meet the check. A step with only an ISO `timestamp` counts as
    `'partial'`: it gives a start but no end, so durations from it are approximate.
    """
    llm = [w for w in walked if is_llm_step(w)]
    times = [llm_times(w) for w in llm]
    exact = sum(t.start is not None and t.end is not None for t in times)
    any_time = sum(t.start is not None or t.end is not None for t in times)
    if llm and exact == len(llm):
        return Precondition(
            name='timestamps', met=True, detail=f'{exact} of {len(llm)} agent steps have invocation timestamps'
        )
    if any_time:
        return Precondition(
            name='timestamps',
            met='partial',
            detail=f'{exact} of {len(llm)} agent steps have invocation timestamps, {any_time - exact} only an ISO start',
        )
    return _coverage('timestamps', 0, len(llm), 'agent steps have a timestamp', required=True)
