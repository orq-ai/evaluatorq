"""Group C signals: autonomy, delegation, parallel work and activity timing."""

from __future__ import annotations

from datetime import timezone
from math import isclose
from operator import itemgetter
from typing import Any

from evaluatorq.formats._shared import atif_content_text, parse_iso
from evaluatorq.signals.models import Evidence, Group, Precondition, SignalFn, SignalResult, result
from evaluatorq.signals.preconditions import subagent_linkage, timestamps, tool_timestamps
from evaluatorq.signals.walk import CallRecord, SignalContext, WalkedStep, invocation, llm_times, tool_times


def _all_agent_steps(ctx: SignalContext) -> list[WalkedStep]:
    return [entry for entry in ctx.walked if entry.step.source == 'agent']


def _agent_steps(ctx: SignalContext) -> list[WalkedStep]:
    return [entry for entry in _all_agent_steps(ctx) if entry.step.llm_call_count != 0]


def _users(ctx: SignalContext) -> list[WalkedStep]:
    return [entry for entry in ctx.walked if not entry.agent_path and entry.step.source == 'user']


def _calls_at(ctx: SignalContext, entry: WalkedStep) -> list[CallRecord]:
    return [record for record in ctx.calls if record.step is entry]


def _steps_per_agent(entry: WalkedStep, ctx: SignalContext) -> int:
    if entry.step.source != 'agent':
        return 0
    return int(entry.step.llm_call_count != 0) + len(_calls_at(ctx, entry))


def _segments(ctx: SignalContext) -> list[tuple[WalkedStep, list[WalkedStep]]]:
    users = _users(ctx)
    segments = []
    positions = {id(entry): index for index, entry in enumerate(ctx.walked)}
    for index, start in enumerate(users):
        begin_at = positions[id(start)]
        finish_at = positions[id(users[index + 1])] if index + 1 < len(users) else len(ctx.walked)
        segments.append((
            start,
            [entry for entry in ctx.walked[begin_at + 1 : finish_at] if entry.step.source == 'agent'],
        ))
    return segments


def _evidence(
    entry: WalkedStep,
    *,
    reason: str = '',
    related: list[WalkedStep] | None = None,
    call_id: str | None = None,
    related_call_ids: list[str] | None = None,
) -> Evidence:
    return Evidence(
        step_id=entry.step.step_id,
        agent_path=entry.agent_path,
        call_id=call_id,
        related=[(item.agent_path, item.step.step_id) for item in related or []],
        related_call_ids=related_call_ids or [],
        reason=reason,
    )


def _has_users(ctx: SignalContext) -> Precondition:
    users = _users(ctx)
    return Precondition(name='has user messages', met=bool(users), detail=f'{len(users)} root user messages')


def autonomous_segment_count(ctx: SignalContext) -> SignalResult:
    users = _users(ctx)
    return result('autonomous_segment_count', 'C', len(users), [_evidence(entry) for entry in users])


def max_autonomous_steps(ctx: SignalContext) -> SignalResult:
    segments = _segments(ctx)
    counts = [(start, entries, sum(_steps_per_agent(entry, ctx) for entry in entries)) for start, entries in segments]
    maximum = max((count for _, _, count in counts), default=0)
    evidence = [
        _evidence(
            start,
            reason=f'{count} steps',
            related=entries,
            related_call_ids=[record.call.tool_call_id for entry in entries for record in _calls_at(ctx, entry)],
        )
        for start, entries, count in counts
        if count == maximum and count
    ]
    return result('max_autonomous_steps', 'C', maximum, evidence, [_has_users(ctx)])


def avg_autonomous_steps(ctx: SignalContext) -> SignalResult:
    segments = _segments(ctx)
    value = (
        round(sum(sum(_steps_per_agent(entry, ctx) for entry in entries) for _, entries in segments) / len(segments), 3)
        if segments
        else None
    )
    return result('avg_autonomous_steps', 'C', value, preconditions=[_has_users(ctx)])


def max_llm_tool_cycles(ctx: SignalContext) -> SignalResult:
    """Longest run of tool-calling LLM responses per agent path; child work cannot join its parent's run."""
    by_path: dict[tuple[str, ...], list[WalkedStep]] = {}
    for entry in _agent_steps(ctx):
        by_path.setdefault(entry.agent_path, []).append(entry)
    best: list[WalkedStep] = []
    for entries in by_path.values():
        run: list[WalkedStep] = []
        for entry in entries:
            if _calls_at(ctx, entry):
                run.append(entry)
                if len(run) > len(best):
                    best = run.copy()
            else:
                run = []
    evidence = (
        [
            _evidence(
                best[0],
                reason=f'{len(best)} cycles',
                related=best[1:],
                related_call_ids=[record.call.tool_call_id for entry in best for record in _calls_at(ctx, entry)],
            )
        ]
        if best
        else []
    )
    return result('max_llm_tool_cycles', 'C', len(best), evidence)


def _last_root_agent(ctx: SignalContext, before: int) -> WalkedStep | None:
    return next(
        (entry for entry in reversed(ctx.walked[:before]) if not entry.agent_path and entry.step.source == 'agent'),
        None,
    )


def _answer(entry: WalkedStep | None) -> bool:
    return bool(entry and not entry.step.tool_calls and atif_content_text(entry.step.message, 'signals').strip())


def terminal_answer_present(ctx: SignalContext) -> SignalResult:
    last = _last_root_agent(ctx, len(ctx.walked))
    has_agent = last is not None
    return result(
        'terminal_answer_present',
        'C',
        _answer(last),
        [_evidence(last)] if last else [],
        [Precondition(name='has agent turns', met=has_agent, detail='root agent steps')],
    )


def human_interruption_count(ctx: SignalContext) -> SignalResult:
    evidence = []
    for user_entry in _users(ctx):
        before = ctx.walked.index(user_entry)
        prior = _last_root_agent(ctx, before)
        if prior is not None and not _answer(prior):
            evidence.append(
                _evidence(
                    user_entry,
                    related=[prior],
                    related_call_ids=[record.call.tool_call_id for record in _calls_at(ctx, prior)],
                )
            )
    return result('human_interruption_count', 'C', len(evidence), evidence, [_has_users(ctx)])


def subagent_step_share(ctx: SignalContext) -> SignalResult:
    steps = _all_agent_steps(ctx)
    total = sum(_steps_per_agent(entry, ctx) for entry in steps)
    sub = [entry for entry in steps if entry.agent_path]
    numerator = sum(_steps_per_agent(entry, ctx) for entry in sub)
    value = round(numerator / total, 4) if total else None
    pcs = [
        Precondition(name='has steps', met=total > 0, detail=f'{total} agent and tool steps'),
        subagent_linkage(ctx),
    ]
    evidence = [
        _evidence(entry, related_call_ids=[record.call.tool_call_id for record in _calls_at(ctx, entry)])
        for entry in sub
    ]
    return result('subagent_step_share', 'C', value, evidence, pcs)


def subagent_message_share(ctx: SignalContext) -> SignalResult:
    total = len(ctx.walked)
    sub = [entry for entry in ctx.walked if entry.agent_path]
    value = round(len(sub) / total, 4) if total else None
    return result(
        'subagent_message_share',
        'C',
        value,
        [_evidence(entry) for entry in sub],
        [Precondition(name='has messages', met=total > 0, detail=f'{total} steps')],
    )


def parallel_tool_batch_count(ctx: SignalContext) -> SignalResult:
    evidence = [
        _evidence(
            entry,
            reason=f'{len(entry.step.tool_calls or [])} calls',
            related_call_ids=[record.call.tool_call_id for record in _calls_at(ctx, entry)],
        )
        for entry in _all_agent_steps(ctx)
        if len(entry.step.tool_calls or []) > 1
    ]
    return result('parallel_tool_batch_count', 'C', len(evidence), evidence)


def max_parallel_tool_calls(ctx: SignalContext) -> SignalResult:
    maximum = max((len(entry.step.tool_calls or []) for entry in _all_agent_steps(ctx)), default=0)
    evidence = [
        _evidence(entry, related_call_ids=[record.call.tool_call_id for record in _calls_at(ctx, entry)])
        for entry in _all_agent_steps(ctx)
        if maximum > 1 and len(entry.step.tool_calls or []) == maximum
    ]
    return result('max_parallel_tool_calls', 'C', maximum, evidence)


Interval = tuple[float, float, WalkedStep, str, str | None, bool]


def _step_epoch(entry: WalkedStep) -> float | None:
    if entry.step.timestamp is None:
        return None
    try:
        moment = parse_iso(entry.step.timestamp)
    except (ValueError, TypeError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def _llm_intervals(ctx: SignalContext) -> tuple[list[Interval], list[float], list[float]]:
    llm: list[Interval] = []
    exact_stamps: list[float] = []
    iso_stamps: list[float] = []
    prior: dict[tuple[str, ...], float] = {}
    for entry in ctx.walked:
        current = _step_epoch(entry)
        times = llm_times(entry)
        start, end = times.start, times.end
        invocation_data = invocation(entry.step)
        if start is not None and not times.approximate and invocation_data.get('start_timestamp') is not None:
            exact_stamps.append(start)
        if end is not None and not times.approximate and invocation_data.get('end_timestamp') is not None:
            exact_stamps.append(end)
        if current is not None:
            iso_stamps.append(current)
        if entry.step.source == 'agent' and entry.step.llm_call_count != 0:
            if start is not None and end is not None and end >= start:
                llm.append((start, end, entry, 'llm', None, False))
            elif (
                invocation(entry.step).get('start_timestamp') is None
                and invocation(entry.step).get('end_timestamp') is None
                and current is not None
                and prior.get(entry.agent_path) is not None
                and current >= prior[entry.agent_path]
            ):
                llm.append((prior[entry.agent_path], current, entry, 'llm', None, True))
        if current is not None:
            prior[entry.agent_path] = current
    return llm, exact_stamps, iso_stamps


def _tool_intervals(ctx: SignalContext) -> tuple[list[Interval], list[float]]:
    tool: list[Interval] = []
    stamps: list[float] = []
    for index, entry in enumerate(ctx.walked):
        following = next((item for item in ctx.walked[index + 1 :] if item.agent_path == entry.agent_path), None)
        next_time = _step_epoch(following) if following else None
        for record in _calls_at(ctx, entry):
            if record.result is None:
                continue
            start, end = tool_times(record.result)
            if start is not None:
                stamps.append(start)
            if end is not None:
                stamps.append(end)
            if start is not None and end is not None and end >= start:
                tool.append((start, end, entry, 'tool', record.call.tool_call_id, False))
            elif start is None and end is None:
                call_time = _step_epoch(entry)
                if call_time is not None and next_time is not None and next_time >= call_time:
                    tool.append((call_time, next_time, entry, 'tool', record.call.tool_call_id, True))
    return tool, stamps


def _intervals(ctx: SignalContext) -> tuple[list[Interval], list[Interval], list[float], list[float]]:
    llm, llm_exact_stamps, iso_stamps = _llm_intervals(ctx)
    tool, tool_stamps = _tool_intervals(ctx)
    return llm, tool, [*llm_exact_stamps, *tool_stamps], iso_stamps


def _union(intervals: list[Interval]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end, _, _, _, _ in sorted(intervals, key=itemgetter(0, 1)):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _length(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def _minus(intervals: list[tuple[float, float]], cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for start, end in intervals:
        cursor = start
        for cut_start, cut_end in cuts:
            if cut_end <= cursor or cut_start >= end:
                continue
            if cut_start > cursor:
                out.append((cursor, cut_start))
            cursor = max(cursor, cut_end)
        if cursor < end:
            out.append((cursor, end))
    return out


def _timing(ctx: SignalContext) -> tuple[dict[str, Any], list[Precondition]]:
    llm, tool, exact_stamps, iso_stamps = _intervals(ctx)
    stamps = [*exact_stamps, *iso_stamps]
    llm_coverage = timestamps(ctx)
    tool_coverage = tool_timestamps(ctx)
    pcs = [
        Precondition(name='usable timestamps', met=bool(stamps), detail=f'{len(stamps)} usable timestamps'),
        llm_coverage,
        tool_coverage,
    ]
    if not stamps or (not llm and not tool):
        pcs.append(Precondition(name='activity intervals', met=False, detail='no LLM or tool interval could be timed'))
        return {}, pcs
    llm_union = _union(llm)
    earliest, latest = min(stamps), max(stamps)
    exact_earliest = any(isclose(earliest, stamp, abs_tol=0.000001) for stamp in exact_stamps)
    exact_latest = any(isclose(latest, stamp, abs_tol=0.000001) for stamp in exact_stamps)
    return (
        {
            'wall': latest - earliest,
            'wall_approximate': not exact_earliest or not exact_latest,
            'active': _length(_union(llm + tool)),
            'llm': _length(llm_union),
            'tool': _length(_minus(_union(tool), llm_union)),
            'llm_intervals': llm,
            'tool_intervals': tool,
            'llm_incomplete': llm_coverage.met is not True,
            'tool_incomplete': tool_coverage.met is not True,
        },
        pcs,
    )


def _time_signal(name: str, ctx: SignalContext, key: str, kinds: tuple[str, ...]) -> SignalResult:
    timing, pcs = _timing(ctx)
    approximate = (
        timing.get('wall_approximate', False)
        if key == 'wall'
        else any(interval[5] for kind in kinds for interval in timing.get(f'{kind}_intervals', []))
    )
    approximate = approximate or timing.get('llm_incomplete', False)
    if key != 'llm':
        approximate = approximate or timing.get('tool_incomplete', False)
    if not timing:
        return result(name, 'C', None, preconditions=pcs, approximate=approximate)
    evidence = [
        _evidence(entry, reason=f'{round((end - start) * 1000)} ms', call_id=call_id)
        for kind in kinds
        for start, end, entry, _, call_id, _ in timing[f'{kind}_intervals']
    ]
    return result(name, 'C', round(timing[key] * 1000), evidence, pcs, approximate=approximate)


def wall_time_ms(ctx: SignalContext) -> SignalResult:
    return _time_signal('wall_time_ms', ctx, 'wall', ())


def active_time_ms(ctx: SignalContext) -> SignalResult:
    return _time_signal('active_time_ms', ctx, 'active', ('llm', 'tool'))


def llm_time_ms(ctx: SignalContext) -> SignalResult:
    return _time_signal('llm_time_ms', ctx, 'llm', ('llm',))


def tool_time_ms(ctx: SignalContext) -> SignalResult:
    return _time_signal('tool_time_ms', ctx, 'tool', ('tool',))


def max_autonomous_duration_ms(ctx: SignalContext) -> SignalResult:
    segments = _segments(ctx)
    candidates: list[tuple[float, WalkedStep, list[WalkedStep]]] = []
    pcs = [_has_users(ctx), timestamps(ctx), tool_timestamps(ctx)]
    for start, entries in segments:
        start_time = _step_epoch(start)
        if start_time is None:
            continue
        end_times = [_step_epoch(entry) for entry in entries]
        end_times = [value for value in end_times if value is not None]
        end_times.extend(
            endpoint
            for entry in entries
            for endpoint in (llm_times(entry).start, llm_times(entry).end)
            if endpoint is not None
        )
        end_times.extend(
            endpoint
            for entry in entries
            for call_record in _calls_at(ctx, entry)
            if call_record.result is not None
            for endpoint in tool_times(call_record.result)
            if endpoint is not None
        )
        candidates.append((max(end_times, default=start_time) - start_time, start, entries))
    if not candidates:
        pcs.append(Precondition(name='timed segments', met=False, detail='no autonomous segment has usable timestamps'))
        return result('max_autonomous_duration_ms', 'C', None, preconditions=pcs)
    duration, start, entries = max(candidates, key=itemgetter(0))
    entry_ids = {id(entry) for entry in entries}
    llm, tool, _, _ = _intervals(ctx)
    timed_intervals = [interval for interval in llm + tool if id(interval[2]) in entry_ids]
    calls_with_timestamps = [
        record.call.tool_call_id
        for entry in entries
        for record in _calls_at(ctx, entry)
        if record.result is not None and any(value is not None for value in tool_times(record.result))
    ]
    related_call_ids = list(
        dict.fromkeys([
            *calls_with_timestamps,
            *(interval[4] for interval in timed_intervals if interval[4] is not None),
        ])
    )
    # The segment starts at an ISO user-step timestamp, which is approximate under the ATIF timing map.
    return result(
        'max_autonomous_duration_ms',
        'C',
        round(duration * 1000),
        [
            _evidence(
                start,
                reason=f'{round(duration * 1000)} ms',
                related=entries,
                related_call_ids=related_call_ids,
            )
        ],
        pcs,
        approximate=True,
    )


SIGNALS: dict[str, tuple[Group, SignalFn]] = {
    'autonomous_segment_count': ('C', autonomous_segment_count),
    'max_autonomous_steps': ('C', max_autonomous_steps),
    'avg_autonomous_steps': ('C', avg_autonomous_steps),
    'max_llm_tool_cycles': ('C', max_llm_tool_cycles),
    'terminal_answer_present': ('C', terminal_answer_present),
    'human_interruption_count': ('C', human_interruption_count),
    'subagent_step_share': ('C', subagent_step_share),
    'subagent_message_share': ('C', subagent_message_share),
    'parallel_tool_batch_count': ('C', parallel_tool_batch_count),
    'max_parallel_tool_calls': ('C', max_parallel_tool_calls),
    'wall_time_ms': ('C', wall_time_ms),
    'active_time_ms': ('C', active_time_ms),
    'llm_time_ms': ('C', llm_time_ms),
    'tool_time_ms': ('C', tool_time_ms),
    'max_autonomous_duration_ms': ('C', max_autonomous_duration_ms),
}
