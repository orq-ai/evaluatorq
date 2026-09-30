"""Small builders for `AtifTrajectory` inputs: `traj([user('hi'), agent(calls=[call('Bash', {...})], results=[ok('x')])])`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from evaluatorq.formats.atif import (
    AtifAgent,
    AtifMetrics,
    AtifObservation,
    AtifObservationResult,
    AtifStep,
    AtifSubagentRef,
    AtifToolCall,
    AtifTrajectory,
)
from evaluatorq.formats._shared import compaction_extra


@dataclass
class StepSpec:
    """A step without an id; `traj` numbers them 1..n."""

    source: Literal['system', 'user', 'agent']
    message: str = ''
    calls: list[AtifToolCall] = field(default_factory=list)
    results: list[AtifObservationResult] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)


def user(message: str = 'hi') -> StepSpec:
    return StepSpec('user', message)


def system(message: str = 'be helpful') -> StepSpec:
    return StepSpec('system', message)


def compaction() -> StepSpec:
    return StepSpec('system', '', kwargs={'extra': compaction_extra([{'type': 'compaction'}])})


def copied(message: str = 'earlier turn') -> StepSpec:
    return StepSpec('user', message, kwargs={'is_copied_context': True})


def call(name: str, arguments: dict[str, Any] | None = None, call_id: str | None = None) -> AtifToolCall:
    return AtifToolCall(tool_call_id=call_id or f'call-{name}', function_name=name, arguments=arguments or {})


def ok(content: str = 'done', *, call_id: str | None = None, **extra: Any) -> AtifObservationResult:
    """A result with `status: ok`; `call_id` defaults to the call at the same position (set by `agent`)."""
    return AtifObservationResult(source_call_id=call_id, content=content, extra={'status': 'ok', **extra})


def failed(content: str = 'boom', *, call_id: str | None = None, **extra: Any) -> AtifObservationResult:
    return AtifObservationResult(source_call_id=call_id, content=content, extra={'status': 'error', **extra})


def bare(content: str = 'done', *, call_id: str | None = None) -> AtifObservationResult:
    """A result with no `extra`, so `is_error` is unknown."""
    return AtifObservationResult(source_call_id=call_id, content=content)


def spawned(trajectory_id: str, *, call_id: str | None = None, content: str = 'sub done') -> AtifObservationResult:
    return AtifObservationResult(
        source_call_id=call_id,
        content=content,
        subagent_trajectory_ref=[AtifSubagentRef(trajectory_id=trajectory_id)],
    )


def agent(
    message: str = 'ok',
    *,
    calls: list[AtifToolCall] | None = None,
    results: list[AtifObservationResult] | None = None,
    model: str | None = None,
    metrics: AtifMetrics | None = None,
    **kwargs: Any,
) -> StepSpec:
    """An agent step. Results without a `source_call_id` are paired to the calls by position."""
    calls = calls or []
    paired = [
        r
        if r.source_call_id is not None or i >= len(calls)
        else r.model_copy(update={'source_call_id': calls[i].tool_call_id})
        for i, r in enumerate(results or [])
    ]
    extra_kwargs = {**kwargs, **({'model_name': model} if model else {}), **({'metrics': metrics} if metrics else {})}
    return StepSpec('agent', message, calls, paired, extra_kwargs)


def traj(
    steps: list[StepSpec],
    *,
    trajectory_id: str | None = None,
    subagents: list[AtifTrajectory] | None = None,
    model: str | None = None,
    tool_definitions: list[dict[str, Any]] | None = None,
) -> AtifTrajectory:
    built = [
        AtifStep(
            step_id=index,
            source=spec.source,
            message=spec.message,
            tool_calls=spec.calls or None,
            observation=AtifObservation(results=spec.results) if spec.results else None,
            **spec.kwargs,
        )
        for index, spec in enumerate(steps, start=1)
    ]
    return AtifTrajectory(
        trajectory_id=trajectory_id,
        agent=AtifAgent(name='test', version='0', model_name=model, tool_definitions=tool_definitions),
        steps=built,
        subagent_trajectories=subagents or None,
    )
