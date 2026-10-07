"""One flat, ordered view of a trajectory and its subagents, for every signal to read.

`walk` flattens the trajectory tree depth-first; `calls` joins each tool call to its result. Both skip
compaction steps and copied context, which are bookkeeping, not agent behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timezone
from typing import TYPE_CHECKING, Any

from evaluatorq.formats._shared import COMPACTION_EXTRA_KEY, RAW_ARGUMENTS_EXTRA_KEY, atif_content_text, parse_iso

if TYPE_CHECKING:
    from evaluatorq.formats.atif import (
        AtifObservationResult,
        AtifStep,
        AtifSubagentRef,
        AtifToolCall,
        AtifTrajectory,
    )
    from evaluatorq.signals.config import SignalsConfig
    from evaluatorq.signals.models import SignalResult

_PROVIDER_PREFIXES: tuple[tuple[str, str], ...] = (
    ('claude', 'anthropic'),
    ('gpt', 'openai'),
    ('o1', 'openai'),
    ('o3', 'openai'),
    ('o4', 'openai'),
    ('gemini', 'google'),
    ('mistral', 'mistral'),
    ('llama', 'meta'),
    ('grok', 'xai'),
    ('deepseek', 'deepseek'),
    ('qwen', 'alibaba'),
)
_RESPONSES_TOOLS_KEY = 'evaluatorq.responses_tools'
_RESPONSES_OUTPUT_ITEMS_KEY = 'evaluatorq.responses_output_items'


@dataclass(frozen=True)
class WalkedStep:
    """A step in walk order, with where it sits in the trajectory tree.

    Attributes:
        step: The step itself.
        agent_path: `trajectory_id`s from the root down to the trajectory holding the step; the root is `()`.
        depth: Subagent nesting level, root 0. For an unlinked subagent this is 1 whatever its real nesting.
        trajectory: The trajectory holding the step (the root or an embedded subagent).
        unlinked: True for steps of an embedded subagent that no observation result references. Those
            subagents are appended after the referenced steps, at depth 1, and every step in such a subtree is
            flagged, so `subagent_linkage` can report them.
    """

    step: AtifStep
    agent_path: tuple[str, ...]
    depth: int
    trajectory: AtifTrajectory
    unlinked: bool = False


@dataclass(frozen=True)
class CallRecord:
    """A tool call joined to its result.

    Attributes:
        seq: Position in trajectory-wide call order.
        step: The walked agent step that made the call.
        call: The call itself.
        result: The result answering it (by `source_call_id`, within the same step), or None for an orphan call.
    """

    seq: int
    step: WalkedStep
    call: AtifToolCall
    result: AtifObservationResult | None

    @property
    def is_error(self) -> bool | None:
        """True when the source marked the result an error, False when it marked a status, else None (unknown).

        Reads `error_type` and `status` from the result's `extra`, where the converters put them.
        """
        extra = self.result.extra if self.result is not None else None
        if not extra:
            return None
        if extra.get('error_type') or extra.get('status') == 'error':
            return True
        if extra.get('status') is not None:
            return False
        return None


@dataclass(frozen=True)
class LlmTimes:
    """When an LLM step started and ended, in epoch seconds. `approximate` when read from the step's ISO timestamp."""

    start: float | None
    end: float | None
    approximate: bool


def raw_tool_arguments(call: AtifToolCall) -> str | None:
    """Original non-JSON tool argument text carried by the format converter, if any."""
    raw = (call.extra or {}).get(RAW_ARGUMENTS_EXTRA_KEY)
    return raw if isinstance(raw, str) else None


def is_skipped(step: AtifStep) -> bool:
    """True for compaction steps and copied context, which no count or walk includes."""
    extra = step.extra or {}
    return COMPACTION_EXTRA_KEY in extra or bool(extra.get('context_management')) or step.is_copied_context is True


def walk(trajectory: AtifTrajectory) -> list[WalkedStep]:
    """Flatten a trajectory and its embedded subagents, depth-first, skipping compaction and copied-context steps.

    A subagent's steps are placed right after the step whose observation references it (by
    `subagent_trajectory_ref[].trajectory_id`), in the order the step references them. A subagent is placed once,
    and a reference to a trajectory that is not embedded (a `trajectory_path` alone) is ignored. Embedded
    subagents that nothing references are appended at the end at depth 1 with `unlinked=True`. A subagent
    referenced only from a skipped step (compaction or copied context) counts as unreferenced, so it is
    unlinked too.
    """
    registry: dict[str, list[AtifTrajectory]] = {}
    embedded: list[AtifTrajectory] = []
    _collect(trajectory, registry, embedded)
    placed: set[int] = {id(trajectory)}
    out: list[WalkedStep] = []

    def visit(current: AtifTrajectory, path: tuple[str, ...], depth: int, *, unlinked: bool) -> None:
        local = {sub.trajectory_id: sub for sub in current.subagent_trajectories or [] if sub.trajectory_id}
        for step in current.steps:
            if is_skipped(step):
                continue
            out.append(WalkedStep(step, path, depth, current, unlinked))
            for ref in _refs(step):
                sub_id = ref.trajectory_id
                if sub_id is None:
                    continue
                candidates = registry.get(sub_id, [])
                sub = local.get(sub_id) or next((item for item in candidates if id(item) not in placed), None)
                if sub is None or id(sub) in placed:
                    continue
                placed.add(id(sub))
                visit(sub, (*path, sub_id), depth + 1, unlinked=unlinked)

    visit(trajectory, (), 0, unlinked=False)
    for sub in embedded:
        if id(sub) not in placed and sub.trajectory_id is not None:
            placed.add(id(sub))
            visit(sub, (sub.trajectory_id,), 1, unlinked=True)
    return out


def _collect(
    trajectory: AtifTrajectory, registry: dict[str, list[AtifTrajectory]], embedded: list[AtifTrajectory]
) -> None:
    for sub in trajectory.subagent_trajectories or []:
        if sub.trajectory_id is not None:
            registry.setdefault(sub.trajectory_id, []).append(sub)
            embedded.append(sub)
            _collect(sub, registry, embedded)


def _refs(step: AtifStep) -> list[AtifSubagentRef]:
    if step.observation is None:
        return []
    return [ref for result in step.observation.results for ref in result.subagent_trajectory_ref or []]


def calls(walked: list[WalkedStep]) -> list[CallRecord]:
    """Every tool call in walk order, each joined to its result by `source_call_id` within the same step.

    The first result for a call id wins. A result with no `source_call_id` answers no call and is not returned.
    """
    records: list[CallRecord] = []
    for entry in walked:
        if not entry.step.tool_calls:
            continue
        results: dict[str, AtifObservationResult] = {}
        for item in entry.step.observation.results if entry.step.observation else []:
            if item.source_call_id is not None:
                results.setdefault(item.source_call_id, item)
        for call in entry.step.tool_calls:
            records.append(CallRecord(len(records), entry, call, results.get(call.tool_call_id)))
    return records


def result_text(result: AtifObservationResult) -> str:
    """A result's content as text (image and audio parts become warned `[type: path]` markers)."""
    return atif_content_text(result.content, 'signals') if result.content is not None else ''


def step_model(entry: WalkedStep) -> str | None:
    """The step's model, else the model of the trajectory holding it."""
    return entry.step.model_name or entry.trajectory.agent.model_name


def is_llm_step(entry: WalkedStep) -> bool:
    """An agent step that made an LLM call: `llm_call_count` above zero, or (when unset) a known model."""
    step = entry.step
    if step.source != 'agent' or step.llm_call_count == 0:
        return False
    return bool(step.llm_call_count) or bool(step_model(entry))


def infer_provider(model: str | None) -> str | None:
    """Provider from a model name: the `provider/` prefix if present, else a known name prefix."""
    if not model:
        return None
    if '/' in model:
        return model.split('/', 1)[0]
    lowered = model.lower()
    return next((provider for prefix, provider in _PROVIDER_PREFIXES if lowered.startswith(prefix)), None)


def invocation(step: AtifStep) -> dict[str, Any]:
    """The step's `extra['invocation']` mapping, or an empty one."""
    value = (step.extra or {}).get('invocation')
    return value if isinstance(value, dict) else {}


def finish_reasons(step: AtifStep) -> list[str]:
    """The step's finish reasons from `extra['invocation']['finish_reasons']`."""
    value = invocation(step).get('finish_reasons')
    return [str(item) for item in value] if isinstance(value, list) else []


def _epoch(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def llm_times(entry: WalkedStep) -> LlmTimes:
    """Invocation timestamps (epoch seconds) when present, else the step's ISO timestamp as an approximate start."""
    info = invocation(entry.step)
    start, end = _epoch(info.get('start_timestamp')), _epoch(info.get('end_timestamp'))
    if start is not None or end is not None:
        return LlmTimes(start, end, approximate=False)
    if entry.step.timestamp is not None:
        moment = parse_iso(entry.step.timestamp)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return LlmTimes(moment.timestamp(), None, approximate=True)
    return LlmTimes(None, None, approximate=False)


def tool_times(result: AtifObservationResult) -> tuple[float | None, float | None]:
    """A tool result's start and end (epoch seconds) from `extra['start_timestamp']` and `['end_timestamp']`."""
    extra = result.extra or {}
    return _epoch(extra.get('start_timestamp')), _epoch(extra.get('end_timestamp'))


def tool_schemas(trajectory: AtifTrajectory) -> dict[str, dict[str, Any]]:
    """Tool name to its definition, from `agent.tool_definitions` (OpenAI-style `function` wrapper or flat)."""
    return _tool_schemas(trajectory.agent.tool_definitions or [])


def call_tool_schemas(record: CallRecord, root: AtifTrajectory) -> dict[str, dict[str, Any]]:
    """Schemas active for one call, preferring the exact Responses response that emitted it.

    A present response-local list is authoritative, including an empty list. ATIF tool definitions remain the
    fallback for trajectories without response-local metadata.
    """
    extra = record.step.step.extra or {}
    response_tools = extra.get(_RESPONSES_TOOLS_KEY)
    if isinstance(response_tools, list):
        return _tool_schemas(response_tools)
    own = tool_schemas(record.step.trajectory)
    return {**tool_schemas(root), **own}


def _tool_schemas(definitions: list[Any]) -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}
    for definition in definitions:
        if not isinstance(definition, dict):
            continue
        wrapped = definition.get('function')
        body = wrapped if isinstance(wrapped, dict) else definition
        name = body.get('name')
        if isinstance(name, str):
            schemas[name] = body
    return schemas


def has_unmapped_tool_activity(walked: list[WalkedStep]) -> list[str]:
    """Describe Responses tool calls kept outside ATIF `tool_calls` because their semantics are unsupported."""
    activity: list[str] = []
    for entry in walked:
        items = (entry.step.extra or {}).get(_RESPONSES_OUTPUT_ITEMS_KEY)
        if not isinstance(items, list):
            continue
        represented_call_ids = {
            call.tool_call_id for call in entry.step.tool_calls or [] if isinstance(call.tool_call_id, str)
        }
        for item in items:
            if not isinstance(item, dict):
                continue
            kind = item.get('type')
            if kind == 'function_call':
                call_id = item.get('call_id')
                unsupported_call = not isinstance(call_id, str) or not call_id or call_id not in represented_call_ids
            else:
                unsupported_call = isinstance(kind, str) and kind.endswith('_call')
            unsupported_output = (
                isinstance(kind, str) and kind.endswith('_call_output') and kind != 'function_call_output'
            )
            if unsupported_call or unsupported_output:
                label = item.get('name') or item.get('server_label') or kind
                activity.append(f'{label} @step {entry.step.step_id}')
    return activity


@dataclass(frozen=True)
class SignalContext:
    """Everything a signal reads, built once per report by `compute_signals` and shared by every signal.

    Attributes:
        trajectory: The root trajectory.
        config: The rule knobs.
        walked: `walk(trajectory)`.
        calls: `calls(walked)`.
        results: Signals already computed in this report, by name, filled by `compute_signals` as it goes in
            registry order. The dict is mutable though the context is frozen, so a signal must only read it. Group
            D (tags) reads the metrics it thresholds from here, and relies on groups A to C being registered first.
    """

    trajectory: AtifTrajectory
    config: SignalsConfig
    walked: list[WalkedStep]
    calls: list[CallRecord]
    results: dict[str, SignalResult] = field(default_factory=dict)

    @classmethod
    def build(cls, trajectory: AtifTrajectory, config: SignalsConfig) -> SignalContext:
        """Walk the trajectory and pair its calls, once."""
        walked = walk(trajectory)
        return cls(trajectory, config, walked, calls(walked))
