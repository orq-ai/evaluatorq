"""Convert between OTel GenAI traces and ATIF trajectories.

OTel -> ATIF reads one agent run per `invoke_agent` scope: `chat` spans become steps in start-time order, and
the conversation comes from each chat span's *new* input (the part the previous chat span had not seen), so a
trace whose chat spans replay the full history (Orq does) is not duplicated. The replayed prefix is checked
against what the previous span saw; a span whose history was rewritten is read by content instead (see
`_new_input`), and a compaction in the input becomes a system step. Tool results come from
`execute_tool` spans or, failing those, from `tool_call_response` parts in later chat inputs, and attach to the
step that issued the call. Nested `invoke_agent` spans become `subagent_trajectories`. Usage is read from chat
spans only: `invoke_agent` spans carry inclusive totals that would double-count.

ATIF -> OTel emits the GenAI semconv layout: one `invoke_agent` span, and under it one `chat` span per agent step
plus one `execute_tool` span per tool call as siblings. User and system steps after the last agent step go into
one final `chat` span that has input and no output, which OTel -> ATIF reads back as those steps and no agent
step. Ids are SHA-256 of the session and trajectory ids. ATIF `extra` data with no semconv home, and
`final_metrics`, travel as JSON-string attributes under `evaluatorq.atif.` and are read back on the way in.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from loguru import logger

from evaluatorq.common.fields import get_field
from evaluatorq.contracts import tool_result_to_text
from evaluatorq.formats._ids import content_seed, stable_hex
from evaluatorq.formats._shared import (
    RAW_ARGUMENTS_EXTRA_KEY,
    atif_content_text,
    atif_tool_arguments,
    compaction_extra,
    final_metrics,
    join_text,
    json_arguments_text,
)
from evaluatorq.formats.atif import (
    AtifAgent,
    AtifContentPart,
    AtifFinalMetrics,
    AtifMetrics,
    AtifObservation,
    AtifObservationResult,
    AtifStep,
    AtifSubagentRef,
    AtifToolCall,
    AtifTrajectory,
)
from evaluatorq.formats.otel import (
    OtelCompactionPart,
    OtelGenericPart,
    OtelMessage,
    OtelPart,
    OtelReasoningPart,
    OtelSpan,
    OtelTextPart,
    OtelToolCallPart,
    OtelToolCallResponsePart,
    OtelTrace,
    OtelUriPart,
    parse_time,
    span_usage,
)

if TYPE_CHECKING:
    from datetime import datetime

_NO_STEPS = 'OtelTrace has no chat spans; ATIF needs at least one step'
_UNKNOWN = 'unknown'
_OTHER_ERROR = '_OTHER'  # the semconv `error.type` fallback when a failed span names no error
_STATUSES = frozenset({'ok', 'error', 'unset'})
_WHITESPACE = re.compile(r'\s+')
_COMPACTION = 'compaction'

# ATIF data OTel has no attribute for, carried as JSON strings so OTel -> ATIF can restore it.
_STEP_EXTRA = 'evaluatorq.atif.step.extra'
_TRAJECTORY_EXTRA = 'evaluatorq.atif.trajectory.extra'
_SUBAGENT_RESULT_OFFSET = 'evaluatorq.atif.subagent_result_offset'
_SUBAGENT_SOURCE_TRAJECTORY_ID = 'evaluatorq.atif.subagent_source_trajectory_id'
_FINAL_METRICS = 'evaluatorq.atif.final_metrics'
_TOOL_CALL_EXTRA = 'evaluatorq.atif.tool_call.extra'
_RESULT_EXTRA = 'evaluatorq.atif.result.extra'
# Step extra keys the chat span already maps (or OTel -> ATIF derives from the span); never carried as JSON.
_MAPPED_STEP_EXTRA = frozenset({'ancestry', 'invocation', 'finish_reasons', 'error_type', 'requested_model'})


@dataclass
class _Scope:
    """The spans of one agent run, and the nested `invoke_agent` spans that become its subagents."""

    members: list[OtelSpan]
    subagents: list[OtelSpan]


@dataclass
class _Draft:
    """One ATIF step being assembled; `span` is the chat span of an agent step."""

    fields: dict[str, Any]
    span: OtelSpan | None = None
    results: list[AtifObservationResult] = field(default_factory=list)
    tool_span_ids: set[str] = field(default_factory=set)
    tool_span_result_indices: dict[str, int] = field(default_factory=dict)
    tool_span_result_groups: dict[str, list[int]] = field(default_factory=dict)


# OTel -> ATIF


def otel_to_atif(trace: OtelTrace, *, agent_name: str = _UNKNOWN, agent_version: str = _UNKNOWN) -> AtifTrajectory:
    """Build an ATIF trajectory from an OTel trace; see the module docstring for the mapping.

    The agent name and version fall back to `gen_ai.agent.name` / `gen_ai.agent.version` of the scope's
    `invoke_agent` span when left at `'unknown'`.

    `evaluatorq.atif.*` attributes written by `atif_to_otel` are merged back into the matching `extra` and
    `final_metrics`; a key the trace itself yields keeps the trace's value (a differing carried value is warned).

    A compaction part in a chat input (an `OtelCompactionPart`, or a Responses `compaction` item wrapped in a
    `data` part) becomes a system step with `extra.context_management = {"type": "compaction", "boundary":
    "replace"}`, the compaction parts under `extra["evaluatorq.compaction"]`, and any plain-text summary as the
    message.

    Lost: chat input messages before the first chat span (prior history, warned), extra output choices
    (warned), span attributes other than model, usage, finish reasons, error type and `evaluatorq.atif.*`, and
    spans that are not `chat`, `execute_tool` or `invoke_agent`.

    Raises:
        ValueError: The trace has no chat span that yields a step.
    """
    tops = _root_scope(trace)
    scope_root = tops[0] if len(tops) == 1 and tops[0].operation == 'invoke_agent' else None
    name = agent_name if agent_name != _UNKNOWN else _attribute(scope_root, 'gen_ai.agent.name') or _UNKNOWN
    version = agent_version if agent_version != _UNKNOWN else _attribute(scope_root, 'gen_ai.agent.version')
    traj = _build(
        trace,
        tops,
        session_id=_session_id(trace, tops),
        trajectory_id=None,
        agent=AtifAgent(name=name, version=version or _UNKNOWN),
    )
    if traj is None:
        raise ValueError(_NO_STEPS)
    return traj


def _root_scope(trace: OtelTrace) -> list[OtelSpan]:
    """Return the top spans of the run: one `invoke_agent` span, or every root merged into a synthetic scope."""
    roots = trace.roots()
    if len(roots) > 1:
        logger.warning('OtelTrace has {} roots; converting them as one run in start-time order', len(roots))
        return roots
    if not roots or roots[0].operation == 'invoke_agent':
        return roots
    agents = [span for span in trace.children(roots[0].span_id) if span.operation == 'invoke_agent']
    return agents if len(agents) == 1 else roots  # Orq wraps its one invoke_agent span in an AgentInvoke root


def _collect(trace: OtelTrace, tops: list[OtelSpan]) -> _Scope:
    """Walk down from `tops`. An `invoke_agent` span below another one starts a subagent and is not descended."""
    scope = _Scope(members=[], subagents=[])
    seen: set[str] = set()
    stack = [(top, top.operation == 'invoke_agent') for top in reversed(tops)]
    while stack:
        span, in_agent = stack.pop()
        if span.span_id in seen:
            continue
        seen.add(span.span_id)
        scope.members.append(span)
        for child in reversed(trace.children(span.span_id)):
            if child.operation == 'invoke_agent' and in_agent:
                scope.subagents.append(child)
            else:
                stack.append((child, in_agent or child.operation == 'invoke_agent'))
    scope.subagents = _by_start(scope.subagents)
    return scope


def _by_start(spans: list[OtelSpan]) -> list[OtelSpan]:
    # sorted() is stable, so ties (and spans without a time, which go last) keep list order
    return sorted(spans, key=lambda s: (s.start_time is None, s.start_time.timestamp() if s.start_time else 0.0))


def _session_id(trace: OtelTrace, tops: list[OtelSpan]) -> str:
    for span in [*tops, *trace.roots()]:
        conversation = span.attributes.get('gen_ai.conversation.id')
        if isinstance(conversation, str) and conversation:
            return conversation
    trace_id = next((span.trace_id for span in tops if span.trace_id), None)
    return trace_id or stable_hex(content_seed([span.span_id for span in trace.spans]), length=16)


def _attribute(span: OtelSpan | None, key: str) -> str | None:
    value = span.attributes.get(key) if span is not None else None
    return value if isinstance(value, str) and value else None


def _build(
    trace: OtelTrace, tops: list[OtelSpan], *, session_id: str, trajectory_id: str | None, agent: AtifAgent
) -> AtifTrajectory | None:
    """Build the trajectory of one scope, or None when it yields no step."""
    scope = _collect(trace, tops)
    by_id = {span.span_id: span for span in trace.spans}
    chats = _by_start([
        span for span in scope.members if span.operation == 'chat' and not _mirrored_router_root(trace, span)
    ])
    drafts, new_inputs = _chat_steps(chats, by_id)
    _attach_results(drafts, scope.members, chats, new_inputs)
    subagents = _subagents(trace, scope, drafts, session_id, trajectory_id, by_id)
    if not drafts:
        return None
    agent_span = tops[0] if len(tops) == 1 and tops[0].operation == 'invoke_agent' else None
    steps = [
        AtifStep(
            step_id=index + 1,
            observation=AtifObservation(results=draft.results) if draft.results else None,
            **draft.fields,
        )
        for index, draft in enumerate(drafts)
    ]
    return AtifTrajectory(
        session_id=session_id,
        trajectory_id=trajectory_id,
        agent=agent,
        steps=steps,
        final_metrics=_final_metrics(steps, agent_span),
        extra=_carried(agent_span, _TRAJECTORY_EXTRA),
        subagent_trajectories=subagents or None,
    )


def _mirrored_router_root(trace: OtelTrace, span: OtelSpan) -> bool:
    """Skip an Orq trace wrapper when its direct child holds the same router chat call."""
    if span.span_type != 'trace' or span.output_messages is None:
        return False
    mirrored = any(
        child.operation == 'chat'
        and _same_transcript(child.input_messages, span.input_messages)
        and _same_transcript(child.output_messages, span.output_messages)
        and _same_parts(child.system_instructions, span.system_instructions)
        for child in trace.children(span.span_id)
    )
    if mirrored:
        logger.info('Router trace span {} mirrors a child chat span; reading the child once', span.span_id)
    return mirrored


def _same_transcript(left: list[OtelMessage] | None, right: list[OtelMessage] | None) -> bool:
    """Compare conversation content, excluding message ids, names and finish metadata."""
    if left is None or right is None:
        return left is right
    return len(left) == len(right) and all(
        _normalized_role(a.role) == _normalized_role(b.role) and _same_parts(a.parts, b.parts)
        for a, b in zip(left, right, strict=True)
    )


def _same_parts(left: list[OtelPart] | None, right: list[OtelPart] | None) -> bool:
    if left is None or right is None:
        return left is right
    return len(left) == len(right) and all(
        _semantic_part(a) == _semantic_part(b) for a, b in zip(left, right, strict=True)
    )


def _semantic_part(part: OtelPart) -> dict[str, Any]:
    """Keep known part fields and the full payload of an unknown part for mirror comparison."""
    payload = part.model_dump(mode='json', exclude_none=True)
    if isinstance(part, OtelGenericPart):
        return payload
    return {key: value for key, value in payload.items() if key in type(part).model_fields}


def _final_metrics(steps: list[AtifStep], agent_span: OtelSpan | None) -> AtifFinalMetrics | None:
    derived = final_metrics(steps)
    carried = _carried(agent_span, _FINAL_METRICS)
    if carried is None:
        return derived
    merged = _merge(derived.model_dump(exclude_none=True) if derived else None, carried, _FINAL_METRICS)
    try:
        return AtifFinalMetrics.model_validate(merged)
    except ValueError as exc:
        logger.warning('{} does not validate ({}); using the totals summed from the steps', _FINAL_METRICS, exc)
        return derived


def _carried(span: OtelSpan | None, key: str) -> dict[str, Any] | None:
    """The JSON object `atif_to_otel` stored under `key`, or None when absent or unreadable (warned)."""
    raw = span.attributes.get(key) if span is not None else None
    if raw is None:
        return None
    try:
        value: Any = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        value = None
    if value == {}:
        return None
    if not isinstance(value, dict):
        logger.warning('Span {} attribute {} is not a JSON object; ignoring it', span.span_id if span else '', key)
        return None
    return cast('dict[str, Any]', value)


def _merge(derived: dict[str, Any] | None, carried: dict[str, Any] | None, where: str) -> dict[str, Any] | None:
    """Carried keys plus derived ones; a derived key wins, and a differing carried value for it is warned."""
    if not carried:
        return derived
    derived = derived or {}
    clashes = sorted(key for key, value in derived.items() if key in carried and carried[key] != value)
    if clashes:
        logger.warning('{}: keeping the values the trace yields for {}, over the carried ones', where, clashes)
    return {**carried, **derived}


def _chat_steps(chats: list[OtelSpan], by_id: dict[str, OtelSpan]) -> tuple[list[_Draft], list[list[OtelMessage]]]:
    """Cut steps from chat spans; also return each span's new input (empty for the first, whose input is history)."""
    drafts: list[_Draft] = []
    new_inputs: list[list[OtelMessage]] = []
    last_system: list[OtelPart] | None = None
    for index, chat in enumerate(chats):
        if chat.system_instructions and chat.system_instructions != last_system:
            drafts.extend(_system_instruction_steps(chat))
            last_system = chat.system_instructions
        if index == 0:
            drafts.extend(_history(chat.input_messages or [], first=True))
            new_inputs.append([])
        else:
            new = _new_input(chats[index - 1], chat)
            drafts.extend(_history(new, first=False))
            new_inputs.append(new)
        agent = _agent_step(chat, by_id)
        if agent is not None:
            drafts.append(agent)
    return drafts, new_inputs


def _new_input(prev: OtelSpan, chat: OtelSpan) -> list[OtelMessage]:
    """Return new input from a cumulative, partial, or per-turn chat span.

    Compare complete message occurrences in order. A rewritten history may overlap the previous one; the
    earliest longest overlap keeps a repeated turn that follows it. A compaction starts a new context and is
    carried as a step.
    """
    seen = [*(prev.input_messages or []), *(prev.output_messages or [])]
    messages = chat.input_messages or []
    signatures = [_signature(message) for message in seen]
    current = [_signature(message) for message in messages]
    if signatures and current[: len(signatures)] == signatures:
        return messages[len(signatures) :]
    prior_input = [_signature(message) for message in prev.input_messages or []]
    if prior_input and len(current) > len(prior_input) and current[: len(prior_input)] == prior_input:
        return messages[len(prior_input) :]
    compacted = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if _is_compaction(messages[index]) and messages[index] not in seen
        ),
        None,
    )
    if compacted is None:
        logger.warning(
            'Chat span {} does not replay the history its predecessor saw; aligning repeated messages in order',
            chat.span_id,
        )
        head: list[OtelMessage] = []
        tail = messages
    else:
        logger.info(
            'Chat span {} input starts from a compaction; reading the history after it as rewritten', chat.span_id
        )
        head = [messages[compacted]]
        tail = messages[compacted + 1 :]
    tail_signatures = [_signature(message) for message in tail]
    for length in range(min(len(signatures), len(tail_signatures)), 0, -1):
        suffix = signatures[-length:]
        for index in range(len(tail_signatures) - length + 1):
            if tail_signatures[index : index + length] == suffix:
                return [*head, *tail[:index], *tail[index + length :]]
    return [*head, *tail]


def _signature(message: OtelMessage) -> tuple[str, str, tuple[str, ...]]:
    """Compare message role, normalised text, and every non-reasoning part's semantic payload."""
    role = _normalized_role(message.role)
    text = _WHITESPACE.sub(
        ' ', join_text(part.content for part in message.parts if isinstance(part, OtelTextPart))
    ).strip()
    payloads = tuple(
        json.dumps(part.model_dump(mode='json', exclude_none=True), sort_keys=True, default=str)
        for part in message.parts
        if not isinstance(part, (OtelTextPart, OtelReasoningPart))
    )
    return role, text, payloads


def _normalized_role(role: str) -> str:
    if role == 'developer':
        return 'system'
    if role == 'agent':
        return 'assistant'
    return role


def _compaction_part(part: OtelPart) -> bool:
    """An OTel compaction part, or a Responses `compaction` item Orq wraps in a `data` part."""
    if isinstance(part, OtelCompactionPart):
        return True
    content = getattr(part, 'content', None) if isinstance(part, OtelGenericPart) and part.type == 'data' else None
    return isinstance(content, dict) and cast('dict[str, Any]', content).get('type') == _COMPACTION


def _is_compaction(message: OtelMessage) -> bool:
    return any(_compaction_part(part) for part in message.parts)


def _compaction_step(message: OtelMessage) -> _Draft:
    """A system step marking a context compaction, in the ATIF `extra.context_management` convention."""
    compactions = [part for part in message.parts if _compaction_part(part)]
    texts, extra = _split_text([part for part in message.parts if not _compaction_part(part)], message.role)
    texts.extend(
        part.content for part in compactions if isinstance(part, OtelCompactionPart) and isinstance(part.content, str)
    )
    extra = {**(extra or {}), **compaction_extra([part.model_dump(mode='json') for part in compactions])}
    return _Draft(fields={'source': 'system', 'message': join_text(texts), 'extra': extra})


def _system_instruction_steps(chat: OtelSpan) -> list[_Draft]:
    if not chat.system_instructions:
        return []
    texts, rest = _split_text(chat.system_instructions, chat.span_id)
    return [_Draft(fields={'source': 'system', 'message': join_text(texts), 'extra': rest})]


def _history(messages: list[OtelMessage], *, first: bool) -> list[_Draft]:
    """Turn new input messages into steps. Tool messages are read as results elsewhere."""
    drafts: list[_Draft] = []
    history = 0
    for message in messages:
        if _is_compaction(message):
            drafts.append(_compaction_step(message))
        elif message.role in ('system', 'developer', 'user'):
            drafts.append(_message_step(message))
        elif first and message.role in ('assistant', 'agent', 'tool'):
            history += 1
        elif message.role in ('assistant', 'agent'):
            drafts.append(_input_agent_step(message))
        elif message.role != 'tool':
            logger.warning('Skipping chat input message with role {!r}: ATIF has no such step source', message.role)
    if history:
        logger.warning('history of {} messages before the first chat span is not converted', history)
    return drafts


def _input_agent_step(message: OtelMessage) -> _Draft:
    """Keep an assistant turn found only in a later chat input after replay alignment."""
    texts: list[str] = []
    reasoning: list[str] = []
    calls: list[AtifToolCall] = []
    other: list[OtelPart] = []
    for part in message.parts:
        if isinstance(part, OtelTextPart):
            texts.append(part.content)
        elif isinstance(part, OtelReasoningPart):
            reasoning.append(part.content)
        elif isinstance(part, OtelToolCallPart):
            calls.append(_tool_call(part, 'input', len(calls)))
        else:
            other.append(part)
    extra = _split_text(other, 'chat input assistant message')[1]
    return _Draft(
        fields={
            'source': 'agent',
            'message': join_text(texts),
            'reasoning_content': '\n\n'.join(reasoning) or None,
            'tool_calls': calls or None,
            'extra': extra,
        }
    )


def _message_step(message: OtelMessage) -> _Draft:
    texts, extra = _split_text(message.parts, message.role)
    text = join_text(texts)
    if message.role == 'developer':
        extra = {**(extra or {}), 'original_role': 'developer'}
    source = 'user' if message.role == 'user' else 'system'
    return _Draft(fields={'source': source, 'message': text, 'extra': extra})


def _split_text(parts: list[OtelPart], where: str) -> tuple[list[str], dict[str, Any] | None]:
    """Return the text parts' contents and, when others exist, an `extra` holding them (warned)."""
    texts = [part.content for part in parts if isinstance(part, OtelTextPart)]
    rest = [part.model_dump(mode='json') for part in parts if not isinstance(part, OtelTextPart)]
    if not rest:
        return texts, None
    logger.warning(
        '{} non-text parts in {} have no ATIF step equivalent; keeping them in extra.non_text_parts', len(rest), where
    )
    return texts, {'non_text_parts': rest}


def _agent_step(chat: OtelSpan, by_id: dict[str, OtelSpan]) -> _Draft | None:
    outputs = chat.output_messages or []
    message = next((m for m in outputs if m.role in ('assistant', 'agent')), None)
    if message is None:
        if outputs:
            logger.warning('Chat span {} has no assistant output message; it gives no agent step', chat.span_id)
        return None
    if len(outputs) > 1:
        logger.warning(
            'Chat span {} has {} output messages; reading the first assistant one', chat.span_id, len(outputs)
        )
    texts: list[str] = []
    reasoning: list[str] = []
    calls: list[AtifToolCall] = []
    other: list[OtelPart] = []
    for part in message.parts:
        if isinstance(part, OtelTextPart):
            texts.append(part.content)
        elif isinstance(part, OtelReasoningPart):
            reasoning.append(part.content)
        elif isinstance(part, OtelToolCallPart):
            calls.append(_tool_call(part, chat.span_id, len(calls)))
        else:
            other.append(part)
    extra = _step_extra(chat, by_id)
    extra.update(_split_text(other, f'chat span {chat.span_id} output')[1] or {})
    extra = _merge(extra, _carried(chat, _STEP_EXTRA), f'chat span {chat.span_id} {_STEP_EXTRA}')
    fields: dict[str, Any] = {
        'source': 'agent',
        'message': join_text(texts),
        'reasoning_content': '\n\n'.join(reasoning) or None,
        'tool_calls': calls or None,
        'model_name': _attribute(chat, 'gen_ai.response.model') or _attribute(chat, 'gen_ai.request.model'),
        'metrics': _metrics(chat),
        'llm_call_count': 1,
        'timestamp': chat.start_time.isoformat() if chat.start_time else None,
        'extra': extra,
    }
    return _Draft(fields=fields, span=chat)


def _tool_call(part: OtelToolCallPart, span_id: str, index: int) -> AtifToolCall:
    call_id = part.id or 'call_' + stable_hex(span_id, part.name, str(index), length=16)
    arguments, raw_arguments = ({}, None) if part.arguments is None else atif_tool_arguments(part.arguments, part.name)
    return AtifToolCall(
        tool_call_id=call_id,
        function_name=part.name,
        arguments=arguments,
        extra={RAW_ARGUMENTS_EXTRA_KEY: raw_arguments} if raw_arguments is not None else None,
    )


def _step_extra(chat: OtelSpan, by_id: dict[str, OtelSpan]) -> dict[str, Any]:
    """NeMo-layout step extra: `ancestry`, `invocation`, plus error type, finish reasons and a differing request model."""
    parent = by_id.get(chat.parent_span_id) if chat.parent_span_id else None
    ancestry = {'function_id': chat.span_id, 'function_name': chat.name}
    if chat.parent_span_id is not None:
        ancestry['parent_id'] = chat.parent_span_id
    if parent is not None:
        ancestry['parent_name'] = parent.name
    invocation: dict[str, Any] = {'status': chat.status}
    if chat.start_time is not None:
        invocation['start_timestamp'] = chat.start_time.timestamp()
    if chat.end_time is not None:
        invocation['end_timestamp'] = chat.end_time.timestamp()
    extra: dict[str, Any] = {'ancestry': ancestry, 'invocation': invocation}
    if chat.error_type is not None:
        extra['error_type'] = chat.error_type
    finish_reasons = chat.attributes.get('gen_ai.response.finish_reasons')
    if finish_reasons is not None:
        extra['finish_reasons'] = finish_reasons
    requested = _attribute(chat, 'gen_ai.request.model')
    responded = _attribute(chat, 'gen_ai.response.model')
    if requested and responded and requested != responded:
        extra['requested_model'] = requested
    return extra


def _metrics(chat: OtelSpan) -> AtifMetrics | None:
    usage = span_usage(chat)
    values = (usage.input_tokens, usage.output_tokens, usage.cached_tokens, usage.reasoning_tokens, usage.cost_usd)
    if all(value is None for value in values):
        return None
    return AtifMetrics(
        prompt_tokens=_tokens(usage.input_tokens, chat.span_id),
        completion_tokens=_tokens(usage.output_tokens, chat.span_id),
        cached_tokens=_tokens(usage.cached_tokens, chat.span_id),
        cost_usd=float(usage.cost_usd) if usage.cost_usd is not None else None,
        extra={'reasoning_tokens': usage.reasoning_tokens} if usage.reasoning_tokens is not None else None,
    )


def _tokens(value: float | None, span_id: str) -> int | None:
    if value is None or float(value).is_integer():
        return None if value is None else int(value)
    logger.warning('Chat span {} reports a fractional token count {}; dropping it', span_id, value)
    return None


def _attach_results(
    drafts: list[_Draft], members: list[OtelSpan], chats: list[OtelSpan], new_inputs: list[list[OtelMessage]]
) -> None:
    """Give each tool call its result: an `execute_tool` span first, else a later chat span's new input.

    Results that match no call (an id-less or unknown-id `execute_tool` span, or a `tool_call_response` part
    whose id no call has) are warned and kept as `source_call_id=None` results on the nearest earlier agent step.
    """
    tool_spans = [span for span in _by_start(members) if span.operation == 'execute_tool']
    by_call: dict[str, list[OtelSpan]] = {}
    for span in tool_spans:
        call_id = span.attributes.get('gen_ai.tool.call.id')
        if isinstance(call_id, str):
            by_call.setdefault(call_id, []).append(span)
    responses = [
        (index, part)
        for index, messages in enumerate(new_inputs)
        for message in messages
        for part in message.parts
        if isinstance(part, OtelToolCallResponsePart)
    ]
    position = {chat.span_id: index for index, chat in enumerate(chats)}
    used_spans: set[str] = set()
    used_parts: set[int] = set()
    next_span_by_call: dict[str, int] = {}
    for draft in drafts:
        if draft.span is None:
            continue
        issued_at = position[draft.span.span_id]
        calls: list[AtifToolCall] = draft.fields['tool_calls'] or []
        pending_results: list[tuple[int, int, AtifObservationResult, str | None]] = []
        result_order = 0
        for n, call in enumerate(calls):
            spans = by_call.get(call.tool_call_id, [])
            span_index = next_span_by_call.get(call.tool_call_id, 0)
            span = spans[span_index] if span_index < len(spans) else None
            if span is not None:
                next_span_by_call[call.tool_call_id] = span_index + 1
                draft.tool_span_ids.add(span.span_id)
            call_extra = _carried(span, _TOOL_CALL_EXTRA)
            if call_extra is not None:
                calls[n] = call.model_copy(update={'extra': {**(call.extra or {}), **call_extra}})
            found = next(
                (
                    index
                    for index, (at, part) in enumerate(responses)
                    if index not in used_parts and at > issued_at and part.id == call.tool_call_id
                ),
                None,
            )
            part = responses[found][1] if found is not None else None
            if found is not None:
                used_parts.add(found)
            result = _result(call_id=call.tool_call_id, span=span, part=part)
            if result is not None:
                response_order = found if found is not None else len(responses) + n
                pending_results.append((response_order, result_order, result, span.span_id if span else None))
                result_order += 1
                additional = _additional_results(span)
                for item in additional:
                    pending_results.append((response_order, result_order, item, span.span_id if span else None))
                    result_order += 1
            if span is not None:
                used_spans.add(span.span_id)
            if found is not None:
                used_parts.add(found)
        for _, _, result, span_id in sorted(pending_results):
            index = len(draft.results)
            draft.results.append(result)
            if span_id is not None:
                draft.tool_span_result_groups.setdefault(span_id, []).append(index)
                draft.tool_span_result_indices.setdefault(span_id, index)
    orphan_spans = [span for span in tool_spans if span.span_id not in used_spans]
    orphan_parts = [entry for n, entry in enumerate(responses) if n not in used_parts]
    _attach_orphans(drafts, position, orphan_spans, orphan_parts)


def _additional_results(span: OtelSpan | None) -> list[AtifObservationResult]:
    """Read extra serialized ATIF results from a tool span, warning and ignoring unreadable data."""
    raw = span.attributes.get('evaluatorq.atif.additional_results') if span is not None else None
    if raw is None:
        return []
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(value, list):
            raise TypeError('expected a list')
        return [AtifObservationResult.model_validate(item) for item in value]
    except (ValueError, TypeError) as exc:
        logger.warning(
            'Tool span {} has unreadable additional ATIF results ({}); ignoring them',
            span.span_id if span is not None else '',
            exc,
        )
        return []


def _attach_orphans(
    drafts: list[_Draft],
    position: dict[str, int],
    spans: list[OtelSpan],
    parts: list[tuple[int, OtelToolCallResponsePart]],
) -> None:
    if not spans and not parts:
        return
    ids = [span.attributes.get('gen_ai.tool.call.id') for span in spans] + [part.id for _, part in parts]
    logger.warning(
        '{} tool results match no tool call (call ids {}); keeping them with no source_call_id on the nearest '
        'earlier agent step',
        len(ids),
        ids,
    )
    agents = [draft for draft in drafts if draft.span is not None]
    if not agents:
        logger.warning('No agent step to hold {} unmatched tool results; dropping them', len(ids))
        return
    for span in spans:
        call_id = span.attributes.get('gen_ai.tool.call.id')
        result = _result(call_id=call_id if isinstance(call_id, str) else None, span=span, part=None, orphan=True)
        if result is not None:
            _agent_started_before(agents, span.start_time).results.append(result)
    for at, part in parts:
        before = [d for d in agents if d.span is not None and position[d.span.span_id] < at]
        result = _result(call_id=part.id, span=None, part=part, orphan=True)
        if result is not None:
            (before[-1] if before else agents[0]).results.append(result)


def _agent_started_before(agents: list[_Draft], started: datetime | None) -> _Draft:
    """The last agent step whose chat span started at or before `started`, else the first agent step."""
    before = [d for d in agents if started and d.span and d.span.start_time and d.span.start_time <= started]
    return before[-1] if before else agents[0]


def _result(
    *, call_id: str | None, span: OtelSpan | None, part: OtelToolCallResponsePart | None, orphan: bool = False
) -> AtifObservationResult | None:
    """The result of one call, or None when nothing reports one (no empty result is made up).

    An orphan keeps its call id, if any, in `extra.orphan_call_id` and gets no `source_call_id`.
    """
    if span is None and part is None:
        return None
    content: str | None = None
    extra: dict[str, Any] = {}
    if span is not None:
        value = span.attributes.get('gen_ai.tool.call.result')
        content = tool_result_to_text(value) if value is not None else None
        if span.status == 'error':
            extra['error_type'] = span.error_type or _OTHER_ERROR
    if content is None and part is not None and part.response is not None:
        content = tool_result_to_text(part.response)
    if orphan and call_id is not None:
        extra['orphan_call_id'] = call_id
    merged = _merge(extra, _carried(span, _RESULT_EXTRA), f'tool span {span.span_id if span else ""} {_RESULT_EXTRA}')
    return AtifObservationResult(source_call_id=None if orphan else call_id, content=content, extra=merged or None)


def _subagents(
    trace: OtelTrace,
    scope: _Scope,
    drafts: list[_Draft],
    session_id: str,
    trajectory_id: str | None,
    by_id: dict[str, OtelSpan],
) -> list[AtifTrajectory]:
    """Build each nested `invoke_agent` span as a subagent trajectory and reference it from a result."""
    subagents: list[AtifTrajectory] = []
    for span in scope.subagents:
        child_id = stable_hex(session_id, trajectory_id or '', span.span_id, length=32)
        agent = AtifAgent(
            name=_attribute(span, 'gen_ai.agent.name') or span.name or _UNKNOWN,
            version=_attribute(span, 'gen_ai.agent.version') or _UNKNOWN,
        )
        child = _build(trace, [span], session_id=session_id, trajectory_id=child_id, agent=agent)
        if child is None:
            logger.warning('Subagent span {} has no chat spans; dropping it', span.span_id)
            continue
        subagents.append(child)
        _reference(AtifSubagentRef(trajectory_id=child_id, session_id=session_id), span, drafts, by_id)
    return subagents


def _reference(ref: AtifSubagentRef, span: OtelSpan, drafts: list[_Draft], by_id: dict[str, OtelSpan]) -> None:
    """Put `ref` on the result of the call whose `execute_tool` span holds the subagent, else on an extra result."""
    parent = by_id.get(span.parent_span_id) if span.parent_span_id else None
    call_id = parent.attributes.get('gen_ai.tool.call.id') if parent and parent.operation == 'execute_tool' else None
    owning_draft = next(
        (
            draft
            for draft in drafts
            if parent is not None and (draft.span is parent or parent.span_id in draft.tool_span_ids)
        ),
        None,
    )
    candidates = [owning_draft] if owning_draft is not None else drafts
    if owning_draft is not None and parent is not None:
        result_offset = span.attributes.get(_SUBAGENT_RESULT_OFFSET)
        result_group = owning_draft.tool_span_result_groups.get(parent.span_id, [])
        result_index = (
            result_group[result_offset]
            if isinstance(result_offset, int) and 0 <= result_offset < len(result_group)
            else owning_draft.tool_span_result_indices.get(parent.span_id)
        )
        if result_index is not None:
            result = owning_draft.results[result_index]
            if result.source_call_id == call_id:
                source_id = span.attributes.get(_SUBAGENT_SOURCE_TRAJECTORY_ID)
                refs = [
                    existing
                    for existing in result.subagent_trajectory_ref or []
                    if source_id is None or existing.trajectory_id != source_id
                ]
                refs.append(ref)
                owning_draft.results[result_index] = result.model_copy(update={'subagent_trajectory_ref': refs})
                return
    matching = [
        (draft, index, result)
        for draft in candidates
        for index, result in enumerate(draft.results)
        if call_id is not None and result.source_call_id == call_id
    ]
    if len(matching) == 1:
        draft, index, result = matching[0]
        refs = [*(result.subagent_trajectory_ref or []), ref]
        draft.results[index] = result.model_copy(update={'subagent_trajectory_ref': refs})
        return
    agents = [draft for draft in drafts if draft.span is not None]
    if not agents:
        logger.warning('Subagent span {} has no agent step to reference it; leaving it unreferenced', span.span_id)
        return
    _agent_started_before(agents, span.start_time).results.append(
        AtifObservationResult(source_call_id=None, subagent_trajectory_ref=[ref])
    )


# ATIF -> OTel


def atif_to_otel(traj: AtifTrajectory) -> OtelTrace:
    """Render an ATIF trajectory as one OTel trace in the GenAI semconv layout; see the module docstring.

    Chat span input is the running history (user and system messages, assistant turns and tool results). A
    system step that opens the trajectory becomes `system_instructions` of every chat span; any later one is a
    `system` (or `developer`) input message, so steps keep their order. User and system steps after the last
    agent step go into a final chat span with no output. Embedded subagents are emitted under the
    `execute_tool` span of the call whose result references them, else under the root span, and share the
    root's trace id.

    Kept as JSON-string attributes (skipped when empty): agent step `extra` beyond the keys mapped to
    semconv (`evaluatorq.atif.step.extra` on the chat span), trajectory `extra` and `final_metrics`
    (`evaluatorq.atif.trajectory.extra`, `evaluatorq.atif.final_metrics` on the `invoke_agent` span), and
    tool-call and result `extra` (`evaluatorq.atif.tool_call.extra`, `evaluatorq.atif.result.extra` on the
    `execute_tool` span). Step `extra.ancestry` is not carried: OTel -> ATIF rebuilds it from the span tree.

    Lost, with a warning per kind: observation results with no `source_call_id` (no tool message or span
    holds them), results on user and system steps, and subagent refs that are not embedded. Lost silently:
    the root `trajectory_id` (it only seeds the ids), `notes`, `agent.model_name`, `agent.tool_definitions`,
    `agent.extra`, user and system step timestamps and `extra` (other than `original_role`), `llm_call_count`,
    `is_copied_context`, and metrics `extra` other than `reasoning_tokens`.
    """
    trace_id = stable_hex(_seed(traj), 'trace', length=32)
    return OtelTrace(spans=_emit_agent(traj, None, trace_id))


def _seed(traj: AtifTrajectory) -> str:
    base = traj.session_id or content_seed(traj.model_dump(mode='json'))
    return stable_hex(base, traj.trajectory_id or '', length=64)


def _emit_agent(traj: AtifTrajectory, parent_span_id: str | None, trace_id: str) -> list[OtelSpan]:
    """Emit the `invoke_agent` span of `traj`, its chat and tool spans, then its subagents' spans."""
    seed = _seed(traj)
    agent_span_id = stable_hex(seed, 'invoke_agent', '0', length=16)
    _warn_dropped(traj)
    spans: list[OtelSpan] = []
    call_spans: dict[tuple[int, str, int], str] = {}
    history: list[OtelMessage] = []
    system: list[OtelPart] = []
    uncarried = False  # a user or system step no chat span has put into its input yet
    for index, step in enumerate(traj.steps):
        if step.source == 'agent':
            spans.extend(_emit_step(step, seed, trace_id, agent_span_id, history, system, call_spans))
            uncarried = False
            continue
        role = _input_role(step)
        if index == 0 and role == 'system':
            system.extend(_input_parts(step.message))
        else:
            history.append(OtelMessage(role=role, parts=_input_parts(step.message)))
        uncarried = True
    if uncarried:
        spans.append(_input_only_chat(seed, trace_id, agent_span_id, history, system))
    attributes: dict[str, Any] = {
        'gen_ai.operation.name': 'invoke_agent',
        'gen_ai.agent.name': traj.agent.name,
        'gen_ai.agent.version': traj.agent.version,
    }
    if traj.session_id is not None:
        attributes['gen_ai.conversation.id'] = traj.session_id
    _put_json(attributes, _TRAJECTORY_EXTRA, traj.extra)
    if traj.final_metrics is not None:
        _put_json(attributes, _FINAL_METRICS, traj.final_metrics.model_dump(mode='json', exclude_none=True))
    starts = [span.start_time for span in spans if span.start_time is not None]
    root = OtelSpan(
        trace_id=trace_id,
        span_id=agent_span_id,
        parent_span_id=parent_span_id,
        name=f'invoke_agent {traj.agent.name}',
        operation='invoke_agent',
        start_time=min(starts) if starts else None,
        attributes=attributes,
    )
    refs = _ref_calls(traj)
    for sub in traj.subagent_trajectories or []:
        ref = refs.get(sub.trajectory_id or '')
        call_ref = ref[:3] if ref is not None else None
        parent = call_spans.get(call_ref, agent_span_id) if call_ref is not None else agent_span_id
        child_spans = _emit_agent(sub, parent, trace_id)
        if ref is not None and call_ref is not None:
            child_spans[0].attributes[_SUBAGENT_RESULT_OFFSET] = ref[3]
            child_spans[0].attributes[_SUBAGENT_SOURCE_TRAJECTORY_ID] = sub.trajectory_id
        spans.extend(child_spans)
    return [root, *spans]


def _warn_dropped(traj: AtifTrajectory) -> None:
    """Warn once per kind of observation data this trajectory loses in OTel."""
    embedded = {sub.trajectory_id for sub in traj.subagent_trajectories or []}
    no_call = on_non_agent = external = 0
    for step in traj.steps:
        results = step.observation.results if step.observation else []
        if step.source != 'agent':
            on_non_agent += len(results)
            continue
        for result in results:
            refs = result.subagent_trajectory_ref or []
            no_call += result.source_call_id is None and not refs
            external += sum(1 for ref in refs if ref.trajectory_id not in embedded)
    name = traj.trajectory_id or traj.session_id or traj.agent.name
    if no_call:
        logger.warning('Trajectory {}: dropping {} observation results with no source_call_id', name, no_call)
    if on_non_agent:
        logger.warning('Trajectory {}: dropping {} observation results on user or system steps', name, on_non_agent)
    if external:
        logger.warning('Trajectory {}: dropping {} references to subagents that are not embedded', name, external)


def _input_role(step: AtifStep) -> str:
    if step.source == 'user':
        return 'user'
    return 'developer' if (step.extra or {}).get('original_role') == 'developer' else 'system'


def _input_parts(message: str | list[AtifContentPart]) -> list[OtelPart]:
    """Parts of a user or system input message; an empty message keeps one empty text part so it still reads back."""
    return _otel_parts(message) or [OtelTextPart(type='text', content='')]


def _input_only_chat(
    seed: str, trace_id: str, parent_span_id: str, history: list[OtelMessage], system: list[OtelPart]
) -> OtelSpan:
    """The chat span carrying user and system steps after the last agent step: input only, no output."""
    return OtelSpan(
        trace_id=trace_id,
        span_id=stable_hex(seed, 'chat', 'input_only', length=16),
        parent_span_id=parent_span_id,
        name='chat',
        operation='chat',
        input_messages=list(history) or None,
        system_instructions=list(system) or None,
        attributes={'gen_ai.operation.name': 'chat'},
    )


def _ref_calls(traj: AtifTrajectory) -> dict[str, tuple[int, str, int, int]]:
    """Map each subagent to its issuing call occurrence and result offset within that call."""
    refs: dict[str, tuple[int, str, int, int]] = {}
    for step in traj.steps:
        call_counts: dict[str, int] = {}
        for call in step.tool_calls or []:
            call_counts[call.tool_call_id] = call_counts.get(call.tool_call_id, 0) + 1
        result_occurrences: dict[str, int] = {}
        for result in step.observation.results if step.observation else []:
            call_id = result.source_call_id
            if call_id is None:
                continue
            occurrence = result_occurrences.get(call_id, 0)
            result_occurrences[call_id] = occurrence + 1
            call_count = call_counts.get(call_id, 0)
            if call_count == 0:
                continue
            call_occurrence = min(occurrence, call_count - 1)
            result_offset = occurrence - call_occurrence
            for ref in result.subagent_trajectory_ref or []:
                if ref.trajectory_id is not None:
                    refs.setdefault(ref.trajectory_id, (step.step_id, call_id, call_occurrence, result_offset))
    return refs


def _emit_step(
    step: AtifStep,
    seed: str,
    trace_id: str,
    parent_span_id: str,
    history: list[OtelMessage],
    system: list[OtelPart],
    call_spans: dict[tuple[int, str, int], str],
) -> list[OtelSpan]:
    """Emit one agent step's chat span and tool spans, then extend `history` with its turn and results."""
    output = OtelMessage(role='assistant', parts=_assistant_parts(step))
    start, end, status = _times(step)
    error_type = _extra_str(step, 'error_type')
    chat = OtelSpan(
        trace_id=trace_id,
        span_id=stable_hex(seed, 'chat', str(step.step_id), length=16),
        parent_span_id=parent_span_id,
        name=f'chat {step.model_name}' if step.model_name else 'chat',
        operation='chat',
        start_time=start,
        end_time=end,
        status=status,
        error_type=error_type,
        input_messages=list(history) or None,
        output_messages=[output],
        system_instructions=list(system) or None,
        attributes=_chat_attributes(step, error_type),
    )
    results: dict[str, list[AtifObservationResult]] = {}
    for result in step.observation.results if step.observation else []:
        if result.source_call_id is not None:
            results.setdefault(result.source_call_id, []).append(result)
    spans = [chat]
    calls = step.tool_calls or []
    occurrences: dict[str, int] = {}
    assigned_results: dict[int, list[AtifObservationResult]] = {}
    response_texts: dict[int, str | None] = {}
    call_occurrences: dict[str, int] = {}
    for index, call in enumerate(calls):
        call_results = results.get(call.tool_call_id, [])
        occurrence = occurrences.get(call.tool_call_id, 0)
        occurrences[call.tool_call_id] = occurrence + 1
        if occurrence < len(call_results):
            assigned_results[index] = [call_results[occurrence]]
    # Any additional results for a reused id belong to its final occurrence, matching the
    # existing additional_results representation on an execute_tool span.
    for call_id, call_results in results.items():
        matching = [index for index, call in enumerate(calls) if call.tool_call_id == call_id]
        if matching and len(call_results) > len(matching):
            assigned_results[matching[-1]].extend(call_results[len(matching) :])

    for index, call in enumerate(calls):
        span_id = stable_hex(seed, 'execute_tool', f'{step.step_id}.{index}', length=16)
        call_results = assigned_results.get(index, [])
        primary = call_results[0] if call_results else None
        text = (
            atif_content_text(primary.content, 'OTel') if primary is not None and primary.content is not None else None
        )
        response_texts[index] = text
        tool_span = _tool_span(
            call,
            primary,
            text,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
        )
        if len(call_results) > 1:
            tool_span.attributes['evaluatorq.atif.additional_results'] = json.dumps(
                [r.model_dump(mode='json', exclude_none=True) for r in call_results[1:]], separators=(',', ':')
            )
        spans.append(tool_span)
        occurrence = call_occurrences.get(call.tool_call_id, 0)
        call_occurrences[call.tool_call_id] = occurrence + 1
        call_spans.setdefault((step.step_id, call.tool_call_id, occurrence), span_id)
    history.append(output)
    primary_result_call = {
        id(call_results[0]): index for index, call_results in assigned_results.items() if call_results
    }
    unmatched = [
        result for call_id, call_results in results.items() if call_id not in occurrences for result in call_results
    ]
    if unmatched:
        logger.warning(
            'ATIF step {} has {} results whose source_call_id matches no tool call; preserving them as tool messages',
            step.step_id,
            len(unmatched),
        )
    unmatched_ids = {id(result) for result in unmatched}
    for result in step.observation.results if step.observation else []:
        index = primary_result_call.get(id(result))
        if index is not None:
            call_id = calls[index].tool_call_id
            response = response_texts[index]
        elif id(result) in unmatched_ids:
            call_id = result.source_call_id or ''
            response = atif_content_text(result.content, 'OTel') if result.content is not None else None
        else:
            continue
        part = OtelToolCallResponsePart(type='tool_call_response', id=call_id, response=response)
        history.append(OtelMessage(role='tool', parts=[part]))
    return spans


def _assistant_parts(step: AtifStep) -> list[OtelPart]:
    parts: list[OtelPart] = []
    if step.reasoning_content:
        parts.append(OtelReasoningPart(type='reasoning', content=step.reasoning_content))
    parts.extend(_otel_parts(step.message))
    parts.extend(
        OtelToolCallPart(type='tool_call', id=call.tool_call_id, name=call.function_name, arguments=call.arguments)
        for call in step.tool_calls or []
    )
    return parts


def _chat_attributes(step: AtifStep, error_type: str | None) -> dict[str, Any]:
    attributes: dict[str, Any] = {'gen_ai.operation.name': 'chat'}
    if step.model_name:
        attributes['gen_ai.request.model'] = _extra_str(step, 'requested_model') or step.model_name
        attributes['gen_ai.response.model'] = step.model_name
    metrics = step.metrics
    if metrics is not None:
        reasoning = (metrics.extra or {}).get('reasoning_tokens')
        usage = {
            'gen_ai.usage.input_tokens': metrics.prompt_tokens,
            'gen_ai.usage.output_tokens': metrics.completion_tokens,
            'gen_ai.usage.cache_read.input_tokens': metrics.cached_tokens,
            'gen_ai.usage.reasoning.output_tokens': reasoning if isinstance(reasoning, (int, float)) else None,
            'gen_ai.usage.total_cost': metrics.cost_usd,
        }
        attributes.update({key: value for key, value in usage.items() if value is not None})
    finish_reasons = (step.extra or {}).get('finish_reasons')
    if isinstance(finish_reasons, list):
        attributes['gen_ai.response.finish_reasons'] = finish_reasons
    if error_type is not None:
        attributes['error.type'] = error_type
    unmapped = {key: value for key, value in (step.extra or {}).items() if key not in _MAPPED_STEP_EXTRA}
    _put_json(attributes, _STEP_EXTRA, unmapped)
    return attributes


def _put_json(attributes: dict[str, Any], key: str, value: dict[str, Any] | None) -> None:
    """Store a non-empty dict as a JSON-string attribute (values JSON cannot encode are written as text)."""
    if value:
        attributes[key] = json.dumps(value, separators=(',', ':'), default=str)


def _tool_span(
    call: AtifToolCall,
    result: AtifObservationResult | None,
    text: str | None,
    *,
    trace_id: str,
    span_id: str,
    parent_span_id: str,
) -> OtelSpan:
    attributes: dict[str, Any] = {
        'gen_ai.operation.name': 'execute_tool',
        'gen_ai.tool.name': call.function_name,
        'gen_ai.tool.call.id': call.tool_call_id,
        'gen_ai.tool.call.arguments': (
            call.extra[RAW_ARGUMENTS_EXTRA_KEY]
            if isinstance(call.extra, dict) and isinstance(call.extra.get(RAW_ARGUMENTS_EXTRA_KEY), str)
            else json_arguments_text(call.arguments)
        ),
    }
    _put_json(attributes, _TOOL_CALL_EXTRA, call.extra)
    error_type = None
    if result is not None:
        _put_json(attributes, _RESULT_EXTRA, result.extra)
        if text is not None:
            attributes['gen_ai.tool.call.result'] = text
        error = (result.extra or {}).get('error_type')
        if isinstance(error, str):
            error_type = error
            attributes['error.type'] = error
    return OtelSpan(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=f'execute_tool {call.function_name}',
        operation='execute_tool',
        status='error' if error_type is not None else 'unset',
        error_type=error_type,
        attributes=attributes,
    )


def _times(step: AtifStep) -> tuple[datetime | None, datetime | None, Literal['ok', 'error', 'unset']]:
    """Start, end and status from `extra.invocation`, else the step timestamp as the start (no end is invented)."""
    invocation = (step.extra or {}).get('invocation')
    if invocation is None:
        return parse_time(step.timestamp), None, 'unset'
    start = parse_time(get_field(invocation, 'start_timestamp')) or parse_time(step.timestamp)
    status = get_field(invocation, 'status')
    return (
        start,
        parse_time(get_field(invocation, 'end_timestamp')),
        cast("Literal['ok', 'error', 'unset']", status) if status in _STATUSES else 'unset',
    )


def _extra_str(step: AtifStep, key: str) -> str | None:
    value = (step.extra or {}).get(key)
    return value if isinstance(value, str) else None


def _otel_parts(message: str | list[AtifContentPart]) -> list[OtelPart]:
    if isinstance(message, str):
        return [OtelTextPart(type='text', content=message)] if message else []
    parts: list[OtelPart] = []
    for part in message:
        if part.type == 'text':
            parts.append(OtelTextPart(type='text', content=part.text or ''))
        elif part.source is not None:
            parts.append(
                OtelUriPart(type='uri', modality=part.type, uri=part.source.path, mime_type=part.source.media_type)
            )
    return parts
