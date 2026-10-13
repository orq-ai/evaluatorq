"""Character-based caps shared by model-input renderers."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, SupportsIndex

from typing_extensions import Self

from evaluatorq.common.messages import MediaContent, iter_content_parts
from evaluatorq.common.redact import scrub_known_secrets

CHARS_PER_ESTIMATED_TOKEN = 4
JEV_STATE_CHARS = 28_000 * CHARS_PER_ESTIMATED_TOKEN
JEV_STATE_QUESTION_CHARS = 32_000 * CHARS_PER_ESTIMATED_TOKEN
JEV_STATE_ALL_QUESTIONS_CHARS = 64_000 * CHARS_PER_ESTIMATED_TOKEN
_REMINDER = re.compile(r'<system-reminder>.*?</system-reminder>', re.DOTALL)
_REASONING = frozenset({'reasoning', 'reasoning_content', 'thinking'})


# A `str` subclass stays JSON-serializable; `UserString` does not.
class _CappedText(str):  # noqa: FURB189
    source_text: str
    source_prefix_chars: int
    source_suffix_chars: int

    def __new__(cls, value: str, source_text: str, prefix_chars: int = 0, suffix_chars: int = 0) -> Self:
        instance = super().__new__(cls, value)
        instance.source_text = source_text
        instance.source_prefix_chars = prefix_chars
        instance.source_suffix_chars = suffix_chars
        return instance

    def __reduce_ex__(self, _protocol: SupportsIndex) -> tuple[Any, tuple[str, str, int, int]]:
        return type(self), (str(self), self.source_text, self.source_prefix_chars, self.source_suffix_chars)


class _MessageLines(_CappedText):
    """Rendered transcript entries that `cap_text` cuts between whole messages instead of mid-line."""

    spans: tuple[tuple[int, int, int], ...]
    parts_cut: bool

    def __new__(
        cls,
        value: str,
        source_text: str,
        prefix_chars: int,
        suffix_chars: int,
        spans: tuple[tuple[int, int, int], ...],
        parts_cut: bool,  # noqa: FBT001 -- `__reduce_ex__` rebuilds it positionally
    ) -> Self:
        instance = super().__new__(cls, value, source_text, prefix_chars, suffix_chars)
        instance.spans = spans
        instance.parts_cut = parts_cut
        return instance

    def __reduce_ex__(self, _protocol: SupportsIndex) -> tuple[Any, tuple[Any, ...]]:
        return type(self), (
            str(self),
            self.source_text,
            self.source_prefix_chars,
            self.source_suffix_chars,
            self.spans,
            self.parts_cut,
        )


def message_lines(text: str, spans: tuple[tuple[int, int, int], ...], *, parts_cut: bool) -> str:
    """Mark rendered transcript text so every later `cap_text` drops whole entries and names the messages it dropped.

    `spans` holds one `(start, end, message index)` per entry, in order. `parts_cut` records that the renderer
    already shortened a message, so `has_truncated_source_text` reports it even when the whole text fits.
    """
    return _MessageLines(text, text, len(text), 0, spans, parts_cut)


def cap_text(text: str, limit: int) -> tuple[str, int]:
    """Keep both ends within a character limit and report the exact omitted count."""
    if limit < 1:
        raise ValueError('model input character cap must be positive')
    if len(text) <= limit:
        return text, 0
    if isinstance(text, _MessageLines) and (by_message := _cap_message_lines(text, limit)) is not None:
        return by_message
    if limit < len('[... 0 chars left out ...]'):
        raise ValueError('model input character cap is too small for an omission marker')
    source = text.source_text if isinstance(text, _CappedText) else text
    source_length = len(source)
    retained = max(0, limit - len('[... 0 chars left out ...]'))
    first = retained // 2
    last = retained - first
    omitted = source_length - retained
    marker = f'[... {omitted} chars left out ...]'
    while first + last + len(marker) > limit and retained:
        retained -= 1
        first = retained // 2
        last = retained - first
        omitted = source_length - retained
        marker = f'[... {omitted} chars left out ...]'
    if first + last + len(marker) > limit:
        raise ValueError('model input character cap is too small for an exact omission marker')
    return _CappedText(f'{source[:first]}{marker}{source[-last:] if last else ""}', source, first, last), omitted


def _cap_message_lines(text: _MessageLines, limit: int) -> tuple[str, int] | None:
    """Keep whole entries from both ends and name the dropped messages; `None` when even the markers do not fit.

    When an end entry alone is longer than its half, that entry is cut inside instead, so both ends stay.
    """
    source, spans = text.source_text, text.spans
    if not spans:
        return None
    last = max(index for _, _, index in spans)
    char_marker = f'[... {len(source)} chars of message {last} left out ...]'
    budget = limit - 2 * len(char_marker) - len(f'[... {len(spans)} messages ({last}-{last}) left out ...]') - 3
    if budget < 2:
        return None
    tail = len(spans)
    while tail > 0 and len(source) - spans[tail - 1][0] <= budget // 2:
        tail -= 1
    cut_end = spans[tail][0] if tail < len(spans) else len(source) - budget // 2
    head = 0
    while head < len(spans) and spans[head][1] <= budget - (len(source) - cut_end):
        head += 1
    cut_start = spans[head - 1][1] if head else budget - (len(source) - cut_end)

    def owner(offset: int) -> tuple[int, int, int] | None:
        return next((span for span in spans if span[0] < offset < span[1]), None)

    head_entry, tail_entry = owner(cut_start), owner(cut_end)
    middle: list[str] = []
    if head_entry is not None and head_entry is tail_entry:
        middle.append(f'[... {cut_end - cut_start} chars left out ...]')
    else:
        if head_entry is not None:
            middle.append(f'[... {head_entry[1] - cut_start} chars of message {head_entry[2]} left out ...]\n')
        if dropped := sorted({index for start, end, index in spans if start >= cut_start and end <= cut_end}):
            which = (
                f'message {dropped[0]}'
                if len(dropped) == 1
                else f'{len(dropped)} messages ({dropped[0]}-{dropped[-1]})'
            )
            middle.append(f'[... {which} left out ...]\n')
        if tail_entry is not None:
            middle.append(f'[... {cut_end - tail_entry[0]} chars of message {tail_entry[2]} left out ...]')
    value = f'{source[:cut_start]}{"" if head_entry else chr(10)}{"".join(middle)}{source[cut_end:]}'
    tail_chars = len(source) - cut_end
    return _MessageLines(value, source, cut_start, tail_chars, spans, text.parts_cut), cut_end - cut_start


def source_text_chars(text: str) -> int:
    """Return original text length from cap metadata, never marker-shaped content."""
    return len(text.source_text) if isinstance(text, _CappedText) else len(text)


def source_text_ranges(text: str) -> tuple[tuple[int, int], ...]:
    """Return the retained source ranges represented by a capped text value."""
    if not isinstance(text, _CappedText):
        return ((0, len(text)),)
    source_length = len(text.source_text)
    ranges: list[tuple[int, int]] = []
    if text.source_prefix_chars:
        ranges.append((0, text.source_prefix_chars))
    if text.source_suffix_chars:
        ranges.append((source_length - text.source_suffix_chars, source_length))
    return tuple(ranges)


def has_truncated_source_text(value: Any) -> bool:
    """Whether capped-text provenance shows source characters were omitted."""
    if isinstance(value, _MessageLines) and value.parts_cut:
        return True
    if isinstance(value, _CappedText):
        return len(value.source_text) > len(value)
    if isinstance(value, Mapping):
        return any(has_truncated_source_text(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(has_truncated_source_text(item) for item in value)
    return False


def serialized_chars(value: Any) -> int:
    """Return the JSON rendering length used by the four-character estimate."""
    return len(json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True))


def classifier_question_wire_payloads(questions: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the exact keyed question objects sent by ``classify_request_body``."""
    wire: dict[str, dict[str, Any]] = {}
    for name, question in questions.items():
        if isinstance(question, Mapping):
            kind, instructions, criteria = (
                question.get('kind', question.get('type')),
                question.get('instructions'),
                question.get('criteria'),
            )
        else:
            kind = getattr(question, 'kind', None)
            instructions = getattr(question, 'instructions', None)
            criteria = getattr(question, 'criteria', None)
        payload: dict[str, Any] = {'type': kind, 'instructions': instructions}
        if criteria is not None:
            payload['criteria'] = criteria
        wire[name] = payload
    return wire


def _question_payloads(questions: Sequence[Any] | Mapping[str, Any]) -> list[Any] | dict[str, Any]:
    if isinstance(questions, Mapping):
        return classifier_question_wire_payloads(questions)
    return [_question_without_state(question) for question in questions]


def _question_without_state(question: Any) -> Any:
    if isinstance(question, Mapping):
        return {key: value for key, value in question.items() if key != 'state'}
    return question


def serialized_question_chars(questions: Sequence[Any] | Mapping[str, Any]) -> int:
    """Measure selected wire questions without counting their shared classifier state again."""
    payloads = _question_payloads(questions)
    return serialized_chars(payloads) if payloads else 0


def is_jev_model(model: str) -> bool:
    """Select only the Jev family for its structured state format."""
    return model.casefold().startswith('typesafe/jev')


def effective_trace_input_chars() -> int:
    """Read the persisted dashboard input cap for standalone model-input calls."""
    from evaluatorq.trace_finder.settings import effective_settings

    return effective_settings().trace_input_chars


def pair_tool_call_results(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[int, dict[str, Any]]]]:
    """Pair results with duplicate call ids FIFO within the latest assistant message group."""
    calls: list[dict[str, Any]] = []
    latest_call_group_by_id: dict[str, tuple[int, list[dict[str, Any]]]] = {}
    next_call_index_by_id: dict[str, int] = {}
    orphan_results: list[tuple[int, dict[str, Any]]] = []

    for index, message in enumerate(messages):
        role = message.get('role')
        if role == 'assistant':
            for call_index, call in enumerate(message.get('tool_calls') or []):
                if not isinstance(call, dict):
                    continue
                record = {'index': index, 'call_index': call_index, 'call': call, 'results': []}
                calls.append(record)
                call_id = call.get('id')
                if isinstance(call_id, str):
                    group = latest_call_group_by_id.get(call_id)
                    if group is None or group[0] != index:
                        latest_call_group_by_id[call_id] = (index, [record])
                        next_call_index_by_id[call_id] = 0
                    else:
                        group[1].append(record)
        elif role == 'tool':
            call_id = message.get('tool_call_id')
            if not isinstance(call_id, str) or call_id not in latest_call_group_by_id:
                orphan_results.append((index, message))
            else:
                _, matching_calls = latest_call_group_by_id[call_id]
                call_index = min(next_call_index_by_id.get(call_id, 0), len(matching_calls) - 1)
                matching_calls[call_index]['results'].append((index, message))
                next_call_index_by_id[call_id] = min(call_index + 1, len(matching_calls))

    return calls, orphan_results


@dataclass(frozen=True)
class EdgeLimit:
    """Keep a text whole up to `total` characters, otherwise its first and last `edge` characters."""

    total: int
    edge: int


@dataclass(frozen=True)
class JevOptions:
    """Per-part trimming limits and message-kind switches for ``jev_state``; defaults are the classifier's.

    `system` also applies to developer messages and `tool_result` to orphan results. Excluding tool calls drops
    their paired results; excluding tool results keeps each call with no results and drops orphan results.
    Excluded messages count in the state's omission marker. Secret scrubbing is not optional. `coverage` puts a
    leading `context` object in the state that says which message kinds it holds, which it leaves out and how each
    kind is trimmed; it is off by default so classifier and Insights inputs stay byte-identical.

    Three format switches, also off by default for the same reason, change only how the state is laid out.
    `located_omission` replaces the one trailing `omission` string by an `omission` entry in `messages` at each gap,
    naming the source indexes on both sides and how many messages it hides. `field_order` writes each entry's keys
    as identifiers first (index, role, type, name, status) and the long text, input and results last.
    `drop_empty` leaves out fields that carry nothing: empty `results`, empty `text` or `input` on entries that are
    not plain messages, and a `call_index` of 0 when the message has one call.
    """

    state_chars: int = JEV_STATE_CHARS
    assistant: EdgeLimit = EdgeLimit(1000, 500)
    user: EdgeLimit = EdgeLimit(2000, 1000)
    system: EdgeLimit = EdgeLimit(8000, 4000)
    tool_input: EdgeLimit = EdgeLimit(200, 100)
    tool_result: EdgeLimit = EdgeLimit(200, 100)
    include_system: bool = True
    include_developer: bool = True
    include_user: bool = True
    include_assistant: bool = True
    include_tool_calls: bool = True
    include_tool_results: bool = True
    coverage: bool = False
    located_omission: bool = False
    field_order: bool = False
    drop_empty: bool = False


DEFAULT_JEV_OPTIONS = JevOptions()


def _edge_phrase(limit: EdgeLimit) -> str:
    return f'whole up to {limit.total} chars, longer keeps {limit.edge} at each end'


def jev_coverage(options: JevOptions) -> dict[str, Any]:
    """Describe what a Jev state built with `options` holds and what it leaves out.

    Reasoning is never part of the state: `jev_state` reads only message content, which renderers strip of reasoning
    parts, and never a message's `reasoning_content`.
    """
    switches = (
        ('system', options.include_system, options.system),
        ('developer', options.include_developer, options.system),
        ('user', options.include_user, options.user),
        ('assistant', options.include_assistant, options.assistant),
        ('tool_calls', options.include_tool_calls, options.tool_input),
        ('tool_results', options.include_tool_results, options.tool_result),
    )
    includes = [name for name, included, _ in switches if included]
    # Results ride on their calls; with calls off only orphan results (no call in the trace) remain.
    if options.include_tool_results and not options.include_tool_calls:
        includes[includes.index('tool_results')] = 'tool_results (only those without a call)'
    context: dict[str, Any] = {
        'includes': includes,
        'left_out': [name for name, included, _ in switches if not included],
        'reasoning': 'not included',
        'trimmed': {name: _edge_phrase(limit) for name, included, limit in switches if included},
    }
    layout = [
        text
        for text, enabled in (
            ('omissions marked in place as omission entries', options.located_omission),
            ('keys ordered with identifiers first and text last', options.field_order),
            ('empty fields left out', options.drop_empty),
        )
        if enabled
    ]
    if layout:
        context['format'] = layout
    return context


def _edge(text: str, limit: EdgeLimit) -> str:
    return edge_text(text, limit.total, limit.edge)


def jev_state(
    messages: list[dict[str, Any]], *, global_char_cap: int, options: JevOptions = DEFAULT_JEV_OPTIONS
) -> dict[str, Any]:
    """Build scrubbed, indexed message entries with each tool call grouped with its results."""
    call_records, orphan_records = pair_tool_call_results(messages)
    entries: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    included_roles = {
        role
        for role, included in (
            ('system', options.include_system),
            ('developer', options.include_developer),
            ('user', options.include_user),
            ('assistant', options.include_assistant),
        )
        if included
    }
    orphan_results = (
        [
            {'index': index, 'role': 'tool', 'type': 'orphan_result', **_result(message, global_char_cap)}
            for index, message in orphan_records
        ]
        if options.include_tool_results
        else []
    )

    for index, message in enumerate(messages):
        role = message.get('role', 'unknown')
        if role in included_roles:
            body, media = _content(message.get('content'))
            body = cap_text(body, global_char_cap)[0]
            if body or media:
                entries.append({
                    'index': index,
                    'role': role,
                    'type': 'message',
                    'text': body,
                    **({'media': media} if media else {}),
                })

    for record in call_records if options.include_tool_calls else []:
        call = record['call']
        function = call.get('function') if isinstance(call.get('function'), dict) else {}
        call_id = call.get('id')
        calls.append({
            'index': record['index'],
            'call_index': record['call_index'],
            'role': 'assistant',
            'type': 'tool_call',
            'id': call_id,
            'name': function.get('name', call.get('name')),
            'input': cap_text(clean_model_text(function.get('arguments', call.get('arguments'))), global_char_cap)[0],
            'results': [_result_entry(message, index, global_char_cap) for index, message in record['results']]
            if options.include_tool_results
            else [],
        })

    entries.extend(calls)
    entries.extend(orphan_results)
    entries.sort(key=lambda item: (item['index'], item.get('call_index', -1), item['type']))
    if options.drop_empty:
        calls_per_message = Counter(call['index'] for call in calls)
        entries = [_compact(entry, drop_call_index=calls_per_message[entry['index']] == 1) for entry in entries]
    bounded = _cap_messages(entries, global_char_cap)
    for entry in bounded.get('messages', []):
        if entry['type'] == 'message':
            limit = {'assistant': options.assistant, 'user': options.user}.get(entry['role'], options.system)
            entry['text'] = _edge(entry['text'], limit)
        elif entry['type'] == 'tool_call':
            if 'input' in entry:
                entry['input'] = _edge(entry['input'], options.tool_input)
            if 'results' in entry:
                entry['results'] = [
                    {**result, 'text': _edge(result['text'], options.tool_result)} if 'text' in result else result
                    for result in entry['results']
                ]
        elif entry['type'] == 'orphan_result' and 'text' in entry:
            entry['text'] = _edge(entry['text'], options.tool_result)
    omitted_before = len(messages) - len(_represented_indexes(bounded.get('messages', [])))
    cap = min(global_char_cap, options.state_chars)
    metadata = {'context': jev_coverage(options)} if options.coverage else None
    if options.located_omission:
        state = _located_state(
            bounded.get('messages', []),
            total=len(messages),
            cap=cap,
            omitted_before=omitted_before,
            metadata=metadata,
        )
    else:
        state = _cap_messages(bounded.get('messages', []), cap, omitted_before=omitted_before, metadata=metadata)
    if options.drop_empty:
        # The size cap can empty a text further; that only shrinks the state.
        state['messages'] = [
            entry if entry['type'] == 'omission' else _compact(entry, drop_call_index=False)
            for entry in state['messages']
        ]
    if options.field_order:
        state['messages'] = [
            entry if entry['type'] == 'omission' else _in_field_order(entry) for entry in state['messages']
        ]
    return state


def _compact(entry: dict[str, Any], *, drop_call_index: bool) -> dict[str, Any]:
    """Leave out the fields of an entry that carry nothing; a plain message keeps its text and its `type`."""
    compact = {
        key: value
        for key, value in entry.items()
        if not (
            (key in {'text', 'input'} and value == '' and entry['type'] != 'message')
            or (key == 'results' and value == [])
            or (key == 'call_index' and value == 0 and drop_call_index)
        )
    }
    if 'results' in compact:
        compact['results'] = [
            {key: value for key, value in result.items() if not (key == 'text' and value == '')}
            for result in compact['results']
        ]
    return compact


_IDENTIFIER_KEYS = ('index', 'call_index', 'role', 'type', 'id', 'name', 'status', 'exit_code', 'error_code', 'media')
_BODY_KEYS = ('text', 'input', 'results')


def _in_field_order(entry: dict[str, Any]) -> dict[str, Any]:
    """Order an entry's keys as identifiers, then status, then the long text, input and results last."""
    ordered = {key: entry[key] for key in _IDENTIFIER_KEYS if key in entry}
    ordered.update({key: value for key, value in entry.items() if key not in ordered and key not in _BODY_KEYS})
    ordered.update({key: entry[key] for key in _BODY_KEYS if key in entry})
    if isinstance(ordered.get('results'), list):
        ordered['results'] = [_in_field_order(result) for result in ordered['results']]
    return ordered


def _located_state(
    entries: list[dict[str, Any]],
    *,
    total: int,
    cap: int,
    omitted_before: int,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fit entries and located markers together without overshooting the available entry budget."""
    best = {**(metadata or {}), 'messages': _with_omission_markers([], total)}
    if serialized_chars(best) > cap:
        raise ValueError('model input character cap cannot fit Jev state metadata and its omission marker')
    low, high = 1, cap
    while low <= high:
        entry_cap = (low + high) // 2
        try:
            state = _cap_messages(entries, entry_cap, omitted_before=omitted_before, metadata=metadata)
        except ValueError:
            # Below the fixed metadata reserve: a larger entry budget is required.
            low = entry_cap + 1
            continue
        located = {
            **{key: value for key, value in state.items() if key not in {'messages', 'omission'}},
            'messages': _with_omission_markers(state['messages'], total),
        }
        if serialized_chars(located) <= cap:
            best = located
            low = entry_cap + 1
        else:
            high = entry_cap - 1
    return best


def _with_omission_markers(kept: list[dict[str, Any]], total: int) -> list[dict[str, Any]]:
    """Mark contiguous absent source indexes, including gaps around grouped tool results."""
    represented = sorted(_represented_indexes(kept))
    neighbours = [None, *represented, None]
    markers: list[tuple[int, dict[str, Any]]] = []
    for after, before in pairwise(neighbours):
        first = 0 if after is None else after + 1
        end = total if before is None else before
        if first < end:
            markers.append((
                first,
                {
                    'type': 'omission',
                    'after_index': after,
                    'before_index': before,
                    'messages_left_out': end - first,
                },
            ))
    located: list[dict[str, Any]] = []
    position = 0
    for entry in kept:
        while position < len(markers) and markers[position][0] < entry['index']:
            located.append(markers[position][1])
            position += 1
        located.append(entry)
    located.extend(marker for _, marker in markers[position:])
    return located


def jev_text_state(text: str, *, global_char_cap: int) -> dict[str, Any]:
    """Encode an already-scoped text view as a Jev message without replacing its content."""
    entry = {'index': 0, 'role': 'user', 'type': 'message', 'text': clean_model_text(text)}
    return _cap_messages([entry], global_char_cap)


def jev_state_char_limit(
    questions: Sequence[Any] | Mapping[str, Any], *, global_char_cap: int = JEV_STATE_CHARS
) -> int:
    """Return the Jev state ceiling after global and model-specific question reserves."""
    question_payloads = _question_payloads(questions)
    values = list(question_payloads.values()) if isinstance(question_payloads, Mapping) else question_payloads
    longest = max((serialized_chars(question) for question in values), default=0)
    all_question_chars = serialized_chars(question_payloads) if values else 0
    if all_question_chars > JEV_STATE_ALL_QUESTIONS_CHARS:
        raise ValueError('classification questions alone exceed the 256000-character limit')
    limit = min(
        JEV_STATE_CHARS,
        global_char_cap - all_question_chars,
        JEV_STATE_QUESTION_CHARS - longest,
        JEV_STATE_ALL_QUESTIONS_CHARS - all_question_chars,
    )
    if limit <= 0:
        raise ValueError('classification questions leave no Jev state budget under the configured input cap')
    return limit


def fit_jev_state(
    state: dict[str, Any],
    questions: Sequence[Any] | Mapping[str, Any],
    *,
    global_char_cap: int = JEV_STATE_CHARS,
) -> dict[str, Any]:
    """Fit the complete Jev state object to the global and question-aware ceilings."""
    limit = jev_state_char_limit(questions, global_char_cap=global_char_cap)
    original = state.get('messages', [])
    omitted_before = omission_count(state.get('omission'))
    metadata = {key: value for key, value in state.items() if key not in {'messages', 'omission'}}
    if any(entry.get('type') == 'omission' for entry in original):
        entries = [entry for entry in original if entry.get('type') != 'omission']
        total = len(_represented_indexes(entries)) + state_omitted_messages(state)
        return _located_state(
            entries, total=total, cap=limit, omitted_before=state_omitted_messages(state), metadata=metadata
        )
    return _cap_messages(original, limit, omitted_before=omitted_before, metadata=metadata)


def fit_classifier_input(
    state: str | dict[str, Any],
    *,
    model: str,
    questions: Sequence[Any] | Mapping[str, Any],
    global_char_cap: int,
) -> str | dict[str, Any]:
    """Fit state and selected question payloads within the configured model-input cap."""
    if is_jev_model(model):
        if not isinstance(state, dict):
            raise TypeError('Jev classifier state must be an object')
        return fit_jev_state(state, questions, global_char_cap=global_char_cap)
    question_payloads = _question_payloads(questions)
    question_chars = serialized_chars(question_payloads) if question_payloads else 0

    state_budget = global_char_cap - question_chars
    if state_budget <= 0:
        raise ValueError('classifier questions leave no state budget under the configured model input cap')
    if isinstance(state, dict) and isinstance(state.get('conversation'), str):
        return _fit_conversation_payload(state, state_budget)
    rendered = (
        state
        if isinstance(state, str)
        else json.dumps(state, ensure_ascii=False, separators=(',', ':'), sort_keys=True)
    )
    return cap_text(rendered, state_budget)[0]


def _fit_conversation_payload(state: dict[str, Any], limit: int) -> dict[str, Any]:
    conversation = state['conversation']
    if serialized_chars(state) <= limit:
        return state
    empty = {**state, 'conversation': ''}
    if serialized_chars(empty) > limit:
        raise ValueError('classifier state metadata cannot fit under the configured model input cap')
    low = len('[... 0 chars left out ...]')
    high = len(conversation)
    best: dict[str, Any] | None = None
    while low <= high:
        text_limit = (low + high) // 2
        candidate = {**state, 'conversation': cap_text(conversation, text_limit)[0]}
        if serialized_chars(candidate) <= limit:
            best = candidate
            low = text_limit + 1
        else:
            high = text_limit - 1
    if best is None:
        raise ValueError('classifier state and omission marker cannot fit under the configured model input cap')
    return best


def _state_payload(messages: list[dict[str, Any]], omitted: int, metadata: dict[str, Any]) -> dict[str, Any]:
    payload = {**metadata, 'messages': messages}
    if omitted:
        payload['omission'] = f'[... {omitted} messages left out ...]'
    return payload


@dataclass
class _MessageSelection:
    left: list[dict[str, Any]]
    right: list[dict[str, Any]]
    lo: int
    hi: int
    left_used: int = 0
    right_used: int = 0
    left_chars: int = 0
    right_chars: int = 0
    left_indexes: dict[int, int] = field(default_factory=dict)
    right_indexes: dict[int, int] = field(default_factory=dict)
    selected_indexes: dict[int, int] = field(default_factory=dict)
    selected_unique: int = 0

    def include_indexes(self, indexes: set[int], side: dict[int, int]) -> None:
        for index in indexes:
            side[index] = side.get(index, 0) + 1
            if self.selected_indexes.get(index, 0) == 0:
                self.selected_unique += 1
            self.selected_indexes[index] = self.selected_indexes.get(index, 0) + 1


def _select_message_edges(
    entries: list[dict[str, Any]], entry_sizes: list[int], indexes_by_entry: list[set[int]], half: int
) -> _MessageSelection:
    selection = _MessageSelection(left=[], right=[], lo=0, hi=len(entries) - 1)
    while selection.lo <= selection.hi:
        size = entry_sizes[selection.lo] + 1
        if selection.left_used + size > half:
            candidate = _fit_entry(entries[selection.lo], half - selection.left_used - 1)
            if candidate is not None:
                candidate_size = serialized_chars(candidate)
                selection.left.append(candidate)
                selection.left_chars += candidate_size
                selection.left_used += candidate_size + 1
                selection.include_indexes(indexes_by_entry[selection.lo], selection.left_indexes)
            selection.lo += 1
        else:
            selection.left.append(entries[selection.lo])
            selection.left_chars += entry_sizes[selection.lo]
            selection.left_used += size
            selection.include_indexes(indexes_by_entry[selection.lo], selection.left_indexes)
            selection.lo += 1
        if selection.lo > selection.hi:
            break
        size = entry_sizes[selection.hi] + 1
        if selection.right_used + size > half:
            candidate = _fit_entry(entries[selection.hi], half - selection.right_used - 1)
            if candidate is not None:
                candidate_size = serialized_chars(candidate)
                selection.right.insert(0, candidate)
                selection.right_chars += candidate_size
                selection.right_used += candidate_size + 1
                selection.include_indexes(indexes_by_entry[selection.hi], selection.right_indexes)
            selection.hi -= 1
        else:
            selection.right.insert(0, entries[selection.hi])
            selection.right_chars += entry_sizes[selection.hi]
            selection.right_used += size
            selection.include_indexes(indexes_by_entry[selection.hi], selection.right_indexes)
            selection.hi -= 1
    return selection


def _cap_messages(
    entries: list[dict[str, Any]],
    cap: int,
    *,
    omitted_before: int = 0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fixed = {key: value for key, value in (metadata or {}).items() if key not in {'messages', 'omission'}}
    empty_without_marker = serialized_chars(_state_payload([], 0, fixed)) - 2
    empty_with_marker = serialized_chars(_state_payload([], 1, fixed)) - 2
    entry_sizes = [serialized_chars(entry) for entry in entries]
    source_indexes = _represented_indexes(entries)

    def payload_size(message_chars: int, message_count: int, omitted: int) -> int:
        array_chars = 2 + message_chars + max(0, message_count - 1)
        root_chars = empty_without_marker if omitted == 0 else empty_with_marker + len(str(omitted)) - 1
        return root_chars + array_chars

    message_chars = sum(entry_sizes)
    if payload_size(message_chars, len(entries), omitted_before) <= cap:
        return _state_payload(entries, omitted_before, fixed)

    marker_reserve = payload_size(0, 0, 999999)
    if cap < marker_reserve:
        raise ValueError('model input character cap is too small for the message omission marker')
    half = max(0, cap - marker_reserve) // 2
    indexes_by_entry = [_represented_indexes([entry]) for entry in entries]
    selection = _select_message_edges(entries, entry_sizes, indexes_by_entry, half)

    def fill_unused_budget() -> None:
        while selection.lo <= selection.hi:
            added = False
            for side_name in ('left', 'right') if selection.left_used <= selection.right_used else ('right', 'left'):
                index = selection.lo if side_name == 'left' else selection.hi
                new_indexes = indexes_by_entry[index]
                new_unique = selection.selected_unique + sum(
                    selection.selected_indexes.get(value, 0) == 0 for value in new_indexes
                )
                omitted_count = omitted_before + len(source_indexes) - new_unique
                candidate_size = payload_size(
                    selection.left_chars + selection.right_chars + entry_sizes[index],
                    len(selection.left) + len(selection.right) + 1,
                    omitted_count,
                )
                if candidate_size > cap:
                    continue
                item = entries[index]
                if side_name == 'left':
                    selection.left.append(item)
                    selection.left_chars += entry_sizes[index]
                    selection.left_used += entry_sizes[index] + 1
                    selection.include_indexes(new_indexes, selection.left_indexes)
                    selection.lo += 1
                else:
                    selection.right.insert(0, item)
                    selection.right_chars += entry_sizes[index]
                    selection.right_used += entry_sizes[index] + 1
                    selection.include_indexes(new_indexes, selection.right_indexes)
                    selection.hi -= 1
                added = True
                break
            if not added:
                break

    fill_unused_budget()

    omitted = omitted_before + len(source_indexes) - selection.selected_unique
    bounded = _state_payload(selection.left + selection.right, omitted, fixed)
    if serialized_chars(bounded) > cap:
        omitted += selection.selected_unique
        bounded = _state_payload([], omitted, fixed)
        if serialized_chars(bounded) > cap:
            raise ValueError('model input character cap cannot fit Jev state metadata and its omission marker')
    return bounded


def _fit_entry(entry: dict[str, Any], char_budget: int) -> dict[str, Any] | None:
    if char_budget < 1:
        return None
    low, high = 0, char_budget
    best: dict[str, Any] | None = None
    while low <= high:
        text_cap = (low + high) // 2
        candidate = _entry_with_text_cap(entry, text_cap)
        if serialized_chars(candidate) <= char_budget:
            best = candidate
            low = text_cap + 1
        else:
            high = text_cap - 1
    return best


def _entry_with_text_cap(entry: dict[str, Any], limit: int) -> dict[str, Any]:
    bounded = dict(entry)
    marker_length = len('[... 0 chars left out ...]')
    if isinstance(bounded.get('text'), str):
        bounded['text'] = cap_text(bounded['text'], limit)[0] if limit >= marker_length else ''
    if isinstance(bounded.get('input'), str):
        bounded['input'] = cap_text(bounded['input'], limit)[0] if limit >= marker_length else ''
    results = bounded.get('results')
    if isinstance(results, list):
        bounded['results'] = [
            {
                **result,
                'text': cap_text(result['text'], limit)[0]
                if isinstance(result, dict) and isinstance(result.get('text'), str) and limit >= marker_length
                else '',
            }
            if isinstance(result, dict)
            else result
            for result in results
        ]
    return bounded


def _represented_indexes(entries: list[dict[str, Any]]) -> set[int]:
    indexes: set[int] = set()
    for entry in entries:
        index = entry.get('index')
        if isinstance(index, int):
            indexes.add(index)
        for result in entry.get('results', []):
            result_index = result.get('index') if isinstance(result, dict) else None
            if isinstance(result_index, int):
                indexes.add(result_index)
    return indexes


def state_omitted_messages(state: Mapping[str, Any]) -> int:
    """Count the messages a Jev state leaves out, from its omission string or from its in-place `omission` entries."""
    placed = sum(
        entry.get('messages_left_out', 0)
        for entry in state.get('messages', [])
        if isinstance(entry, Mapping) and entry.get('type') == 'omission'
    )
    return omission_count(state.get('omission')) + placed


def omission_count(marker: Any) -> int:
    """Parse the omitted-message count from a Jev omission marker string."""
    if not isinstance(marker, str):
        return 0
    match = re.fullmatch(r'\[\.\.\. (\d+) messages left out \.\.\.\]', marker)
    return int(match.group(1)) if match else 0


def _content(value: Any) -> tuple[str, list[str]]:
    text_parts: list[str] = []
    media: list[str] = []
    for part in iter_content_parts(value):
        if isinstance(part, MediaContent):
            marker = _image_marker(part.value) if part.kind == 'image' else f'[{part.kind}]'
            media.append(marker)
            text_parts.append(marker)
        else:
            text_parts.append(clean_model_text(part))
    return '\n'.join(text_parts), media


def _image_marker(block: Any) -> str:
    image_url = _content_field(block, 'image_url')
    raw = _content_field(block, 'image') or _content_field(block, 'data')
    if isinstance(image_url, Mapping):
        raw = image_url.get('url') or image_url.get('data') or raw
    elif isinstance(image_url, str):
        raw = image_url
    elif image_url is not None:
        raw = getattr(image_url, 'url', None) or getattr(image_url, 'data', None) or raw
    size = _inline_image_size(raw)
    return f'[image, {size // 1024} KB]' if size is not None else '[image, size unknown]'


def _content_field(part: Any, key: str) -> Any:
    return part.get(key) if isinstance(part, Mapping) else getattr(part, key, None)


def _inline_image_size(raw: Any) -> int | None:
    if not isinstance(raw, str):
        return None
    encoded = raw.partition(',')[2] if raw.startswith('data:') and ';base64,' in raw else raw
    if not encoded or any(
        character not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for character in encoded
    ):
        return None
    padding = len(encoded) - len(encoded.rstrip('='))
    return len(encoded) * 3 // 4 - min(padding, 2)


def _result(message: dict[str, Any], char_cap: int) -> dict[str, Any]:
    return _result_entry(message, None, char_cap)


def _result_entry(message: dict[str, Any], index: int | None, char_cap: int) -> dict[str, Any]:
    body, _ = _content(message.get('content'))
    result: dict[str, Any] = {'text': cap_text(body, char_cap)[0]}
    if index is not None:
        result['index'] = index
    status, exit_code, error_code = tool_result_metadata(message)
    if status is not None:
        result['status'] = status
    if exit_code is not None:
        result['exit_code'] = exit_code
    if error_code is not None:
        result['error_code'] = error_code
    return result


def clean_model_text(value: Any) -> str:
    if isinstance(value, str):
        text = value
    elif isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True)
    else:
        text = '' if value is None else str(value)
    return _REMINDER.sub('[harness reminder omitted]', scrub_known_secrets(text))


def edge_text(text: str, total: int, side: int) -> str:
    if len(text) <= total:
        return text
    source = text.source_text if isinstance(text, _CappedText) else text
    omitted = len(source) - side * 2
    return _CappedText(f'{source[:side]}[... {omitted} chars left out ...]{source[-side:]}', source, side, side)


def tool_result_metadata(message: dict[str, Any]) -> tuple[Any, Any, Any]:
    metadata = message.get('trace_finder_metadata')
    nested = metadata.get('tool_result') if isinstance(metadata, dict) else None
    sources = [source for source in (nested, message) if isinstance(source, dict)]
    status = exit_code = error_code = None
    for source in sources:
        if status is None:
            status = source.get('status')
            if status is None and source.get('is_error') is not None:
                status = 'error' if source['is_error'] else 'completed'
        if exit_code is None:
            exit_code = source.get('exit_code')
        if error_code is None:
            error_code = source.get('error_code')
    return status, exit_code, error_code
