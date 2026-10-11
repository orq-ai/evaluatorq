"""Character-based caps shared by model-input renderers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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


def cap_text(text: str, limit: int) -> tuple[str, int]:
    """Keep both ends within a character limit and report the exact omitted count."""
    if limit < 1:
        raise ValueError('model input character cap must be positive')
    if len(text) <= limit:
        return text, 0
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


def jev_state(messages: list[dict[str, Any]], *, global_char_cap: int) -> dict[str, Any]:
    """Build scrubbed, indexed message entries with each tool call grouped with its results."""
    call_records, orphan_records = pair_tool_call_results(messages)
    entries: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    orphan_results = [
        {'index': index, 'role': 'tool', 'type': 'orphan_result', **_result(message, global_char_cap)}
        for index, message in orphan_records
    ]

    for index, message in enumerate(messages):
        role = message.get('role', 'unknown')
        if role == 'tool':
            continue
        if role in {'system', 'developer', 'user', 'assistant'}:
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

    for record in call_records:
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
            'results': [_result_entry(message, index, global_char_cap) for index, message in record['results']],
        })

    entries.extend(calls)
    entries.extend(orphan_results)
    entries.sort(key=lambda item: (item['index'], item.get('call_index', -1), item['type']))
    bounded = _cap_messages(entries, global_char_cap)
    for entry in bounded.get('messages', []):
        if entry['type'] == 'message':
            total, side = {'assistant': (1000, 500), 'user': (2000, 1000)}.get(entry['role'], (8000, 4000))
            entry['text'] = edge_text(entry['text'], total, side)
        elif entry['type'] == 'tool_call':
            entry['input'] = edge_text(entry['input'], 200, 100)
            entry['results'] = [{**result, 'text': edge_text(result['text'], 200, 100)} for result in entry['results']]
        elif entry['type'] == 'orphan_result':
            entry['text'] = edge_text(entry['text'], 200, 100)
    omitted_before = len(messages) - len(_represented_indexes(bounded.get('messages', [])))
    return _cap_messages(
        bounded.get('messages', []),
        min(global_char_cap, JEV_STATE_CHARS),
        omitted_before=omitted_before,
    )


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
    omitted_before = _omission_count(state.get('omission'))
    metadata = {key: value for key, value in state.items() if key not in {'messages', 'omission'}}
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


def _omission_count(marker: Any) -> int:
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
