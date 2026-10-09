"""Text views of a trace for the Insights classifier, plus the tool counts read straight from its messages.

Three views, each for one `/classify` call:

- `full_conversation_view`: the readable source messages, tool inputs, and complete tool results for non-Jev classifiers and summaries, capped to the configured trace-input budget.
- `tool_inventory`: which tools ran and how often, for the coding-agent check.
- `tool_activity_chunks`: user context and paired tool activity for coding-tool questions. Non-Jev traces keep their question-specific command excerpts; Jev traces use filtered structured state, 100-character input/result edges, and complete-record chunks that fit the exact question reserve and Jev state-plus-question ceilings.

Nothing here calls a model or any other service. A shell output excerpt is scrubbed of credentials with
`common.redact.scrub_known_secrets` over the whole output before it is cut to its start and end, so a secret across
the cut leaves no fragment.
"""

from __future__ import annotations

import json
import re
import shlex
from collections import Counter
from operator import itemgetter
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, overload

from pydantic import BaseModel

from evaluatorq.common.messages import coerce_content_text, contains_media_content
from evaluatorq.common.model_input import (
    cap_text,
    clean_model_text,
    effective_trace_input_chars,
    is_jev_model,
    jev_state,
    jev_state_char_limit,
    pair_tool_call_results,
    serialized_chars,
    serialized_question_chars,
)
from evaluatorq.common.redact import scrub_known_secrets
from evaluatorq.common.trace_document import TraceDocument, prompt_messages
from evaluatorq.contracts import tool_result_to_text
from evaluatorq.signals.config import SHELL_TOOL_NAMES
from evaluatorq.trace_finder.projection import _tool_call_status, _tool_result_category

if TYPE_CHECKING:
    from collections.abc import Iterator

    from evaluatorq.trace_finder.models import TraceRecord

TOOL_INPUT_CHARS = 300
RISKY_COMMAND_CHARS = 300
# Shell output kept per call, split between its start and end, where exit codes and test summaries sit.
SHELL_OUTPUT_CHARS = 300
# Preserve short text whole instead of adding a cut marker for a few characters.
CUT_SLACK = 40

SKILL_TOOL = 'Skill'
SHELL_TOOLS = SHELL_TOOL_NAMES
# Failure markers searched in the full shell output, including the part the
# excerpt drops. A command can fail inside a call whose status is `completed`,
# so the status alone misses it. Claude Code starts a failed Bash result with
# `Exit code N`; Codex `exec` starts with `Script failed` and nests `"exit_code":N`.
# Substring heuristics: `cat` of a test file can match too, so the
# classifier treats a label as a hint. Upgrade to exit-code fields if traces carry them.
OUTPUT_MARKERS = (
    (
        'nonzero_exit',
        re.compile(
            r'exit(?:ed with)? (?:code|status)[: ]+[1-9]|"exit_code":\s*[1-9]|^Script failed',
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        'tests_failed',
        re.compile(r'\b[1-9]\d* (?:failed|failing|errors?)\b|^FAILED |tests? failed', re.IGNORECASE | re.MULTILINE),
    ),
    ('exception', re.compile(r'Traceback \(most recent call last\)|^panic:|Unhandled exception', re.MULTILINE)),
    (
        'build_failed',
        re.compile(r'\bSyntaxError\b|error\[E\d+\]|error TS\d+|build failed|compilation failed', re.IGNORECASE),
    ),
)
_REMINDER = re.compile(r'<system-reminder>.*?</system-reminder>', re.DOTALL)
_SKILL_ARG = re.compile(r'"skill"\s*:\s*"([^"]+)"')
_STRING_FIELD = re.compile(r'"?(\w+)"?\s*:\s*"((?:[^"\\]|\\.)*)')  # keys unquoted in a Codex exec script
# Wrappers that hide the real program: `uv run pytest` is pytest, `rtk git` is git.
_WRAPPERS: dict[str, str | None] = {
    'uv': 'run',
    'poetry': 'run',
    'uvx': None,
    'npx': None,
    'bunx': None,
    'rtk': None,
    'sudo': None,
    'time': None,
    'env': None,
    'nohup': None,
    'timeout': None,
}
_WITH_SUBCOMMAND = frozenset({
    'git', 'gh', 'npm', 'pnpm', 'yarn', 'bun', 'docker', 'kubectl', 'orq', 'go', 'cargo', 'make', 'brew', 'uv', 'terraform'
})  # fmt: skip
_STATE_BUILTINS = frozenset({'cd', 'source', '.', 'export', 'set', 'unset', 'pushd', 'popd'})
_BLOCK_OPENERS = frozenset({'for', 'while', 'until', 'if', 'case', 'select'})
_BLOCK_WORDS = frozenset({'do', 'then', 'else', 'elif', '{', '(', '!'})
_BLOCK_CLOSERS = frozenset({'done', 'fi', 'esac', '}', ')'})
# Commands kept verbatim instead of collapsed to their program name: they change shared
# state or delete data, so a classifier judging risk needs the line, not `git push`. They
# are still cut to RISKY_COMMAND_CHARS, the same limit as any other tool input.
_RISKY = re.compile(
    r'\bgit\s+(push|reset|clean|rebase|branch\s+-D|checkout\s+--|restore|stash|tag\s+-d|filter-branch)\b'
    r'|\brm\s|\bgh\s+(pr\s+(merge|close|edit|comment|create|review)|release|repo\s+(delete|edit)|issue\s+(close|delete)|api\b)'
    r'|\bkubectl\s+(delete|apply|scale|rollout)|\bdocker\s+(rm|rmi|system\s+prune|volume\s+rm)'
    r'|\bterraform\s+(apply|destroy)|\bdrop\s+(table|database)\b|\btruncate\s+table\b'
    r'|--force\b|\s-f\b.*\bpush\b|\bchmod\b|\bchown\b|\bkill(all)?\b|\bpkill\b'
    r'|\bcurl\b.*-X\s*(POST|PUT|PATCH|DELETE)|\bnpm\s+publish\b|\buv\s+publish\b|\btwine\s+upload\b'
    r'|\bpsql\b|\bmysql\b|\bssh\b|\bscp\b',
    re.IGNORECASE,
)


class ToolStats(BaseModel):
    """Tool, skill and shell-command counts read from one trace's messages, most used first."""

    tools: dict[str, int] = {}
    skills: dict[str, int] = {}
    commands: dict[str, int] = {}


def tool_stats(trace: TraceRecord | TraceDocument) -> ToolStats:
    """Count every tool call by name, every skill the `Skill` tool loaded, and every shell call by its first command."""
    tools: Counter[str] = Counter()
    skills: Counter[str] = Counter()
    commands: Counter[str] = Counter()
    for _, call in _tool_calls(trace):
        name, arguments = _call_name(call), _call_arguments(call)
        if name == SKILL_TOOL:
            skills.update(
                _SKILL_ARG.findall(_raw_arguments(call)) or ([arguments['skill']] if 'skill' in arguments else [])
            )
            continue
        tools[name] += 1
        if name in SHELL_TOOLS:
            commands[first_command(_shell_command(arguments))] += 1
    return ToolStats(
        tools=dict(tools.most_common()), skills=dict(skills.most_common()), commands=dict(commands.most_common())
    )


def first_command(command: str) -> str:
    """Name the program a shell line runs: `cd x && FOO=1 uv run pytest -q` is `pytest`, `gh pr create` is `gh pr`.

    Skips directory and environment setup, wrappers such as `uv run`, loop and
    condition headers, and assignments that only capture a helper's output.
    Returns `(none)` when no segment names a program.
    """
    for segment in re.split(r'&&|\|\||;|\||\n', command.replace('\\\n', ' ')):
        words = _shell_words(segment)
        if not words or words[0] in _BLOCK_OPENERS or re.match(r'^\w+=\S*\$\(', words[0]):
            continue
        while words and words[0] in _BLOCK_WORDS:
            words = words[1:]
        while words and re.match(r'^\w+=', words[0]):
            words = words[1:]
        if not words or words[0] in _STATE_BUILTINS or words[0] in _BLOCK_CLOSERS:
            continue
        words = _strip_wrappers(words)
        if not words:
            continue
        program = PurePosixPath(words[0]).name
        if not re.fullmatch(r'[\w.+-]+', program) or program.startswith('-'):
            continue
        rest = [word for word in words[1:] if not word.startswith('-')]
        return f'{program} {rest[0]}' if program in _WITH_SUBCOMMAND and rest else program
    return '(none)'


def is_risky_command(command: str) -> bool:
    """Whether a shell line pushes, deletes, publishes or otherwise changes state outside the working tree."""
    return bool(_RISKY.search(command))


def tool_inventory(trace: TraceRecord | TraceDocument) -> str:
    """List the tools a trace used with call counts, and the shell programs it ran, one per line."""
    stats = tool_stats(trace)
    if not stats.tools and not stats.skills:
        return 'No tool calls.'
    lines = [f'Tool calls: {sum(stats.tools.values())}']
    lines += [f'- {name} x{count}' for name, count in stats.tools.items()]
    if stats.commands:
        lines.append('Shell commands:')
        lines += [f'- {name} x{count}' for name, count in list(stats.commands.items())[:40]]
    if stats.skills:
        lines.append('Skills loaded:')
        lines += [f'- {name} x{count}' for name, count in stats.skills.items()]
    return '\n'.join(lines)


def full_conversation_view_with_message_spans(
    trace: TraceRecord | TraceDocument,
) -> tuple[str, tuple[tuple[int, int, int], ...], int]:
    """Return complete rendered text, source-message spans and total source-message count."""
    return _render_full_conversation(trace, capture_message_spans=True)


def full_conversation_view(trace: TraceRecord | TraceDocument, budget: int | None = None) -> str:
    """Render source messages, tool inputs, and complete tool results for model input."""
    rendered, _, _ = _render_full_conversation(trace, capture_message_spans=False)
    cap = budget if budget is not None else effective_trace_input_chars()
    return cap_text(rendered, cap)[0]


def _render_full_conversation(
    trace: TraceRecord | TraceDocument,
    *,
    capture_message_spans: bool,
) -> tuple[str, tuple[tuple[int, int, int], ...], int]:
    messages = prompt_messages(trace)
    lines: list[str] = []
    line_indexes = [] if capture_message_spans else None
    for index, message in enumerate(messages):
        role = message.get('role', 'unknown')
        if role == 'tool':
            content = message.get('content')
            rendered = coerce_content_text(content) if contains_media_content(content) else tool_result_to_text(content)
            result = scrub_known_secrets(rendered)
            result = _REMINDER.sub('[harness reminder omitted]', result)
            lines.append(f'{index} tool result {message.get("tool_call_id", "")}: {result}')
            if line_indexes is not None:
                line_indexes.append(index)
        else:
            content = scrub_known_secrets(coerce_content_text(message.get('content')))
            content = _REMINDER.sub('[harness reminder omitted]', content)
            if content.strip():
                lines.append(f'{index} {role}: {content}')
                if line_indexes is not None:
                    line_indexes.append(index)
        for call in message.get('tool_calls') or []:
            name = _call_name(call)
            arguments = _REMINDER.sub('[harness reminder omitted]', scrub_known_secrets(_raw_arguments(call)))
            lines.append(f'{index} tool input {name}: {arguments}')
            if line_indexes is not None:
                line_indexes.append(index)
    conversation = '\n'.join(lines)
    if line_indexes is None:
        return conversation, (), len(messages)
    spans = []
    offset = 0
    for line, index in zip(lines, line_indexes, strict=True):
        spans.append((offset, offset + len(line), index))
        offset += len(line) + 1
    return conversation, tuple(spans), len(messages)

@overload
def tool_activity_chunks(
    trace: TraceRecord | TraceDocument,
    budget: int | None = None,
    *,
    model: None = None,
    trace_input_chars: int | None = None,
    question_payloads: list[Any] | dict[str, Any] | None = None,
) -> list[str]: ...


@overload
def tool_activity_chunks(
    trace: TraceRecord | TraceDocument,
    budget: int | None = None,
    *,
    model: str,
    trace_input_chars: int | None = None,
    question_payloads: list[Any] | dict[str, Any] | None = None,
) -> list[str] | list[dict[str, Any]]: ...


@overload
def tool_activity_chunks(
    trace: TraceRecord | TraceDocument,
    budget: int | None = None,
    *,
    model: str | None = None,
    trace_input_chars: int | None = None,
    question_payloads: list[Any] | dict[str, Any] | None = None,
) -> list[str] | list[dict[str, Any]]: ...


def tool_activity_chunks(
    trace: TraceRecord | TraceDocument,
    budget: int | None = None,
    *,
    model: str | None = None,
    trace_input_chars: int | None = None,
    question_payloads: list[Any] | dict[str, Any] | None = None,
) -> list[str] | list[dict[str, Any]]:
    """Render user context and paired tool activity for the coding-tool questions."""
    global_cap = trace_input_chars if trace_input_chars is not None else effective_trace_input_chars()
    request_budget = min(global_cap, budget if budget is not None else global_cap)
    if request_budget < 1:
        raise ValueError('trace input character cap must be positive')
    questions = question_payloads or []
    jev = is_jev_model(model) if model is not None else False
    messages = prompt_messages(trace)
    if jev:
        units = _jev_activity_units(messages)
        state_budget = jev_state_char_limit(questions, global_char_cap=request_budget)
        if not units:
            if len('No tool calls.') > state_budget:
                raise ValueError('classifier questions leave no Jev state budget for tool activity')
            return ['No tool calls.']
        return _jev_tool_activity_chunks(messages, state_budget=state_budget, units=units)

    chunk_budget = request_budget - serialized_question_chars(questions)
    if chunk_budget <= 0:
        raise ValueError('classifier questions leave no tool-activity state budget under the configured input cap')
    results = _results(messages)
    lines: list[str] = []
    for message in messages:
        role = message.get('role')
        if role == 'user':
            text = _user_text(message)
            if text:
                lines.append(f'USER: {_head_tail(text, 400, 200)}')
        elif role == 'assistant':
            lines.extend(_activity_line(call, results) for call in message.get('tool_calls') or [])
    call_lines = [line for line in lines if line.startswith('CALL')]
    if any(len(line) + 1 > chunk_budget for line in call_lines):
        raise ValueError('one non-Jev tool-activity record exceeds the question-reserved state budget')
    if not call_lines:
        if len('No tool calls.') > chunk_budget:
            raise ValueError('classifier questions leave no non-Jev state budget for tool activity')
        return ['No tool calls.']
    full_view_chars = sum(len(line) for line in lines) + len(lines) - 1
    if full_view_chars <= chunk_budget:
        return ['\n'.join(lines)]
    return _chunk(lines, chunk_budget)


def _jev_tool_activity_chunks(
    messages: list[dict[str, Any]], *, state_budget: int, units: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Split paired tool activity into Jev states without dropping complete call records."""
    if not units:
        return []
    context = [message for message in messages if message.get('role') == 'user']
    groups = _jev_group_activity_units(context, units, state_budget)
    chunks = []
    for number, group in enumerate(groups, start=1):
        state = _jev_activity_state(context, group, state_budget, part=number, total=len(groups))
        if state is None:
            raise ValueError('Jev tool-activity chunk exceeded its reserved classifier state budget')
        chunks.append(state)
    return chunks


def _jev_activity_units(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    call_records, orphan_records = pair_tool_call_results(messages)
    units = [
        {
            'kind': 'call',
            'index': record['index'],
            'order': record['call_index'],
            'record': {**record, 'call': _jev_activity_call(record['call'])},
        }
        for record in call_records
    ]
    units.extend({'kind': 'orphan', 'index': index, 'order': 0, 'message': result} for index, result in orphan_records)
    units.sort(key=itemgetter('index', 'order'))
    return units


def _jev_group_activity_units(
    context: list[dict[str, Any]], units: list[dict[str, Any]], state_budget: int
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for unit in units:
        candidate = [*current, unit]
        if _jev_activity_state(context, candidate, state_budget, part=999_999_999, total=999_999_999) is not None:
            current = candidate
            continue
        if not current:
            raise ValueError('one Jev tool-activity record exceeds the reserved classifier state budget')
        groups.append(current)
        current = [unit]
        if _jev_activity_state(context, current, state_budget, part=999_999_999, total=999_999_999) is None:
            raise ValueError('one Jev tool-activity record exceeds the reserved classifier state budget')
    if current:
        groups.append(current)
    return groups


def _jev_activity_call(call: dict[str, Any]) -> dict[str, Any]:
    """Keep the coding command itself as the bounded input for shell tools."""
    if _call_name(call) not in SHELL_TOOLS:
        return call
    arguments = _call_arguments(call)
    function = call.get('function')
    if isinstance(function, dict):
        return {**call, 'function': {**function, 'arguments': _shell_command(arguments)}}
    return {**call, 'arguments': _shell_command(arguments)}


def _jev_activity_source(
    context: list[dict[str, Any]], units: list[dict[str, Any]]
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[int, int], dict[str, Any]],
    dict[int, dict[str, Any]],
    set[tuple[int, int]],
    set[int],
]:
    source = list(context)
    call_groups: dict[int, list[dict[str, Any]]] = {}
    events: list[tuple[int, int, str, Any, dict[str, Any] | None]] = []
    for unit in units:
        if unit['kind'] == 'call':
            record = unit['record']
            call_groups.setdefault(record['index'], []).append(record)
            events.extend((index, 1, 'result', result, record) for index, result in record['results'])
        else:
            events.append((unit['index'], 1, 'orphan', unit['message'], None))
    events.extend((index, 0, 'calls', records, None) for index, records in call_groups.items())
    events.sort(key=itemgetter(0, 1))

    call_metadata: dict[tuple[int, int], dict[str, Any]] = {}
    result_metadata: dict[int, dict[str, Any]] = {}
    expected_calls: set[tuple[int, int]] = set()
    expected_results: set[int] = set()
    for _, _, kind, message, record in events:
        if kind == 'calls':
            records = sorted(message, key=itemgetter('call_index'))
            message_index = len(source)
            source.append({'role': 'assistant', 'tool_calls': [call_record['call'] for call_record in records]})
            for call_index, call_record in enumerate(records):
                identity = (message_index, call_index)
                expected_calls.add(identity)
                paired = tuple(result for _, result in call_record['results'])
                call_id = call_record['call'].get('id')
                metadata: dict[str, Any] = {'status': _tool_call_status(call_id, paired)}
                category = _tool_result_category(call_id, paired)
                if category is not None:
                    metadata['diagnostic'] = category
                call_metadata[identity] = metadata
            continue

        index = len(source)
        source.append(message)
        expected_results.add(index)
        if kind == 'result' and record is not None and _call_name(record['call']) in SHELL_TOOLS:
            text = _jev_result_text(message).strip()
            markers = [label for label, pattern in OUTPUT_MARKERS if pattern.search(text)]
            if markers:
                result_metadata[index] = {'output_markers': markers}
    return source, call_metadata, result_metadata, expected_calls, expected_results


def _annotate_jev_activity_state(
    state: dict[str, Any],
    call_metadata: dict[tuple[int, int], dict[str, Any]],
    result_metadata: dict[int, dict[str, Any]],
) -> tuple[set[tuple[int, int]], set[int]]:
    visible_calls: set[tuple[int, int]] = set()
    visible_results: set[int] = set()
    for entry in state.get('messages', []):
        if entry.get('type') == 'tool_call':
            index = entry.get('index')
            call_index = entry.get('call_index')
            if isinstance(index, int) and isinstance(call_index, int):
                identity = (index, call_index)
                visible_calls.add(identity)
                entry.update(call_metadata.get(identity, {}))
            for result in entry.get('results', []):
                result_index = result.get('index') if isinstance(result, dict) else None
                if isinstance(result_index, int):
                    visible_results.add(result_index)
                    _clean_result_codes(result)
                    result.update(result_metadata.get(result_index, {}))
        elif entry.get('type') == 'orphan_result':
            index = entry.get('index')
            if isinstance(index, int):
                visible_results.add(index)
                _clean_result_codes(entry)
                entry.update(result_metadata.get(index, {}))
    return visible_calls, visible_results


def _clean_result_codes(result: dict[str, Any]) -> None:
    for key in ('status', 'exit_code', 'error_code'):
        if key in result:
            result[key] = clean_model_text(str(result[key]))


def _jev_activity_state(
    context: list[dict[str, Any]],
    units: list[dict[str, Any]],
    state_budget: int,
    *,
    part: int,
    total: int,
) -> dict[str, Any] | None:
    source, call_metadata, result_metadata, expected_calls, expected_results = _jev_activity_source(context, units)
    state = jev_state(source, global_char_cap=state_budget)
    state['activity_chunk'] = {'part': part, 'total': total}
    visible_calls, visible_results = _annotate_jev_activity_state(state, call_metadata, result_metadata)
    if visible_calls != expected_calls or visible_results != expected_results:
        return None
    return state if serialized_chars(state) <= state_budget else None


def _jev_result_text(message: dict[str, Any]) -> str:
    content = message.get('content')
    return coerce_content_text(content) if contains_media_content(content) else tool_result_to_text(content)


def _chunk(lines: list[str], budget: int) -> list[str]:
    """Split complete activity lines, bounding repeated user context before packing."""
    index = 0
    while index < len(lines) and lines[index].startswith('USER:'):
        index += 1
    body = lines[index:]
    if not body:
        raise ValueError('non-Jev tool activity has no call records to chunk')

    opening = '\n'.join(lines[:index])
    largest_record = max(len(line) + 1 for line in body)
    context_budget = budget - largest_record - 80
    if context_budget < len('[... 0 chars left out ...]'):
        raise ValueError('one non-Jev tool-activity record cannot fit with its chunk header')
    if len(opening) > context_budget:
        opening = cap_text(opening, context_budget)[0]
    capacity = budget - len(opening) - 80
    if capacity < largest_record:
        raise ValueError('one non-Jev tool-activity record cannot fit with its chunk context and header')

    groups: list[list[str]] = [[]]
    used = 0
    for line in body:
        if groups[-1] and used + len(line) + 1 > capacity:
            groups.append([])
            used = 0
        groups[-1].append(line)
        used += len(line) + 1
    chunks = [
        '\n'.join([opening, f'[part {number} of {len(groups)} of the tool calls, in order]', *group]).lstrip('\n')
        for number, group in enumerate(groups, start=1)
    ]
    if any(len(chunk) > budget for chunk in chunks):
        raise ValueError('non-Jev tool-activity chunk exceeded its question-reserved state budget')
    return chunks


def join_cut(head: str, tail: str, omitted: int) -> str:
    """Join the two kept ends of a cut text around a count of the characters left out."""
    return f'{head} [... {omitted} chars ...] {tail}'


def _results(messages: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    return {message.get('tool_call_id'): message for message in messages if message.get('role') == 'tool'}


def _activity_line(call: dict[str, Any], results: dict[Any, dict[str, Any]]) -> str:
    name, arguments = _call_name(call), _call_arguments(call)
    if name in SHELL_TOOLS:
        command = _shell_command(arguments)
        shown = command[:RISKY_COMMAND_CHARS] if is_risky_command(command) else command[:TOOL_INPUT_CHARS]
        call_input = shown.replace('\n', ' ⏎ ')
    else:
        call_input = _raw_arguments(call)[:TOOL_INPUT_CHARS]
    call_id = call.get('id')
    result = results.get(call_id)
    paired = (result,) if result else ()
    status = _tool_call_status(call_id, paired)
    # Result bodies are cut to keep a long trace inside one classifier call.
    # Non-shell results keep only the diagnostic category; shell output decides
    # whether a command worked, so it keeps an excerpt and its failure markers.
    category = _tool_result_category(call_id, paired)
    evidence = [f'diagnostic: {category}'] if category is not None else []
    if result and name in SHELL_TOOLS:
        text = tool_result_to_text(result.get('content'))
        markers = [label for label, pattern in OUTPUT_MARKERS if pattern.search(text)]
        if markers:
            evidence.append(f'output shows: {", ".join(markers)}')
        excerpt = _head_tail(scrub_known_secrets(text.strip()), SHELL_OUTPUT_CHARS // 2, SHELL_OUTPUT_CHARS // 2)
        if excerpt:
            evidence.append(f'output: {excerpt}'.replace('\n', ' ⏎ '))
    return f'CALL {name} [{status}]: {call_input}\n  → {"; ".join(evidence) or "result body omitted"}'




def _tool_calls(trace: TraceRecord | TraceDocument) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    for message in prompt_messages(trace):
        if message.get('role') != 'assistant':
            continue
        for call in message.get('tool_calls') or []:
            if isinstance(call, dict):
                yield message, call


def _call_name(call: dict[str, Any]) -> str:
    function = call.get('function')
    name = function.get('name') if isinstance(function, dict) else call.get('name')
    return name if isinstance(name, str) and name else '(unnamed)'


def _raw_arguments(call: dict[str, Any]) -> str:
    function = call.get('function')
    raw = function.get('arguments') if isinstance(function, dict) else call.get('arguments')
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw)
    except (TypeError, ValueError):
        return ''


def _call_arguments(call: dict[str, Any]) -> dict[str, Any]:
    """Parse a call's arguments; captured arguments can be cut mid-string, so fall back to the leading string fields."""
    raw = _raw_arguments(call)
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        unescape = {'n': '\n', 't': '\t'}
        return {
            key: re.sub(r'\\(.)', lambda escape: unescape.get(escape[1], escape[1]), value)
            for key, value in _STRING_FIELD.findall(raw)
        }
    return parsed if isinstance(parsed, dict) else {}


def _shell_command(arguments: dict[str, Any]) -> str:
    for key in ('command', 'cmd', 'script'):
        value = arguments.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return ' '.join(str(part) for part in value)
    return ''


def _shell_words(segment: str) -> list[str]:
    try:
        words = shlex.split(segment, comments=True)
    except ValueError:
        words = segment.split()
    return [word for word in words if word not in {'(', ')'}]


def _strip_wrappers(words: list[str]) -> list[str]:
    while words and words[0] in _WRAPPERS:
        sub = _WRAPPERS[words[0]]
        words = words[2:] if sub and len(words) > 1 and words[1] == sub else words[1:]
        # wrapper flags, `timeout 60`'s duration and `env -u NAME`'s value
        while words and (
            words[0].startswith('-') or re.fullmatch(r'\d+[smh]?', words[0]) or re.match(r'^\w+=', words[0])
        ):
            words = words[2:] if words[0] == '-u' else words[1:]
    return words


def _user_text(message: dict[str, Any]) -> str:
    return _REMINDER.sub('', coerce_content_text(message.get('content'))).strip()


def _head_tail(text: str, head: int, tail: int) -> str:
    if len(text) <= head + tail + CUT_SLACK:
        return text
    return join_cut(text[:head], text[-tail:], len(text) - head - tail)
