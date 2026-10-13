"""Text views of a trace for the Insights classifier, plus the tool counts read straight from its messages.

Three views, each for one `/decisions` call:

- `full_conversation_view`: the source messages, numbered by message index, for non-Jev classifiers and summaries. Each message type has its own cap (`-1` keeps it whole) and can be included or excluded, with reasoning left out by default; tool calls are either one collapsed line per run or one line per call with a status flag. The whole view is capped to the configured trace-input budget.
- `full_conversation_view`: the source messages as `#<index> <kind>:` entries, for non-Jev classifiers, summaries and Trace Finder. `ConversationViewOptions` and its presets `COMPACT`, `DETAILED` and `WHOLE` choose the message types and each type's cap; the trace-input budget then drops whole entries from the middle.
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
from dataclasses import dataclass
from itertools import groupby
from operator import itemgetter
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, get_args, overload

from pydantic import BaseModel

from evaluatorq.common.messages import coerce_content_text, contains_media_content, reasoning_text
from evaluatorq.common.model_input import (
    cap_text,
    clean_model_text,
    effective_trace_input_chars,
    is_jev_model,
    jev_state,
    jev_state_char_limit,
    message_lines,
    pair_tool_call_results,
    serialized_chars,
    serialized_question_chars,
    tool_result_metadata,
)
from evaluatorq.common.redact import scrub_known_secrets
from evaluatorq.common.trace_document import TraceDocument, to_chat_messages
from evaluatorq.contracts import tool_result_to_text
from evaluatorq.insights.readable_style import (
    MIN_REFERENCED_RESULT_CHARS,
    ReadableStyle,
    TextReducer,
    detect_working_directory,
    plain_arguments,
)
from evaluatorq.signals.config import SHELL_TOOL_NAMES
from evaluatorq.trace_finder.projection import _tool_call_status, _tool_result_category

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterator

    from evaluatorq.trace_finder.models import TraceRecord

TOOL_INPUT_CHARS = 300
USER_TURN_CHARS = 8_000
ASSISTANT_CHARS = 1_000
REASONING_CHARS = 2_000
TOOL_RESULT_CHARS = 300
TARGET_CHARS = 80
MessageKind = Literal['system', 'user', 'assistant', 'reasoning', 'tool_input', 'tool_result']
MESSAGE_KINDS: frozenset[MessageKind] = frozenset(get_args(MessageKind))
_KIND_ORDER: tuple[MessageKind, ...] = ('system', 'user', 'assistant', 'reasoning', 'tool_input', 'tool_result')
# `developer` carries system instructions; `function` is the legacy name for a tool result. Other roles render
# under their own name with the system cap.
_ROLE_KINDS: dict[str, MessageKind] = {
    'system': 'system',
    'developer': 'system',
    'user': 'user',
    'assistant': 'assistant',
    'tool': 'tool_result',
    'function': 'tool_result',
}
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
# are still cut to the tool-input cap.
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
            skills.update(_skill_names(call, arguments))
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


def _message_kinds(kinds: Collection[MessageKind], name: str) -> frozenset[MessageKind]:
    if isinstance(kinds, str):
        raise TypeError(f'{name} takes a collection of message kinds, not a string')
    if unknown := set(kinds) - MESSAGE_KINDS:
        raise ValueError(f'unknown message kinds {sorted(unknown)}, expected some of {sorted(MESSAGE_KINDS)}')
    return frozenset(kinds)


@dataclass(frozen=True)
class ConversationViewOptions:
    """What `full_conversation_view` renders and how much of each message type it keeps.

    `include` names the message types to render; reasoning is left out unless named. Every `*_chars` cap is the
    number of characters kept, split between the start and end of the text, and `-1` keeps that type whole.
    `tools='collapsed'` puts each message's tool calls on one `tools:` line and shows no results, so it rejects
    `tool_result` in `include`; there `tool_input_chars` bounds a risky shell command kept verbatim.
    `tools='per_call'` gives every call and every result its own line.
    """

    include: frozenset[MessageKind] = MESSAGE_KINDS - {'reasoning', 'tool_result'}
    tools: Literal['collapsed', 'per_call'] = 'collapsed'
    system_chars: int = -1
    user_chars: int = USER_TURN_CHARS
    assistant_chars: int = ASSISTANT_CHARS
    reasoning_chars: int = REASONING_CHARS
    tool_input_chars: int = TOOL_INPUT_CHARS
    tool_result_chars: int = TOOL_RESULT_CHARS

    def __post_init__(self) -> None:
        object.__setattr__(self, 'include', _message_kinds(self.include, 'include'))
        if self.tools not in ('collapsed', 'per_call'):
            raise ValueError(f"tools must be 'collapsed' or 'per_call', got {self.tools!r}")
        if self.tools == 'collapsed' and 'tool_result' in self.include:
            raise ValueError("tools='collapsed' shows no tool results; use tools='per_call' or leave out 'tool_result'")
        for kind, cap in self.caps.items():
            if cap != -1 and cap < 2:
                raise ValueError(f'the {kind} cap must be -1 (whole) or at least 2, got {cap}')

    @property
    def caps(self) -> dict[MessageKind, int]:
        return {
            'system': self.system_chars,
            'user': self.user_chars,
            'assistant': self.assistant_chars,
            'reasoning': self.reasoning_chars,
            'tool_input': self.tool_input_chars,
            'tool_result': self.tool_result_chars,
        }


# One line of tool names per message, no results: the cheapest view that still shows what the agent did.
COMPACT = ConversationViewOptions()
# Every call and its result, results cut to 300 characters: summaries and classifiers judge failed and risky calls.
DETAILED = ConversationViewOptions(include=MESSAGE_KINDS - {'reasoning'}, tools='per_call')
# Everything whole, one line per call: Trace Finder counts which messages survive the model-input cap.
WHOLE = ConversationViewOptions(
    include=MESSAGE_KINDS - {'reasoning'},
    tools='per_call',
    system_chars=-1,
    user_chars=-1,
    assistant_chars=-1,
    reasoning_chars=-1,
    tool_input_chars=-1,
    tool_result_chars=-1,
)


def full_conversation_view_with_message_spans(
    trace: TraceRecord | TraceDocument,
) -> tuple[str, tuple[tuple[int, int, int], ...], int]:
    """Return the complete `WHOLE` rendering, its entry spans, and the source-message count.

    Every included message type is kept whole and each tool call gets its own line, so a span covers its source message.
    """
    conversation, spans, count, _, parts_cut = _render_full_conversation(
        trace, include=WHOLE.include, caps=WHOLE.caps, tools='per_call'
    )
    return message_lines(conversation, spans, parts_cut=parts_cut), spans, count


def _view_kinds(
    include: Collection[MessageKind] | None,
    exclude: Collection[MessageKind],
    default_include: frozenset[MessageKind],
) -> frozenset[MessageKind]:
    selected = _message_kinds(include if include is not None else default_include, 'include')
    dropped = _message_kinds(exclude, 'exclude')
    if include is not None and (both := sorted(selected & dropped)):
        raise ValueError(f'include and exclude both name {", ".join(both)}; name each message type in only one')
    return selected - dropped


def _view_caps(
    options: ConversationViewOptions,
    *,
    system_chars: int | None,
    user_chars: int | None,
    assistant_chars: int | None,
    assistant_edge: int | None,
    reasoning_chars: int | None,
    tool_input_chars: int | None,
    tool_result_chars: int | None,
    tool_max_chars: int | None,
) -> dict[MessageKind, int]:
    caps = dict(options.caps)
    if caps['assistant'] != -1:
        caps['assistant'] = max(1, caps['assistant'] // 2)
    caps.update({
        kind: value
        for kind, value in (
            ('system', system_chars),
            ('user', user_chars),
            ('reasoning', reasoning_chars),
            ('tool_input', tool_input_chars),
            ('tool_result', tool_result_chars),
        )
        if value is not None
    })
    if assistant_chars is not None:
        caps['assistant'] = assistant_chars if assistant_chars == -1 else max(1, assistant_chars // 2)
    if assistant_edge is not None:
        caps['assistant'] = assistant_edge
    if tool_max_chars is not None:
        caps['tool_result'] = tool_max_chars
    for kind, cap in caps.items():
        floor = 0 if kind == 'tool_result' else 1
        if cap != -1 and cap < floor:
            raise ValueError(f'the {kind} cap must be -1 (whole) or >= {floor}, got {cap}')
    return caps


def _view_tools(
    tools: Literal['collapsed', 'per_call'] | None,
    default_tools: Literal['collapsed', 'per_call'],
    kinds: frozenset[MessageKind],
) -> Literal['collapsed', 'per_call']:
    selected = tools if tools is not None else default_tools
    if selected not in ('collapsed', 'per_call'):
        raise ValueError(f"tools must be 'collapsed' or 'per_call', got {selected!r}")
    if selected == 'collapsed' and 'tool_result' in kinds:
        raise ValueError("tools='collapsed' shows no tool results; use tools='per_call' or leave out 'tool_result'")
    return selected


def full_conversation_view(
    trace: TraceRecord | TraceDocument,
    budget: int | None = None,
    options: ConversationViewOptions = COMPACT,
    *,
    include: Collection[MessageKind] | None = None,
    exclude: Collection[MessageKind] = (),
    system_chars: int | None = None,
    user_chars: int | None = None,
    assistant_chars: int | None = None,
    assistant_edge: int | None = None,
    reasoning_chars: int | None = None,
    tools: Literal['collapsed', 'per_call'] | None = None,
    tool_input_chars: int | None = None,
    tool_result_chars: int | None = None,
    tool_max_chars: int | None = None,
    coverage_header: bool = False,
    style: ReadableStyle | None = None,
) -> str:
    """Render one preset with source-indexed entries, optional overrides and opt-in readable styling."""
    kinds = _view_kinds(include, exclude, options.include)
    caps = _view_caps(
        options,
        system_chars=system_chars,
        user_chars=user_chars,
        assistant_chars=assistant_chars,
        assistant_edge=assistant_edge,
        reasoning_chars=reasoning_chars,
        tool_input_chars=tool_input_chars,
        tool_result_chars=tool_result_chars,
        tool_max_chars=tool_max_chars,
    )
    selected_tools = _view_tools(tools, options.tools, kinds)
    rendered, spans, message_count, present, parts_cut = _render_full_conversation(
        trace, include=kinds, caps=caps, tools=selected_tools, style=style
    )
    rendered = message_lines(rendered, spans, parts_cut=parts_cut)
    cap = budget if budget is not None else effective_trace_input_chars()
    if not coverage_header:
        return cap_text(rendered, cap)[0]
    header_settings: dict[str, Any] = {
        'kinds': kinds,
        'caps': caps,
        'tools': selected_tools,
        'present': present,
        'message_count': message_count,
        'style': style or ReadableStyle(),
    }
    header = _coverage_header(**header_settings, cut=False)
    body, omitted = _cap_under_header(rendered, header, cap)
    if omitted:
        header = _coverage_header(**header_settings, cut=True)
        body, _ = _cap_under_header(rendered, header, cap)
    return message_lines(f'{header}\n{body}' if body else header, (), parts_cut=parts_cut or bool(omitted))


def _cap_under_header(rendered: str, header: str, cap: int) -> tuple[str, int]:
    room = cap - len(header) - 1
    if room < 1:
        raise ValueError('the view budget is too small for the coverage header')
    return cap_text(rendered, room)


_KIND_LABELS: dict[MessageKind, str] = {
    'system': 'system',
    'user': 'user',
    'assistant': 'assistant',
    'reasoning': 'reasoning',
    'tool_input': 'tool input',
    'tool_result': 'tool result',
}


def _cap_phrase(kind: MessageKind, cap: int) -> str:
    label = _KIND_LABELS[kind]
    if cap == -1:
        return f'{label} whole'
    if kind == 'assistant':
        return f'{label} {cap} chars at each end'
    if kind == 'tool_input':
        return f'{label} {cap} chars'
    return f'{label} {cap} chars (start and end kept)'


def _coverage_header(
    *,
    kinds: frozenset[MessageKind],
    caps: dict[MessageKind, int],
    tools: Literal['collapsed', 'per_call'],
    present: frozenset[MessageKind],
    message_count: int,
    cut: bool,
    style: ReadableStyle,
) -> str:
    """Describe the rendered view from the settings in effect and from which kinds the trace turned out to hold."""
    shown = [
        kind
        for kind in _KIND_ORDER
        if kind in kinds and not (kind == 'tool_result' and not _shows_results(kinds, caps, tools))
    ]
    included = [_KIND_LABELS[kind] for kind in shown if kind in present]
    absent = [_KIND_LABELS[kind] for kind in shown if kind not in present]
    left_out = [_KIND_LABELS[kind] for kind in _KIND_ORDER if kind not in shown]
    # A collapsed run names tools without their inputs, so only per-call mode has an input cut to report.
    cuts = [
        _cap_phrase(kind, caps[kind])
        for kind in shown
        if kind in present and not (kind == 'tool_input' and tools == 'collapsed')
    ]
    parts = [f'includes {", ".join(included) or "nothing"}']
    if left_out:
        parts.append(f'left out: {", ".join(left_out)}')
    if absent:
        parts.append(f'none in this trace: {", ".join(absent)}')
    if cuts:
        parts.append(f'cuts: {", ".join(cuts)}')
    if 'tool_input' in shown:
        parts.append(
            ('tools: one block per call' if style.blocks else 'tools: one line per call')
            if tools == 'per_call'
            else 'tools: calls collapsed to one `tools:` line per run, results left out'
        )
    pairing_enabled = (
        style.blocks and tools == 'per_call' and 'tool_input' in kinds and _shows_results(kinds, caps, tools)
    )
    layout = [note for note in style.format_notes() if note != 'results paired with calls' or pairing_enabled]
    if layout:
        parts.append(f'format: {", ".join(layout)}')
    if style.pair_results and not pairing_enabled:
        parts.append(
            'pairing disabled: requires block layout'
            if not style.blocks
            else 'pairing disabled: requires shown tool inputs and results in per-call mode'
        )
    if reductions := style.reduction_notes():
        parts.append(f'reduced: {", ".join(reductions)}')
    parts.append(f'{message_count} source messages')
    if cut:
        parts.append('view cut to budget')
    return f'[context: {"; ".join(parts)}]'


def _shows_results(
    kinds: frozenset[MessageKind], caps: dict[MessageKind, int], tools: Literal['collapsed', 'per_call']
) -> bool:
    return 'tool_result' in kinds and tools == 'per_call' and caps['tool_result'] != 0


def _block(header: str, body: str) -> str:
    """Keep source-message headers separate from every body line, including marker-shaped source text."""
    return '\n'.join([header, *(f'  {row}' for row in body.split('\n'))]) if body else header


def _header_text(value: Any) -> str:
    """Scrub untrusted header fields without letting them insert another header line."""
    return clean_model_text(value).replace('\r', ' ⏎ ').replace('\n', ' ⏎ ')


def _result_outcome(message: dict[str, Any]) -> str:
    """Read status and exit/error codes before truncating the result body."""
    status = _tool_call_status('result', ({**message, 'tool_call_id': 'result'},))
    _, exit_code, error_code = tool_result_metadata(message)
    if exit_code is None:
        content = tool_result_to_text(message.get('content'))
        try:
            decoded = json.loads(content)
        except ValueError:
            decoded = None
        if isinstance(decoded, dict):
            exit_code = decoded.get('exit_code')
            outcome = decoded.get('outcome')
            if exit_code is None and isinstance(outcome, dict):
                exit_code = outcome.get('exit_code')
        elif match := re.match(r'Exit (?:code|status)[: ]+(-?\d+)\b', content, re.IGNORECASE):
            exit_code = match[1]
    if exit_code is not None and not isinstance(exit_code, bool) and re.fullmatch(r'-?\d+', str(exit_code)):
        if str(exit_code).lstrip('-').strip('0'):
            status = 'error'
    if error_code not in (None, False, ''):
        status = 'error'
    parts = [status]
    if exit_code is not None:
        parts.append(f'exit {_header_text(exit_code)}')
    if error_code is not None:
        parts.append(f'error {_header_text(error_code)}')
    return ', '.join(parts)


def _call_outcome(results: list[tuple[int, dict[str, Any]]]) -> str:
    """Keep all distinct outcomes when a call has more than one result."""
    outcomes = dict.fromkeys(_result_outcome(result) for _, result in results)
    return '; '.join(sorted(outcomes, key=lambda outcome: not outcome.startswith('error'))) or 'pending'


def _call_runs(messages: list[dict[str, Any]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Fold only adjacent identical calls; source prose breaks a run even when the view excludes it."""
    runs: dict[tuple[int, int], list[tuple[int, int]]] = {}
    head: tuple[int, int] | None = None
    signature: tuple[str, str] | None = None
    for index, message in enumerate(messages):
        if message.get('role') == 'tool':
            continue
        if reasoning_text(message) or coerce_content_text(message.get('content')).strip():
            head = None
        calls = message.get('tool_calls') or []
        if message.get('role') != 'assistant' or not calls:
            head = None
            continue
        for call_index, call in enumerate(calls):
            if not isinstance(call, dict):
                head = None
                continue
            key = (index, call_index)
            call_signature = (_prepare(_call_name(call)), _prepare(_raw_arguments(call)))
            if head is not None and call_signature == signature:
                runs[head].append(key)
            else:
                head, signature = key, call_signature
                runs[head] = [key]
    return {first: members for first, members in runs.items() if len(members) > 1}


def _run_status(statuses: list[str]) -> str:
    return 'error' if 'error' in statuses else 'pending' if 'pending' in statuses else 'completed'


def _view_working_directory(
    trace: TraceRecord | TraceDocument, records: dict[tuple[int, int], dict[str, Any]]
) -> str | None:
    metadata = trace.metadata if isinstance(trace, TraceDocument) else trace
    capture = metadata.capture_metadata
    recorded = [value for key in ('cwd', 'workdir', 'working_directory') if isinstance(value := capture.get(key), str)]
    if str(capture.get('source', '')).startswith('local:') and metadata.project.startswith('/'):
        recorded.append(metadata.project)
    return detect_working_directory((_raw_arguments(record['call']) for record in records.values()), recorded=recorded)


class _ConversationRenderer:
    """Render one view while keeping pairing, pending calls and result references local to that view."""

    def __init__(
        self,
        *,
        trace: TraceRecord | TraceDocument,
        include: frozenset[MessageKind],
        caps: dict[MessageKind, int],
        tools: Literal['collapsed', 'per_call'],
        style: ReadableStyle | None,
    ) -> None:
        self.include = include
        self.caps = caps
        self.tools = tools
        self.explicit_style = style is not None and any((
            style.blocks,
            style.plain_tool_input,
            style.labelled_cuts,
            style.strip_terminal_noise,
            style.shorten_cwd,
            style.collapse_repeats,
            style.collapse_blobs,
        ))
        self.style = style or ReadableStyle()
        style = self.style
        self.messages = to_chat_messages(trace)
        calls, _ = pair_tool_call_results(self.messages)
        if style.blocks:
            for record in calls:
                if not isinstance(record['call'].get('id'), str) or not record['call']['id']:
                    record['results'] = []
        self.records: dict[tuple[int, int], dict[str, Any]] = {
            (record['index'], record['call_index']): record for record in calls
        }
        self.statuses = {
            key: _tool_call_status(record['call'].get('id'), tuple(result for _, result in record['results']))
            for key, record in self.records.items()
        }
        self.call_labels = {key: f'c{number}' for number, key in enumerate(self.records, 1)}
        self.result_labels: dict[int, str] = {
            result_index: self.call_labels[key]
            for key, record in self.records.items()
            for result_index, _ in record['results']
        }
        self.result_call_details: dict[int, tuple[str, str]] = {
            result_index: (
                f'{_call_name(record["call"])} (call {record["index"]}.{record["call_index"] + 1})',
                self.statuses[key],
            )
            for key, record in self.records.items()
            for result_index, _ in record['results']
        }
        self.show_results = _shows_results(include, caps, tools)
        self.pair = style.pair_results and style.blocks and tools == 'per_call' and 'tool_input' in include
        self.paired_results = set(self.result_labels) if self.pair else set()
        self.cwd = _view_working_directory(trace=trace, records=self.records) if style.shorten_cwd else None
        self.reducer = TextReducer.for_style(style, self.cwd)
        self.parts_cut = False
        self.runs = _call_runs(self.messages) if style.collapse_repeats and tools == 'per_call' else {}
        self.folded = {member for members in self.runs.values() for member in members[1:]}
        self.seen_results: dict[str, str] = {}
        self.lines: list[tuple[str, int]] = []
        self.present: set[MessageKind] = set()
        self.pending: list[str] = []
        self.pending_index = 0

    def _cut_label(self, kind: MessageKind) -> str | None:
        return _KIND_LABELS[kind] if self.style.labelled_cuts else None

    def _fit(self, text: str, cap: int) -> str:
        fitted, cut = _fit(text, cap)
        self.parts_cut |= cut
        return fitted

    def _add(self, *, index: int, label: str, text: str) -> None:
        rendered = (
            _block(f'[{index}] {_header_text(label)}', text) if self.style.blocks else f'#{index} {label}: {text}'
        )
        self.lines.append((rendered, index))

    def _flush(self) -> None:
        if self.pending:
            runs = ((entry, len(list(repeats))) for entry, repeats in groupby(self.pending))
            names = ', '.join(f'{entry} x{count}' if count > 1 else entry for entry, count in runs)
            self._add(index=self.pending_index, label='tools', text=names)
            self.pending.clear()

    def _result_text(self, *, index: int, message: dict[str, Any]) -> str:
        content = message.get('content')
        text = coerce_content_text(content) if contains_media_content(content) else tool_result_to_text(content)
        prepared = _prepare(text, self.reducer) if self.explicit_style else clean_model_text(text).strip()
        # Compare before lossy reductions and cuts: distinct blobs or truncated middles must not compare equal.
        if self.style.collapse_repeats:
            scrubbed = _prepare(text)
            if len(scrubbed) >= MIN_REFERENCED_RESULT_CHARS:
                earlier = self.seen_results.get(scrubbed)
                if earlier is None:
                    label = self.result_labels.get(index)
                    self.seen_results[scrubbed] = (
                        f'[{index}] ({label})' if self.style.blocks and label else f'[{index}]'
                    )
                else:
                    reference = f'same as result {earlier}'
                    if len(reference) < len(prepared):
                        return (
                            self._fit(reference, self.caps['tool_result'])
                            if not self.explicit_style
                            else _cut(reference, self.caps['tool_result'], self._cut_label('tool_result'))
                        )
        return (
            self._fit(prepared, self.caps['tool_result'])
            if not self.explicit_style
            else _cut(prepared, self.caps['tool_result'], self._cut_label('tool_result'))
        )

    def _add_result(self, *, index: int, message: dict[str, Any]) -> None:
        body = self._result_text(index=index, message=message)
        if self.style.blocks:
            label = self.result_labels.get(index)
            subject = f'result of {label}' if label else 'orphan (no matching call)'
            rendered = _block(f'[{index}] tool result [{_result_outcome(message)}] {subject}', body)
        else:
            label, status = self.result_call_details.get(index, ('(no matching call)', 'completed'))
            flag = '' if status == 'completed' else f' [{status}]'
            rendered = f'#{index} tool result {label}{flag}: {body}'
        self.lines.append((rendered, index))
        self.present.add('tool_result')

    def _tool_input_text(self, call: dict[str, Any]) -> str:
        cap = self.caps['tool_input']
        if not (self.style.plain_tool_input or self.style.labelled_cuts or self.reducer):
            return self._fit(_input_text(call), cap)
        shell_command = not self.style.plain_tool_input and _call_name(call) in SHELL_TOOLS
        if self.style.plain_tool_input:
            text = _prepare(_raw_arguments(call), self.reducer, transform=plain_arguments)
        else:
            text = _prepare(
                _shell_command(_call_arguments(call)) if shell_command else _raw_arguments(call), self.reducer
            )
        if shell_command and not self.style.blocks:
            text = text.replace('\n', ' ⏎ ')
        if cap != -1 and len(text) > cap:
            text = _labelled_cut(text, cap, 'tool input', first_only=True) if self.style.labelled_cuts else text[:cap]
        return text

    def _add_message_text(self, *, index: int, message: dict[str, Any]) -> None:
        if 'reasoning' in self.include and (reasoning := reasoning_text(message)):
            self._flush()
            text = (
                self._fit(reasoning, self.caps['reasoning'])
                if not self.explicit_style
                else _clean(reasoning, self.caps['reasoning'], label=self._cut_label('reasoning'))
            )
            self._add(index=index, label='reasoning', text=text)
            self.present.add('reasoning')
        role = str(message.get('role') or 'unknown')
        kind = _ROLE_KINDS.get(role, 'system')
        content = coerce_content_text(message.get('content'))
        if kind in self.include and content.strip():
            self._flush()
            cap = self.caps[kind]
            if not self.explicit_style:
                text = self._fit(content, cap * 2 if kind == 'assistant' and cap != -1 else cap)
            else:
                text = _clean(content, -1 if kind == 'assistant' else cap, label=self._cut_label(kind))
                if kind == 'assistant' and cap != -1:
                    text = _head_tail(text, cap, cap, self._cut_label(kind))
            self._add(index=index, label=role, text=text)
            self.present.add(kind)

    def _add_paired_results(self, members: list[tuple[int, int]]) -> None:
        if self.pair and self.show_results:
            for member in members:
                for result_index, result in self.records[member]['results']:
                    self._add_result(index=result_index, message=result)

    def _add_call(self, *, index: int, call: dict[str, Any], key: tuple[int, int]) -> None:
        members = self.runs.get(key, [key])
        status = _run_status([self.statuses[member] for member in members])
        times = f' x{len(members)}' if len(members) > 1 else ''
        arguments = self._tool_input_text(call=call)
        if self.style.blocks:
            span = (
                self.call_labels[key]
                if len(members) == 1
                else f'{self.call_labels[key]}-{self.call_labels[members[-1]]}'
            )
            outcomes = sorted(
                dict.fromkeys(_call_outcome(self.records[member]['results']) for member in members),
                key=lambda outcome: not outcome.startswith('error'),
            )
            header = f'[{index}] tool call [{"; ".join(outcomes)}] {_header_text(_call_name(call))} ({span}){times}'
            rendered = _block(header, arguments)
        else:
            flag = '' if status == 'completed' else f' [{status}]'
            rendered = f'#{index} tool input {_call_label(call, index, key[1])}{times}{flag}: {arguments}'
        self.lines.append((rendered, index))
        self.present.add('tool_input')
        self._add_paired_results(members=members)

    def _add_calls(self, *, index: int, message: dict[str, Any]) -> None:
        if 'tool_input' not in self.include:
            return
        for call_index, call in enumerate(message.get('tool_calls') or []):
            if not isinstance(call, dict):
                continue
            if self.tools == 'collapsed':
                if not self.pending:
                    self.pending_index = index
                key = (index, call_index)
                status = self.statuses.get(key, 'completed')
                flag = '' if status == 'completed' else f' [{status}]'
                self.pending.append(f'{_call_line(call, self.caps["tool_input"])}{flag}')
                self.present.add('tool_input')
                continue
            key = (index, call_index)
            if key not in self.folded:
                self._add_call(index=index, call=call, key=key)

    def render(self) -> tuple[str, tuple[tuple[int, int, int], ...], int, frozenset[MessageKind], bool]:
        for index, message in enumerate(self.messages):
            role = str(message.get('role') or 'unknown')
            if _ROLE_KINDS.get(role) == 'tool_result':
                if self.show_results and index not in self.paired_results:
                    self._add_result(index=index, message=message)
                continue
            self._add_message_text(index=index, message=message)
            self._add_calls(index=index, message=message)
        self._flush()
        if not self.style.blocks:
            self.lines = [(line.replace('\n', '\n  '), index) for line, index in self.lines]
        if self.lines and self.cwd:
            self.lines.insert(0, (f'paths under {self.cwd} are shown as ./', 0))
        conversation = '\n'.join(line for line, _ in self.lines)
        spans = []
        offset = 0
        for line, index in self.lines:
            spans.append((offset, offset + len(line), index))
            offset += len(line) + 1
        return conversation, tuple(spans), len(self.messages), frozenset(self.present), self.parts_cut


def _render_full_conversation(
    trace: TraceRecord | TraceDocument,
    *,
    include: frozenset[MessageKind],
    caps: dict[MessageKind, int],
    tools: Literal['collapsed', 'per_call'],
    style: ReadableStyle | None = None,
) -> tuple[str, tuple[tuple[int, int, int], ...], int, frozenset[MessageKind], bool]:
    """Render text and source-message spans without deriving message boundaries from body text."""
    return _ConversationRenderer(trace=trace, include=include, caps=caps, tools=tools, style=style).render()


def _prepare(text: str, reducer: TextReducer | None = None, transform: Callable[[str], str] | None = None) -> str:
    """Scrub credentials, mark injected reminders, then apply the opt-in reductions to what is left.

    `transform` reshapes a scrubbed text first (a JSON argument into plain lines); the text is scrubbed again after
    the reductions that run before scrubbing, so nothing they join back together is left unscrubbed.
    """
    if reducer is not None or transform is not None:
        text = scrub_known_secrets(text)
    if transform is not None:
        text = transform(text)
    if reducer is not None:
        text = reducer.before_scrub(text)
    text = _REMINDER.sub('[harness reminder omitted]', scrub_known_secrets(text))
    return reducer.after_scrub(text) if reducer is not None else text


def _cut(prepared: str, cap: int, label: str | None = None) -> str:
    """Keep the start and end of a prepared text within `cap` (-1 keeps it whole)."""
    if cap == -1:
        return prepared
    return _head_tail(prepared if label is not None else prepared.strip(), cap // 2, cap - cap // 2, label)


def _clean(text: str, cap: int, *, label: str | None = None, reducer: TextReducer | None = None) -> str:
    """Scrub credentials, mark injected reminders, then keep the start and end within `cap` (-1 keeps it whole)."""
    return _cut(_prepare(text, reducer), cap, label)


def _fit(text: str, cap: int) -> tuple[str, bool]:
    """Match the production renderer's per-kind cap and omission marker."""
    text = clean_model_text(text).strip()
    if cap == -1 or len(text) <= cap + CUT_SLACK:
        return text, False
    return join_cut(text[: cap // 2], text[-(cap - cap // 2) :], len(text) - cap), True


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
    messages = to_chat_messages(trace)
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
    call_records, _ = pair_tool_call_results(messages)
    paired = {
        (record['index'], record['call_index']): tuple(result for _, result in record['results'])
        for record in call_records
    }
    lines: list[str] = []
    for index, message in enumerate(messages):
        role = message.get('role')
        if role == 'user':
            text = _user_text(message)
            if text:
                lines.append(f'USER: {_head_tail(text, 400, 200)}')
        elif role == 'assistant':
            lines.extend(
                _activity_line(call, paired[index, call_index])
                for call_index, call in enumerate(message.get('tool_calls') or [])
                if isinstance(call, dict)
            )
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
            text = _result_text(message).strip()
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


def _result_text(message: dict[str, Any]) -> str:
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
    return f'{head} [... {omitted} chars left out ...] {tail}'


def _call_label(call: dict[str, Any], index: int, call_index: int) -> str:
    return f'{_call_name(call)} (call {index}.{call_index + 1})'


def _input_text(call: dict[str, Any]) -> str:
    arguments = _call_arguments(call)
    return _shell_command(arguments) if _call_name(call) in SHELL_TOOLS else _raw_arguments(call)


def _call_input(call: dict[str, Any], limit: int = TOOL_INPUT_CHARS) -> str:
    text = _input_text(call)
    if _call_name(call) in SHELL_TOOLS:
        text = text.replace('\n', ' ⏎ ')
    return scrub_known_secrets(text)[:limit]


def _activity_line(call: dict[str, Any], paired: tuple[dict[str, Any], ...]) -> str:
    name, call_input = _call_name(call), _call_input(call)
    call_id = call.get('id')
    status = _tool_call_status(call_id, paired)
    result = paired[0] if paired else None
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


def _call_line(call: dict[str, Any], limit: int) -> str:
    """One call inside a collapsed `tools:` line: the program, skill or target, or a risky command cut to `limit`."""
    name, arguments = _call_name(call), _call_arguments(call)
    if name in SHELL_TOOLS:
        command = _shell_command(arguments)
        if is_risky_command(command):
            return f'{name}: `{_fit(command, limit)[0].replace(chr(10), " ⏎ ")}`'
        return f'{name}: {first_command(command)}'
    if name == SKILL_TOOL:
        skills = _skill_names(call, arguments)
        return f'{name}: {skills[0] if skills else "?"}'
    for key in ('file_path', 'path', 'url', 'query', 'description', 'pattern'):
        value = arguments.get(key)
        if isinstance(value, str):
            return f'{name}: {_fit(value, TARGET_CHARS)[0]}'
    return name


def _skill_names(call: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    names = _SKILL_ARG.findall(_raw_arguments(call)) or ([arguments['skill']] if 'skill' in arguments else [])
    return [scrub_known_secrets(str(name)) for name in names]


def _tool_calls(trace: TraceRecord | TraceDocument) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    for message in to_chat_messages(trace):
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
    """The shell line a call runs, scrubbed of credentials so no view derived from it can carry one."""
    for key in ('command', 'cmd', 'script'):
        value = arguments.get(key)
        if isinstance(value, str):
            return scrub_known_secrets(value)
        if isinstance(value, list):
            return scrub_known_secrets(' '.join(str(part) for part in value))
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


def _labelled_cut(text: str, limit: int, label: str, *, first_only: bool = False) -> str:
    """Fit source characters and a kind-specific exact omission marker within the part's final cap."""
    if len(text) <= limit:
        return text
    marker = f'[... {len(text)} chars of {label} left out ...]'
    if len(marker) > limit:
        raise ValueError('the message cap is too small for a labelled omission marker')
    retained = limit - len(marker)
    while True:
        omitted = len(text) - retained
        marker = f'[... {omitted} chars of {label} left out ...]'
        room = limit - len(marker)
        if retained <= room:
            break
        retained = room
    first = retained if first_only else retained // 2
    last = 0 if first_only else retained - first
    return f'{text[:first]}{marker}{text[-last:] if last else ""}'


def _head_tail(text: str, head: int, tail: int, label: str | None = None) -> str:
    """Keep both text ends; opt-in labelled markers count against the final per-part cap."""
    if label is not None:
        return _labelled_cut(text, head + tail, label)
    if len(text) <= head + tail + CUT_SLACK:
        return text
    omitted = len(text) - head - tail
    return join_cut(text[:head], text[-tail:], omitted)
