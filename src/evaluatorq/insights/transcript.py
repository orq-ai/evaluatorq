"""Text views of a trace for the Insights classifier, plus the tool counts read straight from its messages.

Three views, each for one `/classify` call:

- `conversation_view`: what the user said and what the agent answered. User
  turns stay whole, assistant text keeps its start and end, tool calls shrink
  to one line each and tool outputs are dropped. The opening user turns are
  never cut; an over-budget trace loses its middle.
- `tool_inventory`: which tools ran and how often, for the coding-agent check.
- `tool_activity_chunks`: every tool call with its input and status, the start
  and end of each shell output, and fixed labels for failures found in it, for the
  questions that need to see what a command did; a long trace is split into
  several chunks rather than losing more than 30% of its middle.

Nothing here calls a model.
"""

from __future__ import annotations

import json
import re
import shlex
from collections import Counter
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from evaluatorq.common.trace_document import TraceDocument, prompt_messages
from evaluatorq.contracts import content_to_text, tool_result_to_text
from evaluatorq.signals.config import SHELL_TOOL_NAMES
from evaluatorq.trace_finder.projection import _tool_call_status, _tool_result_category

if TYPE_CHECKING:
    from collections.abc import Iterator

    from evaluatorq.trace_finder.models import TraceRecord

# Characters, sized for about 25k classifier tokens: tool-call text runs near 3 characters a token
# (measured worst case 2.9), and `/classify` rejected views of 29k+ tokens with max_tokens_exceeded.
VIEW_BUDGET = 75_000
ASSISTANT_EDGE = 300
TOOL_INPUT_CHARS = 300
OPENING_SHARE = 0.4
# Cut at most this share of a tool-activity view from the middle; longer views are split into chunks.
MAX_CUT_SHARE = 0.3
RISKY_COMMAND_CHARS = 300
# Shell output kept per call, split between its start and end, where exit codes and test summaries sit.
SHELL_OUTPUT_CHARS = 300
# Credential-shaped text masked in the shell excerpt; the classifier never needs the value.
_CREDENTIAL = re.compile(
    r'\b(?:sk|pk|rk|ghp|gho|ghs|xox[abprs])[-_][A-Za-z0-9_-]{8,}'
    r'|(?i:(bearer\s+|(?:token|secret|password|api[_-]?key|credential)s?\s*[=:]\s*))\S+'
)
USER_TURN_CHARS = 8_000

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


def conversation_view(trace: TraceRecord | TraceDocument, budget: int = VIEW_BUDGET) -> str:
    """Render the conversation as plain text within `budget` characters.

    User turns are kept whole up to `USER_TURN_CHARS` (a pasted log keeps its
    start and end), minus injected `<system-reminder>` blocks;
    assistant text keeps its first and last `ASSISTANT_EDGE` characters, and a
    run of tool calls becomes one `[tools]` line with repeats counted. Tool
    outputs are dropped. A risky shell command is kept verbatim rather than
    collapsed to its program name, up to `RISKY_COMMAND_CHARS` characters.
    """
    lines: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            runs = Counter(pending)
            lines.append('  [tools] ' + ', '.join(f'{name} x{n}' if n > 1 else name for name, n in runs.items()))
            pending.clear()

    for message in prompt_messages(trace):
        role = message.get('role')
        if role == 'user':
            text = _user_text(message)
            if text:
                flush()
                lines.append(f'USER: {_head_tail(text, USER_TURN_CHARS // 2, USER_TURN_CHARS // 2)}')
        elif role == 'assistant':
            text = content_to_text(message.get('content')).strip()
            if text:
                flush()
                lines.append(f'ASSISTANT: {_head_tail(text, ASSISTANT_EDGE, ASSISTANT_EDGE)}')
            pending.extend(_call_line(call) for call in message.get('tool_calls') or [])
    flush()
    return _fit(lines, budget, keep_opening_users=True)


def tool_activity_chunks(trace: TraceRecord | TraceDocument, budget: int = VIEW_BUDGET) -> list[str]:
    """Render every tool call with its input and status, with user turns for context.

    Used for the questions that need to see what a command did: whether an error
    stayed unfixed, and whether an action was risky. Assistant prose is left out,
    and each tool result body is replaced by its diagnostic category; a shell
    call also keeps the start and end of its output and the failure markers
    found in all of it.
    A view that fits after cutting at most `MAX_CUT_SHARE` of it from the middle is
    one chunk; a longer one is split into consecutive chunks of at most `budget`
    characters, each starting with the opening user turns, for the caller to ask
    separately and merge.
    """
    messages = prompt_messages(trace)
    results = {message.get('tool_call_id'): message for message in messages if message.get('role') == 'tool'}
    lines: list[str] = []
    for message in messages:
        role = message.get('role')
        if role == 'user':
            text = _user_text(message)
            if text:
                lines.append(f'USER: {_head_tail(text, 400, 200)}')
        elif role == 'assistant':
            lines.extend(_activity_line(call, results) for call in message.get('tool_calls') or [])
    if not any(line.startswith('CALL') for line in lines):
        return ['No tool calls.']
    if sum(len(line) + 1 for line in lines) * (1 - MAX_CUT_SHARE) <= budget:
        return [_fit(lines, budget, keep_opening_users=True)]
    return _chunk(lines, budget)


def _chunk(lines: list[str], budget: int) -> list[str]:
    """Split `lines` into consecutive chunks within `budget`, each repeating the opening user turns."""
    index = 0
    while index < len(lines) and lines[index].startswith('USER:'):
        index += 1
    opening = '\n'.join(lines[:index])
    capacity = budget - len(opening) - 80  # room for the part header
    groups: list[list[str]] = [[]]
    used = 0
    for line in lines[index:]:
        if groups[-1] and used + len(line) + 1 > capacity:
            groups.append([])
            used = 0
        groups[-1].append(line)
        used += len(line) + 1
    return [
        '\n'.join([opening, f'[part {number} of {len(groups)} of the tool calls, in order]', *group]).lstrip('\n')
        for number, group in enumerate(groups, start=1)
    ]


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
        half = SHELL_OUTPUT_CHARS // 2
        excerpt = _CREDENTIAL.sub(
            lambda match: (match.group(1) or '') + '[redacted]', _head_tail(text.strip(), half, half)
        )
        evidence.append(f'output: {excerpt}'.replace('\n', ' ⏎ '))
    return f'CALL {name} [{status}]: {call_input}\n  → {"; ".join(evidence) or "result body omitted"}'


def _fit(lines: list[str], budget: int, *, keep_opening_users: bool) -> str:
    """Join `lines`, cutting from the middle when over `budget`; the opening user turns always survive."""
    text = '\n'.join(lines)
    if len(text) <= budget:
        return text
    opening: list[str] = []
    index = 0
    if keep_opening_users:
        # Everything up to the first agent line: the task as the user stated it.
        while index < len(lines) and lines[index].startswith('USER:'):
            opening.append(lines[index])
            index += 1
    head_budget = max(int(budget * OPENING_SHARE), sum(len(line) + 1 for line in opening))
    used = sum(len(line) + 1 for line in opening)
    while index < len(lines) and used + len(lines[index]) + 1 <= head_budget:
        opening.append(lines[index])
        used += len(lines[index]) + 1
        index += 1
    closing: list[str] = []
    remaining = budget - used - 60
    for line in reversed(lines[index:]):
        if len(line) + 1 > remaining:
            break
        closing.append(line)
        remaining -= len(line) + 1
    closing.reverse()
    omitted = len(lines) - len(opening) - len(closing)
    return '\n'.join([*opening, f'[... {omitted} lines omitted ...]', *closing])


def _call_line(call: dict[str, Any]) -> str:
    name, arguments = _call_name(call), _call_arguments(call)
    if name in SHELL_TOOLS:
        command = _shell_command(arguments)
        if is_risky_command(command):
            return f'{name}: `{command[:RISKY_COMMAND_CHARS].replace(chr(10), " ⏎ ")}`'
        return f'{name}: {first_command(command)}'
    if name == SKILL_TOOL:
        skills = _SKILL_ARG.findall(_raw_arguments(call))
        return f'{name}: {skills[0] if skills else "?"}'
    for key in ('file_path', 'path', 'url', 'query', 'description', 'pattern'):
        value = arguments.get(key)
        if isinstance(value, str):
            return f'{name}: {value[:80]}'
    return name


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
    return _REMINDER.sub('', content_to_text(message.get('content'))).strip()


def _head_tail(text: str, head: int, tail: int) -> str:
    if len(text) <= head + tail + 40:
        return text
    return f'{text[:head]} [... {len(text) - head - tail} chars ...] {text[-tail:]}'
