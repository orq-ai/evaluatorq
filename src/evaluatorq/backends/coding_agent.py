"""Run a local coding-agent CLI (Claude Code, Codex CLI, OpenCode) as an ``AgentTarget``.

Each ``respond()`` renders the whole transcript into one prompt, runs a fresh agent process in a
per-clone temporary working directory, parses the agent's JSONL stdout into an ``AgentResponse``
and maps failures to ``cli.*`` error codes via ``map_error``. There is no session resume.

Two launchers: ``direct`` runs the binary with the caller's environment and credentials; ``orq``
runs ``orq launch <agent>`` so every model call goes through the Orq gateway with the workspace's
skills and MCP server attached.

Retry lives in ``common.target_call.call_target_with_retry`` only. ``respond()`` never retries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from evaluatorq.common.sanitize import delimit
from evaluatorq.common.target_call import NonRetryableTargetError
from evaluatorq.contracts import (
    AgentResponse,
    AgentTarget,
    Message,
    ToolCallOutputItem,
    Usage,
    content_to_text,
    tool_result_to_text,
)
from evaluatorq.openresponses.convert_models import FunctionCallStatus

if TYPE_CHECKING:
    from collections.abc import Callable

AgentName = Literal['claude', 'codex', 'opencode']
Launcher = Literal['direct', 'orq']

_STDERR_EXCERPT_CHARS = 4000


class CodingAgentError(Exception):
    """A coding-agent turn failed. ``code`` is one of the ``cli.*`` codes ``map_error`` reports.

    Codes: ``cli.not_found``, ``cli.timeout`` (both non-retryable, see
    `CodingAgentUnavailableError`), ``cli.exit.<code>``, ``cli.no_result``, ``cli.parse_error``,
    ``cli.agent_error``.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f'{code}: {message}')
        self.code = code
        self.message = message


class CodingAgentUnavailableError(CodingAgentError, NonRetryableTargetError):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """``cli.not_found`` and ``cli.timeout``: a retry replays the same outcome, so the loop stops."""


@dataclass(frozen=True)
class OrqLaunchOptions:
    """Flags for ``orq launch``; honoured only under ``launcher='orq'``.

    ``mcp=False`` renders ``--no-mcp``, ``skills=False`` renders ``--no-skills``, ``base_url``
    renders ``--base-url <url>``, ``fetch_models=False`` renders ``--no-fetch-models``. Profile and
    workspace selection are not launch flags: set ``ORQ_PROFILE`` or ``ORQ_API_KEY`` through the
    target's ``env``.
    """

    mcp: bool = True
    skills: bool = True
    base_url: str | None = None
    fetch_models: bool = True

    def to_flags(self) -> list[str]:
        flags: list[str] = []
        if not self.mcp:
            flags.append('--no-mcp')
        if not self.skills:
            flags.append('--no-skills')
        if self.base_url:
            flags += ['--base-url', self.base_url]
        if not self.fetch_models:
            flags.append('--no-fetch-models')
        return flags


@dataclass(frozen=True)
class _AgentSpec:
    binary: str
    output_args: tuple[str, ...]
    model_flag: str
    permission_flag: str | None
    system_prompt_flag: str | None
    skills_dir: str
    tools: tuple[str, ...]
    # Argv that makes the binary read the prompt from stdin under ``direct``. Empty means the
    # binary reads stdin when no positional prompt is given (claude ``-p``, opencode ``run``).
    stdin_marker: tuple[str, ...] = ()


_AGENTS: dict[str, _AgentSpec] = {
    'claude': _AgentSpec(
        binary='claude',
        output_args=('-p', '--output-format', 'stream-json', '--verbose'),
        model_flag='--model',
        permission_flag='--permission-mode',
        system_prompt_flag='--append-system-prompt',
        skills_dir='.claude/skills',
        tools=('Bash', 'Read', 'Edit', 'Write', 'Glob', 'Grep', 'WebFetch'),
    ),
    'codex': _AgentSpec(
        binary='codex',
        output_args=('exec', '--json', '--skip-git-repo-check'),
        model_flag='-m',
        permission_flag='--sandbox',
        system_prompt_flag=None,
        skills_dir='.agents/skills',
        tools=('shell', 'apply_patch'),
        stdin_marker=('-',),
    ),
    'opencode': _AgentSpec(
        binary='opencode',
        output_args=('run', '--format', 'json'),
        model_flag='--model',
        permission_flag=None,
        system_prompt_flag=None,
        skills_dir='.agents/skills',
        tools=('bash', 'read', 'edit', 'write', 'glob', 'grep'),
    ),
}


def build_argv(
    *,
    agent: AgentName,
    launcher: Launcher,
    model: str | None,
    permission_mode: str | None,
    system_prompt: str | None,
    extra_args: list[str] | None,
    orq: OrqLaunchOptions | None,
    prompt: str,
) -> tuple[list[str], str | None]:
    """Return ``(argv, stdin_text)``. ``stdin_text`` is ``None`` when the child's stdin must be closed.

    Under ``direct`` the prompt travels on stdin (claude and opencode read it there; codex needs the
    ``-`` positional), so a long transcript never hits ``ARG_MAX``. Under ``orq`` the codex and
    opencode prompts go through ``orq launch -p`` and stdin is closed, because codex appends any
    open stdin to the prompt. Claude under ``orq`` still reads stdin.
    """
    spec = _AGENTS[agent]
    if permission_mode is not None and spec.permission_flag is None:
        raise ValueError(f'{agent} has no permission-mode flag; pass its own flags through extra_args instead')

    agent_args: list[str] = list(spec.output_args)
    if launcher == 'direct' and model:
        agent_args += [spec.model_flag, model]
    if permission_mode is not None and spec.permission_flag is not None:
        agent_args += [spec.permission_flag, permission_mode]
    if system_prompt and spec.system_prompt_flag:
        agent_args += [spec.system_prompt_flag, system_prompt]
    agent_args += list(extra_args or [])

    if launcher == 'direct':
        return [spec.binary, *agent_args, *spec.stdin_marker], prompt

    orq_flags = ['--model', model] if model else []
    orq_flags += (orq or OrqLaunchOptions()).to_flags()
    if agent == 'claude':
        return ['orq', 'launch', 'claude', *orq_flags, '--', *agent_args], prompt
    # ``orq launch`` already invokes the agent's subcommand (``codex exec`` / ``opencode run``), so
    # the leading entry of ``output_args`` is dropped here to avoid passing it twice.
    return ['orq', 'launch', agent, *orq_flags, '-p', prompt, '--', *agent_args[1:]], None


_PROMPT_INSTRUCTION = (
    'You are continuing the conversation below. It is a JSON array of chat messages in order; '
    '"tool" entries are the results of your own earlier tool calls. Reply to the last "user" message. '
    'Do not restate the transcript.'
)


def _message_to_dict(message: Message) -> dict[str, Any]:
    if message.role == 'tool':
        return {
            'role': 'tool',
            'tool_call_id': message.tool_call_id or '',
            'name': message.name,
            'content': content_to_text(message.content),
        }
    out: dict[str, Any] = {
        'role': message.role,
        'content': None if message.content is None else content_to_text(message.content),
    }
    if message.tool_calls:
        out['tool_calls'] = [
            {'id': tc.id, 'name': tc.function.name, 'arguments': tc.function.arguments} for tc in message.tool_calls
        ]
    return out


def render_prompt(messages: list[Message], *, system_prompt: str | None, inline_system: bool) -> str:
    """Render the transcript as a delimited JSON conversation followed by the reply instruction.

    ``inline_system`` prepends ``system_prompt`` as a ``system`` entry for agents without a
    system-prompt flag (codex, opencode). Claude receives it via ``--append-system-prompt`` instead.
    """
    entries: list[dict[str, Any]] = []
    if inline_system and system_prompt:
        entries.append({'role': 'system', 'content': system_prompt})
    entries += [_message_to_dict(m) for m in messages]
    block = delimit(json.dumps(entries, ensure_ascii=False, indent=None), tag='conversation')
    return f'{_PROMPT_INSTRUCTION}\n{block}'


@dataclass
class ParsedTurn:
    """What one agent run said, before the exit-code and error-order checks in ``respond()``."""

    text: str | None = None
    tool_calls: list[ToolCallOutputItem] = field(default_factory=list)
    usage: Usage | None = None
    session_id: str | None = None
    model: str | None = None
    cost_usd: float | None = None
    agent_error: str | None = None


def _tool_call(
    *,
    call_id: str,
    name: str,
    arguments: Any,
    result: str | None,
    status: Literal['in_progress', 'completed', 'incomplete'],
) -> ToolCallOutputItem:
    return ToolCallOutputItem(
        id=call_id,
        call_id=call_id,
        name=name,
        arguments=arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False),
        status=FunctionCallStatus(status),
        result=result,
    )


def _parse_claude(events: list[dict[str, Any]]) -> ParsedTurn:
    turn = ParsedTurn()
    calls: dict[str, ToolCallOutputItem] = {}
    order: list[str] = []
    last_text: list[str] = []
    for event in events:
        kind = event.get('type')
        if kind == 'assistant':
            last_text = []
            for block in event.get('message', {}).get('content', []) or []:
                if block.get('type') == 'tool_use':
                    call_id = str(block.get('id'))
                    calls[call_id] = _tool_call(
                        call_id=call_id,
                        name=str(block.get('name')),
                        arguments=block.get('input', {}),
                        result=None,
                        status='in_progress',
                    )
                    order.append(call_id)
                elif block.get('type') == 'text':
                    last_text.append(str(block.get('text', '')))
        elif kind == 'user':
            for block in event.get('message', {}).get('content', []) or []:
                if isinstance(block, dict) and block.get('type') == 'tool_result':
                    call_id = str(block.get('tool_use_id'))
                    if call_id in calls:
                        status: Literal['completed', 'incomplete'] = (
                            'incomplete' if block.get('is_error') else 'completed'
                        )
                        calls[call_id] = calls[call_id].model_copy(
                            update={
                                'result': tool_result_to_text(block.get('content')),
                                'status': FunctionCallStatus(status),
                            }
                        )
        elif kind == 'result':
            turn.session_id = event.get('session_id')
            turn.cost_usd = event.get('total_cost_usd')
            result_text = event.get('result')
            turn.text = result_text if isinstance(result_text, str) else (''.join(last_text) or None)
            if event.get('is_error'):
                turn.agent_error = result_text if isinstance(result_text, str) else str(event.get('subtype'))
            usage_block = event.get('usage')
            turn.usage = Usage.extract(usage_block, calls=1) if usage_block else None
            model_usage = event.get('modelUsage') or {}
            turn.model = next(iter(model_usage), None)
            for denial in event.get('permission_denials') or []:
                call_id = str(denial.get('tool_use_id') or f'denied-{len(order)}')
                denied = _tool_call(
                    call_id=call_id,
                    name=str(denial.get('tool_name')),
                    arguments=denial.get('tool_input', {}),
                    result='[denied by claude]',
                    status='incomplete',
                )
                if call_id not in calls:
                    order.append(call_id)
                calls[call_id] = denied
        else:
            logger.warning(f'claude skipped unknown event type: {kind}')
    if turn.text is None and last_text:
        turn.text = ''.join(last_text)
    turn.tool_calls = [calls[c] for c in order]
    return turn


def _codex_status(item: dict[str, Any]) -> Literal['in_progress', 'completed', 'incomplete']:
    status = item.get('status')
    if status == 'completed' and (item.get('exit_code') in (None, 0)):
        return 'completed'
    if status in ('in_progress', None):
        return 'in_progress'
    return 'incomplete'


def _parse_codex(events: list[dict[str, Any]]) -> ParsedTurn:
    turn = ParsedTurn()
    calls: dict[str, ToolCallOutputItem] = {}
    order: list[str] = []
    for event in events:
        kind = event.get('type')
        if kind == 'thread.started':
            turn.session_id = event.get('thread_id')
        elif kind == 'turn.failed':
            turn.agent_error = str((event.get('error') or {}).get('message') or 'turn.failed')
        elif kind == 'turn.completed':
            usage = event.get('usage')
            if usage is not None:
                missing_fields = [field for field in ('input_tokens', 'output_tokens') if field not in usage]
                if missing_fields:
                    for field in missing_fields:
                        logger.warning(f'codex turn.completed usage missing required field: {field}')
                else:
                    turn.usage = Usage(
                        input_tokens=int(usage['input_tokens']),
                        output_tokens=int(usage['output_tokens']),
                        total_tokens=int(usage['input_tokens']) + int(usage['output_tokens']),
                        cached_tokens=int(usage.get('cached_input_tokens', 0)),
                        cache_creation_tokens=int(usage.get('cache_write_input_tokens', 0)),
                        reasoning_tokens=int(usage.get('reasoning_output_tokens', 0)),
                        calls=1,
                    )
        elif kind in ('item.started', 'item.completed'):
            item = event.get('item') or {}
            item_id = str(item.get('id'))
            item_type = item.get('type')
            if item_type == 'agent_message':
                turn.text = str(item.get('text', ''))
            elif item_type == 'error':
                logger.warning(f'codex reported: {item.get("message")}')
            elif item_type == 'command_execution':
                calls[item_id] = _tool_call(
                    call_id=item_id,
                    name='shell',
                    arguments={'command': item.get('command')},
                    result=item.get('aggregated_output') if kind == 'item.completed' else None,
                    status=_codex_status(item),
                )
            elif item_type == 'file_change':
                calls[item_id] = _tool_call(
                    call_id=item_id,
                    name='apply_patch',
                    arguments={'changes': item.get('changes', [])},
                    result='applied' if item.get('status') == 'completed' else None,
                    status=_codex_status(item),
                )
            elif item_type == 'mcp_tool_call':
                error = item.get('error') or {}
                calls[item_id] = _tool_call(
                    call_id=item_id,
                    name=f'{item.get("server")}.{item.get("tool")}',
                    arguments=item.get('arguments', {}),
                    result=error.get('message') if error else tool_result_to_text(item.get('result')),
                    status='incomplete' if error else _codex_status(item),
                )
            else:
                logger.warning(f'codex skipped unknown item type: {item_type}')
                continue
            if item_id in calls and item_id not in order:
                order.append(item_id)
        else:
            logger.warning(f'codex skipped unknown event type: {kind}')
    turn.tool_calls = [calls[c] for c in order]
    return turn


def _parse_opencode(events: list[dict[str, Any]]) -> ParsedTurn:
    turn = ParsedTurn()
    texts: list[str] = []
    totals = {'input': 0, 'output': 0, 'reasoning': 0, 'read': 0, 'write': 0}
    cost = 0.0
    saw_valid_usage = False
    saw_step_finish = False
    saw_cost = False
    for event in events:
        turn.session_id = turn.session_id or event.get('sessionID')
        part = event.get('part') or {}
        kind = event.get('type')
        if kind == 'text':
            texts.append(str(part.get('text', '')))
        elif kind == 'tool_use':
            state = part.get('state') or {}
            status_raw = state.get('status')
            status: Literal['in_progress', 'completed', 'incomplete'] = (
                'completed' if status_raw == 'completed' else 'incomplete' if status_raw == 'error' else 'in_progress'
            )
            call_id = str(part.get('callID') or part.get('id'))
            turn.tool_calls.append(
                _tool_call(
                    call_id=call_id,
                    name=str(part.get('tool')),
                    arguments=state.get('input', {}),
                    result=tool_result_to_text(state.get('output'))
                    if state.get('output') is not None
                    else tool_result_to_text(state.get('error'))
                    if 'error' in state
                    else None,
                    status=status,
                )
            )
        elif kind == 'step_finish':
            saw_step_finish = True
            tokens = part.get('tokens') or {}
            if 'input' in tokens and 'output' in tokens:
                saw_valid_usage = True
                totals['input'] += int(tokens['input'])
                totals['output'] += int(tokens['output'])
                totals['reasoning'] += int(tokens.get('reasoning', 0))
                cache = tokens.get('cache') or {}
                totals['read'] += int(cache.get('read', 0))
                totals['write'] += int(cache.get('write', 0))
            if 'cost' in part:
                saw_cost = True
                cost += float(part['cost'] or 0)
        elif kind == 'error':
            turn.agent_error = str((part.get('error') or event.get('error') or {}).get('message') or 'error')
        else:
            logger.warning(f'opencode skipped unknown event type: {kind}')
    if texts:
        turn.text = '\n'.join(texts)
    if saw_valid_usage:
        turn.usage = Usage(
            input_tokens=totals['input'],
            output_tokens=totals['output'],
            total_tokens=totals['input'] + totals['output'],
            cached_tokens=totals['read'],
            cache_creation_tokens=totals['write'],
            reasoning_tokens=totals['reasoning'],
            calls=1,
        )
    elif saw_step_finish:
        logger.warning('opencode step_finish usage missing input and output token fields; usage is unknown')
    if saw_cost:
        turn.cost_usd = cost
    return turn


_PARSERS: dict[str, Callable[[list[dict[str, Any]]], ParsedTurn]] = {
    'claude': _parse_claude,
    'codex': _parse_codex,
    'opencode': _parse_opencode,
}


def parse_events(agent: AgentName, events: list[dict[str, Any]]) -> ParsedTurn:
    """Pure: one agent's JSONL events to a `ParsedTurn`. Unknown event types are skipped."""
    return _PARSERS[agent](events)


class CodingAgentTarget(AgentTarget):  # placeholder, replaced in Task 6
    async def respond(self, messages: list[Message]) -> AgentResponse:
        raise NotImplementedError

    def new(self) -> CodingAgentTarget:
        return type(self)()
