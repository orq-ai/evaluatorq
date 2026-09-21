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

import asyncio
import contextlib
import json
import os
import shutil
import signal
import tempfile
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from evaluatorq.common.sanitize import delimit
from evaluatorq.common.target_call import NonRetryableTargetError
from evaluatorq.common.tracing import record_token_usage, set_span_attrs, with_llm_span
from evaluatorq.contracts import (
    DEFAULT_TARGET_TIMEOUT_MS,
    AgentContext,
    AgentResponse,
    AgentTarget,
    Message,
    TextOutputItem,
    ToolCallOutputItem,
    ToolInfo,
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


def _remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


class CodingAgentTarget(AgentTarget):
    """Run a local coding-agent CLI as the system under test.

    Each ``respond()`` spawns a fresh process (no session resume), in a working directory that is
    private to this instance: an empty temp dir, or a copy of ``workdir``. ``new()`` clones carry
    the same configuration and get their own copy, so concurrent jobs never share a tree. ``close()``
    kills a live process group and removes the temp dir unless ``keep_workdir`` is set; the runner
    calls it through ``common.target_call.close_target``. A ``weakref.finalize`` is the leak backstop.

    Privilege is expressed in each agent's own vocabulary and defaults to the agent's own default:

    - ``permission_mode``: claude ``--permission-mode`` (``default``, ``acceptEdits``, ``plan``,
      ``bypassPermissions``, ``dontAsk``); codex ``--sandbox`` (``read-only``, ``workspace-write``,
      ``danger-full-access``); opencode has no such flag and raises ``ValueError``.
    - ``extra_args``: appended to the agent argv untouched. OpenCode blocks on its first permission
      prompt when run headless; pass ``extra_args=['--auto']`` to auto-approve.

    ``system_prompt`` is claude's ``--append-system-prompt``; codex and opencode have no equivalent, so
    it is prepended to the rendered conversation as a ``system`` entry and a warning says so.

    ``launcher='orq'`` runs ``orq launch <agent>`` so model calls route through the Orq gateway with
    the workspace skills and MCP server attached; ``orq`` tunes its flags and is ignored (with a warning)
    under ``direct``. ``model`` is the agent's own flag under ``direct`` and ``orq launch --model
    provider/id`` under ``orq``.

    ``env`` overlays ``os.environ``; caller-supplied values win. ``timeout_ms`` is the per-turn ceiling
    and defaults to the same value the outer retry helper uses.
    """

    def __init__(
        self,
        agent: AgentName,
        *,
        launcher: Launcher = 'direct',
        orq: OrqLaunchOptions | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        permission_mode: str | None = None,
        extra_args: list[str] | None = None,
        workdir: Path | None = None,
        keep_workdir: bool = False,
        skills: list[Path] | None = None,
        timeout_ms: int = DEFAULT_TARGET_TIMEOUT_MS,
        env: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        if agent not in _AGENTS:
            raise ValueError(f'Unknown coding agent {agent!r}; expected one of {sorted(_AGENTS)}')
        if launcher not in ('direct', 'orq'):
            raise ValueError(f'Unknown launcher {launcher!r}; expected "direct" or "orq"')
        self._spec = _AGENTS[agent]
        if permission_mode is not None and self._spec.permission_flag is None:
            raise ValueError(f'{agent} has no permission-mode flag; pass its own flags through extra_args instead')
        if orq is not None and launcher == 'direct':
            logger.warning(f'CodingAgentTarget({agent}): orq options given but launcher is "direct"; ignoring them')
        if system_prompt and self._spec.system_prompt_flag is None:
            logger.warning(
                f'CodingAgentTarget({agent}): no system-prompt flag; prepending system_prompt to the conversation instead'
            )
        self._kwargs: dict[str, Any] = {
            'launcher': launcher,
            'orq': orq,
            'model': model,
            'system_prompt': system_prompt,
            'permission_mode': permission_mode,
            'extra_args': list(extra_args) if extra_args else None,
            'workdir': workdir,
            'keep_workdir': keep_workdir,
            'skills': list(skills) if skills else None,
            'timeout_ms': timeout_ms,
            'env': dict(env) if env else None,
        }
        self._agent: AgentName = agent
        self._launcher: Launcher = launcher
        self._orq = orq
        self._model = model
        self._system_prompt = system_prompt
        self._permission_mode = permission_mode
        self._extra_args = list(extra_args or [])
        self._source_workdir = Path(workdir) if workdir is not None else None
        self._keep_workdir = keep_workdir
        self._skills = [Path(s) for s in skills or []]
        self._timeout_ms = timeout_ms
        self._env = dict(env or {})
        self._workdir: Path | None = None
        self._finalizer: weakref.finalize | None = None  # pyright: ignore[reportMissingTypeArgument]
        self._proc: asyncio.subprocess.Process | None = None

    @property
    def workdir(self) -> Path | None:
        """The private working directory, ``None`` until the first turn creates it."""
        return self._workdir

    def _ensure_workdir(self) -> Path:
        if self._workdir is not None:
            return self._workdir
        dst = Path(tempfile.mkdtemp(prefix=f'evaluatorq-{self._agent}-'))
        if self._source_workdir is not None:
            shutil.copytree(self._source_workdir, dst, symlinks=True, dirs_exist_ok=True)
        skills_dir = dst / self._spec.skills_dir
        skills_dir.mkdir(parents=True, exist_ok=True)
        for skill in self._skills:
            link = skills_dir / skill.name
            if link.exists() or link.is_symlink():
                raise FileExistsError(f'skill {skill.name!r} already exists in {skills_dir}; refusing to shadow it')
            link.symlink_to(skill.resolve(), target_is_directory=True)
        if not self._keep_workdir:
            self._finalizer = weakref.finalize(self, _remove_tree, dst)
        self._workdir = dst
        return dst

    def new(self) -> CodingAgentTarget:
        return type(self)(self._agent, **self._kwargs)

    async def close(self) -> None:
        """Kill a live process group and release the temp workdir. Idempotent."""
        proc = self._proc
        if proc is not None:
            _kill_group(proc)
            self._proc = None
        if self._workdir is None:
            return
        if self._keep_workdir:
            logger.info(f'CodingAgentTarget({self._agent}): keeping workdir {self._workdir}')
        else:
            _remove_tree(self._workdir)
            if self._finalizer is not None:
                self._finalizer.detach()
        self._workdir = None

    async def get_agent_context(self) -> AgentContext:
        tools = [ToolInfo(name=name) for name in self._spec.tools]
        tools += [ToolInfo(name=skill.name, action_type='skill') for skill in self._skills]
        return AgentContext(
            key=f'coding-agent:{self._agent}',
            system_prompt=self._system_prompt or '',
            tools=tools,
            model=self._model,
        )

    def map_error(self, exc: Exception) -> tuple[str, str] | None:
        if isinstance(exc, CodingAgentError):
            return exc.code, exc.message
        return None

    async def respond(self, messages: list[Message]) -> AgentResponse:
        workdir = self._ensure_workdir()
        prompt = render_prompt(
            messages, system_prompt=self._system_prompt, inline_system=self._spec.system_prompt_flag is None
        )
        argv, stdin_text = build_argv(
            agent=self._agent,
            launcher=self._launcher,
            model=self._model,
            permission_mode=self._permission_mode,
            system_prompt=self._system_prompt,
            extra_args=self._extra_args,
            orq=self._orq,
            prompt=prompt,
        )
        async with with_llm_span(
            model=self._model or self._agent,
            operation='chat',
            input_messages=[m.to_chat_completion() for m in messages],
            attributes={
                'orq.redteam.llm_purpose': 'target',
                'evaluatorq.coding_agent.agent': self._agent,
                'evaluatorq.coding_agent.launcher': self._launcher,
            },
        ) as span:
            returncode, stdout, stderr = await self._run(argv, stdin_text, cwd=workdir)
            set_span_attrs(span, {'evaluatorq.coding_agent.exit_code': returncode})
            stderr_excerpt = stderr[-_STDERR_EXCERPT_CHARS:]
            if returncode != 0:
                raise CodingAgentError(f'cli.exit.{returncode}', f'{argv[0]} exited {returncode}: {stderr_excerpt}')
            events = _parse_jsonl(stdout)
            if stdout.strip() and not events:
                raise CodingAgentError('cli.parse_error', f'no JSON events in stdout: {stdout[:_STDERR_EXCERPT_CHARS]}')
            try:
                turn = parse_events(self._agent, events)
            except Exception as exc:  # Any parser crash is a parse error, not a target crash.
                raise CodingAgentError('cli.parse_error', f'could not parse {self._agent} output: {exc!r}') from exc
            if turn.agent_error is not None:
                raise CodingAgentError('cli.agent_error', f'{turn.agent_error} {stderr_excerpt}'.strip())
            if turn.text is None or not turn.text.strip():
                raise CodingAgentError('cli.no_result', f'exit 0 but no final assistant message: {stderr_excerpt}')
            if turn.usage is None:
                logger.warning(f'CodingAgentTarget({self._agent}): no usage in output; usage is None')
            else:
                record_token_usage(span, usage=turn.usage, total_cost=turn.cost_usd)
            set_span_attrs(
                span, {'evaluatorq.coding_agent.cost_usd': turn.cost_usd, 'gen_ai.response.id': turn.session_id}
            )
            return AgentResponse(
                output=[*turn.tool_calls, TextOutputItem(text=turn.text, annotations=[])],
                usage=turn.usage,
                model=turn.model or self._model,
                response_id=turn.session_id,
            )

    async def _run(self, argv: list[str], stdin_text: str | None, *, cwd: Path) -> tuple[int, str, str]:
        """Run one agent process in its own process group; kill the group on timeout, error or cancel."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env={**os.environ, **self._env},
                stdin=asyncio.subprocess.PIPE if stdin_text is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise CodingAgentUnavailableError('cli.not_found', f'{argv[0]!r} not found on PATH') from exc
        self._proc = proc
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(stdin_text.encode() if stdin_text is not None else None),
                timeout=self._timeout_ms / 1000,
            )
        except asyncio.TimeoutError as exc:
            raise CodingAgentUnavailableError(
                'cli.timeout', f'{argv[0]} produced no result within {self._timeout_ms / 1000:.0f}s'
            ) from exc
        finally:
            _kill_group(proc)
            self._proc = None
        return proc.returncode or 0, stdout.decode(errors='replace'), stderr.decode(errors='replace')


def _parse_jsonl(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(f'CodingAgentTarget: skipping non-JSON stdout line: {line[:200]}')
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
