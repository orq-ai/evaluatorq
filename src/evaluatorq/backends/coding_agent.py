"""Run a local coding-agent CLI (Claude Code, Codex CLI, OpenCode) as an ``AgentTarget``.

Each ``respond()`` renders the whole transcript into one prompt, runs a fresh agent process in a
per-clone temporary working directory, parses the agent's JSONL stdout into an ``AgentResponse``
and maps failures to ``cli.*`` error codes via ``map_error``. There is no session resume.

Two launchers: ``direct`` runs the binary with the caller's environment and credentials; ``orq``
runs ``orq launch <agent>`` so every model call goes through the Orq gateway with the workspace's
skills and MCP server attached.

Retry lives in ``common.target_call.call_target_with_retry`` only. ``respond()`` never retries.

With container mode, one long-lived container belongs to each target clone and each turn runs through
``docker exec``. ``close()``, cancellation, and early turn termination remove that container.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
import types
import uuid
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from evaluatorq.backends.container import (
    RESERVED_ENV,
    DockerOptions,
    LiveContainer,
    build_exec_argv,
    build_run_argv,
    forwarded_env_names,
    host_ids,
    isolation_breaking_flags,
    register,
    release_containers,
    remove_containers_async,
    sweep_orphans,
    unregister,
    unsafe_mounts,
    write_beat,
)
from evaluatorq.common.cli_tty import shell_join
from evaluatorq.common.sanitize import delimit
from evaluatorq.common.target_call import NonRetryableTargetError
from evaluatorq.common.tracing import record_token_usage, set_span_attrs, with_llm_span
from evaluatorq.contracts import (
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
    from collections.abc import Awaitable, Callable

AgentName = Literal['claude', 'codex', 'opencode']
Launcher = Literal['direct', 'orq']

# Idle limit: the turn ends when the agent writes nothing for this long. Coding agents opt out of the
# retry helper's wall clock (manages_own_timeout), so this is not tied to DEFAULT_TARGET_TIMEOUT_MS.
DEFAULT_CODING_AGENT_TIMEOUT_MS = 300_000
# Hard cap: wall clock of one turn, however busy the agent is.
DEFAULT_CODING_AGENT_MAX_TURN_MS = 7_200_000
STDERR_EXCERPT_CHARS = 4000
READ_CHUNK_BYTES = 65_536


class CodingAgentError(Exception):
    """A coding-agent turn failed. ``code`` is one of the ``cli.*`` codes ``map_error`` reports.

    Codes: ``cli.not_found``, ``cli.timeout`` (idle limit and hard cap; non-retryable, see
    `CodingAgentUnavailableError`), ``cli.container_start``, ``cli.image_missing``,
    ``cli.prompt_too_long``, ``cli.agent_not_found``, ``cli.exit.<code>``, ``cli.no_result``,
    ``cli.parse_error``, and ``cli.agent_error``.
    """

    def __init__(self, code: str, message: str, *, kill_reason: str | None = None) -> None:
        super().__init__(f'{code}: {message}')
        self.code = code
        self.message = message
        self.kill_reason = kill_reason


class CodingAgentUnavailableError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    CodingAgentError, NonRetryableTargetError
):
    """Non-retryable codes: ``cli.not_found``, ``cli.timeout``, ``cli.prompt_too_long``,
    ``cli.agent_not_found``, ``cli.image_missing``, ``cli.container_start``, and ``cli.unsafe_shim``. Retrying these
    outcomes repeats the same failure, so the retry loop stops.
    """


class OrqLaunchOptions(BaseModel):
    """Flags for ``orq launch``; honoured only under ``launcher='orq'``.

    ``mcp=False`` renders ``--no-mcp``, ``skills=False`` renders ``--no-skills``, ``base_url``
    renders ``--base-url <url>``, ``fetch_models=False`` renders ``--no-fetch-models``. Profile and
    workspace selection are not launch flags: set ``ORQ_PROFILE`` or ``ORQ_API_KEY`` through the
    target's ``env``.
    """

    model_config = ConfigDict(frozen=True)

    mcp: bool = True
    skills: bool = True
    base_url: str | None = None
    fetch_models: bool = True

    def __init__(
        self,
        mcp: bool = True,  # noqa: FBT001, FBT002 - preserve the public dataclass constructor
        skills: bool = True,  # noqa: FBT001, FBT002 - preserve the public dataclass constructor
        base_url: str | None = None,
        fetch_models: bool = True,  # noqa: FBT001, FBT002 - preserve the public dataclass constructor
    ) -> None:
        """Keep the public dataclass's positional constructor while validating with Pydantic."""
        super().__init__(mcp=mcp, skills=skills, base_url=base_url, fetch_models=fetch_models)

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


class AgentSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    binary: str
    output_args: tuple[str, ...]
    model_flag: str
    permission_flag: str | None
    system_prompt_flag: str | None
    skills_dir: str
    tools: tuple[str, ...]
    container_permission: str | None
    container_extra_args: tuple[str, ...]
    provider_env: tuple[str, ...]
    # Empty means the binary reads stdin whenever no positional prompt is given.
    stdin_marker: tuple[str, ...] = ()


AGENTS: types.MappingProxyType[str, AgentSpec] = types.MappingProxyType({
    'claude': AgentSpec(
        binary='claude',
        output_args=('-p', '--output-format', 'stream-json', '--verbose'),
        model_flag='--model',
        permission_flag='--permission-mode',
        system_prompt_flag='--append-system-prompt',
        skills_dir='.claude/skills',
        tools=('Bash', 'Read', 'Edit', 'Write', 'Glob', 'Grep', 'WebFetch'),
        container_permission='bypassPermissions',
        container_extra_args=(),
        provider_env=('ANTHROPIC_API_KEY',),
    ),
    'codex': AgentSpec(
        binary='codex',
        output_args=('exec', '--json', '--skip-git-repo-check'),
        model_flag='-m',
        permission_flag='--sandbox',
        system_prompt_flag=None,
        skills_dir='.agents/skills',
        tools=('shell', 'apply_patch'),
        # Codex's own sandbox often cannot start inside a container; the container is the sandbox.
        container_permission='danger-full-access',
        container_extra_args=(),
        provider_env=('OPENAI_API_KEY',),
        stdin_marker=('-',),
    ),
    'opencode': AgentSpec(
        binary='opencode',
        output_args=('run', '--format', 'json'),
        model_flag='--model',
        permission_flag=None,
        system_prompt_flag=None,
        skills_dir='.agents/skills',
        tools=('bash', 'read', 'edit', 'write', 'glob', 'grep'),
        container_permission=None,
        container_extra_args=('--auto',),
        provider_env=('ANTHROPIC_API_KEY', 'OPENAI_API_KEY'),
    ),
})
if set(AGENTS) != set(get_args(AgentName)):
    raise RuntimeError(f'AGENTS {sorted(AGENTS)} does not match AgentName {sorted(get_args(AgentName))}')


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
    spec = AGENTS[agent]
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
    # orq launch injects the subcommand (``exec`` / ``run``) itself, so the leading entry is dropped.
    return ['orq', 'launch', agent, *orq_flags, '-p', prompt, '--', *agent_args[1:]], None


# cmd.exe re-parses a batch launcher's command line and CPython does not escape for it (BatBadBut).
CMD_METACHARACTERS = frozenset('"%&|<>^!\r\n')
BATCH_SUFFIXES = frozenset({'.bat', '.cmd'})


def is_batch_launcher(executable: str) -> bool:
    return Path(executable).suffix.lower() in BATCH_SUFFIXES


def refuse_unsafe_batch_args(argv: list[str]) -> None:
    if not is_batch_launcher(argv[0]):
        return
    for arg in argv[1:]:
        if found := sorted(CMD_METACHARACTERS & set(arg)):
            raise CodingAgentUnavailableError(
                'cli.unsafe_shim',
                f'{argv[0]} is a Windows batch-file launcher, and cmd.exe would interpret '
                f'{", ".join(repr(c) for c in found)} in the argument {arg!r}. Remove those characters from '
                '`extra_args` or `model`, or install the agent CLI as a native .exe so evaluatorq can call it directly.',
            )


PROMPT_INSTRUCTION = (
    'You are continuing the conversation below. It is a JSON array of chat messages in order; '
    '"tool" entries are the results of your own earlier tool calls. Reply to the last "user" message. '
    'Do not restate the transcript.'
)


def message_to_dict(message: Message) -> dict[str, Any]:
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
    system-prompt flag (codex, opencode), or for claude behind a Windows ``.cmd`` launcher; otherwise
    Claude receives it via ``--append-system-prompt``.
    """
    entries: list[dict[str, Any]] = []
    if inline_system and system_prompt:
        entries.append({'role': 'system', 'content': system_prompt})
    entries += [message_to_dict(m) for m in messages]
    block = delimit(json.dumps(entries, ensure_ascii=False, indent=None), tag='conversation')
    return f'{PROMPT_INSTRUCTION}\n{block}'


class ParsedTurn(BaseModel):
    """What one agent run said, before the exit-code and error-order checks in ``respond()``."""

    text: str | None = None
    tool_calls: list[ToolCallOutputItem] = Field(default_factory=list)
    usage: Usage | None = None
    session_id: str | None = None
    model: str | None = None
    cost_usd: float | None = None
    agent_error: str | None = None


def tool_call(
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


def usage_or_none(agent: str, usage_block: Any, required: tuple[str, str]) -> Usage | None:
    """``Usage.extract`` for one agent usage record; ``None`` plus a warning when a required count is absent."""
    if usage_block is None:
        return None
    missing_fields = [field for field in required if field not in usage_block]
    if missing_fields:
        logger.warning(f'{agent} usage missing required field(s): {", ".join(missing_fields)}; usage is None')
        return None
    return Usage.extract(usage_block, calls=1)


def opencode_usage(tokens: Any) -> dict[str, Any] | None:
    """Spell OpenCode's ``step_finish.tokens`` block in the canonical names ``Usage.extract`` reads.

    OpenCode uses bare ``input``/``output``/``reasoning`` and a nested ``cache`` object; those words are
    too generic to live in the shared alias table, so the rename stays here.
    """
    if not isinstance(tokens, dict):
        return None
    cache: dict[str, Any] = tokens['cache'] if isinstance(tokens.get('cache'), dict) else {}
    renamed: dict[str, Any] = {
        'input_tokens': tokens.get('input'),
        'output_tokens': tokens.get('output'),
        'cache_read_input_tokens': cache.get('read'),
        'cache_creation_input_tokens': cache.get('write'),
    }
    if tokens.get('reasoning') is not None:
        renamed['output_tokens_details'] = {'reasoning_tokens': tokens['reasoning']}
    # `total` is left out: OpenCode adds cache reads on top of input there, which breaks the
    # `total == input + output` invariant Usage keeps, so the extractor's own sum is used instead.
    return {k: v for k, v in renamed.items() if v is not None}


def heaviest_model(model_usage: dict[str, dict[str, Any]]) -> str | None:
    """The ``modelUsage`` entry with the most tokens; the first key is often an auxiliary model."""
    if not model_usage:
        return None
    return max(model_usage, key=lambda m: model_usage[m].get('inputTokens', 0) + model_usage[m].get('outputTokens', 0))


# Stream events claude emits every turn that carry no text, tool call or usage: `system` (init, hooks,
# status) and `rate_limit_event`. Skipped silently so the unknown-event warning stays meaningful.
CLAUDE_HOUSEKEEPING_EVENTS = frozenset({'system', 'rate_limit_event'})


def parse_claude(events: list[dict[str, Any]]) -> ParsedTurn:
    turn = ParsedTurn()
    calls: dict[str, ToolCallOutputItem] = {}
    order: list[str] = []
    last_text: list[str] = []
    content_events = [e for e in events if e.get('type') not in CLAUDE_HOUSEKEEPING_EVENTS]
    for event in content_events:
        kind = event.get('type')
        if kind == 'assistant':
            turn.model = event.get('message', {}).get('model') or turn.model
            last_text = []
            for block in event.get('message', {}).get('content', []) or []:
                if block.get('type') == 'tool_use':
                    call_id = str(block.get('id'))
                    calls[call_id] = tool_call(
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
            turn.usage = usage_or_none('claude', event.get('usage'), ('input_tokens', 'output_tokens'))
            turn.model = turn.model or heaviest_model(event.get('modelUsage') or {})
            for denial in event.get('permission_denials') or []:
                call_id = str(denial.get('tool_use_id') or f'denied-{len(order)}')
                denied = tool_call(
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


def codex_status(item: dict[str, Any]) -> Literal['in_progress', 'completed', 'incomplete']:
    status = item.get('status')
    if status == 'completed' and (item.get('exit_code') in (None, 0)):
        return 'completed'
    if status in ('in_progress', None):
        return 'in_progress'
    return 'incomplete'


def parse_codex(events: list[dict[str, Any]]) -> ParsedTurn:
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
            turn.usage = usage_or_none('codex', event.get('usage'), ('input_tokens', 'output_tokens'))
        elif kind in ('item.started', 'item.completed'):
            item = event.get('item') or {}
            item_id = str(item.get('id'))
            item_type = item.get('type')
            if item_type == 'agent_message':
                turn.text = str(item.get('text', ''))
            elif item_type == 'error':
                logger.warning(f'codex reported: {item.get("message")}')
            elif item_type == 'command_execution':
                calls[item_id] = tool_call(
                    call_id=item_id,
                    name='shell',
                    arguments={'command': item.get('command')},
                    result=item.get('aggregated_output') if kind == 'item.completed' else None,
                    status=codex_status(item),
                )
            elif item_type == 'file_change':
                calls[item_id] = tool_call(
                    call_id=item_id,
                    name='apply_patch',
                    arguments={'changes': item.get('changes', [])},
                    result='applied' if item.get('status') == 'completed' else None,
                    status=codex_status(item),
                )
            elif item_type == 'mcp_tool_call':
                error = item.get('error') or {}
                calls[item_id] = tool_call(
                    call_id=item_id,
                    name=f'{item.get("server")}.{item.get("tool")}',
                    arguments=item.get('arguments', {}),
                    result=error.get('message') if error else tool_result_to_text(item.get('result')),
                    status='incomplete' if error else codex_status(item),
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


def parse_opencode(events: list[dict[str, Any]]) -> ParsedTurn:
    turn = ParsedTurn()
    last_text: str | None = None
    cost = 0.0
    saw_step_finish = False
    saw_cost = False
    for event in events:
        turn.session_id = turn.session_id or event.get('sessionID')
        part = event.get('part') or {}
        kind = event.get('type')
        if kind == 'text':
            last_text = str(part.get('text', ''))
        elif kind == 'tool_use':
            state = part.get('state') or {}
            status_raw = state.get('status')
            status: Literal['in_progress', 'completed', 'incomplete'] = (
                'completed' if status_raw == 'completed' else 'incomplete' if status_raw == 'error' else 'in_progress'
            )
            call_id = str(part.get('callID') or part.get('id'))
            turn.tool_calls.append(
                tool_call(
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
            step_usage = usage_or_none(
                'opencode', opencode_usage(part.get('tokens')), ('input_tokens', 'output_tokens')
            )
            if step_usage is not None:
                turn.usage = step_usage if turn.usage is None else turn.usage + step_usage
            if 'cost' in part:
                saw_cost = True
                cost += float(part['cost'] or 0)
        elif kind == 'error':
            turn.agent_error = str((part.get('error') or event.get('error') or {}).get('message') or 'error')
        else:
            logger.warning(f'opencode skipped unknown event type: {kind}')
    if last_text is not None:
        turn.text = last_text
    if saw_step_finish and turn.usage is None:
        logger.warning('opencode step_finish carried no usable token counts; usage is unknown')
    if saw_cost:
        turn.cost_usd = cost
    return turn


PARSERS: types.MappingProxyType[str, Callable[[list[dict[str, Any]]], ParsedTurn]] = types.MappingProxyType({
    'claude': parse_claude,
    'codex': parse_codex,
    'opencode': parse_opencode,
})


def parse_events(agent: AgentName, events: list[dict[str, Any]]) -> ParsedTurn:
    """Pure: one agent's JSONL events to a `ParsedTurn`. Unknown event types are skipped."""
    return PARSERS[agent](events)


def remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def remove_tree_if_owner(pid: int, path: Path) -> None:
    """Finalizer callback that cannot remove a parent process's workdir after fork."""
    if os.getpid() == pid:
        remove_tree(path)


def release_containers_if_owner(pid: int, owned: list[str]) -> None:
    """Finalizer callback that cannot remove containers inherited across fork."""
    if os.getpid() == pid:
        release_containers(owned)


async def finish_task_uninterruptibly(task: asyncio.Future[Any]) -> None:
    """Wait for a shielded operation despite repeated caller cancellation, consuming its result."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:  # noqa: PERF203 — the caller may cancel this wait more than once.
            continue
        except (OSError, subprocess.TimeoutExpired):
            break
    with contextlib.suppress(BaseException):
        task.result()


async def await_shielded(awaitable: Awaitable[Any]) -> Any:
    """Await an operation through cancellation, including repeated cancellation requests."""
    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await finish_task_uninterruptibly(task)
        raise


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

    ``env`` overlays ``os.environ``; caller-supplied values win. ``timeout_ms`` is the idle limit: the
    maximum time without agent output. ``max_turn_ms`` is the wall-clock hard cap. Both raise the
    non-retryable ``cli.timeout`` error. Coding agents enforce these limits themselves, so the runner's
    ``target_agent_timeout_ms`` does not apply.
    """

    # The idle and hard limits below fully bound each respond call.
    manages_own_timeout = True

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
        timeout_ms: int = DEFAULT_CODING_AGENT_TIMEOUT_MS,
        max_turn_ms: int = DEFAULT_CODING_AGENT_MAX_TURN_MS,
        env: dict[str, str] | None = None,
        container: DockerOptions | None = None,
    ) -> None:
        super().__init__()
        if agent not in AGENTS:
            raise ValueError(f'Unknown coding agent {agent!r}; expected one of {sorted(AGENTS)}')
        if launcher not in ('direct', 'orq'):
            raise ValueError(f'Unknown launcher {launcher!r}; expected "direct" or "orq"')
        if timeout_ms <= 0 or max_turn_ms <= 0:
            raise ValueError('timeout_ms and max_turn_ms must be positive')
        self._spec = AGENTS[agent]
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
            'max_turn_ms': max_turn_ms,
            'env': dict(env) if env else None,
            'container': container,
        }
        self._container = container
        if container is not None:
            if container.allow_privilege_escalation:
                logger.warning(
                    f'CodingAgentTarget({agent}): allow_privilege_escalation=True drops no-new-privileges; '
                    'the writable /etc/passwd lets the agent become root inside the container'
                )
            if permission_mode is None:
                permission_mode = self._spec.container_permission
            extra_args = [
                *(extra_args or []),
                *(a for a in self._spec.container_extra_args if a not in (extra_args or [])),
            ]
            for flag in isolation_breaking_flags(container.run_args):
                logger.warning(f'CodingAgentTarget({agent}): run_args {flag!r} undoes the container isolation')
            dropped = [key for key in RESERVED_ENV if key in (env or {})]
            if dropped:
                logger.warning(
                    f'CodingAgentTarget({agent}): env {", ".join(dropped)} not forwarded into the container; '
                    'it applies to the docker client only'
                )
        if permission_mode is not None and self._spec.permission_flag is None:
            raise ValueError(f'{agent} has no permission-mode flag; pass its own flags through extra_args instead')
        if orq is not None and launcher == 'direct':
            logger.warning(f'CodingAgentTarget({agent}): orq options given but launcher is "direct"; ignoring them')
        if system_prompt and self._spec.system_prompt_flag is None:
            logger.warning(
                f'CodingAgentTarget({agent}): no system-prompt flag; '
                'prepending system_prompt to the conversation instead'
            )
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
        self.timeout_ms = timeout_ms
        self.max_turn_ms = max_turn_ms
        self._env = dict(env or {})
        self._workdir: Path | None = None
        self._forward_env = (
            forwarded_env_names(
                container, launcher=launcher, provider_env=self._spec.provider_env, caller_env=self._env
            )
            if container
            else []
        )
        self._root: Path | None = None
        self._creator_pid = os.getpid()
        self._container_name: str | None = None
        self._restarts = 0
        self._image_checked = False
        self._owned: list[str] = []
        self._container_finalizer: weakref.finalize[Any, Any] | None = (
            weakref.finalize(self, release_containers_if_owner, self._creator_pid, self._owned) if container else None
        )
        self._finalizer: weakref.finalize | None = None  # pyright: ignore[reportMissingTypeArgument]
        self._proc: asyncio.subprocess.Process | None = None

    @property
    def workdir(self) -> Path | None:
        """The private working directory, ``None`` until the first turn creates it."""
        return self._workdir

    async def _docker(self, argv: list[str], *, timeout_s: float = 60) -> subprocess.CompletedProcess[str]:
        """Run one bounded Docker control call with the host environment."""
        opts = self._container
        if opts is None:
            raise RuntimeError('container options are required for Docker calls')
        operation = asyncio.create_task(
            asyncio.to_thread(subprocess.run, argv, capture_output=True, text=True, timeout=timeout_s, check=False)
        )
        try:
            return await asyncio.shield(operation)
        except asyncio.CancelledError:
            # `to_thread` cannot stop subprocess.run. Wait for its bounded result before allowing cleanup
            # to issue `rm`, so a slow `docker run` cannot create a container after cleanup has passed.
            await finish_task_uninterruptibly(operation)
            raise
        except FileNotFoundError as exc:
            raise CodingAgentUnavailableError('cli.not_found', f'{opts.binary!r} not found on PATH') from exc
        except (subprocess.TimeoutExpired, OSError) as exc:
            operation_name = next(
                (arg for arg in argv if arg in {'image', 'run', 'inspect', 'exec', 'rm', 'ps'}), 'control'
            )
            detail = str(exc)[:STDERR_EXCERPT_CHARS].strip() or type(exc).__name__
            raise CodingAgentUnavailableError(
                'cli.container_start', f'{opts.binary} {operation_name} control call failed: {detail}'
            ) from exc

    async def _ensure_container(self, root: Path) -> str:
        opts = self._container
        if opts is None:
            raise RuntimeError('container options are required to start a container')
        if self._container_name is not None:
            state = await self._docker([*opts.cli(), 'inspect', '-f', '{{.State.Running}}', self._container_name])
            if state.returncode == 0 and state.stdout.strip() == 'true':
                return self._container_name
            old_name = self._container_name
            marker = root / 'home' / '.evq-exit'
            cause = marker.read_text(encoding='utf-8').strip() if marker.exists() else None
            marker.unlink(missing_ok=True)
            log_kill(
                self._agent,
                'lease_expired' if cause else 'container_lost',
                f'container {old_name} stopped between turns ({cause or "cause unknown"}); recreating it. '
                'Files in the workdir and home survive; processes the agent started in the background do not',
            )
            await self._drop_container_shielded()
            self._restarts += 1
        elif not self._image_checked:
            inspected = await self._docker([*opts.cli(), 'image', 'inspect', opts.image])
            if inspected.returncode != 0:
                build_dir = Path(__file__).parent / 'docker'
                detail = inspected.stderr[-STDERR_EXCERPT_CHARS:].strip()
                if any(phrase in detail.lower() for phrase in ('no such image', 'no such object', 'image not known')):
                    raise CodingAgentUnavailableError(
                        'cli.image_missing',
                        f'image {opts.image!r} not found. Build it with '
                        f'`{shell_join(["eq", "coding-agent", "build-image", "--tag", opts.image])}` '
                        f'or `{shell_join([opts.binary, "build", "-t", opts.image, str(build_dir)])}`',
                    )
                raise CodingAgentUnavailableError(
                    'cli.container_start',
                    f'{opts.binary} image inspect failed: {detail or "Docker returned no error details"}',
                )
            self._image_checked = True
            await asyncio.to_thread(sweep_orphans, opts.binary, opts.context)
        for source in unsafe_mounts(opts.run_args, root):
            logger.warning(
                f'CodingAgentTarget({self._agent}): run_args mount source {source!r} is outside the private workdir; '
                'the agent can reach it'
            )
        name = f'{opts.name_prefix}-{self._agent}-{uuid.uuid4().hex[:8]}'
        lease_dir = root / 'lease' / name
        lease_dir.mkdir()
        beat = lease_dir / 'beat'
        write_beat(beat, 0)
        register(name, LiveContainer(binary=opts.binary, context=opts.context, beat=beat))
        self._owned.append(name)
        self._container_name = name
        uid, gid = host_ids()
        try:
            started = await self._docker(
                build_run_argv(opts, name=name, root=root, lease_dir=lease_dir, uid=uid, gid=gid)
            )
        except BaseException:
            await self._drop_container_shielded()
            raise
        if started.returncode != 0:
            await self._drop_container_shielded()
            raise CodingAgentUnavailableError(
                'cli.container_start', f'{opts.binary} run failed: {started.stderr[-STDERR_EXCERPT_CHARS:].strip()}'
            )
        try:
            state = await self._docker([*opts.cli(), 'inspect', '-f', '{{.State.Running}}', name])
        except BaseException:
            await self._drop_container_shielded()
            raise
        if state.returncode != 0 or state.stdout.strip() != 'true':
            await self._drop_container_shielded()
            detail = state.stderr[-STDERR_EXCERPT_CHARS:].strip() or state.stdout[-STDERR_EXCERPT_CHARS:].strip()
            raise CodingAgentUnavailableError(
                'cli.container_start',
                f'{opts.binary} run exited successfully but container {name} was not running'
                f'{f": {detail}" if detail else ""}',
            )
        return name

    async def _drop_container(self) -> None:
        """Unregister and remove this target's current container."""
        name, self._container_name = self._container_name, None
        live = unregister(name) if name else None
        if name and live:
            await remove_containers_async(live.binary, live.context, [name])

    async def _drop_container_shielded(self) -> None:
        await await_shielded(self._drop_container())

    async def _prepare_container_exec(
        self, argv: list[str]
    ) -> tuple[list[str], Callable[[], Awaitable[None]] | None, str | None]:
        if self._container is None:
            return argv, None, None
        root = self._root
        if root is None:
            raise RuntimeError('container workdir root was not initialized')
        try:
            name = await self._ensure_container(root)
        except asyncio.CancelledError:
            await self._drop_container_shielded()
            raise
        return (
            build_exec_argv(self._container, name=name, env_names=self._forward_env, agent_argv=argv),
            (self._drop_container),
            name,
        )

    def _ensure_workdir(self) -> Path:
        if self._workdir is not None:
            return self._workdir
        root = Path(tempfile.mkdtemp(prefix=f'evaluatorq-{self._agent}-'))
        dst = root / 'work' if self._container is not None else root
        if self._container is not None:
            dst.mkdir()
            (root / 'home').mkdir()
            (root / 'lease').mkdir()
        try:
            if self._source_workdir is not None:
                shutil.copytree(self._source_workdir, dst, symlinks=True, dirs_exist_ok=True)
            skills_dir = dst / self._spec.skills_dir
            if not skills_dir.resolve().is_relative_to(dst.resolve()):
                raise ValueError(
                    f'{self._spec.skills_dir} in the workdir is a symlink that leaves the private copy; '
                    'refusing to write skill links through it'
                )
            skills_dir.mkdir(parents=True, exist_ok=True)
            for skill in self._skills:
                link = skills_dir / skill.name
                if link.exists() or link.is_symlink():
                    raise FileExistsError(f'skill {skill.name!r} already exists in {skills_dir}; refusing to shadow it')
                if self._container is not None:
                    shutil.copytree(skill.resolve(), link, symlinks=True)
                else:
                    link.symlink_to(skill.resolve(), target_is_directory=True)
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)
            raise
        if not self._keep_workdir:
            self._finalizer = weakref.finalize(self, remove_tree_if_owner, self._creator_pid, root)
        self._root = root
        self._workdir = dst
        return dst

    def new(self) -> CodingAgentTarget:
        return type(self)(self._agent, **self._kwargs)

    def _require_creator_process(self, operation: str) -> None:
        if os.getpid() != self._creator_pid:
            raise RuntimeError(
                f'CodingAgentTarget.{operation}() cannot use a target inherited across os.fork(); '
                'call target.new() in the child process to create a process-owned target'
            )

    async def close(self) -> None:
        """Kill a live process group and release the temp workdir. Idempotent."""
        self._require_creator_process('close')
        proc = self._proc
        if proc is not None:
            await kill_group(proc)
            self._proc = None
        if self._container is not None:
            await self._drop_container_shielded()
        if self._root is None:
            return
        lease = self._root / 'lease'
        if lease.exists():
            remove_tree(lease)
        if self._keep_workdir:
            logger.info(f'CodingAgentTarget({self._agent}): keeping workdir {self._workdir}')
        else:
            remove_tree(self._root)
        if self._finalizer is not None:
            self._finalizer.detach()
        if self._container_finalizer is not None:
            self._container_finalizer.detach()
        self._workdir = None
        self._root = None

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
        self._require_creator_process('respond')
        workdir = self._ensure_workdir()
        env = {**os.environ, **self._env}
        # Looked up on the child PATH: Windows would search the parent's, and for .exe files only.
        host_binary = (
            shutil.which('orq' if self._launcher == 'orq' else self._spec.binary, path=env.get('PATH'))
            if self._container is None
            else None
        )
        # A batch launcher cannot carry free text safely, so the system prompt rides in the stdin transcript.
        inline_system = self._spec.system_prompt_flag is None or (
            host_binary is not None and is_batch_launcher(host_binary)
        )
        prompt = render_prompt(messages, system_prompt=self._system_prompt, inline_system=inline_system)
        argv, stdin_text = build_argv(
            agent=self._agent,
            launcher=self._launcher,
            model=self._model,
            permission_mode=self._permission_mode,
            system_prompt=None if inline_system else self._system_prompt,
            extra_args=self._extra_args,
            orq=self._orq,
            prompt=prompt,
        )
        if host_binary is not None:
            argv = [host_binary, *argv[1:]]
        on_early_exit: Callable[[], Awaitable[None]] | None = None
        agent_binary = argv[0]
        argv, on_early_exit, name = await self._prepare_container_exec(argv)
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
            turn_succeeded = False
            try:
                if self._container is not None:
                    set_span_attrs(
                        span,
                        {
                            'evaluatorq.coding_agent.container.image': self._container.image,
                            'evaluatorq.coding_agent.container.name': name,
                            'evaluatorq.coding_agent.container.restarts': self._restarts,
                        },
                    )
                returncode, stdout, stderr = await self._run(argv, stdin_text, cwd=workdir, env=env)
                set_span_attrs(span, {'evaluatorq.coding_agent.exit_code': returncode})
                stderr_excerpt = stderr[-STDERR_EXCERPT_CHARS:]
                if self._container is not None and returncode in (126, 127):
                    raise CodingAgentUnavailableError(
                        'cli.agent_not_found',
                        f'{agent_binary!r} or evq-entrypoint is missing from image {self._container.image!r}: '
                        f'{stderr_excerpt}',
                    )
                if self._container is not None and returncode == 137:
                    log_kill(self._agent, 'container_lost', f'container {name} was removed while the agent ran')
                    await self._drop_container_shielded()
                    set_span_attrs(span, {'evaluatorq.coding_agent.kill_reason': 'container_lost'})
                    raise CodingAgentUnavailableError(
                        'cli.timeout',
                        'the container was removed while the agent was running',
                        kill_reason='container_lost',
                    )
                if returncode != 0:
                    raise CodingAgentError(f'cli.exit.{returncode}', f'{argv[0]} exited {returncode}: {stderr_excerpt}')
                events = parse_jsonl(stdout)
                if stdout.strip() and not events:
                    raise CodingAgentError(
                        'cli.parse_error', f'no JSON events in stdout: {stdout[:STDERR_EXCERPT_CHARS]}'
                    )
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
                response = AgentResponse(
                    output=[*turn.tool_calls, TextOutputItem(text=turn.text, annotations=[])],
                    usage=turn.usage,
                    model=turn.model or self._model,
                    response_id=turn.session_id,
                )
                turn_succeeded = True
                return response
            except CodingAgentError as exc:
                set_span_attrs(span, {'evaluatorq.coding_agent.kill_reason': exc.kill_reason})
                raise
            except asyncio.CancelledError:
                set_span_attrs(span, {'evaluatorq.coding_agent.kill_reason': 'cancelled'})
                raise
            finally:
                if not turn_succeeded and on_early_exit is not None:
                    await await_shielded(on_early_exit())

    async def _run(
        self,
        argv: list[str],
        stdin_text: str | None,
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> tuple[int, str, str]:
        """Run one agent process; end it on the idle limit, the hard cap, an error or cancellation.

        Stdout and stderr are drained concurrently, and chunks from either stream reset the idle deadline.
        Stdout is read in chunks, so neither a long JSONL line nor a partial one can stall or crash the read.
        The caller owns container cleanup after this method has killed the process group on early exit.
        """
        refuse_unsafe_batch_args(argv)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env=env,
                stdin=asyncio.subprocess.PIPE if stdin_text is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise CodingAgentUnavailableError('cli.not_found', f'{argv[0]!r} not found on PATH') from exc
        except OSError as exc:
            if exc.errno != errno.E2BIG:
                raise
            raise CodingAgentUnavailableError(
                'cli.prompt_too_long', f'{argv[0]} argv exceeds the OS limit; the rendered transcript is too long'
            ) from exc
        self._proc = proc
        if proc.stdout is None or proc.stderr is None:
            await kill_group(proc)
            self._proc = None
            raise RuntimeError('coding-agent process was created without stdout and stderr pipes')
        activity = [asyncio.get_running_loop().time()]
        stderr_task = asyncio.create_task(
            drain_tail(
                proc.stderr,
                STDERR_EXCERPT_CHARS * 4,
                on_chunk=lambda: activity.__setitem__(0, asyncio.get_running_loop().time()),
            )
        )
        stdin_task = (
            asyncio.create_task(feed_stdin(proc.stdin, stdin_text.encode()))
            if stdin_text is not None and proc.stdin is not None
            else None
        )
        idle_s, hard_s = self.timeout_ms / 1000, self.max_turn_ms / 1000
        stdout = b''
        finished = False
        proc_wait_task: asyncio.Task[int] | None = None
        try:
            stdout, started = await read_stdout(
                proc.stdout, self._agent, idle_s=idle_s, hard_s=hard_s, activity=activity
            )
            proc_wait_task = asyncio.create_task(proc.wait())
            await wait_until_deadline(proc_wait_task, self._agent, stdout, started, activity, idle_s, hard_s)
            await wait_until_deadline(stderr_task, self._agent, stdout, started, activity, idle_s, hard_s)
            stderr = stderr_task.result()
            finished = True
        except asyncio.CancelledError:
            log_kill(self._agent, 'cancelled', 'the caller cancelled the turn')
            raise
        finally:
            await kill_group(proc, force=not finished)
            self._proc = None
            if stdin_task is not None:
                stdin_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stdin_task
            if not finished:
                stderr_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stderr_task
                if proc_wait_task is not None:
                    proc_wait_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await proc_wait_task
        return proc.returncode or 0, stdout.decode(errors='replace'), stderr.decode(errors='replace')


def log_kill(agent: str, reason: str, detail: str) -> None:
    """The one log line every early end of an agent run writes, so a report row can say which limit fired."""
    logger.warning(f'CodingAgentTarget({agent}) {reason}: {detail}')


def deadline_error(
    agent: str, stdout: bytes | bytearray, started: float, last_output: float, idle_s: float, hard_s: float, now: float
) -> CodingAgentUnavailableError | None:
    if now >= started + hard_s:
        event_count = stdout.count(b'\n')
        detail = f'turn exceeded {hard_s:.0f}s ({event_count} events)'
        log_kill(agent, 'hard_cap', detail)
        return CodingAgentUnavailableError('cli.timeout', detail, kill_reason='hard_cap')
    if now >= last_output + idle_s:
        stamp = time.strftime('%H:%M:%S', time.localtime(time.time() - (now - last_output)))
        detail = f'no output for {idle_s:.0f}s (last event: {describe_last_event(stdout)} at {stamp})'
        log_kill(agent, 'idle_timeout', detail)
        return CodingAgentUnavailableError('cli.timeout', detail, kill_reason='idle_timeout')
    return None


async def read_stdout(
    stream: asyncio.StreamReader, agent: str, *, idle_s: float, hard_s: float, activity: list[float]
) -> tuple[bytes, float]:
    started = asyncio.get_running_loop().time()
    stdout = bytearray()
    while True:
        now = asyncio.get_running_loop().time()
        expired = deadline_error(agent, stdout, started, activity[0], idle_s, hard_s, now)
        if expired is not None:
            raise expired
        wait = min(activity[0] + idle_s, started + hard_s) - now
        try:
            chunk = await asyncio.wait_for(stream.read(READ_CHUNK_BYTES), timeout=wait)
        except asyncio.TimeoutError:
            continue
        if not chunk:
            return bytes(stdout), started
        stdout.extend(chunk)
        activity[0] = asyncio.get_running_loop().time()


async def wait_until_deadline(
    task: asyncio.Task[Any],
    agent: str,
    stdout: bytes,
    started: float,
    activity: list[float],
    idle_s: float,
    hard_s: float,
) -> Any:
    loop = asyncio.get_running_loop()
    while True:
        now = loop.time()
        expired = deadline_error(agent, stdout, started, activity[0], idle_s, hard_s, now)
        if expired is not None:
            raise expired
        if task.done():
            return task.result()
        remaining = min(started + hard_s, activity[0] + idle_s) - now
        await asyncio.wait({task}, timeout=remaining)


def describe_last_event(stdout: bytes | bytearray) -> str:
    """Return the last complete JSON event type and the tool or item it names, when present."""
    for line in reversed(stdout.splitlines()):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        item = event.get('item')
        part = event.get('part')
        detail = (item.get('type') if isinstance(item, dict) else None) or (
            part.get('tool') if isinstance(part, dict) else None
        )
        message = event.get('message')
        content = message.get('content') if isinstance(message, dict) else None
        if detail is None and isinstance(content, list) and content and isinstance(content[-1], dict):
            detail = content[-1].get('name') or content[-1].get('type')
        return f'{event.get("type")} {detail}' if detail else str(event.get('type'))
    return 'none'


async def drain_tail(
    stream: asyncio.StreamReader, keep_bytes: int, *, on_chunk: Callable[[], None] | None = None
) -> bytes:
    """Drain a stream to EOF, retaining only its final ``keep_bytes``."""
    tail = bytearray()
    while chunk := await stream.read(READ_CHUNK_BYTES):
        if on_chunk is not None:
            on_chunk()
        tail.extend(chunk)
        del tail[:-keep_bytes]
    return bytes(tail)


async def feed_stdin(stdin: asyncio.StreamWriter, data: bytes) -> None:
    """Write the prompt, tolerating an agent that exits before reading it."""
    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
        stdin.write(data)
        await stdin.drain()
    stdin.close()


def parse_jsonl(stdout: str) -> list[dict[str, Any]]:
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


def taskkill_tree(pid: int) -> bool:
    """End a Windows process tree with taskkill /T. Return whether it succeeded, logging why when it did not."""
    try:
        result = subprocess.run(
            ['taskkill', '/T', '/F', '/PID', str(pid)], capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            f'CodingAgentTarget: taskkill could not end process tree {pid}, killing the agent process only: {exc}'
        )
        return False
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[:STDERR_EXCERPT_CHARS]
        logger.warning(
            f'CodingAgentTarget: taskkill exited {result.returncode} for process tree {pid}, '
            f'killing the agent process only; its children may still be running: {detail}'
        )
        return False
    return True


def kill_direct(proc: asyncio.subprocess.Process) -> None:
    """Fallback when taskkill fails: end the agent process itself. Its children are out of reach."""
    with contextlib.suppress(ProcessLookupError):
        proc.kill()


async def kill_group(proc: asyncio.subprocess.Process, *, force: bool = False) -> None:
    if proc.returncode is not None and not force:
        return
    if not hasattr(os, 'killpg'):
        # Windows has no process groups, so this is best effort: taskkill /T ends the tree it can still
        # reach from the pid. Once the CLI has exited, its children are unreachable and the pid may be reused.
        if proc.returncode is None:
            operation = asyncio.ensure_future(asyncio.to_thread(taskkill_tree, proc.pid))
            try:
                tree_killed = await asyncio.shield(operation)
            except asyncio.CancelledError:
                # `to_thread` cannot stop taskkill. Wait for its bounded result so cleanup runs after the kill.
                await finish_task_uninterruptibly(operation)
                if operation.cancelled() or operation.exception() is not None or not operation.result():
                    kill_direct(proc)
                raise
            if not tree_killed:
                kill_direct(proc)
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
