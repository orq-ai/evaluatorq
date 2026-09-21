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
from dataclasses import dataclass
from typing import Any, Literal

from evaluatorq.common.sanitize import delimit
from evaluatorq.common.target_call import NonRetryableTargetError
from evaluatorq.contracts import AgentResponse, AgentTarget, Message, content_to_text

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


class CodingAgentTarget(AgentTarget):  # placeholder, replaced in Task 6
    async def respond(self, messages: list[Message]) -> AgentResponse:
        raise NotImplementedError

    def new(self) -> CodingAgentTarget:
        return type(self)()
