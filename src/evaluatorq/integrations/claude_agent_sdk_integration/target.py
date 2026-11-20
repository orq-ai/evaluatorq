"""Run an agent built on the Claude Agent SDK (``claude-agent-sdk``) as an ``AgentTarget``.

The agent is whatever a ``ClaudeAgentOptions`` describes: its system prompt, tools, MCP servers, hooks and
permission callbacks. Each ``respond()`` runs one ``query()`` and turns the SDK's message stream into an
``AgentResponse`` with every tool call paired to its result.

The target keeps the conversation: one SDK session per instance, and every turn after the first resumes the
last successful session as a fork. A turn that fails leaves that session untouched, so a retry starts from the
same point instead of from a session that already holds half the failed turn.

Retry lives in ``common.target_call.call_target_with_retry`` only. ``respond()`` never retries, and the Claude Code
runtime's own retry of a failed model request is off (``CLAUDE_CODE_MAX_RETRIES='0'``) unless ``options.env`` sets it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ClaudeSDKError,
    CLINotFoundError,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)
from loguru import logger

from evaluatorq.backends.coding_agent import parse_claude
from evaluatorq.common.target_call import NonRetryableTargetError
from evaluatorq.common.tracing import record_token_usage, set_span_attrs, with_llm_span
from evaluatorq.contracts import (
    AgentContext,
    AgentResponse,
    AgentTarget,
    ConversationHistoryMode,
    Message,
    TextOutputItem,
    ToolInfo,
    content_to_text,
)

Isolation = Literal['sandbox', 'none']

# What the Claude Code runtime would otherwise read from, or write to, the host user's own configuration.
ISOLATION_ENV = {
    'CLAUDE_CODE_DISABLE_AUTO_MEMORY': '1',
    'ENABLE_CLAUDEAI_MCP_SERVERS': 'false',
}

# The Claude Code runtime retries a failed model request itself before the turn fails: with its defaults a wrong key took
# 174 s to come back. call_target_with_retry owns retry for target calls, so the runtime's layer is off unless the
# caller's options.env sets it.
RUNTIME_RETRY_ENV = {'CLAUDE_CODE_MAX_RETRIES': '0'}

# The command sandbox covers shell commands only. These two make the file tools (Read, Write, Edit) refuse a path
# outside the working directory; without them a live run wrote and read files elsewhere on the host.
FILE_TOOL_BLOCKS = {'blockWritesOutsideWorkingDirectories': True, 'blockReadsOutsideWorkingDirectories': True}

# Forced on in sandbox mode. Without failIfUnavailable the runtime only prints a warning when the sandbox cannot start
# (Linux without bubblewrap or socat) and runs every shell command on the host; a live run in such a container wrote
# a file outside the working directory. With it the runtime exits at startup instead.
SANDBOX_REQUIRED = {'enabled': True, 'allowUnsandboxedCommands': False, 'failIfUnavailable': True}

# What the runtime prints when failIfUnavailable stops it; matched to report the failure as non-retryable.
SANDBOX_UNAVAILABLE_MARKERS = ('Sandbox required but unavailable', 'sandbox.failIfUnavailable')

# Model request failures a retry repeats: the runtime's own `AssistantMessage.error` values for them.
NON_RETRYABLE_API_ERRORS = frozenset({'authentication_failed', 'billing_error', 'invalid_request'})

# The project files that `setting_sources=['project']` would load from the working directory.
PROJECT_SETTING_FILES = ('CLAUDE.md', 'CLAUDE.local.md', '.claude/settings.json', '.claude/settings.local.json')

# get_agent_context() reads the tool list from a runtime pointed at this address, where no model can answer, so
# reading it costs no model request. Port 9 (discard) has nothing listening on a normal host.
PROBE_ENDPOINT = 'http://127.0.0.1:9'
PROBE_ENDPOINT_ENV = (
    'ANTHROPIC_BASE_URL',
    'ANTHROPIC_BEDROCK_BASE_URL',
    'ANTHROPIC_VERTEX_BASE_URL',
    'ANTHROPIC_FOUNDRY_BASE_URL',
)
PROBE_TIMEOUT_S = 60.0

_warned_project_files: set[Path] = set()


class ClaudeAgentSDKError(Exception):
    """A Claude Agent SDK turn failed. ``code`` is one of the ``agent_sdk.*`` codes ``map_error`` reports.

    Codes: ``agent_sdk.cli_not_found`` and ``agent_sdk.sandbox_unavailable`` (non-retryable, see
    `ClaudeAgentSDKUnavailableError`), ``agent_sdk.authentication_failed``, ``agent_sdk.billing_error`` and
    ``agent_sdk.invalid_request`` (non-retryable, see `ClaudeAgentSDKRequestError`), ``agent_sdk.sdk_error`` (the SDK
    raised), ``agent_sdk.api_error`` (the model request failed in a way a retry can fix, such as a rate limit, a server
    error or an unreachable gateway), ``agent_sdk.<subtype>`` for a run that ended on another error result (for example
    ``agent_sdk.error_max_turns``) and ``agent_sdk.no_result`` (the run ended with no final text).
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f'{code}: {message}')
        self.code = code
        self.message = message


class ClaudeAgentSDKUnavailableError(ClaudeAgentSDKError, NonRetryableTargetError):
    """The Claude Code runtime is not installed, or the sandbox ``isolation='sandbox'`` needs cannot start here.

    Retrying repeats the same failure.
    """


class ClaudeAgentSDKRequestError(ClaudeAgentSDKError, NonRetryableTargetError):
    """The model request was refused for a reason a retry repeats: authentication, billing or an invalid request."""


def message_to_event(message: object) -> dict[str, Any] | None:
    """Render one SDK message as the stream-json event `parse_claude` reads, or ``None`` for one it has no use for.

    The SDK parses the Claude Code runtime's stream-json lines into typed messages; this puts the three kinds that
    carry a turn's content back into that shape, so the SDK target and `CodingAgentTarget` share one parser.
    """
    if isinstance(message, AssistantMessage):
        blocks: list[dict[str, Any]] = []
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                blocks.append({'type': 'tool_use', 'id': block.id, 'name': block.name, 'input': block.input})
            elif isinstance(block, TextBlock):
                blocks.append({'type': 'text', 'text': block.text})
        return {'type': 'assistant', 'message': {'model': message.model, 'content': blocks}, 'error': message.error}
    if isinstance(message, UserMessage):
        if isinstance(message.content, str):
            return None
        results = [
            {
                'type': 'tool_result',
                'tool_use_id': block.tool_use_id,
                'content': block.content,
                'is_error': bool(block.is_error),
            }
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        return {'type': 'user', 'message': {'content': results}} if results else None
    if isinstance(message, ResultMessage):
        return {
            'type': 'result',
            'subtype': message.subtype,
            'session_id': message.session_id,
            'total_cost_usd': message.total_cost_usd,
            'result': message.result,
            'is_error': message.is_error,
            'usage': message.usage,
            'modelUsage': message.model_usage,
            'permission_denials': message.permission_denials,
            'api_error_status': message.api_error_status,
        }
    return None


class ClaudeAgentSDKTarget(AgentTarget):
    """Wrap an agent described by ``ClaudeAgentOptions`` as the system under test.

    Usage:

    ```python
    from claude_agent_sdk import ClaudeAgentOptions
    from evaluatorq.integrations.claude_agent_sdk_integration import ClaudeAgentSDKTarget

    options = ClaudeAgentOptions(system_prompt="You are a support agent.", model="claude-sonnet-5-5", max_turns=8)
    target = ClaudeAgentSDKTarget(options)
    report = await red_team(target=target)
    ```

    Each instance owns one conversation and two private temp directories: the agent's working directory (empty, or
    a copy of ``workdir``, without its symbolic links) and its Claude configuration directory, where the SDK keeps the session. ``new()`` clones
    carry the same options and get their own, so parallel attacks never share files or a session. ``close()`` removes
    both unless ``keep_workdir`` is set; the runner calls it through ``common.target_call.close_target``.

    ``isolation='sandbox'`` (the default) turns on the SDK's command sandbox, refuses unsandboxed commands, makes the
    runtime stop at startup when the sandbox cannot run (the turn raises `ClaudeAgentSDKUnavailableError`), and makes
    the file tools refuse paths outside the working directory. It also stops the runtime loading any settings, memory
    files and account connectors, the project's own ``CLAUDE.md`` and ``.claude/settings.json`` included, unless the
    options already set ``setting_sources``. Project settings are left out because their hooks run on the host,
    outside the sandbox. ``isolation='none'`` leaves the options as given and only moves the two directories. In both
    modes the runtime's own retry of a failed model request is off (``CLAUDE_CODE_MAX_RETRIES='0'``), so
    ``call_target_with_retry`` is the only retry layer; a ``CLAUDE_CODE_MAX_RETRIES`` in ``options.env`` wins. Neither
    confines Python tools the options define in-process: those run in this process, with its access. Run the whole
    evaluation in a container when the agent must not be able to reach the host at all.

    The private configuration directory means the runtime does not see a ``claude`` login stored under ``~/.claude``.
    Authenticate with ``ANTHROPIC_API_KEY`` or a gateway set through ``options.env``.

    Not safe for concurrent ``respond()`` on one instance; use ``new()``.
    """

    history_mode = ConversationHistoryMode.TARGET

    def __init__(
        self,
        options: ClaudeAgentOptions,
        *,
        workdir: Path | None = None,
        isolation: Isolation = 'sandbox',
        keep_workdir: bool = False,
    ) -> None:
        super().__init__()
        if isolation not in ('sandbox', 'none'):
            raise ValueError(f'Unknown isolation {isolation!r}; expected "sandbox" or "none"')
        if workdir is not None and not Path(workdir).is_dir():
            raise ValueError(f'workdir {str(workdir)!r} is not a directory')
        self._options = options
        self._source_workdir = Path(workdir) if workdir is not None else None
        self._isolation: Isolation = isolation
        self._keep_workdir = keep_workdir
        self._root: Path | None = None
        self._session_id: str | None = None
        if isolation == 'sandbox' and options.setting_sources is None and self._source_workdir is not None:
            _warn_skipped_project_files(self._source_workdir)

    @property
    def workdir(self) -> Path | None:
        """The agent's private working directory, once the first turn has created it."""
        return self._root / 'workdir' if self._root is not None else None

    @property
    def session_id(self) -> str | None:
        """The SDK session the next turn resumes; ``None`` before the first successful turn."""
        return self._session_id

    def _ensure_dirs(self) -> Path:
        if self._root is None:
            # _root stays unset until the copy is whole, so a retried turn builds the directories again.
            self._root = _make_dirs(self._source_workdir)
        return self._root

    def turn_options(self) -> ClaudeAgentOptions:
        """The options for the next turn: the caller's, moved into this instance's directories and session."""
        return self._options_in(self._ensure_dirs(), resume=self._session_id)

    def _options_in(self, root: Path, *, resume: str | None) -> ClaudeAgentOptions:
        changes: dict[str, Any] = {
            'cwd': root / 'workdir',
            'env': {**RUNTIME_RETRY_ENV, **self._options.env, 'CLAUDE_CONFIG_DIR': str(root / 'config')},
            'resume': resume,
            'fork_session': resume is not None,
        }
        if self._isolation == 'sandbox':
            changes['env'] = {**ISOLATION_ENV, **changes['env']}
            if self._options.setting_sources is None:
                changes['setting_sources'] = []
            changes['sandbox'] = {**(self._options.sandbox or {}), **SANDBOX_REQUIRED}
            changes['settings'] = settings_with_file_tool_blocks(self._options.settings)
        return dataclasses.replace(self._options, **changes)

    async def respond(self, messages: list[Message]) -> AgentResponse:
        """Send the latest user turn to the agent's session and return what the agent did and said.

        The turn runs in a ``with_llm_span`` span that carries its token usage and the cost the runtime reported
        (``evaluatorq.claude_agent_sdk.cost_usd``), the way `CodingAgentTarget` records them. ``AgentResponse.usage``
        carries the token counts only, so token-based pricing still applies to it.
        """
        if not messages or messages[-1].role != 'user':
            raise ValueError("ClaudeAgentSDKTarget.respond requires messages[-1].role == 'user'")
        prompt = content_to_text(messages[-1].content)
        async with with_llm_span(
            model=self._options.model or 'claude-agent-sdk',
            operation='chat',
            input_messages=[m.to_chat_completion() for m in messages],
            attributes={'orq.redteam.llm_purpose': 'target', 'evaluatorq.claude_agent_sdk.isolation': self._isolation},
        ) as span:
            events = await self._run_turn(prompt)
            turn = parse_claude(events)
            if turn.text is None or not turn.text.strip():
                raise ClaudeAgentSDKError('agent_sdk.no_result', 'the run ended without a final assistant message')
            if turn.session_id is None:
                logger.warning('ClaudeAgentSDKTarget: no session id in the result; the next turn starts a new session')
            if turn.usage is None:
                logger.warning('ClaudeAgentSDKTarget: no usage in the result; usage is None')
            else:
                record_token_usage(span, usage=turn.usage, total_cost=turn.cost_usd)
            set_span_attrs(
                span, {'evaluatorq.claude_agent_sdk.cost_usd': turn.cost_usd, 'gen_ai.response.id': turn.session_id}
            )
            self._session_id = turn.session_id
            return AgentResponse(
                output=[*turn.tool_calls, TextOutputItem(text=turn.text, annotations=[])],
                usage=turn.usage,
                model=turn.model or self._options.model,
                response_id=turn.session_id,
            )

    async def _run_turn(self, prompt: str) -> list[dict[str, Any]]:
        """Run one ``query()`` and return its events, or raise the `ClaudeAgentSDKError` that says why it failed."""
        events: list[dict[str, Any]] = []
        try:
            async for message in query(prompt=prompt, options=self.turn_options()):
                event = message_to_event(message)
                if event is not None:
                    events.append(event)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            raise
        except CLINotFoundError as exc:
            raise ClaudeAgentSDKUnavailableError('agent_sdk.cli_not_found', str(exc)) from exc
        except ClaudeSDKError as exc:
            # A run that ends on an error result raises after yielding that result; report the result, which says why.
            failed = _failure(events)
            if failed is not None:
                raise failed from exc
            detail = f'{type(exc).__name__}: {exc} {getattr(exc, "stderr", None) or ""}'.strip()
            if self._isolation == 'sandbox' and any(marker in detail for marker in SANDBOX_UNAVAILABLE_MARKERS):
                raise ClaudeAgentSDKUnavailableError(
                    'agent_sdk.sandbox_unavailable',
                    'isolation="sandbox" needs the command sandbox, which cannot start on this host (on Linux, '
                    f'install bubblewrap and socat): {detail}',
                ) from exc
            raise ClaudeAgentSDKError('agent_sdk.sdk_error', detail) from exc
        failed = _failure(events)
        if failed is not None:
            raise failed
        return events

    def map_error(self, exc: Exception) -> tuple[str, str] | None:
        if isinstance(exc, ClaudeAgentSDKError):
            return exc.code, exc.message
        return None

    async def get_agent_context(self) -> AgentContext:
        """Describe the agent: system prompt, model, and every tool the runtime gives it.

        The tool list is the one the Claude Code runtime announces when a session starts: the built-in tools that
        ``tools`` and ``disallowed_tools`` leave, plus each MCP server's tools as ``mcp__<server>__<tool>``. To read
        it without a model request, the runtime starts with the options of a real turn, pointed at an address where
        no model answers, and is stopped as soon as it has announced its tools. Hooks and ``can_use_tool`` are left
        out of that start, since no tool runs. When the runtime cannot be started, the tools the options name are
        listed instead, with a warning.
        """
        options = self._options
        try:
            init = await asyncio.wait_for(self._announced_session(), timeout=PROBE_TIMEOUT_S)
        except (ClaudeSDKError, ClaudeAgentSDKError, OSError, asyncio.TimeoutError) as exc:
            logger.warning(
                'ClaudeAgentSDKTarget: could not read the tool list from the Claude Code runtime ({}: {}); listing '
                'only the tools the options name, without the built-in defaults or MCP tools',
                type(exc).__name__,
                exc,
            )
            tools = [name for name in options.tools or [] if name not in options.disallowed_tools]
        else:
            tools = [str(name) for name in init.get('tools', [])]
            for server in init.get('mcp_servers', []):
                if server.get('status') != 'connected':
                    logger.warning(
                        'ClaudeAgentSDKTarget: MCP server {!r} is {}; its tools are not in the agent context',
                        server.get('name'),
                        server.get('status'),
                    )
        return AgentContext(
            key='claude_agent_sdk',
            display_name='Claude Agent SDK agent',
            description='Claude Agent SDK target',
            system_prompt=system_prompt_text(options.system_prompt),
            tools=[ToolInfo(name=name) for name in tools],
            model=options.model,
        )

    async def _announced_session(self) -> dict[str, Any]:
        """Start the runtime with no reachable model, return the ``init`` data it announces, and stop it."""
        root = await asyncio.to_thread(_make_dirs, self._source_workdir)
        try:
            options = self._options_in(root, resume=None)
            options = dataclasses.replace(
                options,
                # The caller's retry setting is overridden too: retrying the dead address only delays the probe.
                env={**options.env, **dict.fromkeys(PROBE_ENDPOINT_ENV, PROBE_ENDPOINT), **RUNTIME_RETRY_ENV},
                hooks=None,
                can_use_tool=None,
            )
            client = ClaudeSDKClient(options=options)
            await client.connect()
            try:
                await client.query('List your tools.')
                async for message in client.receive_messages():
                    if isinstance(message, SystemMessage) and message.subtype == 'init':
                        return message.data
            finally:
                await client.disconnect()
            raise ClaudeAgentSDKError('agent_sdk.no_result', 'the runtime ended without announcing its tools')
        finally:
            await asyncio.to_thread(shutil.rmtree, root, ignore_errors=True)

    def new(self) -> ClaudeAgentSDKTarget:
        """Return an independent instance: same options, its own directories and a new session."""
        return type(self)(
            self._options,
            workdir=self._source_workdir,
            isolation=self._isolation,
            keep_workdir=self._keep_workdir,
        )

    async def close(self) -> None:
        """Remove this instance's directories unless ``keep_workdir`` is set, and forget its session."""
        root, self._root, self._session_id = self._root, None, None
        if root is not None and not self._keep_workdir:
            await asyncio.to_thread(shutil.rmtree, root, ignore_errors=True)


def _make_dirs(source_workdir: Path | None) -> Path:
    """Create a private root holding ``workdir`` (a copy of ``source_workdir`` without its links, or empty) and ``config``.

    A failed copy is removed before the error propagates, so nothing partial is left behind.
    """
    root = Path(tempfile.mkdtemp(prefix='evq-agent-sdk-'))
    try:
        if source_workdir is not None:
            shutil.copytree(source_workdir, root / 'workdir', ignore=_symlinks)
        else:
            (root / 'workdir').mkdir()
        (root / 'config').mkdir()
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return root


def _warn_skipped_project_files(workdir: Path) -> None:
    """Warn once per ``workdir`` that its CLAUDE.md and project settings are not loaded in sandbox mode."""
    key = workdir.resolve()
    found = [name for name in PROJECT_SETTING_FILES if (key / name).is_file()]
    if not found or key in _warned_project_files:
        return
    _warned_project_files.add(key)
    logger.warning(
        'ClaudeAgentSDKTarget: isolation="sandbox" does not load {} from {}: project settings can define hooks, '
        'which the runtime runs on the host outside the sandbox. Pass setting_sources=["project"] to load them '
        'anyway, or put the instructions in system_prompt',
        ', '.join(found),
        workdir,
    )


def system_prompt_text(prompt: object) -> str:
    """The text of ``ClaudeAgentOptions.system_prompt`` that evaluatorq can know.

    A plain string or a ``custom`` prompt is returned whole and a ``file`` prompt is read. Of the ``claude_code``
    preset only the ``append`` text is known, because the preset itself lives in the runtime; a warning says so.
    """
    if prompt is None:
        return ''
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, dict):
        logger.warning('ClaudeAgentSDKTarget: unknown system_prompt shape {}; reporting none', type(prompt).__name__)
        return ''
    kind = prompt.get('type')
    if kind == 'custom':
        return str(prompt.get('prompt') or '')
    if kind == 'preset':
        logger.warning(
            'ClaudeAgentSDKTarget: the {!r} preset system prompt lives in the runtime; the agent context carries '
            'only its append text',
            prompt.get('preset'),
        )
        return str(prompt.get('append') or '')
    if kind == 'file':
        try:
            return Path(str(prompt.get('path'))).read_text(encoding='utf-8')
        except OSError as exc:
            logger.warning('ClaudeAgentSDKTarget: could not read the system prompt file: {}', exc)
            return ''
    logger.warning('ClaudeAgentSDKTarget: unknown system_prompt type {!r}; reporting none', kind)
    return ''


def _symlinks(directory: str, names: list[str]) -> list[str]:
    """The entries of ``directory`` that are symbolic links, for `shutil.copytree` to leave out.

    A link copied as a link would still point at its target on the host, a way out of the private working
    directory. Following it would copy host files the caller never put in the project. So links are not copied.
    """
    links = [name for name in names if (Path(directory) / name).is_symlink()]
    if links:
        logger.warning(
            'ClaudeAgentSDKTarget: not copying {} symbolic link(s) from {} into the working directory',
            len(links),
            directory,
        )
    return links


def _failure(events: list[dict[str, Any]]) -> ClaudeAgentSDKError | None:
    """The error to raise when the run's result event reports one, else ``None``.

    A failed model request ends the run with subtype ``success`` and ``is_error`` set. The runtime names its kind on
    the assistant message (``authentication_failed``, ``billing_error``, ``invalid_request``, ``rate_limit``,
    ``server_error``, ``unknown``) and its HTTP status on the result. Authentication, billing and invalid-request
    failures, and any other 4xx except 408 and 429, raise `ClaudeAgentSDKRequestError`, which is not retried.
    """
    result = next((event for event in reversed(events) if event['type'] == 'result'), None)
    if result is None or not result['is_error']:
        return None
    subtype = str(result['subtype'] or 'error')
    detail = result['result'] if isinstance(result['result'], str) and result['result'] else subtype
    if subtype != 'success':
        return ClaudeAgentSDKError(f'agent_sdk.{subtype}', detail)
    api_error = next(
        (event['error'] for event in reversed(events) if event['type'] == 'assistant' and event.get('error')), None
    )
    status = result.get('api_error_status')
    if api_error in NON_RETRYABLE_API_ERRORS:
        return ClaudeAgentSDKRequestError(f'agent_sdk.{api_error}', detail)
    if isinstance(status, int) and 400 <= status < 500 and status not in (408, 429):
        return ClaudeAgentSDKRequestError('agent_sdk.invalid_request', f'HTTP {status}: {detail}')
    return ClaudeAgentSDKError('agent_sdk.api_error', detail)


def settings_with_file_tool_blocks(settings: str | None) -> str:
    """Return ``settings`` as a JSON string with `FILE_TOOL_BLOCKS` forced on under ``permissions``.

    ``ClaudeAgentOptions.settings`` is a JSON string or a path to a settings file. Either is read, so the caller's
    other settings are kept.

    Raises:
        ValueError: ``settings`` is neither a JSON object nor a readable JSON settings file.
    """
    loaded: Any = {}
    if settings is not None:
        text = settings
        if not settings.lstrip().startswith('{'):
            try:
                text = Path(settings).read_text(encoding='utf-8')
            except OSError as exc:
                raise ValueError(
                    f'options.settings {settings!r} is not JSON and could not be read as a file: {exc}'
                ) from exc
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f'options.settings is not valid JSON: {exc}') from exc
    if not isinstance(loaded, dict) or not isinstance(loaded.get('permissions', {}), dict):
        # ValueError like the two refusals above: all three mean the caller's settings cannot carry the blocks.
        raise ValueError('options.settings must be a JSON object whose "permissions" is an object')  # noqa: TRY004
    return json.dumps({**loaded, 'permissions': {**loaded.get('permissions', {}), **FILE_TOOL_BLOCKS}})
