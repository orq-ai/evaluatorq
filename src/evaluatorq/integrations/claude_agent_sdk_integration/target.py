"""Run an agent built on the Claude Agent SDK (``claude-agent-sdk``) as an ``AgentTarget``.

The agent is whatever a ``ClaudeAgentOptions`` describes: its system prompt, tools, MCP servers, hooks and
permission callbacks. Each ``respond()`` runs one ``query()`` and turns the SDK's message stream into an
``AgentResponse`` with every tool call paired to its result.

The target keeps the conversation: one SDK session per instance, and every turn after the first resumes the
last successful session as a fork. A turn that fails leaves that session untouched, so a retry starts from the
same point instead of from a session that already holds half the failed turn.

Retry lives in ``common.target_call.call_target_with_retry`` only. ``respond()`` never retries.
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
    ClaudeSDKError,
    CLINotFoundError,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)
from loguru import logger

from evaluatorq.backends.coding_agent import parse_claude
from evaluatorq.common.target_call import NonRetryableTargetError
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

# The command sandbox covers shell commands only. These two make the file tools (Read, Write, Edit) refuse a path
# outside the working directory; without them a live run wrote and read files elsewhere on the host.
FILE_TOOL_BLOCKS = {'blockWritesOutsideWorkingDirectories': True, 'blockReadsOutsideWorkingDirectories': True}


class ClaudeAgentSDKError(Exception):
    """A Claude Agent SDK turn failed. ``code`` is one of the ``agent_sdk.*`` codes ``map_error`` reports.

    Codes: ``agent_sdk.cli_not_found`` (non-retryable, see `ClaudeAgentSDKUnavailableError`),
    ``agent_sdk.sdk_error`` (the SDK raised), ``agent_sdk.api_error`` (the model request failed, for example a wrong
    gateway address), ``agent_sdk.<subtype>`` for a run that ended on another error result (for example
    ``agent_sdk.error_max_turns``) and ``agent_sdk.no_result`` (the run ended with no final text).
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f'{code}: {message}')
        self.code = code
        self.message = message


class ClaudeAgentSDKUnavailableError(ClaudeAgentSDKError, NonRetryableTargetError):
    """The Claude Code runtime the SDK drives is not installed. Retrying repeats the same failure."""


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
        return {'type': 'assistant', 'message': {'model': message.model, 'content': blocks}}
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

    ``isolation='sandbox'`` (the default) turns on the SDK's command sandbox, refuses unsandboxed commands, and makes
    the file tools refuse paths outside the working directory. It also stops the runtime loading the host user's
    settings, memory files and account connectors, unless the options already set ``setting_sources``. ``isolation='none'`` leaves the options as given and only moves the two
    directories. Neither confines Python tools the options define in-process: those run in this process, with its
    access. Run the whole evaluation in a container when the agent must not be able to reach the host at all.

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
            root = Path(tempfile.mkdtemp(prefix='evq-agent-sdk-'))
            self._root = root  # set first so close() removes a partial copy if copytree raises
            if self._source_workdir is not None:
                shutil.copytree(self._source_workdir, root / 'workdir', ignore=_symlinks)
            else:
                (root / 'workdir').mkdir()
            (root / 'config').mkdir()
        return self._root

    def turn_options(self) -> ClaudeAgentOptions:
        """The options for the next turn: the caller's, moved into this instance's directories and session."""
        root = self._ensure_dirs()
        changes: dict[str, Any] = {
            'cwd': root / 'workdir',
            'env': {**self._options.env, 'CLAUDE_CONFIG_DIR': str(root / 'config')},
            'resume': self._session_id,
            'fork_session': self._session_id is not None,
        }
        if self._isolation == 'sandbox':
            changes['env'] = {**ISOLATION_ENV, **changes['env']}
            if self._options.setting_sources is None:
                changes['setting_sources'] = []
            changes['sandbox'] = {
                **(self._options.sandbox or {}),
                'enabled': True,
                'allowUnsandboxedCommands': False,
            }
            changes['settings'] = settings_with_file_tool_blocks(self._options.settings)
        return dataclasses.replace(self._options, **changes)

    async def respond(self, messages: list[Message]) -> AgentResponse:
        """Send the latest user turn to the agent's session and return what the agent did and said."""
        if not messages or messages[-1].role != 'user':
            raise ValueError("ClaudeAgentSDKTarget.respond requires messages[-1].role == 'user'")
        prompt = content_to_text(messages[-1].content)
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
            failed = _error_result(events)
            if failed is None:
                raise ClaudeAgentSDKError('agent_sdk.sdk_error', f'{type(exc).__name__}: {exc}') from exc
            raise ClaudeAgentSDKError(*failed) from exc
        failed = _error_result(events)
        if failed is not None:
            raise ClaudeAgentSDKError(*failed)
        turn = parse_claude(events)
        if turn.text is None or not turn.text.strip():
            raise ClaudeAgentSDKError('agent_sdk.no_result', 'the run ended without a final assistant message')
        if turn.session_id is None:
            logger.warning('ClaudeAgentSDKTarget: no session id in the result; the next turn starts a new session')
        self._session_id = turn.session_id
        return AgentResponse(
            output=[*turn.tool_calls, TextOutputItem(text=turn.text, annotations=[])],
            usage=turn.usage,
            model=turn.model or self._options.model,
            response_id=turn.session_id,
        )

    def map_error(self, exc: Exception) -> tuple[str, str] | None:
        if isinstance(exc, ClaudeAgentSDKError):
            return exc.code, exc.message
        return None

    async def get_agent_context(self) -> AgentContext:
        """Describe the agent from its options: system prompt, model and the tools it names."""
        options = self._options
        named = options.tools if isinstance(options.tools, list) else options.allowed_tools
        return AgentContext(
            key='claude_agent_sdk',
            display_name='Claude Agent SDK agent',
            description='Claude Agent SDK target',
            system_prompt=options.system_prompt if isinstance(options.system_prompt, str) else '',
            tools=[ToolInfo(name=name) for name in named],
            model=options.model,
        )

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


def _error_result(events: list[dict[str, Any]]) -> tuple[str, str] | None:
    """``(code, message)`` when the run's result event reports an error, else ``None``."""
    result = next((event for event in reversed(events) if event['type'] == 'result'), None)
    if result is None or not result['is_error']:
        return None
    subtype = str(result['subtype'] or 'error')
    detail = result['result'] if isinstance(result['result'], str) and result['result'] else subtype
    # A failed model request ends the run with subtype ``success`` and ``is_error`` set; name it for what it is.
    return f'agent_sdk.{"api_error" if subtype == "success" else subtype}', detail


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
