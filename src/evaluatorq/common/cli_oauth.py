"""Orq CLI backed clients for dashboard Insights OAuth sessions.

The CLI owns OAuth token storage and refresh. This module never reads its
credential/session files and never copies an OAuth token into the environment.
Each SDK-shaped method is translated into one non-interactive ``orq`` command.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import date, datetime
from types import SimpleNamespace
from typing import Any, Literal, NamedTuple

from loguru import logger

# Environment credentials and profile selection override the CLI OAuth session.
# Every OAuth call removes them so the CLI has one unambiguous auth source.
_OVERRIDE_ENV = (
    'ORQ_API_KEY',
    'ORQ_PROFILE',
    'ORQ_SERVER',
    'ORQ_WORKSPACE',
    'ORQ_PROJECT',
    'ORQ_VERBOSE',
    'ORQ_JMESPATH',
)


def _oauth_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in _OVERRIDE_ENV:
        env.pop(name, None)
    return env


class OrqCLIError(RuntimeError):
    """The authenticated Orq CLI could not complete a request."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class _CLI:
    def __init__(self, *, server_url: str | None, workspace: str | None, project: str | None) -> None:
        binary = shutil.which('orq')
        if binary is None:
            raise OrqCLIError('The orq CLI is not installed. Install it and sign in before using CLI OAuth.')
        self.binary = binary
        self.server_url = server_url
        self.workspace = workspace
        self.project = project
        self._response_hooks: list[Any] = []

    async def call(
        self,
        args: list[str],
        payload: dict[str, Any] | None = None,
        *,
        capture: tuple[str, str] | None = None,
    ) -> Any:
        # An explicit empty profile bypasses a globally selected API-key
        # profile and makes the CLI choose its saved OAuth session for the
        # requested server. This does not change the CLI's persistent config.
        command = [self.binary, '--no-input', '--output-format', 'json', '--profile', '']
        if self.server_url:
            command.extend(['--server', self.server_url])
        if self.workspace:
            command.extend(['--workspace', self.workspace])
        if self.project:
            command.extend(['--project', self.project])
        command.extend(args)
        if payload is not None:
            command.append('--stdin')

        env = _oauth_env()
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE if payload is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            body = (
                json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode() if payload is not None else None
            )
            stdout, stderr = await process.communicate(body)
        except OSError as exc:
            raise OrqCLIError(f'Could not start the orq CLI: {exc}') from exc
        if process.returncode:
            detail = _safe_error(stderr.decode(errors='replace'))
            raise OrqCLIError(detail or f'orq CLI exited with status {process.returncode}')
        try:
            result = _attribute_value(json.loads(stdout))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OrqCLIError('The orq CLI returned an unreadable JSON response.') from exc
        if capture is not None:
            operation_id, path = capture
            response = SimpleNamespace(
                json=lambda: result,
                request=SimpleNamespace(url=SimpleNamespace(path=f'/v3{path}')),
            )
            context = SimpleNamespace(operation_id=operation_id)
            for hook in tuple(getattr(self, '_response_hooks', ())):
                callback = getattr(hook, 'after_success', hook)
                if not callable(callback):
                    raise TypeError('Orq CLI response hook has no callable after_success method.')
                callback(context, response)
        return result

    def register_response_hook(self, hook: Any) -> None:
        if hook not in self._response_hooks:
            self._response_hooks.append(hook)

    def unregister_response_hook(self, hook: Any) -> None:
        with suppress(ValueError):
            self._response_hooks.remove(hook)


class _Traces:
    def __init__(self, cli: _CLI) -> None:
        self._cli = cli

    async def query_async(
        self,
        *,
        from_: Any,
        to: Any,
        oql: str,
        limit: int | None = None,
        page_token: str | None = None,
        timeout_ms: int | None = None,
        **_: Any,
    ) -> Any:
        del timeout_ms
        payload = {'from': _json_value(from_), 'to': _json_value(to), 'oql': oql}
        if limit is not None:
            payload['limit'] = limit
        if page_token is not None:
            payload['page_token'] = page_token
        return await self._cli.call(['traces', 'query-oql'], payload, capture=('TracesQueryOql', '/traces/query'))

    async def get_span_async(self, *, trace_id: str, span_id: str, timeout_ms: int | None = None, **_: Any) -> Any:
        del timeout_ms
        return await self._cli.call(
            ['traces', 'get-span', trace_id, span_id],
            capture=('TracesGetSpan', f'/traces/{trace_id}/spans/{span_id}'),
        )

    async def list_spans_async(
        self,
        *,
        trace_id: str,
        limit: int | None = None,
        page_token: str | None = None,
        timeout_ms: int | None = None,
        **_: Any,
    ) -> Any:
        del timeout_ms
        args = ['traces', 'list-spans', trace_id]
        if limit is not None:
            args.extend(['--limit', str(limit)])
        if page_token is not None:
            args.extend(['--page-token', page_token])
        return await self._cli.call(args)

    async def list_facet_values_async(
        self,
        *,
        field: str,
        from_: Any = None,
        to: Any = None,
        limit: int | None = None,
        filter_operator: str | None = None,
        timeout_ms: int | None = None,
        **_: Any,
    ) -> Any:
        del timeout_ms
        args = ['traces', 'list-facet-values', field]
        for key, value in (('--from', from_), ('--to', to), ('--limit', limit), ('--filter-operator', filter_operator)):
            if value is not None:
                args.extend([key, str(_json_value(value))])
        return await self._cli.call(args)


class _Projects:
    def __init__(self, cli: _CLI) -> None:
        self._cli = cli

    async def list_async(self, *, limit: int | None = None, starting_after: str | None = None, **_: Any) -> Any:
        args = ['projects', 'list']
        if limit is not None:
            args.extend(['--limit', str(limit)])
        if starting_after:
            args.extend(['--starting-after', starting_after])
        return await self._cli.call(args)


class _OrqClient:
    def __init__(self, cli: _CLI) -> None:
        self.traces = _Traces(cli)
        self.projects = _Projects(cli)
        self.sdk_configuration = SimpleNamespace(_hooks=_CLIHooks(cli))

    async def aclose(self) -> None:
        """Satisfy the Orq client lifecycle interface; subprocesses are per call."""

    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None


class _CLIHooks:
    def __init__(self, cli: _CLI) -> None:
        self._cli = cli

    def register_after_success_hook(self, hook: Any) -> None:
        self._cli.register_response_hook(hook)

    def unregister_after_success_hook(self, hook: Any) -> None:
        self._cli.unregister_response_hook(hook)


class _Endpoint:
    def __init__(self, cli: _CLI, command: list[str]) -> None:
        self._cli = cli
        self._command = command

    async def create(self, **params: Any) -> Any:
        body = dict(params)
        extra_body = body.pop('extra_body', None)
        if isinstance(extra_body, dict):
            body = {**body, **extra_body}
        return await self._cli.call(self._command, _json_value(body))

    async def parse(self, **params: Any) -> Any:
        response_model = params.pop('response_model', None)
        supplied_format = params.get('response_format')
        if (
            response_model is None
            and isinstance(supplied_format, type)
            and hasattr(supplied_format, 'model_validate_json')
        ):
            response_model = supplied_format
            params['response_format'] = _schema_format(response_model)
        if response_model is not None and 'response_format' not in params:
            params['response_format'] = _schema_format(response_model)
        response = await self.create(**params)
        if response_model is not None:
            for choice in getattr(response, 'choices', ()):
                message = choice.message
                content = getattr(message, 'content', None)
                try:
                    message.parsed = response_model.model_validate_json(content) if content else None
                except ValueError:
                    message.parsed = None
        return response


class _LLMClient:
    def __init__(self, cli: _CLI) -> None:
        self.chat = SimpleNamespace()
        self.chat.completions = _Endpoint(cli, ['chat', 'create'])
        self.embeddings = _Endpoint(cli, ['embeddings', 'create'])
        self.responses = _Endpoint(cli, ['responses', 'create'])
        self.base_url = f'{(cli.server_url or "https://my.orq.ai").rstrip("/")}/v3/router'
        self._cli = cli

    async def post(self, path: str, *, cast_to: Any = object, body: dict[str, Any], options: Any = None) -> Any:
        """Route Orq's classify endpoint through the CLI OAuth session."""

        del cast_to
        if path != '/classify':
            raise ValueError(f'Unsupported CLI OAuth model endpoint: {path}')
        args = ['request', 'POST', f'/v3/router{path}', '--force']
        # The CLI owns authentication. Caller headers may contain credentials,
        # and forwarding them in argv would expose them through process inspection.
        del options
        result = await self._cli.call(args, _json_value(body))
        if isinstance(result, dict) and 'body' in result and 'status' in result:
            return result['body']
        return result

    async def get_model_catalogue(self) -> Any:
        """Fetch the model catalogue through the CLI's OAuth session."""

        result = await self._cli.call(['request', 'GET', '/v2/models', '--force'])
        if isinstance(result, dict) and 'status' in result and 'body' in result:
            status = result['status']
            if isinstance(status, bool) or not isinstance(status, int) or status < 200 or status >= 300:
                raise OrqCLIError(
                    f'Orq CLI model catalogue request failed with HTTP {status}.',
                    status_code=status if isinstance(status, int) and not isinstance(status, bool) else None,
                )
            if result.get('ok') is False:
                raise OrqCLIError('Orq CLI model catalogue request failed.')
            return result['body']
        return result

    async def close(self) -> None:
        """No persistent transport is held by the CLI adapter."""


def build_cli_oauth_clients(
    server_url: str | None = None,
    workspace: str | None = None,
    project: str | None = None,
) -> tuple[Any, Any]:
    """Build Orq SDK/OpenAI-shaped adapters using the active CLI OAuth session.

    CLI session refresh is delegated to ``orq``. Calls are serialized only by
    the CLI itself; no OAuth token is read, logged, or exported to a child env.
    """
    cli = _CLI(server_url=server_url, workspace=workspace, project=project)
    return _OrqClient(cli), _LLMClient(cli)


_WHOAMI_FIELDS = (
    '{user_id:user.id,workspace:active_workspace_key,project:active_project_id,'
    'source:credential.source,authenticated:authenticated}'
)
# The CLI exits 1 for both a dead login and a network failure; only its message tells them apart.
_SIGNED_OUT = re.compile(r"(?i)expired or was revoked|invalid refresh token|run 'orq auth login'")


def _whoami(binary: str, server_url: str, timeout: float) -> subprocess.CompletedProcess[str]:
    """Ask the CLI who is signed in on *server_url*; the CLI refreshes an expired access token first."""
    return subprocess.run(
        [
            binary,
            '--no-input',
            '--output-format',
            'json',
            '--profile',
            '',
            '--server',
            server_url,
            '--jmespath',
            _WHOAMI_FIELDS,
            'auth',
            'whoami',
        ],
        env=_oauth_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def oauth_subject(server_url: str) -> dict[str, str | None]:
    """Read non-secret OAuth account and active scope IDs for a run launch guard."""

    binary = shutil.which('orq')
    if binary is None:
        raise OrqCLIError('The orq CLI is not installed. Install it and sign in before using CLI OAuth.')
    try:
        result = _whoami(binary, server_url, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OrqCLIError('Could not check the Orq CLI OAuth sign-in.') from exc
    if result.returncode:
        raise OrqCLIError('The Orq CLI OAuth sign-in needs attention. Run orq auth login.')
    try:
        data = json.loads(result.stdout)
    except ValueError as exc:
        raise OrqCLIError('The Orq CLI returned an unreadable OAuth identity.') from exc
    if not isinstance(data, dict) or not data.get('authenticated'):
        raise OrqCLIError('The Orq CLI OAuth sign-in needs attention. Run orq auth login.')
    user_id = data.get('user_id')
    if not isinstance(user_id, str) or not user_id:
        raise OrqCLIError('The Orq CLI did not identify the signed-in user. Run orq auth login.')
    return {key: data.get(key) for key in ('user_id', 'workspace', 'project')}


SessionStatus = Literal['valid', 'signed-out', 'unknown', 'unreadable']
# The CLI's statuses for a session file it cannot decode; such a row has a host but no server.
_UNREADABLE = frozenset({'invalid', 'unreadable'})


class OAuthSession(NamedTuple):
    """One saved Orq CLI OAuth login. The CLI keeps exactly one per server host.

    ``server`` is empty for an ``unreadable`` session: the CLI could not decode
    its file, so nothing can authenticate with it until it is signed in again.
    """

    server: str
    host: str
    user: str | None
    workspace: str | None
    status: SessionStatus
    active: bool


def _session_status(binary: str, server_url: str, timeout: float) -> SessionStatus:
    """Try the login once: a refresh that works is valid, a rejected one is signed out, anything else unknown."""
    try:
        result = _whoami(binary, server_url, timeout)
    except (OSError, subprocess.TimeoutExpired):
        return 'unknown'
    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
        except ValueError:
            return 'unknown'
        return 'valid' if isinstance(data, dict) and data.get('authenticated') else 'signed-out'
    return 'signed-out' if _SIGNED_OUT.search(result.stderr) else 'unknown'


def list_oauth_sessions(timeout: float = 5.0) -> tuple[OAuthSession, ...]:
    """Saved CLI OAuth logins with a checked token status, or none when the CLI is missing or fails.

    The CLI's own listing only says whether the short-lived access token has
    expired, not whether its refresh token still works. Every login the CLI does
    not report as ``ok`` is therefore tried once, in parallel, with ``orq auth
    whoami``; a successful refresh saves new tokens, exactly as the CLI's next
    real call would. Reads only CLI output, never its session files.
    """
    binary = shutil.which('orq')
    if binary is None:
        return ()
    try:
        result = subprocess.run(
            [binary, '--no-input', '--output-format', 'json', 'auth', 'sessions'],
            env=_oauth_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        listing = json.loads(result.stdout) if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        logger.warning('Could not list Orq CLI OAuth sessions: {}', exc)
        return ()
    rows = listing.get('sessions') if isinstance(listing, dict) else None
    if not isinstance(rows, list):
        logger.warning('Could not list Orq CLI OAuth sessions (exit {})', result.returncode)
        return ()
    rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get('host'), str)
        and row['host']
        and isinstance(row.get('status'), str)
        and (row['status'] in _UNREADABLE or (isinstance(row.get('server'), str) and row['server']))
    ]
    to_check = [row['server'] for row in rows if row['status'] not in _UNREADABLE and row['status'] != 'ok']
    checked: dict[str, SessionStatus] = {}
    if to_check:
        with ThreadPoolExecutor(max_workers=len(to_check)) as pool:
            checked = dict(
                zip(to_check, pool.map(lambda server: _session_status(binary, server, timeout), to_check), strict=True)
            )
    sessions: list[OAuthSession] = []
    for row in rows:
        user, workspace = row.get('user'), row.get('workspace')
        unreadable = row['status'] in _UNREADABLE
        sessions.append(
            OAuthSession(
                '' if unreadable else row['server'],
                row['host'],
                user if isinstance(user, str) and user else None,
                workspace if isinstance(workspace, str) and workspace else None,
                'unreadable' if unreadable else checked.get(row['server'], 'valid'),
                bool(row.get('active')),
            )
        )
    return tuple(sessions)


def _json_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, type) and hasattr(value, 'model_json_schema'):
        return _schema_format(value)
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json', exclude_none=True)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items() if item is not None}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class _AttributeMap(dict[str, Any]):  # noqa: FURB189 - must remain a dict for raw trace capture
    """JSON object that supports both SDK-style attributes and mapping access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _attribute_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _AttributeMap({key: _attribute_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_attribute_value(item) for item in value]
    return value


def _schema_format(model: Any) -> dict[str, Any]:
    from openai.lib._pydantic import to_strict_json_schema

    name = getattr(model, '__name__', 'StructuredOutput')
    schema = to_strict_json_schema(model)
    return {'type': 'json_schema', 'json_schema': {'name': name, 'schema': schema, 'strict': True}}


_SECRET = re.compile(
    r'(?i)(bearer\s+)[A-Za-z0-9._~+/-]+=*|((?:api[_-]?key|access[_-]?token|refresh[_-]?token)["\' :=]+)[^\s,;]+'
)


def _safe_error(value: str) -> str:
    """Keep a useful CLI error while masking any credential-shaped text."""
    cleaned = _SECRET.sub(lambda match: (match.group(1) or match.group(2) or '') + '[redacted]', value.strip())
    return cleaned[:1200]
