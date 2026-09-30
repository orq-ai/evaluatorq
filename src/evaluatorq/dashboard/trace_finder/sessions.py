"""Browser-session ownership for the Traces dashboard surface."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

SESSION_COOKIE = 'evaluatorq_dashboard_session'
SESSION_TTL_SECONDS = 30 * 60
MAX_SESSIONS = 32
_SESSION_ID = re.compile(r'^[A-Za-z0-9_-]{43}$')


def _signature(session_id: str, signing_key: bytes) -> str:
    digest = hmac.new(signing_key, session_id.encode('ascii'), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode('ascii').rstrip('=')


def signed_session_cookie(session_id: str, signing_key: bytes) -> str:
    """Return the opaque ID and its message authentication code."""
    return f'{session_id}.{_signature(session_id, signing_key)}'


def verified_session_id(cookie: str | None, signing_key: bytes) -> str | None:
    """Return a well-formed, signed session ID, or ``None`` for untrusted input."""
    if not cookie or '.' not in cookie:
        return None
    session_id, supplied_signature = cookie.rsplit('.', 1)
    if not _SESSION_ID.fullmatch(session_id):
        return None
    if not hmac.compare_digest(supplied_signature, _signature(session_id, signing_key)):
        return None
    return session_id


@dataclass
class _SessionEntry:
    session_id: str
    last_access: float
    store: Any | None = None
    create_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_requests: int = 0
    retired: bool = False


class TraceSessionRegistry:
    """Bounded, process-local RunStore registry keyed by signed cookie identity."""

    def __init__(
        self,
        *,
        max_sessions: int = MAX_SESSIONS,
        ttl_seconds: float = SESSION_TTL_SECONDS,
        monotonic: Any = time.monotonic,
    ) -> None:
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._monotonic = monotonic
        self._lock = asyncio.Lock()
        self._entries: dict[str, _SessionEntry] = {}

    async def get(self, session_id: str, factory: Any, *, request_state: Any | None = None) -> Any | None:
        """Return one session's store, creating it once without holding the registry lock."""
        while True:
            now = self._monotonic()
            retired: list[Any] = []
            async with self._lock:
                expired = [key for key, entry in self._entries.items() if now - entry.last_access >= self.ttl_seconds]
                for key in expired:
                    entry = self._entries.pop(key)
                    entry.retired = True
                    if entry.store is not None and entry.active_requests == 0:
                        retired.append(entry.store)
                entry = self._entries.get(session_id)
                if entry is None:
                    while len(self._entries) >= self.max_sessions:
                        oldest_id = min(self._entries, key=lambda key: self._entries[key].last_access)
                        oldest = self._entries.pop(oldest_id)
                        oldest.retired = True
                        if oldest.store is not None and oldest.active_requests == 0:
                            retired.append(oldest.store)
                    entry = _SessionEntry(session_id=session_id, last_access=now)
                    self._entries[session_id] = entry
                else:
                    entry.last_access = now
                if request_state is not None:
                    active_entries = request_state.setdefault('trace_session_entries', [])
                    if not any(active is entry for active in active_entries):
                        entry.active_requests += 1
                        active_entries.append(entry)
            await _close_stores(retired)

            async with entry.create_lock:
                async with self._lock:
                    if self._entries.get(session_id) is not entry:
                        continue
                    entry.last_access = self._monotonic()
                    if entry.store is not None:
                        return entry.store
                store = await factory()
                async with self._lock:
                    if self._entries.get(session_id) is entry:
                        entry.last_access = self._monotonic()
                        if store is not None:
                            entry.store = store
                        return store
                if store is not None:
                    await _close_stores([store])

    async def release(self, entry: _SessionEntry) -> None:
        """Release a request lease and close a retired store when its last request ends."""
        store = None
        async with self._lock:
            entry.active_requests = max(0, entry.active_requests - 1)
            if entry.retired and entry.active_requests == 0:
                store, entry.store = entry.store, None
        if store is not None:
            await _close_stores([store])

    async def detach_all(self) -> list[Any]:
        """Remove every entry, deferring active stores until their request leases end."""
        stores: list[Any] = []
        async with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
            for entry in entries:
                entry.retired = True
                if entry.store is not None and entry.active_requests == 0:
                    stores.append(entry.store)
                    entry.store = None
        return stores

    async def close_all(self) -> None:
        """Retire all sessions and close their stores."""
        async with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
            stores = [entry.store for entry in entries if entry.store is not None]
            for entry in entries:
                entry.retired = True
                entry.store = None
        await _close_stores(stores)


async def _close_stores(stores: list[Any]) -> None:
    results = await asyncio.gather(*(store.close() for store in stores), return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            exc = result
            logger.opt(exception=True).warning('Could not close a retired Traces session store: {}', exc)


def _scope_state(scope: Scope) -> dict[str, Any]:
    """Normalize scope state to the mutable mapping Starlette Request expects."""
    state = scope.get('state')
    if isinstance(state, dict):
        return state
    backing = getattr(state, '_state', None)
    if isinstance(backing, dict):
        scope['state'] = backing
        return backing
    state_dict: dict[str, Any] = {}
    scope['state'] = state_dict
    return state_dict


class TraceSessionMiddleware:
    """Verify the opaque session cookie and add it to Traces and finder responses."""

    def __init__(self, app: ASGIApp, *, app_state: Any) -> None:
        self.app = app
        self.app_state = app_state

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return
        path = scope.get('path', '')
        if path != '/traces' and not path.startswith('/traces/') and path != '/find' and not path.startswith('/find/'):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get('headers', ()))
        raw_cookie = headers.get(b'cookie', b'').decode('latin-1')
        cookie_value = next(
            (
                part.strip().partition('=')[2]
                for part in raw_cookie.split(';')
                if part.strip().startswith(f'{SESSION_COOKIE}=')
            ),
            None,
        )
        signing_key = self.app_state.dashboard_session_signing_key
        session_id = verified_session_id(cookie_value, signing_key)
        fresh_cookie = session_id is None
        if fresh_cookie:
            session_id = secrets.token_urlsafe(32)
        state = _scope_state(scope)
        state['dashboard_session_id'] = session_id

        async def send_with_cookie(message: Message) -> None:
            if message['type'] == 'http.response.start' and fresh_cookie:
                cookie = (
                    f'{SESSION_COOKIE}={signed_session_cookie(session_id, signing_key)}; Path=/; HttpOnly; SameSite=Lax'
                )
                if scope.get('scheme') == 'https':
                    cookie += '; Secure'
                message = dict(message)
                message['headers'] = [*message.get('headers', []), (b'set-cookie', cookie.encode('latin-1'))]
            await send(message)

        try:
            await self.app(scope, receive, send_with_cookie)
        finally:
            entries = _scope_state(scope).pop('trace_session_entries', [])
            for entry in entries:
                await self.app_state.trace_sessions.release(entry)
