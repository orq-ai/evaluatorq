from __future__ import annotations

import asyncio
from typing import Any

import pytest

from evaluatorq.dashboard.trace_finder.sessions import (
    TraceSessionRegistry,
    signed_session_cookie,
    verified_session_id,
)


class Store:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_session_cookie_requires_a_valid_signature() -> None:
    key = b'test signing key'
    session_id = 'a' * 43
    cookie = signed_session_cookie(session_id, key)

    assert verified_session_id(cookie, key) == session_id
    assert verified_session_id(f'{cookie}tampered', key) is None
    assert verified_session_id(cookie, b'other key') is None
    assert verified_session_id('malformed', key) is None


@pytest.mark.asyncio
async def test_session_registry_expires_idle_entries_and_evicts_lru() -> None:
    now = 0.0
    registry = TraceSessionRegistry(max_sessions=2, ttl_seconds=10, monotonic=lambda: now)
    stores = {name: Store(name) for name in ('a', 'b', 'c', 'a-new')}

    async def build(name: str) -> Store:
        return stores[name]

    await registry.get('a', lambda: build('a'))
    await registry.get('b', lambda: build('b'))
    now = 1.0
    await registry.get('a', lambda: build('a-new'))
    await registry.get('c', lambda: build('c'))
    assert stores['a'].closed is False
    assert stores['b'].closed is True

    now = 12.0
    await registry.get('a', lambda: build('a-new'))
    assert stores['a-new'].closed is False
    assert stores['c'].closed is True

    await registry.close_all()
    assert stores['a-new'].closed is True


@pytest.mark.asyncio
async def test_evicted_in_flight_creation_cannot_replace_the_current_entry() -> None:
    registry = TraceSessionRegistry(max_sessions=1)
    entered = asyncio.Event()
    release = asyncio.Event()
    old_store = Store('old')
    current_store = Store('current')
    calls = 0

    async def slow_factory() -> Store:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()
            return old_store
        return current_store

    first = asyncio.create_task(registry.get('session-a', slow_factory))
    await entered.wait()
    await registry.get('session-b', lambda: asyncio.sleep(0, result=Store('other')))
    current = await registry.get('session-a', lambda: asyncio.sleep(0, result=current_store))
    release.set()

    assert await first is current_store
    assert current is current_store
    assert old_store.closed is True
    assert current_store.closed is False


@pytest.mark.asyncio
async def test_settings_retirement_defers_close_until_active_request_releases_store() -> None:
    registry = TraceSessionRegistry()
    store = Store('active')
    request_state: dict[str, Any] = {}

    assert await registry.get('session-a', lambda: asyncio.sleep(0, result=store), request_state=request_state) is store
    assert await registry.detach_all() == []
    assert store.closed is False

    await registry.release(request_state['trace_session_entries'][0])
    assert store.closed is True


@pytest.mark.asyncio
async def test_expired_active_entry_stays_available_until_last_request_releases() -> None:
    now = 0.0
    registry = TraceSessionRegistry(max_sessions=2, ttl_seconds=10, monotonic=lambda: now)
    store = Store('active')
    first_state: dict[str, Any] = {}
    second_state: dict[str, Any] = {}

    assert await registry.get(
        'session-a', lambda: asyncio.sleep(0, result=store), request_state=first_state
    ) is store
    now = 1.0
    assert await registry.get(
        'session-a', lambda: asyncio.sleep(0, result=Store('replacement')), request_state=second_state
    ) is store
    now = 11.0
    assert store.closed is False

    await registry.release(first_state['trace_session_entries'][0])
    assert store.closed is False
    await registry.release(second_state['trace_session_entries'][0])
    assert store.closed is True


@pytest.mark.asyncio
async def test_capacity_with_only_active_entries_preserves_them_until_release() -> None:
    registry = TraceSessionRegistry(max_sessions=1)
    store = Store('active')
    request_state: dict[str, Any] = {}

    assert await registry.get(
        'session-a', lambda: asyncio.sleep(0, result=store), request_state=request_state
    ) is store
    assert await registry.get('session-b', lambda: asyncio.sleep(0, result=Store('other'))) is None
    assert store.closed is False

    await registry.release(request_state['trace_session_entries'][0])
    replacement = Store('replacement')
    assert await registry.get('session-b', lambda: asyncio.sleep(0, result=replacement)) is replacement
    assert store.closed is True
