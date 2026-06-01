"""Tests for evaluatorq.common.async_utils (await_maybe, warn_if_sync_hooks)."""

from __future__ import annotations

import warnings

import pytest

from evaluatorq.common.async_utils import await_maybe, warn_if_sync_hooks


@pytest.mark.asyncio
async def test_await_maybe_returns_plain_value_unchanged():
    assert await await_maybe(42) == 42
    assert await await_maybe(None) is None
    assert await await_maybe(False) is False


@pytest.mark.asyncio
async def test_await_maybe_awaits_coroutine():
    async def coro() -> str:
        return 'done'

    assert await await_maybe(coro()) == 'done'


@pytest.mark.asyncio
async def test_await_maybe_awaits_arbitrary_awaitable():
    class Awaitable:
        def __await__(self):
            yield
            return 'awaited'

    assert await await_maybe(Awaitable()) == 'awaited'


@pytest.mark.asyncio
async def test_await_maybe_propagates_sync_exception():
    def boom() -> int:
        raise ValueError('sync boom')

    # A sync callable that raises does so before await_maybe sees a value.
    with pytest.raises(ValueError, match='sync boom'):
        await await_maybe(boom())


@pytest.mark.asyncio
async def test_await_maybe_propagates_async_exception():
    async def boom() -> int:
        raise ValueError('async boom')

    with pytest.raises(ValueError, match='async boom'):
        await await_maybe(boom())


def test_warn_if_sync_hooks_warns_for_sync_method():
    class SyncHooks:
        def on_confirm(self):  # noqa: ANN201
            return True

    with pytest.warns(DeprecationWarning, match='on_confirm'):
        warn_if_sync_hooks(SyncHooks(), ('on_confirm',))


def test_warn_if_sync_hooks_silent_for_async_method():
    class AsyncHooks:
        async def on_confirm(self):  # noqa: ANN201
            return True

    with warnings.catch_warnings():
        warnings.simplefilter('error')  # any warning becomes an error
        warn_if_sync_hooks(AsyncHooks(), ('on_confirm',))


def test_warn_if_sync_hooks_lists_only_sync_methods():
    class MixedHooks:
        async def on_run_start(self):  # noqa: ANN201
            return None

        def on_run_complete(self):  # noqa: ANN201
            return None

    with pytest.warns(DeprecationWarning) as record:
        warn_if_sync_hooks(MixedHooks(), ('on_run_start', 'on_run_complete'))
    msg = str(record[0].message)
    assert 'on_run_complete' in msg
    assert 'on_run_start' not in msg


def test_warn_if_sync_hooks_ignores_missing_methods():
    class Partial:
        async def on_confirm(self):  # noqa: ANN201
            return True

    # Method not implemented at all -> not flagged (getattr returns None).
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        warn_if_sync_hooks(Partial(), ('on_confirm', 'on_run_complete'))
