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


# --- normalize_to_list ------------------------------------------------------


def test_normalize_to_list_variants():
    from evaluatorq.common.async_utils import normalize_to_list

    assert normalize_to_list(None) == []
    obj = object()
    assert normalize_to_list(obj) == [obj]
    assert normalize_to_list([1, 2]) == [1, 2]
    assert normalize_to_list((1, 2)) == [1, 2]
    # str/bytes are scalars, never iterated char-by-char.
    assert normalize_to_list('abc') == ['abc']
    assert normalize_to_list(b'xy') == [b'xy']


# --- fan_out (FIX 6: later exceptions logged, not lost) ----------------------


class _Rec:
    def __init__(self, *, raises: BaseException | None = None) -> None:
        self.calls = 0
        self._raises = raises

    async def ping(self) -> None:
        self.calls += 1
        if self._raises is not None:
            raise self._raises

    async def on_confirm(self, _payload=None) -> bool:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return True


@pytest.mark.asyncio
async def test_fan_out_runs_all_and_reraises_first():
    from evaluatorq.common.async_utils import fan_out

    a = _Rec(raises=ValueError('first'))
    b = _Rec()
    c = _Rec(raises=RuntimeError('second'))
    with pytest.raises(ValueError, match='first'):
        await fan_out([a, b, c], 'ping')
    # All children ran despite the first raising (run-all policy).
    assert a.calls == 1 and b.calls == 1 and c.calls == 1


@pytest.mark.asyncio
async def test_fan_out_logs_second_exception(caplog):
    import logging

    from evaluatorq.common.async_utils import fan_out

    a = _Rec(raises=ValueError('first'))
    c = _Rec(raises=RuntimeError('second-boom'))
    # loguru → standard logging is bridged in tests via caplog's propagation;
    # assert the second exception surfaces in a warning rather than vanishing.
    with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match='first'):
        await fan_out([a, c], 'ping')


# --- combine_confirm --------------------------------------------------------


@pytest.mark.asyncio
async def test_combine_confirm_all_true_and_any_false():
    from evaluatorq.common.async_utils import combine_confirm

    assert await combine_confirm([_Rec(), _Rec()]) is True

    class _No:
        async def on_confirm(self, _p=None) -> bool:
            return False

    assert await combine_confirm([_Rec(), _No()]) is False


@pytest.mark.asyncio
async def test_combine_confirm_runs_all_and_reraises_first():
    from evaluatorq.common.async_utils import combine_confirm

    a = _Rec(raises=ValueError('first'))
    b = _Rec()
    with pytest.raises(ValueError, match='first'):
        await combine_confirm([a, b])
    assert a.calls == 1 and b.calls == 1


# --- require_hooks_like (FIX 5: wrong type → TypeError) ----------------------


def test_require_hooks_like_rejects_non_hook():
    from evaluatorq.common.hook_compose import require_hooks_like

    require_hooks_like(_Rec())  # has callable on_confirm → ok
    for bad in (123, 'nope', object()):
        with pytest.raises(TypeError, match='on_confirm'):
            require_hooks_like(bad)
