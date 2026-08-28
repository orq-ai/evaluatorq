"""Lifecycle contract tests for evaluatorq tracing."""

# ruff: noqa: S101, SLF001, S102

from __future__ import annotations

import asyncio
import importlib
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest

from evaluatorq.tracing import TracingContext, tracing_session
from evaluatorq.tracing import setup as tracing_setup

evaluatorq_module = importlib.import_module('evaluatorq.evaluatorq')


async def _enter_and_exit_session() -> None:
    async with tracing_session('concurrent-run'):
        pass


@asynccontextmanager
async def _enabled_test_session(*_args: object, **_kwargs: object):  # noqa: RUF029
    yield TracingContext(
        run_id='session-run',
        run_name='session-run',
        enabled=True,
        parent_context=None,
        trace_type='evaluatorq',
    )


@pytest.mark.asyncio
async def test_tracing_session_initializes_yields_context_and_flushes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized = AsyncMock(return_value=True)
    flushed = AsyncMock()
    monkeypatch.setattr('evaluatorq.tracing.setup.init_tracing_if_needed', initialized)
    monkeypatch.setattr('evaluatorq.tracing.setup.flush_tracing', flushed)

    async with tracing_session('red-team', trace_type='redteam') as context:
        assert context.enabled is True
        assert context.run_name == 'red-team'
        assert context.trace_type == 'redteam'

    initialized.assert_awaited_once()
    flushed.assert_awaited_once()


@pytest.mark.asyncio
async def test_tracing_session_flushes_even_when_body_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr('evaluatorq.tracing.setup.init_tracing_if_needed', AsyncMock(return_value=True))
    flushed = AsyncMock()
    monkeypatch.setattr('evaluatorq.tracing.setup.flush_tracing', flushed)

    class BodyError(Exception):
        pass

    with pytest.raises(BodyError):
        async with tracing_session('red-team'):
            raise BodyError

    flushed.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_sessions_never_call_private_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown = AsyncMock()
    monkeypatch.setattr('evaluatorq.tracing.setup.init_tracing_if_needed', AsyncMock(return_value=False))
    monkeypatch.setattr('evaluatorq.tracing.setup._shutdown_tracing', shutdown)

    await asyncio.gather(*[_enter_and_exit_session() for _ in range(2)])

    shutdown.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_enabled_sessions_initialize_and_flush_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized = AsyncMock(return_value=True)
    flushed = AsyncMock()
    monkeypatch.setattr('evaluatorq.tracing.setup.init_tracing_if_needed', initialized)
    monkeypatch.setattr('evaluatorq.tracing.setup.flush_tracing', flushed)

    await asyncio.gather(*[_enter_and_exit_session() for _ in range(2)])

    assert initialized.await_count == 2
    assert flushed.await_count == 2


@pytest.mark.asyncio
async def test_evaluatorq_passes_the_yielded_session_context_to_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_contexts: list[TracingContext] = []

    async def capture_processing(*args: object) -> list[object]:  # noqa: RUF029
        tracing_context = args[6]
        assert isinstance(tracing_context, TracingContext)
        seen_contexts.append(tracing_context)
        return []

    monkeypatch.setattr(evaluatorq_module, 'tracing_session', _enabled_test_session, raising=False)
    monkeypatch.setattr(evaluatorq_module, 'process_data_point', capture_processing)

    await evaluatorq_module.evaluatorq(
        'session-run',
        data=[{'inputs': {'value': 1}}],
        jobs=[lambda _data, _row: None],
        print_results=False,
        _send_results=False,
    )

    assert seen_contexts == [
        TracingContext(
            run_id='session-run',
            run_name='session-run',
            enabled=True,
            parent_context=None,
            trace_type='evaluatorq',
        )
    ]


@pytest.mark.parametrize('raw', [None, '', '   ', 'invalid', '0', '-1'])
def test_env_int_uses_default_for_unset_or_non_positive_values(
    monkeypatch: pytest.MonkeyPatch, raw: str | None
) -> None:
    if raw is None:
        monkeypatch.delenv('X', raising=False)
    else:
        monkeypatch.setenv('X', raw)

    assert tracing_setup._env_int('X', 4096) == 4096


def test_env_int_returns_positive_parsed_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('X', '8192')

    assert tracing_setup._env_int('X', 4096) == 8192


@pytest.mark.parametrize('raw', ['', '   ', 'invalid', '0', '-1'])
def test_env_int_warns_on_set_but_invalid_value(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    warning = Mock()
    monkeypatch.setenv('X', raw)
    monkeypatch.setattr(tracing_setup.logger, 'warning', warning)

    assert tracing_setup._env_int('X', 4096) == 4096
    warning.assert_called_once()


def _fake_tracing_sdk(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Fake out the exporter and provider, returning the captured processor kwargs.

    Drives real ``init_tracing_if_needed`` so the values the documented
    ``ORQ_OTEL_*`` knobs resolve to are observed where they are actually used.
    """
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http import trace_exporter
    from opentelemetry.sdk import trace as sdk_trace
    from opentelemetry.sdk.trace import export as trace_export

    processor_options: dict[str, int] = {}

    class FakeSpanProcessor:
        def __init__(self, exporter: object, **kwargs: int) -> None:
            del exporter
            processor_options.update(kwargs)

    class FakeProvider:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def add_span_processor(self, processor: object) -> None:
            del processor

    monkeypatch.setattr(trace_export, 'BatchSpanProcessor', FakeSpanProcessor)
    monkeypatch.setattr(trace_exporter, 'OTLPSpanExporter', Mock())
    monkeypatch.setattr(sdk_trace, 'TracerProvider', FakeProvider)
    monkeypatch.setattr(trace, 'get_tracer', Mock())
    monkeypatch.setattr(trace, 'set_tracer_provider', Mock())
    monkeypatch.setattr(tracing_setup, '_sdk', None)
    monkeypatch.setattr(tracing_setup, '_tracer', None)
    monkeypatch.setattr(tracing_setup, '_is_initialized', False)
    monkeypatch.setattr(tracing_setup, '_initialization_attempted', False)
    # Opt out of the suite-wide export guard: this drives real setup with the
    # exporter and provider faked out above.
    monkeypatch.delenv('ORQ_DISABLE_TRACING', raising=False)
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'https://example.test')
    return processor_options


@pytest.mark.asyncio
async def test_initialization_uses_documented_batching_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the defaults published in docs/tracing.md and docs/configuration.md."""
    processor_options = _fake_tracing_sdk(monkeypatch)
    warning = Mock()
    monkeypatch.setattr(tracing_setup.logger, 'warning', warning)
    for name in (
        'ORQ_OTEL_MAX_QUEUE_SIZE',
        'ORQ_OTEL_MAX_BATCH_SIZE',
        'ORQ_OTEL_SCHEDULE_DELAY_MS',
    ):
        monkeypatch.delenv(name, raising=False)

    assert await tracing_setup.init_tracing_if_needed() is True
    assert processor_options == {
        'max_queue_size': 4096,
        'max_export_batch_size': 512,
        'schedule_delay_millis': 5000,
    }
    # The clamp warning is conditional: defaults must not trip it.
    warning.assert_not_called()


@pytest.mark.asyncio
async def test_initialization_caps_export_batch_size_to_queue_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor_options = _fake_tracing_sdk(monkeypatch)
    warning = Mock()
    monkeypatch.setattr(tracing_setup.logger, 'warning', warning)
    monkeypatch.setenv('ORQ_OTEL_MAX_QUEUE_SIZE', '100')
    monkeypatch.setenv('ORQ_OTEL_MAX_BATCH_SIZE', '200')
    monkeypatch.setenv('ORQ_OTEL_SCHEDULE_DELAY_MS', '300')

    assert await tracing_setup.init_tracing_if_needed() is True
    assert processor_options == {
        'max_queue_size': 100,
        'max_export_batch_size': 100,
        'schedule_delay_millis': 300,
    }
    # The clamp is a degraded path, so it announces itself.
    warning.assert_called_once()
    assert 'clamping' in warning.call_args.args[0]


@pytest.mark.asyncio
async def test_flush_tracing_uses_worker_thread_and_warns_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = Mock()
    to_thread = AsyncMock(return_value=False)
    warning = Mock()
    monkeypatch.setattr(tracing_setup, '_sdk', provider)
    monkeypatch.setattr(tracing_setup.asyncio, 'to_thread', to_thread)
    monkeypatch.setattr(tracing_setup.logger, 'warning', warning)
    monkeypatch.delenv('ORQ_OTEL_FLUSH_TIMEOUT_MS', raising=False)

    await tracing_setup.flush_tracing()

    to_thread.assert_awaited_once_with(provider.force_flush, 5000)
    warning.assert_called_once_with(
        'OTEL span flush timed out after {}ms; some spans may not have been exported.', 5000
    )


@pytest.mark.asyncio
async def test_flush_tracing_does_not_warn_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    warning = Mock()
    monkeypatch.setattr(tracing_setup, '_sdk', Mock())
    monkeypatch.setattr(tracing_setup.asyncio, 'to_thread', AsyncMock(return_value=True))
    monkeypatch.setattr(tracing_setup.logger, 'warning', warning)

    await tracing_setup.flush_tracing()

    warning.assert_not_called()


@pytest.mark.asyncio
async def test_flush_tracing_bounds_a_hanging_force_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SDK ignores the timeout it is handed, so ``asyncio.wait_for`` enforces it."""
    warning = Mock()

    async def never_returns(*_args: object, **_kwargs: object) -> bool:
        await asyncio.sleep(30)
        return True

    monkeypatch.setattr(tracing_setup, '_sdk', Mock())
    monkeypatch.setattr(tracing_setup.asyncio, 'to_thread', never_returns)
    monkeypatch.setattr(tracing_setup.logger, 'warning', warning)
    monkeypatch.setenv('ORQ_OTEL_FLUSH_TIMEOUT_MS', '10')

    await tracing_setup.flush_tracing()

    warning.assert_called_once_with(
        'OTEL span flush timed out after {}ms; some spans may not have been exported.', 10
    )


@pytest.mark.asyncio
async def test_flush_tracing_warns_when_force_flush_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    warning = Mock()
    monkeypatch.setattr(tracing_setup, '_sdk', Mock())
    monkeypatch.setattr(tracing_setup.asyncio, 'to_thread', AsyncMock(side_effect=RuntimeError('boom')))
    monkeypatch.setattr(tracing_setup.logger, 'warning', warning)

    await tracing_setup.flush_tracing()

    warning.assert_called_once()
    assert 'flush failed' in warning.call_args.args[0]


@pytest.mark.asyncio
async def test_shutdown_with_active_sdk_flushes_and_shuts_provider_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = Mock()
    flush = AsyncMock()
    monkeypatch.setattr(tracing_setup, '_sdk', provider)
    monkeypatch.setattr(tracing_setup, '_tracer', Mock())
    monkeypatch.setattr(tracing_setup, '_is_initialized', True)
    monkeypatch.setattr(tracing_setup, 'flush_tracing', flush)

    await tracing_setup._shutdown_tracing()

    flush.assert_awaited_once()
    provider.shutdown.assert_called_once()
    assert tracing_setup._sdk is None
    assert tracing_setup._initialization_attempted is True
    assert await tracing_setup.init_tracing_if_needed() is False


@pytest.mark.asyncio
async def test_shutdown_without_sdk_disables_reinitialization_for_process_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enabled = Mock(return_value=False)
    monkeypatch.setattr(tracing_setup, '_sdk', None)
    monkeypatch.setattr(tracing_setup, '_tracer', Mock())
    monkeypatch.setattr(tracing_setup, '_is_initialized', True)
    monkeypatch.setattr(tracing_setup, '_initialization_attempted', True)
    monkeypatch.setattr(tracing_setup, 'is_tracing_enabled', enabled)

    await tracing_setup._shutdown_tracing()

    assert tracing_setup._initialization_attempted is True
    assert tracing_setup._tracer is None
    assert tracing_setup._is_initialized is False

    assert await tracing_setup.init_tracing_if_needed() is False

    enabled.assert_not_called()


def test_shutdown_tracing_is_not_publicly_importable() -> None:
    with pytest.raises(ImportError):
        exec('from evaluatorq.tracing import _shutdown_tracing')


@pytest.mark.asyncio
async def test_the_suite_never_installs_a_live_span_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The autouse guard in the root conftest must hold even with a real key in
    the environment. Without it, the first test to reach setup installs a
    process-wide OTLP exporter and every later span is queued for upload to
    my.orq.ai, flushed at interpreter shutdown.
    """
    monkeypatch.setenv('ORQ_API_KEY', 'looks-real-enough')
    monkeypatch.setattr(tracing_setup, '_initialization_attempted', False)
    monkeypatch.setattr(tracing_setup, '_is_initialized', False)

    assert await tracing_setup.init_tracing_if_needed() is False
    assert tracing_setup._sdk is None
