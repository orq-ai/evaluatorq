"""Lifecycle contract tests for evaluatorq tracing."""

# ruff: noqa: S101, SLF001, S102

from __future__ import annotations

import asyncio
import builtins
import importlib
from collections.abc import Mapping, Sequence
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


# The ORQ_OTEL_* int knobs read through common.env_config.env_int (min_value=1); the reader's
# contract (unset/empty/invalid/non-positive -> default + warning) is covered in
# tests/common/test_env_config.py. The tracing-layer resolution is exercised end-to-end below.


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
    # Opt out of the suite-wide export guard: exporter and provider are faked above.
    monkeypatch.delenv('ORQ_DISABLE_TRACING', raising=False)
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'https://example.test')
    return processor_options


def _capture_tracing_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, dict[str, object]]:
    """Patch the SDK classes and capture exporter construction kwargs.

    Processor kwargs are not captured here — the batching knobs they carry are
    covered separately via ``_fake_tracing_sdk``.
    """
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http import trace_exporter
    from opentelemetry.sdk import trace as sdk_trace
    from opentelemetry.sdk.trace import export as trace_export

    construction: dict[str, dict[str, object]] = {'exporter': {}}

    class FakeExporter:
        def __init__(self, **kwargs: object) -> None:
            construction['exporter'].update(kwargs)

    class FakeSpanProcessor:
        def __init__(self, exporter: object, **kwargs: object) -> None:
            del exporter
            del kwargs

    class FakeProvider:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def add_span_processor(self, processor: object) -> None:
            del processor

    monkeypatch.setattr(trace_export, 'BatchSpanProcessor', FakeSpanProcessor)
    monkeypatch.setattr(trace_exporter, 'OTLPSpanExporter', FakeExporter)
    monkeypatch.setattr(sdk_trace, 'TracerProvider', FakeProvider)
    monkeypatch.setattr(trace, 'get_tracer', Mock())
    monkeypatch.setattr(trace, 'set_tracer_provider', Mock())
    monkeypatch.setattr(tracing_setup, '_sdk', None)
    monkeypatch.setattr(tracing_setup, '_tracer', None)
    monkeypatch.setattr(tracing_setup, '_is_initialized', False)
    monkeypatch.setattr(tracing_setup, '_initialization_attempted', False)
    monkeypatch.delenv('ORQ_DISABLE_TRACING', raising=False)
    return construction


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('endpoint', 'expected_endpoint', 'expected_headers'),
    [
        (
            'https://orq.ai/v2/otel',
            'https://orq.ai/v2/otel/v1/traces',
            {'Authorization': 'Bearer test-key'},
        ),
        (
            'https://tenant.orq.ai/v2/otel/v1/traces',
            'https://tenant.orq.ai/v2/otel/v1/traces',
            {'Authorization': 'Bearer test-key'},
        ),
        ('https://collector.example/v2/otel', 'https://collector.example/v2/otel/v1/traces', {}),
    ],
)
async def test_initialization_resolves_endpoint_and_domain_scoped_auth_headers(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    expected_endpoint: str,
    expected_headers: dict[str, str],
) -> None:
    construction = _capture_tracing_construction(monkeypatch)
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', endpoint)
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    monkeypatch.delenv('OTEL_EXPORTER_OTLP_HEADERS', raising=False)

    assert await tracing_setup.init_tracing_if_needed() is True
    assert construction['exporter'] == {
        'endpoint': expected_endpoint,
        'headers': expected_headers,
        'timeout': 5,
    }


@pytest.mark.asyncio
async def test_initialization_parses_headers_with_equals_and_skips_empty_or_malformed_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction = _capture_tracing_construction(monkeypatch)
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'https://collector.example')
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_HEADERS', 'token=abc=def,malformed,=empty-key,empty-value=, spaced = value ')

    assert await tracing_setup.init_tracing_if_needed() is True
    assert construction['exporter']['headers'] == {
        'token': 'abc=def',
        'spaced': 'value',
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('endpoint', 'base_url', 'expected_source', 'expected_traces_endpoint'),
    [
        ('https://collector.example', 'https://ignored.example', 'OTEL_EXPORTER_OTLP_ENDPOINT', 'https://collector.example/v1/traces'),
        (None, 'https://collector.example', 'ORQ_BASE_URL', 'https://collector.example/v2/otel/v1/traces'),
        (None, None, 'default (ORQ_API_KEY)', 'https://my.orq.ai/v2/otel/v1/traces'),
    ],
)
async def test_initialization_debug_prints_endpoint_source(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str | None,
    base_url: str | None,
    expected_source: str,
    expected_traces_endpoint: str,
) -> None:
    _capture_tracing_construction(monkeypatch)
    if endpoint is None:
        monkeypatch.delenv('OTEL_EXPORTER_OTLP_ENDPOINT', raising=False)
    else:
        monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', endpoint)
    if base_url is None:
        monkeypatch.delenv('ORQ_BASE_URL', raising=False)
    else:
        monkeypatch.setenv('ORQ_BASE_URL', base_url)
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    monkeypatch.setenv('ORQ_DEBUG', '1')
    monkeypatch.delenv('OTEL_EXPORTER_OTLP_HEADERS', raising=False)
    printed = Mock()
    monkeypatch.setattr(builtins, 'print', printed)

    assert await tracing_setup.init_tracing_if_needed() is True
    expected_prints = [
        ('[evaluatorq] OTEL tracing enabled',),
        (f'[evaluatorq] OTEL endpoint: {expected_traces_endpoint}',),
        (f'[evaluatorq] OTEL endpoint source: {expected_source}',),
    ]
    if expected_source == 'default (ORQ_API_KEY)':
        expected_prints.append(('[evaluatorq] Authorization: Bearer ***',))
    assert [call.args for call in printed.call_args_list] == expected_prints


@pytest.mark.asyncio
async def test_initialization_returns_false_when_tracing_is_not_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracing_setup, '_sdk', None)
    monkeypatch.setattr(tracing_setup, '_tracer', None)
    monkeypatch.setattr(tracing_setup, '_is_initialized', False)
    monkeypatch.setattr(tracing_setup, '_initialization_attempted', False)
    monkeypatch.delenv('ORQ_DISABLE_TRACING', raising=False)
    monkeypatch.delenv('ORQ_API_KEY', raising=False)
    monkeypatch.delenv('OTEL_EXPORTER_OTLP_ENDPOINT', raising=False)

    assert await tracing_setup.init_tracing_if_needed() is False


@pytest.mark.asyncio
async def test_initialization_returns_false_when_endpoint_resolution_is_falsy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracing_setup, '_sdk', None)
    monkeypatch.setattr(tracing_setup, '_tracer', None)
    monkeypatch.setattr(tracing_setup, '_is_initialized', False)
    monkeypatch.setattr(tracing_setup, '_initialization_attempted', False)
    monkeypatch.delenv('ORQ_DISABLE_TRACING', raising=False)
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    monkeypatch.delenv('OTEL_EXPORTER_OTLP_ENDPOINT', raising=False)
    monkeypatch.setattr(tracing_setup, '_get_otlp_endpoint', Mock(return_value=None))

    assert await tracing_setup.init_tracing_if_needed() is False


@pytest.mark.asyncio
async def test_initialization_returns_false_when_opentelemetry_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracing_setup, '_sdk', None)
    monkeypatch.setattr(tracing_setup, '_tracer', None)
    monkeypatch.setattr(tracing_setup, '_is_initialized', False)
    monkeypatch.setattr(tracing_setup, '_initialization_attempted', False)
    monkeypatch.delenv('ORQ_DISABLE_TRACING', raising=False)
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'https://collector.example')

    real_import = builtins.__import__

    def fail_opentelemetry_import(
        name: str,
        globals_: Mapping[str, object] | None = None,
        locals_: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = None,
        level: int = 0,
    ) -> object:
        if name == 'opentelemetry':
            raise ImportError('OpenTelemetry unavailable')
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', fail_opentelemetry_import)

    assert await tracing_setup.init_tracing_if_needed() is False


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
