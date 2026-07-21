"""
OpenTelemetry SDK setup with lazy initialization and auto-detection.

Tracing is automatically enabled when:
1. OTEL_EXPORTER_OTLP_ENDPOINT is set (explicit endpoint)
2. ORQ_API_KEY is set (traces sent to Orq platform automatically)

Tracing can be explicitly disabled by setting:
- ORQ_DISABLE_TRACING=1 or ORQ_DISABLE_TRACING=true

When ORQ_API_KEY is provided:
- Uses ORQ_BASE_URL with /v2/otel path appended for the OTEL endpoint
- Falls back to https://my.orq.ai/v2/otel if ORQ_BASE_URL is not set
- Authorization header is automatically added with the API key

Set ORQ_DEBUG=1 to enable debug logging for tracing setup.

Provider lifecycle and batching:
- The ``TracerProvider`` lives for the process. Runs force-flush their buffered spans;
  the SDK atexit hook performs provider shutdown, so runtime code must not shut it down.
- Tune the long-lived batch processor with ``ORQ_OTEL_MAX_QUEUE_SIZE`` (default 4096),
  ``ORQ_OTEL_SCHEDULE_DELAY_MS`` (default 5000), and ``ORQ_OTEL_MAX_BATCH_SIZE``
  (default 512).
- ``ORQ_OTEL_FLUSH_TIMEOUT_MS`` controls per-run force-flush (default 5000). A timeout
  logs a warning because some spans may not have been exported.
- Exporter headers are bound at initialization: changing ``ORQ_API_KEY`` requires a
  process restart. A hard ``SIGKILL`` can lose spans still buffered by the batch exporter.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

# Module-level state
_sdk: Any = None
_tracer: Tracer | None = None
_is_initialized = False
_initialization_attempted = False


def _env_int(name: str, default: int) -> int:
    """Read a positive int from the environment, falling back to *default*."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _is_tracing_explicitly_disabled() -> bool:
    """Check if tracing is explicitly disabled via ORQ_DISABLE_TRACING."""
    disable_value = os.environ.get('ORQ_DISABLE_TRACING', '')
    return disable_value in ('1', 'true')


def is_tracing_enabled() -> bool:
    """
    Check if tracing should be enabled based on environment variables.
    Tracing is enabled when ORQ_API_KEY or OTEL_EXPORTER_OTLP_ENDPOINT is set,
    unless explicitly disabled via ORQ_DISABLE_TRACING.
    """
    if _is_tracing_explicitly_disabled():
        return False
    return bool(os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT') or os.environ.get('ORQ_API_KEY'))


def _get_otlp_endpoint() -> str | None:
    """
    Get the default OTLP endpoint based on environment configuration.

    Priority:
    1. OTEL_EXPORTER_OTLP_ENDPOINT - explicit endpoint override
    2. ORQ_BASE_URL - use as-is with /v2/otel path appended
    3. Default to production my.orq.ai when only ORQ_API_KEY is set
    """
    # Explicit endpoint takes precedence
    explicit_endpoint = os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT')
    if explicit_endpoint:
        return explicit_endpoint

    # If ORQ_API_KEY is set, use ORQ_BASE_URL or default endpoint
    if os.environ.get('ORQ_API_KEY'):
        base_url = os.environ.get('ORQ_BASE_URL')
        if base_url:
            # Use ORQ_BASE_URL as-is; OTEL ingest is served on the same my.orq.ai
            # host as the rest of the Orq API (verified end-to-end: spans POSTed to
            # my.orq.ai/v2/otel return 200 and are queryable). No my.→api. rewrite.
            return f'{base_url.rstrip("/")}/v2/otel'
        # Default to production endpoint (unified my.orq.ai host).
        return 'https://my.orq.ai/v2/otel'

    return None


async def init_tracing_if_needed() -> bool:  # noqa: RUF029
    """
    Initialize the OpenTelemetry SDK if not already initialized.
    Uses dynamic imports to handle optional dependencies gracefully.

    Returns:
        True if tracing was successfully initialized, False otherwise
    """
    global _sdk, _tracer, _is_initialized, _initialization_attempted

    if _initialization_attempted:
        return _is_initialized
    _initialization_attempted = True

    debug = os.environ.get('ORQ_DEBUG')

    if not is_tracing_enabled():
        if debug:
            if _is_tracing_explicitly_disabled():
                print('[evaluatorq] OTEL tracing disabled via ORQ_DISABLE_TRACING')
            else:
                print('[evaluatorq] OTEL tracing not enabled (no ORQ_API_KEY or OTEL_EXPORTER_OTLP_ENDPOINT)')
        return False

    endpoint = _get_otlp_endpoint()
    if not endpoint:
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Service name and version attribute keys
        SERVICE_NAME = 'service.name'
        SERVICE_VERSION = 'service.version'

        # Ensure endpoint has the traces path
        traces_endpoint = endpoint if endpoint.endswith('/v1/traces') else f'{endpoint}/v1/traces'

        # Build headers for the exporter
        headers: dict[str, str] = {}
        api_key = os.environ.get('ORQ_API_KEY')

        # Add Authorization header only for verified orq.ai domains
        # Use proper domain validation to prevent leaking API keys to malicious endpoints
        from urllib.parse import urlparse

        parsed_endpoint = urlparse(traces_endpoint)
        is_orq_domain = parsed_endpoint.netloc.endswith('.orq.ai') or parsed_endpoint.netloc == 'orq.ai'
        if api_key and is_orq_domain:
            headers['Authorization'] = f'Bearer {api_key}'

        # Parse OTEL_EXPORTER_OTLP_HEADERS if present
        # Format: "key1=value1,key2=value2" - values may contain "=" (e.g., JWT tokens)
        env_headers = os.environ.get('OTEL_EXPORTER_OTLP_HEADERS')
        if env_headers:
            for pair in env_headers.split(','):
                eq_index = pair.find('=')
                if eq_index > 0:
                    key = pair[:eq_index].strip()
                    value = pair[eq_index + 1 :].strip()
                    if key and value:
                        headers[key] = value

        resource = Resource.create({
            SERVICE_NAME: os.environ.get('OTEL_SERVICE_NAME', 'evaluatorq'),
            SERVICE_VERSION: os.environ.get('OTEL_SERVICE_VERSION', '1.0.0'),
        })

        if debug:
            if os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT'):
                source = 'OTEL_EXPORTER_OTLP_ENDPOINT'
            elif os.environ.get('ORQ_BASE_URL'):
                source = 'ORQ_BASE_URL'
            else:
                source = 'default (ORQ_API_KEY)'
            print('[evaluatorq] OTEL tracing enabled')
            print(f'[evaluatorq] OTEL endpoint: {traces_endpoint}')
            print(f'[evaluatorq] OTEL endpoint source: {source}')
            if 'Authorization' in headers:
                print('[evaluatorq] Authorization: Bearer ***')

        exporter = OTLPSpanExporter(
            endpoint=traces_endpoint,
            headers=headers,
            timeout=5,  # 5 second timeout for telemetry
        )

        # Use BatchSpanProcessor to export spans asynchronously in batches.
        # Queue/scheduling are env-tunable: in a long-lived process (e.g. the
        # dashboard) that runs many red-team/simulation runs without ever tearing
        # the provider down, the default 2048-span queue can overflow and silently
        # drop spans. Larger defaults + env overrides reduce that risk.
        max_queue_size = _env_int('ORQ_OTEL_MAX_QUEUE_SIZE', 4096)
        requested_batch_size = _env_int('ORQ_OTEL_MAX_BATCH_SIZE', 512)
        span_processor = BatchSpanProcessor(
            exporter,
            max_queue_size=max_queue_size,
            schedule_delay_millis=_env_int('ORQ_OTEL_SCHEDULE_DELAY_MS', 5000),
            max_export_batch_size=min(requested_batch_size, max_queue_size),
        )

        provider = TracerProvider(resource=resource)
        provider.add_span_processor(span_processor)
        trace.set_tracer_provider(provider)

        _sdk = provider
        _tracer = trace.get_tracer('evaluatorq')
        _is_initialized = True

        return True

    except ImportError as e:
        # OTEL packages not installed
        if debug:
            print(f'[evaluatorq] OpenTelemetry not available: {e}')
        return False
    except Exception as e:
        if debug:
            print(f'[evaluatorq] OpenTelemetry initialization failed: {e}')
        return False


async def flush_tracing() -> None:
    """
    Force flush all pending spans, blocking until export completes or times out.

    ``force_flush`` is a synchronous, blocking SDK call, so it runs on a worker
    thread to avoid stalling the event loop. A ``False`` return means the flush
    timed out with spans still unexported — surfaced as a warning rather than
    silently dropped.
    """
    if _sdk is None:
        return
    provider = _sdk  # TracerProvider
    timeout_ms = _env_int('ORQ_OTEL_FLUSH_TIMEOUT_MS', 5000)
    try:
        ok = await asyncio.to_thread(provider.force_flush, timeout_ms)
        if ok is False:
            logger.warning(
                'OTEL span flush timed out after {}ms; some spans may not have been exported.',
                timeout_ms,
            )
    except Exception as e:
        logger.debug('Error flushing traces: {}', e)


async def _shutdown_tracing() -> None:
    """
    Tear down the OpenTelemetry SDK and reset module state.

    DO NOT call this during a run. Process-exit teardown is handled automatically
    by the SDK ``TracerProvider`` atexit hook (``shutdown_on_exit=True``). Calling
    it mid-run tears the global provider down and black-holes every span emitted
    afterwards. Retained only for the explicit-teardown/test path; resets
    ``_initialization_attempted`` so a subsequent ``init_tracing_if_needed()`` can
    re-initialize.
    """
    global _sdk, _tracer, _is_initialized, _initialization_attempted

    if _sdk is not None:
        try:
            await flush_tracing()
            provider = _sdk  # TracerProvider
            provider.shutdown()
        except Exception as e:
            logger.debug('Error shutting down tracing: {}', e)

    _sdk = None
    _tracer = None
    _is_initialized = False
    _initialization_attempted = False


def get_tracer() -> Tracer | None:
    """
    Get the tracer instance if tracing is initialized.
    Returns None if tracing is not enabled.
    """
    return _tracer


def is_tracing_initialized() -> bool:
    """Check if tracing has been successfully initialized."""
    return _is_initialized
