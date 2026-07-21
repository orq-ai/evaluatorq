"""
Tracing context utilities.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@dataclass
class TracingContext:
    """Context for tracing an evaluation run."""

    run_id: str
    """Unique identifier for the evaluation run"""

    run_name: str
    """Human-readable name for the evaluation run"""

    enabled: bool
    """Whether tracing is enabled"""

    parent_context: Any | None = None
    """Parent OTEL context, if any"""

    trace_type: str = 'evaluatorq'
    """Trace type identifier for ``orq.trace_type`` span attribute"""


def generate_run_id() -> str:
    """Generate a unique run ID for an evaluation run."""
    return str(uuid.uuid4())


async def capture_parent_context() -> Any | None:  # noqa: RUF029
    """
    Capture the current OTEL context as a parent context.
    Returns None if OTEL is not available.
    """
    try:
        from opentelemetry import context

        return context.get_current()
    except ImportError:
        return None


@asynccontextmanager
async def tracing_session(run_name: str, *, trace_type: str = 'evaluatorq') -> AsyncGenerator[TracingContext, None]:
    """Framework-owned tracing lifecycle, shared by ``evaluatorq()``, ``red_team()``,
    and ``simulate()``.

    Initializes tracing on enter (idempotent) and flushes buffered spans on exit. It
    NEVER shuts the provider down: process-exit teardown is handled by the SDK
    ``TracerProvider`` atexit hook (``shutdown_on_exit=True``). This makes the lifecycle
    correct at any nesting depth and for sequential/concurrent runs — nothing tears the
    provider down while work is still in flight.

    Note this manages *lifecycle* only; it opens no spans — callers open their own
    spans against ``ctx.parent_context``. The process-lifetime batch processor can be
    tuned with ``ORQ_OTEL_MAX_QUEUE_SIZE``, ``ORQ_OTEL_SCHEDULE_DELAY_MS``, and
    ``ORQ_OTEL_MAX_BATCH_SIZE``. On session exit, force-flush uses
    ``ORQ_OTEL_FLUSH_TIMEOUT_MS`` and logs a warning if it times out, because spans may
    remain unexported.

    Known limitations (long-lived processes): a rotated ``ORQ_API_KEY`` only takes effect
    after a process restart (the exporter binds headers once at initialization and the
    provider is never rotated), and spans still buffered at a hard ``SIGKILL`` are lost —
    inherent to any batch exporter.

    Yields:
        The :class:`TracingContext` for the run.
    """
    from evaluatorq.tracing.setup import flush_tracing, init_tracing_if_needed

    enabled = await init_tracing_if_needed()
    ctx = TracingContext(
        run_id=generate_run_id(),
        run_name=run_name,
        enabled=enabled,
        parent_context=await capture_parent_context() if enabled else None,
        trace_type=trace_type,
    )
    try:
        yield ctx
    finally:
        if enabled:
            await flush_tracing()
