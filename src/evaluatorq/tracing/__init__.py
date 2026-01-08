"""
OpenTelemetry tracing support for evaluatorq.

Tracing is automatically enabled when:
1. OTEL_EXPORTER_OTLP_ENDPOINT is set (explicit endpoint)
2. ORQ_API_KEY is set (traces sent to Orq platform automatically)

Tracing can be explicitly disabled by setting:
- ORQ_DISABLE_TRACING=1 or ORQ_DISABLE_TRACING=true

Set ORQ_DEBUG=1 to enable debug logging for tracing setup.
"""

from .context import TracingContext, capture_parent_context, generate_run_id
from .setup import (
    flush_tracing,
    get_tracer,
    init_tracing_if_needed,
    is_tracing_enabled,
    is_tracing_initialized,
    shutdown_tracing,
)
from .spans import (
    set_evaluation_attributes,
    set_job_name_attribute,
    with_evaluation_span,
    with_job_span,
)

__all__ = [
    # Setup functions
    "init_tracing_if_needed",
    "flush_tracing",
    "shutdown_tracing",
    "get_tracer",
    "is_tracing_enabled",
    "is_tracing_initialized",
    # Context functions
    "TracingContext",
    "capture_parent_context",
    "generate_run_id",
    # Span functions
    "with_job_span",
    "with_evaluation_span",
    "set_evaluation_attributes",
    "set_job_name_attribute",
]
