"""OTel span helpers for the agent simulation module.

Domain-specific span builders (with_simulation_span, with_llm_span) live here.
Generic recording utilities are imported from evaluatorq.common.tracing.

Span hierarchy:
    orq.simulation.pipeline (root)
      ├── orq.simulation.run (per datapoint)
      │   ├── orq.simulation.first_message_generation
      │   └── orq.simulation.turn (per turn)
      │       ├── orq.simulation.target_call
      │       ├── orq.simulation.judge_evaluation
      │       └── orq.simulation.user_simulator_call
      └── chat/responses {model}  (LLM client spans, GenAI semconv)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.common.tracing import AttrMap, AttrValue
from evaluatorq.tracing.setup import get_tracer

if TYPE_CHECKING:
	from collections.abc import AsyncGenerator

	from opentelemetry.trace import Span

_otel_import_warned = False

_PROVIDER_ALIASES: dict[str, str] = {
	"azure": "azure.ai.openai",
}


def _derive_provider(model: str) -> str:
	if "/" in model:
		prefix = model.split("/", 1)[0]
		return _PROVIDER_ALIASES.get(prefix, prefix)
	return "openai"


@asynccontextmanager
async def with_simulation_span(  # noqa: RUF029
	name: str,
	attributes: AttrMap | None = None,
) -> AsyncGenerator[Span | None, None]:
	"""Execute a block within a simulation span (SpanKind.INTERNAL).

	Records exceptions (including asyncio.CancelledError) and sets span status.

	Yields:
		The active span, or None when tracing is disabled.
	"""
	tracer = get_tracer()
	if tracer is None:
		yield None
		return

	try:
		from opentelemetry.trace import SpanKind, Status, StatusCode
	except ImportError as exc:
		global _otel_import_warned
		if not _otel_import_warned:
			logger.warning("OpenTelemetry import failed; tracing disabled: %s", exc)
			_otel_import_warned = True
		yield None
		return

	clean_attrs: dict[str, AttrValue] = {
		k: v for k, v in (attributes or {}).items() if v is not None
	}

	with tracer.start_as_current_span(
		name,
		kind=SpanKind.INTERNAL,
		attributes=clean_attrs,
	) as span:
		try:
			yield span
			span.set_status(Status(StatusCode.OK))
		except BaseException as e:
			span.set_status(Status(StatusCode.ERROR, str(e)))
			span.record_exception(e)
			span.set_attribute("error.type", type(e).__name__)
			raise


@asynccontextmanager
async def with_llm_span(  # noqa: RUF029
	*,
	model: str,
	operation: str = "chat",
	provider: str | None = None,
	temperature: float | None = None,
	max_tokens: int | None = None,
	purpose: str | None = None,
) -> AsyncGenerator[Span | None, None]:
	"""Execute a block within a GenAI LLM span (SpanKind.CLIENT).

	Span name is "{operation} {model}". Sets orq.simulation.llm_purpose when
	purpose is provided.

	Yields:
		The active span, or None when tracing is disabled.
	"""
	tracer = get_tracer()
	if tracer is None:
		yield None
		return

	try:
		from opentelemetry.trace import SpanKind, Status, StatusCode
	except ImportError as exc:
		global _otel_import_warned
		if not _otel_import_warned:
			logger.warning("OpenTelemetry import failed; tracing disabled: %s", exc)
			_otel_import_warned = True
		yield None
		return

	resolved_provider = provider or _derive_provider(model)
	span_name = f"{operation} {model}"

	attrs: dict[str, Any] = {
		"gen_ai.operation.name": operation,
		"gen_ai.system": resolved_provider,
		"gen_ai.provider.name": resolved_provider,
		"gen_ai.request.model": model,
	}
	if temperature is not None:
		attrs["gen_ai.request.temperature"] = temperature
	if max_tokens is not None:
		attrs["gen_ai.request.max_tokens"] = max_tokens
	if purpose:
		# Domain-neutral key (parity with openresponses.with_llm_span) so the
		# platform can query orq.llm.purpose across all domains; the legacy
		# orq.simulation.llm_purpose is emitted alongside for dashboard back-compat.
		attrs["orq.llm.purpose"] = purpose
		attrs["orq.simulation.llm_purpose"] = purpose

	with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT, attributes=attrs) as span:
		try:
			yield span
			span.set_status(Status(StatusCode.OK))
		except BaseException as e:
			span.set_status(Status(StatusCode.ERROR, str(e)))
			span.record_exception(e)
			span.set_attribute("error.type", type(e).__name__)
			raise
