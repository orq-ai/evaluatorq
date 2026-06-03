"""OTel span helpers for the OpenResponses runtime.

Provides with_llm_span for the Responses API call path, and
record_openresponses_request/response helpers that record the full
Responses API payload alongside the standard gen_ai.* attributes.
Imports recording utilities from common/tracing.py; no simulation import.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.common.tracing import (
	_capture_message_content,
	record_llm_response,
	truncate_for_span,
)
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

	Mirrors simulation.tracing.with_llm_span without the simulation dependency.
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


def record_openresponses_request(span: Span | None, payload: dict[str, Any]) -> None:
	"""Record a Responses API request with both generic and Orq-specific attrs."""
	if span is None:
		return
	model = payload.get("model")
	if model:
		span.set_attribute("gen_ai.request.model", str(model))
	max_output_tokens = payload.get("max_output_tokens")
	if isinstance(max_output_tokens, int):
		span.set_attribute("gen_ai.request.max_tokens", max_output_tokens)
	if not _capture_message_content():
		return
	input_items = payload.get("input") or []
	serialized_input = truncate_for_span(
		json.dumps(input_items, ensure_ascii=False, default=str)
	)
	span.set_attribute("gen_ai.input.messages", serialized_input)
	span.set_attribute("input", serialized_input)
	span.set_attribute(
		"orq.openresponses.request",
		truncate_for_span(json.dumps(payload, ensure_ascii=False, default=str)),
	)


def record_openresponses_response(span: Span | None, response: Any) -> None:
	"""Record a Responses API response with standard gen_ai.* attributes."""
	if span is None:
		return
	record_llm_response(span, response)
	try:
		payload = (
			response.model_dump(mode="json")
			if hasattr(response, "model_dump")
			else response
		)
	except Exception as exc:
		logger.debug(
			"record_openresponses_response: model_dump failed ({}); falling back to repr",
			exc,
		)
		payload = repr(response)
	if _capture_message_content():
		span.set_attribute(
			"orq.openresponses.response",
			truncate_for_span(json.dumps(payload, ensure_ascii=False, default=str)),
		)
