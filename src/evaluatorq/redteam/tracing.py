"""Red teaming span utilities for OpenTelemetry instrumentation.

Domain-specific span builders (with_redteam_span, with_llm_span) live here.
Generic recording utilities are imported from evaluatorq.common.tracing.

Span hierarchy:
- orq.redteam.pipeline (root or child of parent context)      [runner.py]
  +-- orq.redteam.context_retrieval                            [runner.py]
  +-- orq.redteam.datapoint_generation                         [runner.py]
  |   +-- orq.redteam.capability_classification                [strategy_planner.py]
  |   |   +-- chat (llm_purpose=classify_tools)                [capability_classifier.py]
  |   |   +-- chat (llm_purpose=infer_resources)               [capability_classifier.py]
  |   +-- orq.redteam.strategy_planning                        [strategy_planner.py]
  |       +-- chat (llm_purpose=generate_strategies)           [objective_generator.py]
  +-- orq.job (framework)                                      [processings.py]
  |   +-- orq.redteam.attack                                   [pipeline.py]
  |   |   +-- orq.redteam.target_call                          [pipeline.py]
  |   |   |   +-- agent <key> (llm_purpose=target)             [orq.py]
  |   |   |   +-- chat (llm_purpose=target)                    [openai.py]
  |   |   +-- orq.redteam.attack_turn x N                      [orchestrator.py]
  |   |       +-- orq.redteam.adversarial_generation           [orchestrator.py]
  |   |       |   +-- chat (llm_purpose=adversarial)           [orchestrator.py]
  |   |       +-- orq.redteam.target_call                      [orchestrator.py]
  |   |           +-- agent <key> (llm_purpose=target)         [orq.py]
  |   |           +-- chat (llm_purpose=target)                [openai.py]
  |   +-- orq.evaluation (framework)                           [processings.py]
  |       +-- orq.redteam.security_evaluation                  [pipeline.py]
  |           +-- chat (llm_purpose=evaluation)                [evaluator.py]
  +-- orq.redteam.memory_cleanup                               [runner.py]
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.common.tracing import capture_message_content, truncate_for_span
from evaluatorq.tracing.setup import get_tracer

if TYPE_CHECKING:
	from collections.abc import AsyncGenerator

	from opentelemetry.trace import Span


def _derive_provider(model: str) -> str:
	# NOTE: intentionally does NOT expand provider aliases (e.g. azure ->
	# azure.ai.openai) the way simulation/openresponses do, so `azure/gpt-4o`
	# yields gen_ai.system="azure" on redteam spans. Pre-existing divergence
	# tracked separately; aligning would itself be an observability delta.
	if "/" in model:
		return model.split("/", 1)[0]
	return "openai"


@asynccontextmanager
async def with_redteam_span(  # noqa: RUF029
	name: str,
	attributes: dict[str, Any] | None = None,
	parent_context: Any | None = None,
) -> AsyncGenerator[Span | None, None]:
	"""Execute code within a red teaming span (SpanKind.INTERNAL).

	Yields the span when tracing is enabled, None otherwise.
	Exceptions propagate and are recorded on the span with ERROR status.
	"""
	tracer = get_tracer()
	if tracer is None:
		yield None
		return

	try:
		from opentelemetry import context as otel_context
		from opentelemetry.trace import SpanKind, Status, StatusCode
	except ImportError:
		yield None
		return

	ctx = parent_context or otel_context.get_current()

	with tracer.start_as_current_span(
		name,
		context=ctx,
		kind=SpanKind.INTERNAL,
		attributes=attributes or {},
	) as span:
		try:
			yield span
			span.set_status(Status(StatusCode.OK))
		except BaseException as e:
			span.set_attribute("error.type", type(e).__name__)
			span.set_status(Status(StatusCode.ERROR, str(e)))
			span.record_exception(e)
			raise


@asynccontextmanager
async def with_llm_span(  # noqa: RUF029
	*,
	model: str,
	operation: str = "chat",
	provider: str | None = None,
	temperature: float | None = None,
	max_tokens: int | None = None,
	input_messages: list[Any] | None = None,
	attributes: dict[str, Any] | None = None,
	parent_context: Any | None = None,
) -> AsyncGenerator[Span | None, None]:
	"""Execute code within a GenAI LLM span (SpanKind.CLIENT).

	Span name is "{operation} {model}". input_messages are serialized to
	gen_ai.input.messages and input attributes.
	"""
	tracer = get_tracer()
	if tracer is None:
		yield None
		return

	try:
		from opentelemetry import context as otel_context
		from opentelemetry.trace import SpanKind, Status, StatusCode
	except ImportError:
		yield None
		return

	ctx = parent_context or otel_context.get_current()
	resolved_provider = provider or _derive_provider(model)
	span_name = f"{operation} {model}"

	genai_attrs: dict[str, Any] = {
		"gen_ai.operation.name": operation,
		"gen_ai.system": resolved_provider,
		"gen_ai.provider.name": resolved_provider,
		"gen_ai.request.model": model,
	}
	if temperature is not None:
		genai_attrs["gen_ai.request.temperature"] = float(temperature)
	if max_tokens is not None:
		genai_attrs["gen_ai.request.max_tokens"] = max_tokens
	if input_messages is not None and capture_message_content():
		serialized = json.dumps(
			_sanitize_messages(input_messages), ensure_ascii=False
		)
		genai_attrs["gen_ai.input.messages"] = serialized
		genai_attrs["input"] = serialized
	if attributes:
		genai_attrs.update(attributes)
	# Dual-emit the domain-neutral purpose key (parity with simulation/openresponses
	# with_llm_span) so cross-domain `orq.llm.purpose` queries include redteam spans.
	# Redteam callers pass purpose via the open `attributes` dict as
	# `orq.redteam.llm_purpose`; mirror it onto `orq.llm.purpose` when not already set.
	redteam_purpose = genai_attrs.get("orq.redteam.llm_purpose")
	if redteam_purpose is not None and "orq.llm.purpose" not in genai_attrs:
		genai_attrs["orq.llm.purpose"] = redteam_purpose

	with tracer.start_as_current_span(
		span_name,
		context=ctx,
		kind=SpanKind.CLIENT,
		attributes=genai_attrs,
	) as span:
		try:
			yield span
			span.set_status(Status(StatusCode.OK))
		except BaseException as e:
			span.set_attribute("error.type", type(e).__name__)
			span.set_status(Status(StatusCode.ERROR, str(e)))
			span.record_exception(e)
			raise


def _sanitize_messages(messages: list[Any]) -> list[dict[str, str]]:
	"""JSON-safe list of {role, content} for gen_ai.input.messages."""
	sanitized: list[dict[str, str]] = []
	for msg in messages:
		if hasattr(msg, "get") and callable(msg.get):
			role = msg.get("role", "")
			content = msg.get("content", "")
		else:
			role = getattr(msg, "role", "")
			content = getattr(msg, "content", "")
		sanitized.append({"role": str(role), "content": truncate_for_span(content)})
	return sanitized
