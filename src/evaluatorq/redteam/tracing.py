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

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from evaluatorq.common.tracing import with_llm_span as _common_with_llm_span
from evaluatorq.tracing.setup import get_tracer

if TYPE_CHECKING:
	from collections.abc import AsyncGenerator

	from opentelemetry.trace import Span


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

	Delegates to ``evaluatorq.common.tracing.with_llm_span`` after mapping the
	redteam-specific ``orq.redteam.llm_purpose`` key onto the neutral
	``orq.llm.purpose`` key so cross-domain purpose queries include redteam spans.
	"""
	# Map the redteam-domain key onto the neutral key before passing attributes
	# to common. This is the only redteam-specific behaviour; common must not do it.
	resolved_attrs: dict[str, Any] = dict(attributes or {})
	redteam_purpose = resolved_attrs.get("orq.redteam.llm_purpose")
	if redteam_purpose is not None and "orq.llm.purpose" not in resolved_attrs:
		resolved_attrs["orq.llm.purpose"] = redteam_purpose

	async with _common_with_llm_span(
		model=model,
		operation=operation,
		provider=provider,
		temperature=temperature,
		max_tokens=max_tokens,
		input_messages=input_messages,
		attributes=resolved_attrs,
		parent_context=parent_context,
	) as span:
		yield span
