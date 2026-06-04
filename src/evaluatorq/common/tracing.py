"""Generic OTel span-recording utilities shared by all evaluatorq domains.

Domain-specific span builders (with_simulation_span, with_redteam_span,
with_llm_span) stay in their respective domain tracing modules and may import
from here. This module must not import from redteam, simulation, or openresponses.
"""

from __future__ import annotations

import functools
import json
import os
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.common.fields import get_field as _field

if TYPE_CHECKING:
	from opentelemetry.trace import Span

AttrValue = str | int | float | bool
AttrMap = dict[str, AttrValue | None]

_TRUNCATION_MARKER = "... [truncated]"
_DEFAULT_SPAN_MAX_TEXT_CHARS = 8192


@functools.lru_cache(maxsize=1)
def _default_span_max_text_chars() -> int | None:
	"""Read EVALUATORQ_SPAN_MAX_TEXT_CHARS once. Default 8192. Set 0 to disable.

	Call _default_span_max_text_chars.cache_clear() in tests after changing the env var.
	"""
	raw = os.getenv("EVALUATORQ_SPAN_MAX_TEXT_CHARS")
	if raw is None or raw == "":
		return _DEFAULT_SPAN_MAX_TEXT_CHARS
	try:
		value = int(raw)
	except ValueError:
		logger.warning(
			"EVALUATORQ_SPAN_MAX_TEXT_CHARS={!r} is not a valid int; using default {}",
			raw,
			_DEFAULT_SPAN_MAX_TEXT_CHARS,
		)
		return _DEFAULT_SPAN_MAX_TEXT_CHARS
	if value < 0:
		logger.warning(
			"EVALUATORQ_SPAN_MAX_TEXT_CHARS={!r} must be non-negative; using default {}",
			value,
			_DEFAULT_SPAN_MAX_TEXT_CHARS,
		)
		return _DEFAULT_SPAN_MAX_TEXT_CHARS
	return value


def truncate_for_span(text: object, *, max_chars: int | None = None) -> str:
	"""Truncate text for span attribute storage.

	Defaults to EVALUATORQ_SPAN_MAX_TEXT_CHARS env var (or 8192 if unset).
	Set 0 to disable truncation. Negative values raise ValueError.
	Output never exceeds max_chars; the marker is reserved within the budget.
	"""
	if isinstance(text, str):
		s = text
	else:
		try:
			s = str(text)
		except Exception as e:  # pragma: no cover  # noqa: BLE001
			s = f"<unrepresentable {type(text).__name__}: {e}>"
	if max_chars is None:
		max_chars = _default_span_max_text_chars()
	if max_chars is None or max_chars == 0:
		return s
	if max_chars < 0:
		raise ValueError(f"max_chars must be non-negative, got {max_chars}")
	if len(s) <= max_chars:
		return s
	marker_len = len(_TRUNCATION_MARKER)
	if max_chars <= marker_len:
		return _TRUNCATION_MARKER[:max_chars]
	return s[: max_chars - marker_len] + _TRUNCATION_MARKER


def capture_message_content() -> bool:
	"""Honor OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT.

	Defaults to True (matches TypeScript impl, RES-595). Set the env var to
	'false' to opt out when exporting to a third-party backend. Public so domain
	span builders (redteam/openresponses) can gate input-message capture too.
	"""
	flag = os.environ.get("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")
	if flag is None:
		return True
	return flag.lower() == "true" or flag == "1"


# Back-compat private alias (pre-existing callers/tests import the underscored name).
_capture_message_content = capture_message_content


def _serialize_messages(messages: list[dict[str, Any]]) -> str:
	return json.dumps(
		[
			{
				"role": str(m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")),
				"content": truncate_for_span(
					m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
				),
			}
			for m in messages
		],
		ensure_ascii=False,
	)


def _serialize_tool_call_content(tool_calls: list[dict[str, str]]) -> str:
	return json.dumps({"tool_calls": tool_calls}, ensure_ascii=False)


def _extract_chat_tool_call_payloads(tool_calls: Any) -> list[dict[str, str]]:
	payloads: list[dict[str, str]] = []
	for tool_call in tool_calls or []:
		function = _field(tool_call, "function")
		name = _field(function, "name") or _field(tool_call, "name")
		arguments = _field(function, "arguments") or _field(tool_call, "arguments")
		payload: dict[str, str] = {}
		if name:
			payload["name"] = str(name)
		if arguments is not None:
			payload["arguments"] = str(arguments)
		if payload:
			payloads.append(payload)
	return payloads


def _extract_response_tool_call_payloads(output_items: list[Any]) -> list[dict[str, str]]:
	payloads: list[dict[str, str]] = []
	for item in output_items:
		call_id = _field(item, "call_id")
		name = _field(item, "name")
		arguments = _field(item, "arguments")
		if call_id or name or arguments is not None:
			payload: dict[str, str] = {}
			if call_id:
				payload["call_id"] = str(call_id)
			if name:
				payload["name"] = str(name)
			if arguments is not None:
				payload["arguments"] = str(arguments)
			payloads.append(payload)
	return payloads


def _extract_output_messages(response: Any) -> list[dict[str, str]]:
	"""Extract output message dicts from Chat Completions or Responses API shape."""
	output_messages: list[dict[str, str]] = []
	choices = _field(response, "choices")
	if choices:
		for choice in choices:
			message = _field(choice, "message")
			content = _field(message, "content") if message else None
			if content:
				role = _field(message, "role") or "assistant"
				output_messages.append({"role": str(role), "content": str(content)})
				continue
			tool_payloads = _extract_chat_tool_call_payloads(
				_field(message, "tool_calls") if message else None
			)
			if tool_payloads:
				role = _field(message, "role") or "assistant"
				output_messages.append({
					"role": str(role),
					"content": _serialize_tool_call_content(tool_payloads),
				})
	else:
		output_text = _field(response, "output_text")
		if isinstance(output_text, str) and output_text:
			output_messages.append({"role": "assistant", "content": output_text})
		else:
			output_items = _field(response, "output") or []
			parts: list[str] = []
			for item in output_items:
				content = _field(item, "content")
				if content:
					for part in content:
						text = _field(part, "text")
						if isinstance(text, str) and text:
							parts.append(text)
				else:
					text = _field(item, "text")
					if isinstance(text, str) and text:
						parts.append(text)
			joined = "".join(parts)
			if joined:
				output_messages.append({"role": "assistant", "content": joined})
			else:
				tool_payloads = _extract_response_tool_call_payloads(output_items)
				if tool_payloads:
					output_messages.append({
						"role": "assistant",
						"content": _serialize_tool_call_content(tool_payloads),
					})
	return output_messages


def set_span_attrs(span: Span | None, attrs: AttrMap) -> None:
	"""Batch-set span attributes. Skips None values. Safe no-op when span is None."""
	if span is None:
		return
	for key, value in attrs.items():
		if value is not None:
			span.set_attribute(key, value)


def record_token_usage(
	span: Span | None,
	*,
	prompt_tokens: int | None = None,
	completion_tokens: int | None = None,
	total_tokens: int | None = None,
	calls: int = 0,
	cache_read_input_tokens: int | None = None,
	cache_creation_input_tokens: int | None = None,
) -> None:
	"""Record token usage on a span. Safe no-op when span is None.

	Superset of both former redteam and simulation impls: sets OTel GenAI
	attribute names, their aliases, bare keys, call count, and cache details.
	"""
	if span is None:
		return
	prompt = prompt_tokens if prompt_tokens is not None else 0
	completion = completion_tokens if completion_tokens is not None else 0
	total = total_tokens if total_tokens is not None else prompt + completion
	span.set_attribute("gen_ai.usage.input_tokens", prompt)
	span.set_attribute("gen_ai.usage.output_tokens", completion)
	span.set_attribute("gen_ai.usage.prompt_tokens", prompt)
	span.set_attribute("gen_ai.usage.completion_tokens", completion)
	span.set_attribute("gen_ai.usage.total_tokens", total)
	span.set_attribute("prompt_tokens", prompt)
	span.set_attribute("completion_tokens", completion)
	span.set_attribute("input_tokens", prompt)
	span.set_attribute("output_tokens", completion)
	span.set_attribute("total_tokens", total)
	if calls:
		span.set_attribute("gen_ai.usage.calls", calls)
		span.set_attribute("calls", calls)
	if cache_read_input_tokens is not None:
		span.set_attribute("gen_ai.usage.cache_read.input_tokens", cache_read_input_tokens)
		# Legacy attribute name emitted by the former redteam impl — kept for platform dashboard compat.
		span.set_attribute("gen_ai.usage.prompt_tokens_details.cached_tokens", cache_read_input_tokens)
	if cache_creation_input_tokens is not None:
		span.set_attribute("gen_ai.usage.cache_creation.input_tokens", cache_creation_input_tokens)


def record_llm_response(
	span: Span | None,
	response: Any,
	*,
	output_content: str | None = None,
) -> None:
	"""Record LLM response attributes on a span.

	Superset of both former impls: duck-typed (_field handles dicts + objects),
	handles Chat Completions and Responses API shapes, honors the PII capture
	gate, accepts an optional output_content override for backward compat with
	redteam callers that pass the output string explicitly.
	"""
	if span is None:
		return

	response_id = _field(response, "id")
	if response_id:
		span.set_attribute("gen_ai.response.id", response_id)
	response_model = _field(response, "model")
	if response_model:
		span.set_attribute("gen_ai.response.model", response_model)

	usage = _field(response, "usage")
	if usage is not None:
		prompt = _field(usage, "prompt_tokens")
		if prompt is None:
			prompt = _field(usage, "input_tokens")
		completion = _field(usage, "completion_tokens")
		if completion is None:
			completion = _field(usage, "output_tokens")
		total = _field(usage, "total_tokens")
		details = _field(usage, "prompt_tokens_details")
		if details is None:
			details = _field(usage, "input_tokens_details")
		cache_read = _field(details, "cached_tokens") if details else None
		cache_creation = _field(usage, "cache_creation_input_tokens")
		record_token_usage(
			span,
			prompt_tokens=prompt,
			completion_tokens=completion,
			total_tokens=total,
			cache_read_input_tokens=cache_read,
			cache_creation_input_tokens=cache_creation,
		)
		completion_details = _field(usage, "completion_tokens_details")
		if completion_details is not None:
			reasoning = _field(completion_details, "reasoning_tokens")
			if reasoning is not None:
				span.set_attribute(
					"gen_ai.usage.completion_tokens_details.reasoning_tokens",
					int(reasoning),
				)

	if _capture_message_content():
		if output_content is not None:
			serialized = json.dumps(
				[{"role": "assistant", "content": truncate_for_span(output_content)}],
				ensure_ascii=False,
			)
			span.set_attribute("gen_ai.output.messages", serialized)
			span.set_attribute("output", serialized)
		else:
			output_messages = _extract_output_messages(response)
			if output_messages:
				serialized = _serialize_messages(output_messages)
				span.set_attribute("gen_ai.output.messages", serialized)
				span.set_attribute("output", serialized)

	finish_reasons: list[str] = []
	choices = _field(response, "choices")
	if choices:
		for choice in choices:
			reason = _field(choice, "finish_reason")
			if reason:
				finish_reasons.append(reason)
	else:
		status = _field(response, "status")
		if isinstance(status, str) and status:
			finish_reasons.append(status)
	if finish_reasons:
		span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)


def record_llm_input(span: Span | None, messages: list[dict[str, Any]]) -> None:
	"""Record LLM input messages. Suppressed when capture gate is off."""
	if span is None or not messages:
		return
	if not _capture_message_content():
		return
	serialized = _serialize_messages(messages)
	span.set_attribute("gen_ai.input.messages", serialized)
	span.set_attribute("input", serialized)


def record_llm_output(span: Span | None, output: str) -> None:
	"""Record a single LLM output string. Suppressed when capture gate is off."""
	if span is None or not output:
		return
	if not _capture_message_content():
		return
	serialized = _serialize_messages([{"role": "assistant", "content": output}])
	span.set_attribute("gen_ai.output.messages", serialized)
	span.set_attribute("output", serialized)


async def get_trace_context_headers() -> dict[str, str]:  # noqa: RUF029
	"""Return W3C trace context headers for the current active span.

	Empty dict when OTel is not available. Used to propagate trace context
	into outgoing HTTP requests.
	"""
	try:
		from opentelemetry import context, propagate
	except ImportError:
		return {}
	headers: dict[str, str] = {}
	propagate.inject(headers, context=context.get_current())
	return headers
